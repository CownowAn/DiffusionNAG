import torch
import sys
import numpy as np
import random

sys.path.append('.')
import sampling
import datasets_nas
from models import cate
from models import digcn
from models import digcn_meta
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import sde_lib
from utils import *
from analysis.arch_functions import BasicArchMetricsMeta
from all_path import *


def get_sampling_fn_meta(config):
    ## Set SDE
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(
            beta_min=config.model.beta_min, 
            beta_max=config.model.beta_max, 
            N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(
            beta_min=config.model.beta_min, 
            beta_max=config.model.beta_max,
            N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(
            sigma_min=config.model.sigma_min, 
            sigma_max=config.model.sigma_max,
            N=config.model.num_scales)
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    ## Get data normalizer inverse
    inverse_scaler = datasets_nas.get_data_inverse_scaler(config)

    ## Get sampling function
    sampling_shape = (config.eval.batch_size, config.data.max_node, config.data.n_vocab)
    sampling_fn = sampling.get_sampling_fn(
        config=config, 
        sde=sde, 
        shape=sampling_shape,
        inverse_scaler=inverse_scaler, 
        eps=sampling_eps, 
        conditional=True,
        data_name=config.sampling.check_dataname, 
        num_sample=config.model.num_sample)

    return sampling_fn, sde


def get_score_model(config):
    try:
        score_config = torch.load(config.scorenet_ckpt_path)['config']
        ckpt_path = config.scorenet_ckpt_path
    except:
        config.scorenet_ckpt_path = SCORENET_CKPT_PATH
        score_config = torch.load(config.scorenet_ckpt_path)['config']
        ckpt_path = config.scorenet_ckpt_path

    score_model = mutils.create_model(score_config)
    score_ema = ExponentialMovingAverage(
        score_model.parameters(), decay=score_config.model.ema_rate)
    score_state = dict(
        model=score_model, ema=score_ema, step=0, config=score_config)
    score_state = restore_checkpoint(
        ckpt_path, score_state, 
        device=config.device, resume=True)
    score_ema.copy_to(score_model.parameters())
    return score_model, score_ema, score_config


def get_surrogate(config):
    surrogate_model = mutils.create_model(config)
    return surrogate_model


def get_adj(except_inout=False):
    _adj = np.asarray(
            [[0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0]]
        )
    _adj = torch.tensor(_adj, dtype=torch.float32, device=torch.device('cpu'))
    if except_inout: _adj = _adj[1:-1, 1:-1]
    return _adj


def generate_archs_meta(
        config, 
        sampling_fn, 
        score_model, 
        score_ema, 
        meta_surrogate_model,
        num_samples,
        args=None,
        task=None,
        patient_factor=20,
        batch_size=256,):

    metrics = BasicArchMetricsMeta()

    ## Get the adj and mask
    adj_s = get_adj()
    mask_s = aug_mask(adj_s)[0]
    adj_c = get_adj()
    mask_c = aug_mask(adj_c)[0]
    assert (adj_s == adj_c).all() and (mask_s == mask_c).all()
    adj_s, mask_s, adj_c, mask_c = \
        adj_s.to(config.device), mask_s.to(config.device), adj_c.to(config.device), mask_c.to(config.device)

    score_ema.copy_to(score_model.parameters())
    score_model.eval()
    meta_surrogate_model.eval()
    c_scale = args.classifier_scale

    num_sampling_rounds = int(np.ceil(num_samples / batch_size) * patient_factor) if num_samples > batch_size else int(patient_factor)
    round = 0
    all_samples = []
    while True and round < num_sampling_rounds:
        round += 1
        sample = sampling_fn(score_model, 
                             mask_s, 
                             meta_surrogate_model,
                             classifier_scale=c_scale,
                             task=task)
        quantized_sample = quantize(sample)
        _, _, valid_arch_str, _ = metrics.compute_validity(quantized_sample)
        if len(valid_arch_str) > 0: all_samples += valid_arch_str
        # to sample various architectures
        c_scale -= args.scale_step
        seed = int(random.random() * 10000)
        reset_seed(seed)
        # stop sampling if we have enough samples
        if (len(set(all_samples)) >= num_samples):
            break

    return list(set(all_samples))


def save_checkpoint(ckpt_dir, state, epoch, is_best):
    saved_state = {}
    for k in state:
        if k in ['optimizer', 'model', 'ema']:
            saved_state.update({k: state[k].state_dict()})
        else:
            saved_state.update({k: state[k]})
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(saved_state, os.path.join(ckpt_dir, f'checkpoint_{epoch}.pth.tar'))
    if is_best:
        shutil.copy(os.path.join(ckpt_dir, f'checkpoint_{epoch}.pth.tar'), os.path.join(ckpt_dir, 'model_best.pth.tar'))
    
    # remove the ckpt except is_best state
    for ckpt_file in sorted(os.listdir(ckpt_dir)):
        if not ckpt_file.startswith('checkpoint'):
            continue
        if os.path.join(ckpt_dir, ckpt_file) != os.path.join(ckpt_dir, 'model_best.pth.tar'):
            os.remove(os.path.join(ckpt_dir, ckpt_file))


def restore_checkpoint(ckpt_dir, state, device, resume=False):
    if not resume:
        os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
        return state
    elif not os.path.exists(ckpt_dir):
        if not os.path.exists(os.path.dirname(ckpt_dir)):
            os.makedirs(os.path.dirname(ckpt_dir))
        logging.warning(f"No checkpoint found at {ckpt_dir}. "
                        f"Returned the same state as input")
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        for k in state:
            if k in ['optimizer', 'model', 'ema']:
                state[k].load_state_dict(loaded_state[k])
            else:
                state[k] = loaded_state[k]
        return state


def floyed(r):
    """
    :param r: a numpy NxN matrix with float 0,1
    :return: a numpy NxN matrix with float 0,1
    """
    if type(r) == torch.Tensor:
        r = r.cpu().numpy()
    N = r.shape[0]
    for k in range(N):
        for i in range(N):
            for j in range(N):
                if r[i, k] > 0 and r[k, j] > 0:
                    r[i, j] = 1
    return r


def aug_mask(adj, algo='floyed', data='NASBench201'):
    if len(adj.shape) == 2:
        adj = adj.unsqueeze(0)
    
    if data.lower() in ['nasbench201', 'ofa']:
        assert len(adj.shape) == 3
        r = adj[0].clone().detach()
        if algo == 'long_range':
            mask_i = torch.from_numpy(long_range(r)).float().to(adj.device)
        elif algo == 'floyed':
            mask_i = torch.from_numpy(floyed(r)).float().to(adj.device)
        else:
            mask_i = r
        masks = [mask_i] * adj.size(0)
        return torch.stack(masks)
    else:
        masks = []
        for r in adj:
            if algo == 'long_range':
                mask_i = torch.from_numpy(long_range(r)).float().to(adj.device)
            elif algo == 'floyed':
                mask_i = torch.from_numpy(floyed(r)).float().to(adj.device)
            else:
                mask_i = r
            masks.append(mask_i)
        return torch.stack(masks)


def long_range(r):
    """
    :param r: a numpy NxN matrix with float 0,1
    :return: a numpy NxN matrix with float 0,1
    """
    # r = np.array(r)
    if type(r) == torch.Tensor:
        r = r.cpu().numpy()
    N = r.shape[0]
    for j in range(1, N):
        col_j = r[:, j][:j]
        in_to_j = [i for i, val in enumerate(col_j) if val > 0]
        if len(in_to_j) > 0:
            for i in in_to_j:
                col_i = r[:, i][:i]
                in_to_i = [i for i, val in enumerate(col_i) if val > 0]
                if len(in_to_i) > 0:
                    for k in in_to_i:
                        r[k, j] = 1
    return r


def quantize(x):
    """Covert the PyTorch tensor x, adj matrices to numpy array.

    Args:
        x: [Batch_size, Max_node, N_vocab]
    """
    x_list = []

    # discretization
    x[x >= 0.5] = 1.
    x[x < 0.5] = 0.

    for i in range(x.shape[0]):
        x_tmp = x[i]
        x_tmp = x_tmp.cpu().numpy()
        x_list.append(x_tmp)

    return x_list


def reset_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True