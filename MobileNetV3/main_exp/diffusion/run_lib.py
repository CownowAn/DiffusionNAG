import torch
import numpy as np
import sys
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
sys.path.append('.')
import sampling

import datasets_nas
from models import pgsn
from models import digcn
from models import cate
from models import dagformer
from models import digcn
from models import digcn_meta
from models import regressor
from models.GDSS import scorenetx
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import sde_lib
from utils import *
import losses

from analysis.arch_functions import BasicArchMetricsOFA
import losses
from analysis.arch_functions import NUM_STAGE, MAX_LAYER_PER_STAGE
from all_path import *


def get_sampling_fn(config, p=1, prod_w=False, weight_ratio_abs=False):
    # Setup SDEs
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
    
    # create data normalizer and its inverse
    inverse_scaler = datasets_nas.get_data_inverse_scaler(config)
    
    sampling_shape = (
        config.eval.batch_size, config.data.max_node, config.data.n_vocab) # ofa: 1024, 20, 28
    sampling_fn = sampling.get_sampling_fn(
        config, sde, sampling_shape, inverse_scaler, 
        sampling_eps, config.data.name, conditional=True, 
        p=p, prod_w=prod_w, weight_ratio_abs=weight_ratio_abs)
    
    return sampling_fn, sde


def get_sampling_fn_meta(config, p=1, prod_w=False, weight_ratio_abs=False, init=False, n_init=5):
    # Setup SDEs
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
    
    # create data normalizer and its inverse
    inverse_scaler = datasets_nas.get_data_inverse_scaler(config)
    
    if init:
        sampling_shape = (
            n_init, config.data.max_node, config.data.n_vocab) 
    else:
        sampling_shape = (
            config.eval.batch_size, config.data.max_node, config.data.n_vocab) # ofa: 1024, 20, 28
    sampling_fn = sampling.get_sampling_fn(
        config, sde, sampling_shape, inverse_scaler, 
        sampling_eps, config.data.name, conditional=True, 
        is_meta=True, data_name=config.sampling.check_dataname, 
        num_sample=config.model.num_sample)
    
    return sampling_fn, sde


def get_score_model(config, pos_enc_type=2):
    # Build sampling functions and Load pre-trained score network & predictor network
    score_config = torch.load(config.scorenet_ckpt_path)['config']
    ckpt_path = config.scorenet_ckpt_path
    score_config.sampling.corrector = 'langevin'
    score_config.model.pos_enc_type = pos_enc_type

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


def get_predictor(config):
    classifier_model = mutils.create_model(config)

    return classifier_model 


def get_adj(data_name, except_inout):
    if data_name == 'NASBench201':
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
        if except_inout:
            _adj = _adj[1:-1, 1:-1]
    elif data_name == 'ofa':
        assert except_inout 
        num_nodes = NUM_STAGE * MAX_LAYER_PER_STAGE
        _adj = torch.zeros(num_nodes, num_nodes)
        for i in range(num_nodes-1):
            _adj[i, i+1] = 1
        return _adj
    return _adj    

def generate_archs(
        config, sampling_fn, score_model, score_ema, classifier_model,
        num_samples, patient_factor, batch_size=512, classifier_scale=None,
        task=None):
    
    metrics = BasicArchMetricsOFA()
    # algo = 'none'
    adj_s = get_adj(config.data.name, config.data.except_inout)
    mask_s = aug_mask(adj_s, algo=config.data.aug_mask_algo)[0]
    adj_c = get_adj(config.data.name, config.data.except_inout)
    mask_c = aug_mask(adj_c, algo=config.data.aug_mask_algo)[0]
    assert (adj_s == adj_c).all() and (mask_s == mask_c).all()
    adj_s, mask_s, adj_c, mask_c = \
        adj_s.to(config.device), mask_s.to(config.device), adj_c.to(config.device), mask_c.to(config.device)
    
    # Generate and save samples
    score_ema.copy_to(score_model.parameters())
    if num_samples > batch_size:
        num_sampling_rounds = int(np.ceil(num_samples / batch_size) * patient_factor)
    else:
        num_sampling_rounds = int(patient_factor)
    print(f'==> Sampling for {num_sampling_rounds} rounds...')
    
    r = 0
    all_samples = []
    classifier_scales = list(range(100000, 0, -int(classifier_scale)))
    
    while True and r < num_sampling_rounds:
        classifier_scale = classifier_scales[r]
        print(f'==> round {r} classifier_scale {classifier_scale}')
        sample, _, sample_chain, (score_grad_norm_p, classifier_grad_norm_p, score_grad_norm_c, classifier_grad_norm_c) \
            = sampling_fn(score_model, mask_s, classifier_model, 
                        eval_chain=True, 
                        number_chain_steps=config.sampling.number_chain_steps,
                        classifier_scale=classifier_scale,
                        task=task, sample_bs=num_samples)
        try:
            sample_list = quantize(sample, adj_s) # quantization
            _, validity, valid_arch_str, _, _ = metrics.compute_validity(sample_list, adj_s, mask_s)
        except:
            import pdb; pdb.set_trace()
            validity = 0.
            valid_arch_str = []
        print(f' ==> [Validity]: {round(validity, 4)}')

        if len(valid_arch_str) > 0:
            all_samples += valid_arch_str
        print(f' ==> [# Unique Arch]: {len(set(all_samples))}')
        
        if (len(set(all_samples)) >= num_samples):
            break

        r += 1
    
    return list(set(all_samples))[:num_samples]


def noise_aware_meta_predictor_fit(config, 
                              predictor_model=None,
                              xtrain=None, 
                              seed=None, 
                              sde=None, 
                              batch_size=5, 
                              epochs=50,
                              save_best_p_corr=False,
                              save_path=None,):
    assert save_best_p_corr
    reset_seed(seed)

    data_loader = DataLoader(xtrain, 
                             batch_size=batch_size, 
                             shuffle=True, 
                             drop_last=True)

    # create data normalizer and its inverse
    scaler = datasets_nas.get_data_scaler(config)

    # Initialize model.
    optimizer = losses.get_optimizer(config, predictor_model.parameters())
    state = dict(optimizer=optimizer, 
                 model=predictor_model, 
                 step=0, 
                 config=config)

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    train_step_fn = losses.get_step_fn_predictor(sde, train=True, optimize_fn=optimize_fn,
                                    reduce_mean=reduce_mean, continuous=continuous,
                                    likelihood_weighting=likelihood_weighting,
                                    data=config.data.name, label_list=config.data.label_list, 
                                    noised=config.training.noised,
                                    t_spot=config.training.t_spot,
                                    is_meta=True)

    # temp 
    # epochs = len(xtrain) * 100
    is_best = False
    best_p_corr = -1
    ckpt_dir = os.path.join(save_path, 'loop')
    print(f'==> Training for {epochs} epochs')
    for epoch in range(epochs):
        pred_list, labels_list = list(), list()
        for step, batch in enumerate(data_loader):
            x = batch['x'].to(config.device) # (5, 5, 20, 9)???
            adj = get_adj(config.data.name, config.data.except_inout)
            task = batch['task']
            extra = batch
            mask = aug_mask(adj, 
                            algo=config.data.aug_mask_algo, 
                            data=config.data.name)
            x = scaler(x.to(config.device))
            adj = adj.to(config.device)
            mask = mask.to(config.device)
            task = task.to(config.device)
            batch = (x, adj, mask, extra, task)
            # Execute one training step
            loss, pred, labels = train_step_fn(state, batch)
            pred_list += [v.detach().item() for v in pred.squeeze()]
            labels_list += [v.detach().item() for v in labels.squeeze()]
        p_corr = pearsonr(np.array(pred_list), np.array(labels_list))[0]
        s_corr = spearmanr(np.array(pred_list), np.array(labels_list))[0]
        if epoch % 50 == 0: print(f'==> [Epoch-{epoch}] P corr: {round(p_corr, 4)} | S corr: {round(s_corr, 4)}')

        if save_best_p_corr:
            if p_corr > best_p_corr:
                is_best = True
                best_p_corr = p_corr
                os.makedirs(ckpt_dir, exist_ok=True)
                save_checkpoint(ckpt_dir, state, epoch, is_best)
    if save_best_p_corr:
        loaded_state = torch.load(os.path.join(ckpt_dir, 'model_best.pth.tar'), map_location=config.device)
        predictor_model.load_state_dict(loaded_state['model'])


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
