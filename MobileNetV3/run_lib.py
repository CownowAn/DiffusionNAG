import os
import torch
import numpy as np
import random
import logging
import time

from absl import flags

from torch_geometric.loader import DataLoader 
import pickle
from scipy.stats import pearsonr, spearmanr
import wandb
import pandas as pd
import torch
from torch.utils.data import DataLoader #, Subset

from models import pgsn
from models import cate
from models import dagformer
from models import digcn
from models import digcn_meta
from models import regressor
from models.GDSS import scorenetx
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets_nas
import sde_lib
from utils import *
from logger import Logger
from analysis.arch_metrics import SamplingArchMetrics, SamplingArchMetricsMeta

FLAGS = flags.FLAGS


def set_exp_name(config, classifier_config_nf=None):
    exp_name = f'./exp/{config.task}/{config.folder_name}'
    wandb_exp_name =  exp_name
        
    os.makedirs(exp_name, exist_ok=True)

    config.exp_name = exp_name
    
    set_random_seed(config)
    
    return exp_name, wandb_exp_name


def set_random_seed(config):
    seed = config.seed
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sde_train(config):
    """Runs the training pipeline.

    Args:
        config: Configuration to use.
        workdir: Working directory for checkpoints and TF summaries.
            If this contains checkpoint training will be resumed from the latest checkpoint.
    """
    # Wandb logger
    exp_name, wandb_exp_name = set_exp_name(config)
    wandb_logger = Logger(
        log_dir=exp_name,
        exp_name=wandb_exp_name,
        write_textfile=True,
        use_wandb=config.log.use_wandb,
        wandb_project_name=config.log.wandb_project_name)
    wandb_logger.update_config(config, is_args=True)
    wandb_logger.write_str(str(vars(config)))
    wandb_logger.write_str('-' * 100)

    # Create directories for experimental logs
    sample_dir = os.path.join(exp_name, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    # Initialize model.
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0, config=config)

    # Create checkpoints directly
    checkpoint_dir = os.path.join(exp_name, "checkpoints")
    # Intermediate checkpoints to resume training
    checkpoint_meta_dir = os.path.join(exp_name, "checkpoints-meta", "checkpoint.pth")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_meta_dir), exist_ok=True)
    # Resume training when intermediate checkpoints are detected
    if config.resume:
        state = restore_checkpoint(config.resume_ckpt_path, state, config.device, resume=config.resume)
    initial_step = int(state['step'])

    train_ds, eval_ds, test_ds = datasets_nas.get_dataset(config)
    train_loader, eval_loader, test_loader = datasets_nas.get_dataloader(config, train_ds, eval_ds, test_ds)
    n_node_pmf = None # temp
    print(f'==> # of training elem: {len(train_ds)}')
    train_iter = iter(train_loader)
    # create data normalizer and its inverse
    scaler = datasets_nas.get_data_scaler(config)
    inverse_scaler = datasets_nas.get_data_inverse_scaler(config)

    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                       reduce_mean=reduce_mean, continuous=continuous,
                                       likelihood_weighting=likelihood_weighting,
                                       data=config.data.name)
    eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                      reduce_mean=reduce_mean, continuous=continuous,
                                      likelihood_weighting=likelihood_weighting,
                                      data=config.data.name)

    # Build sampling functions
    if config.training.snapshot_sampling:
        sampling_shape = (config.training.eval_batch_size, config.data.max_node, config.data.n_vocab)
        sampling_fn = sampling.get_sampling_fn(
            config, sde, sampling_shape, inverse_scaler, sampling_eps, config.data.name)

    num_train_steps = config.training.n_iters

    # Build analysis tools
    sampling_metrics = SamplingArchMetrics(config, train_ds, exp_name)
    # visualization_tools =  ArchVisualization(config, remove_none=False, exp_name=exp_name)
    
    # -------- Train --------- #
    logging.info("Starting training loop at step %d." % (initial_step,))
    element = {'train': ['training_loss'],
                'eval': ['eval_loss'],
                'test': ['test_loss'],
                'sample': ['r_valid', 'r_unique', 'r_novel'],
                'valid_error': ['multi_node_type', 'INVALID_1OR2', 'INVALID_3AND4', 'x_elem_sum']}

    is_best = False
    min_test_loss = 1e05
    for step in range(initial_step, num_train_steps + 1):
        try:
            x, adj, extra = next(train_iter)
        except StopIteration:
            train_iter = train_loader.__iter__()
            x, adj, extra = next(train_iter)
        mask = aug_mask(adj, algo=config.data.aug_mask_algo, data=config.data.name)
        x, adj, mask = scaler(x.to(config.device)), adj.to(config.device), mask.to(config.device)
        # mask = cate_mask(adj)
        # adj, mask = dense_adj(graphs, config.data.max_node, scaler, config.data.dequantization)
        batch = (x, adj, mask)
        # Execute one training step
        loss = train_step_fn(state, batch)
        wandb_logger.update(key="training_loss", v=loss.item())
        if step % config.training.log_freq == 0:
            logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))

        # Report the loss on evaluation dataset periodically
        if step % config.training.eval_freq == 0:
            for eval_x, eval_adj, eval_extra in eval_loader:
                eval_mask = aug_mask(eval_adj, algo=config.data.aug_mask_algo, data=config.data.name)
                eval_x, eval_adj, eval_mask = scaler(eval_x.to(config.device)), eval_adj.to(config.device), eval_mask.to(config.device)
                eval_batch = (eval_x, eval_adj, eval_mask)
                eval_loss = eval_step_fn(state, eval_batch)
                logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss.item()))
                wandb_logger.update(key="eval_loss", v=eval_loss.item())
            for test_x, test_adj, test_extra in test_loader:
                test_mask = aug_mask(test_adj, algo=config.data.aug_mask_algo, data=config.data.name)
                test_x, test_adj, test_mask = scaler(test_x.to(config.device)), test_adj.to(config.device), test_mask.to(config.device)
                test_batch = (test_x, test_adj, test_mask)
                test_loss = eval_step_fn(state, test_batch)
                logging.info("step: %d, test_loss: %.5e" % (step, test_loss.item()))
                wandb_logger.update(key="test_loss", v=test_loss.item())

            if wandb_logger.logs['test_loss'].avg < min_test_loss:
                is_best = True

        # Save a checkpoint periodically and generate samples
        if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
            # Save the checkpoint.
            save_step = step // config.training.snapshot_freq
            # save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)
            save_checkpoint(checkpoint_dir, state, step, save_step, is_best)

            # Generate and save samples
            if config.training.snapshot_sampling:
                ema.store(score_model.parameters())
                ema.copy_to(score_model.parameters())

                sample, sample_steps, _ = sampling_fn(score_model, mask) # sample: [batch_size, num_node, n_vocab]
                sample_list = quantize(sample, adj, 
                    alpha=config.sampling.alpha, qtype=config.sampling.qtype) # quantization
                this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
                os.makedirs(this_sample_dir, exist_ok=True)
                # check samples
                arch_metric = sampling_metrics(arch_list=sample_list, adj=adj, mask=mask, this_sample_dir=this_sample_dir, test=False)
                r_valid, r_unique, r_novel = arch_metric[0][0], arch_metric[0][1],  arch_metric[0][2]
                if len(arch_metric[0]) > 3:
                    error_type_1 = arch_metric[0][3]
                    error_type_2 = arch_metric[0][4]
                    error_type_3 = arch_metric[0][5]
                    x_elem_sum = int(torch.sum(torch.tensor(sample_list)))
                else:
                    error_type_1 = None

                logging.info("step: %d, r_valid: %.5e" % (step, r_valid))
                logging.info("step: %d, r_unique: %.5e" % (step, r_unique))
                logging.info("step: %d, r_novel: %.5e" % (step, r_novel))
                if error_type_1 is not None:
                    logging.info("step: %d, multi_node_type: %.5e" % (step, error_type_1))
                    logging.info("step: %d, INVALID_1OR2: %.5e" % (step, error_type_2))
                    logging.info("step: %d, INVALID_3AND4: %.5e" % (step, error_type_3))
                    logging.info("step: %d, x_elem_sum: %d" % (step, x_elem_sum))
                # writer.add_scalar("r_valid", r_valid, step)
                # res = nasbench201.get_prop(sample_valid_str_list=sample_valid_str)
                if config.log.use_wandb:
                    # wandb_logger.log_sample(sample)
                    wandb_logger.update(key="r_valid", v=r_valid)
                    wandb_logger.update(key="r_unique", v=r_unique)
                    wandb_logger.update(key="r_novel", v=r_novel)
                    if error_type_1 is not None:
                        wandb_logger.update(key="multi_node_type", v=error_type_1)
                        wandb_logger.update(key="INVALID_1OR2", v=error_type_2)
                        wandb_logger.update(key="INVALID_3AND4", v=error_type_3)
                        wandb_logger.update(key="x_elem_sum", v=x_elem_sum)
                    if config.log.log_valid_sample_prop:
                        wandb_logger.log_valid_sample_prop(arch_metric, x_axis='latency', y_axis='test_acc')

        if step % config.training.eval_freq == 0:
            wandb_logger.write_log(element=element, step=step)
        else:
            wandb_logger.write_log(element={'train': ['training_loss']}, step=step)
        wandb_logger.reset()
    wandb_logger.save_log()


def meta_predictor_train(config):
    
    # Wandb logger
    exp_name, wandb_exp_name = set_exp_name(config)
    wandb_logger = Logger(
        log_dir=exp_name,
        exp_name=wandb_exp_name,
        write_textfile=True,
        use_wandb=config.log.use_wandb,
        wandb_project_name=config.log.wandb_project_name)
    wandb_logger.update_config(config, is_args=True)
    wandb_logger.write_str(str(vars(config)))
    wandb_logger.write_str('-' * 100)

    # Create directories for experimental logs
    sample_dir = os.path.join(exp_name, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    # Initialize model.
    predictor_model = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, predictor_model.parameters())
    state = dict(optimizer=optimizer, model=predictor_model, step=0, config=config)
    # Create checkpoints directly

    checkpoint_dir = os.path.join(exp_name, "checkpoints")
    # Intermediate checkpoints to resume training
    checkpoint_meta_dir = os.path.join(exp_name, "checkpoints-meta", "checkpoint.pth")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_meta_dir), exist_ok=True)
    # Resume training when intermediate checkpoints are detected
    state = restore_checkpoint(checkpoint_meta_dir, state, config.device, resume=config.resume)
    initial_step = int(state['step'])

    # Build dataloader and iterators
    train_ds, eval_ds, test_ds = datasets_nas.get_meta_dataset(config)
    train_loader, eval_loader, test_loader = datasets_nas.get_dataloader(config, train_ds, eval_ds, test_ds)

    train_iter = iter(train_loader)
    # create data normalizer and its inverse
    scaler = datasets_nas.get_data_scaler(config)
    inverse_scaler = datasets_nas.get_data_inverse_scaler(config)

    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

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
    eval_step_fn = losses.get_step_fn_predictor(sde, train=False, optimize_fn=optimize_fn,
                                      reduce_mean=reduce_mean, continuous=continuous,
                                      likelihood_weighting=likelihood_weighting,
                                      data=config.data.name, label_list=config.data.label_list, 
                                      noised=config.training.noised,
                                      t_spot=config.training.t_spot,
                                      is_meta=True)

    # Build sampling functions and Load pre-trained score network
    if config.training.snapshot_sampling:
        sampling_shape = (config.training.eval_batch_size, config.data.max_node, config.data.n_vocab)
        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, 
                                               sampling_eps, config.data.name, conditional=True, 
                                               is_meta=True, data_name='cifar10', num_sample=config.model.num_sample)
        # Score model
        score_config = torch.load(config.scorenet_ckpt_path)['config']
        check_config(score_config, config)
        score_model = mutils.create_model(score_config)
        score_ema = ExponentialMovingAverage(score_model.parameters(), decay=score_config.model.ema_rate)
        score_state = dict(model=score_model, ema=score_ema, step=0, config=score_config)
        score_state = restore_checkpoint(config.scorenet_ckpt_path, score_state, device=config.device, resume=True)
        score_ema.copy_to(score_model.parameters())

    num_train_steps = config.training.n_iters

    # Build analysis tools
    sampling_metrics = SamplingArchMetricsMeta(config, train_ds, exp_name)

    # -------- Train --------- #
    logging.info("Starting training loop at step %d." % (initial_step,))
    element = {'train': ['training_loss'],
                'eval': ['eval_loss']}

    is_best = False
    max_eval_p_corr = -1
    for step in range(initial_step, num_train_steps + 1):
        try:
            x, adj, extra, task = next(train_iter)
        except StopIteration:
            train_iter = train_loader.__iter__()
            x, adj, extra, task = next(train_iter)
        mask = aug_mask(adj, algo=config.data.aug_mask_algo, data=config.data.name)
        x, adj, mask = scaler(x.to(config.device)), adj.to(config.device), mask.to(config.device)
        # task = task.to(config.device) if config.data.name == 'NASBench201' else [_.to(config.device) for _ in task]
        task = [_.to(config.device) for _ in task] if config.data.name == 'ofa' else task.to(config.device)
        # mask = cate_mask(adj)
        # adj, mask = dense_adj(graphs, config.data.max_node, scaler, config.data.dequantization)
        batch = (x, adj, mask, extra, task)
        # Execute one training step
        loss, pred, labels = train_step_fn(state, batch)
        wandb_logger.update(key="training_loss", v=loss.item())
        if step % config.training.log_freq == 0:
            logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))

        # Save a temporary checkpoint to resume training after pre-emption periodically
        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
            save_checkpoint(checkpoint_meta_dir, state, step, save_step, is_best)

        # Report the loss on evaluation dataset periodically
        if step % config.training.eval_freq == 0:
            eval_pred_list, eval_labels_list = list(), list()
            for eval_x, eval_adj, eval_extra, eval_task in eval_loader:
                eval_mask = aug_mask(eval_adj, algo=config.data.aug_mask_algo, data=config.data.name)
                eval_x, eval_adj, eval_mask = scaler(eval_x.to(config.device)), eval_adj.to(config.device), eval_mask.to(config.device)
                eval_task = [_.to(config.device) for _ in eval_task]
                eval_batch = (eval_x, eval_adj, eval_mask, eval_extra, eval_task)
                eval_loss, eval_pred, eval_labels = eval_step_fn(state, eval_batch)
                eval_pred_list += [v.detach().item() for v in eval_pred.squeeze()]
                eval_labels_list += [v.detach().item() for v in eval_labels.squeeze()]
                logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss.item()))
                wandb_logger.update(key="eval_loss", v=eval_loss.item())
            
            eval_p_corr = pearsonr(np.array(eval_pred_list), np.array(eval_labels_list))[0]
            eval_s_corr = spearmanr(np.array(eval_pred_list), np.array(eval_labels_list))[0]

            if eval_p_corr > max_eval_p_corr:
                is_best = True
                max_eval_p_corr = eval_p_corr

        # Save a checkpoint periodically and generate samples
        if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
            # Save the checkpoint.
            save_step = step // config.training.snapshot_freq
            save_checkpoint(checkpoint_dir, state, step, save_step, is_best)

            # Generate and save samples
            if config.training.snapshot_sampling:
                score_ema.store(score_model.parameters())
                score_ema.copy_to(score_model.parameters())
                
                sample, sample_steps, sample_chain, (score_grad_norm_p, classifier_grad_norm_p, score_grad_norm_c, classifier_grad_norm_c) = \
                    sampling_fn(score_model, mask, predictor_model, eval_chain=False, number_chain_steps=config.sampling.number_chain_steps,
                                classifier_scale=config.sampling.classifier_scale)
                sample_list = quantize(sample, adj) # quantization
                this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
                os.makedirs(this_sample_dir, exist_ok=True)
                arch_metric = sampling_metrics(arch_list=sample_list, adj=adj, mask=mask, 
                                                this_sample_dir=this_sample_dir, test=False, 
                                                check_dataname=config.sampling.check_dataname)

                r_valid, r_unique, r_novel = arch_metric[0][0], arch_metric[0][1],  arch_metric[0][2]
                test_acc_list = arch_metric[2]['test_acc_list']

        if step % config.training.eval_freq == 0:
            wandb_logger.write_log(element=element, step=step)
        else:
            wandb_logger.write_log(element={'train': ['training_loss']}, step=step)
        wandb_logger.reset()


def check_config(config1, config2):
    assert config1.training.sde == config2.training.sde
    assert config1.training.continuous == config2.training.continuous
    assert config1.data.centered == config2.data.centered
    assert config1.data.max_node == config2.data.max_node
    assert config1.data.n_vocab == config2.data.n_vocab

run_train_dict = {
    'sde': sde_train,
    'meta_predictor': meta_predictor_train
}


def train(config):
    run_train_dict[config.model_type](config)


