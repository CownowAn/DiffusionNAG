"""Evaluate trained score network"""

import ml_collections
import torch

from all_path import SCORENET_CKPT_PATH

def get_config():
    config = ml_collections.ConfigDict()

    # general
    config.folder_name = 'test'
    config.model_type = 'scorenet'
    config.task = 'eval_scorenet'
    config.exp_name = None
    config.seed = 42
    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    config.resume = False
    config.scorenet_ckpt_path = SCORENET_CKPT_PATH

    # training
    config.training = training = ml_collections.ConfigDict()
    training.sde = 'vesde'
    training.continuous = True
    training.reduce_mean = True
    training.noised = True

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.method = 'pc'
    sampling.predictor = 'euler_maruyama'
    sampling.corrector = 'langevin'
    sampling.n_steps_each = 1
    sampling.noise_removal = True
    sampling.probability_flow = False
    sampling.snr = 0.16

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.batch_size = 256
    evaluate.enable_sampling = True
    evaluate.num_samples = 256

    # data
    config.data = data = ml_collections.ConfigDict()
    data.centered = True
    data.dequantization = False

    data.root = '../data/transfer_nag/nasbench201_info.pt'
    data.name = 'NASBench201'
    data.split_ratio = 1.0
    data.dataset_idx = 'random' # 'sorted' | 'random'
    data.max_node = 8
    data.n_vocab = 7 # number of operations
    data.START_TYPE = 0
    data.END_TYPE = 1
    data.num_graphs = 15625
    data.num_channels = 1
    data.label_list = ['test-acc']
    data.tg_dataset = 'cifar10'
    # aug_mask
    data.aug_mask_algo = 'floyd' # 'long_range' | 'floyd'

    # model
    config.model = model = ml_collections.ConfigDict()
    model.num_scales = 1000
    model.beta_min = 0.1
    model.beta_max = 5.0
    model.sigma_min = 0.1
    model.sigma_max = 5.0

    return config
