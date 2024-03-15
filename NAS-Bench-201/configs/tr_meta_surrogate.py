"""Training PGSN on Community Small Dataset with GraphGDP"""

import ml_collections
import torch
from all_path import SCORENET_CKPT_PATH
from all_path import NASBENCH201_INFO


def get_config():
    config = ml_collections.ConfigDict()

    # config.search_space = None

    # general
    config.folder_name = 'test'
    config.model_type = 'meta_surrogate'
    config.task = 'tr_meta_surrogate'
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
    training.batch_size = 256
    training.eval_batch_size = 100
    training.n_iters = 10000
    training.snapshot_freq = 500
    training.log_freq = 100
    training.eval_freq = 100
    training.snapshot_sampling = True
    training.likelihood_weighting = False

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.method = 'pc'
    sampling.predictor = 'euler_maruyama'
    sampling.corrector = 'langevin' 
    sampling.n_steps_each = 1
    sampling.noise_removal = True
    sampling.probability_flow = False
    sampling.snr = 0.16

    # for conditional sampling
    sampling.classifier_scale = 10000.0
    sampling.regress = True
    sampling.labels = 'max'
    sampling.weight_ratio = False
    sampling.weight_scheduling = False
    sampling.check_dataname = 'cifar10'

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.batch_size = 512
    evaluate.enable_sampling = True
    evaluate.num_samples = 1024

    # data
    config.data = data = ml_collections.ConfigDict()
    data.centered = True
    data.dequantization = False

    data.root = NASBENCH201_INFO
    data.name = 'NASBench201'
    data.max_node = 8
    data.n_vocab = 7
    data.START_TYPE = 0
    data.END_TYPE = 1
    data.num_channels = 1
    data.label_list = ['meta-acc']
    # aug_mask
    data.aug_mask_algo = 'floyd' # 'long_range' | 'floyd'

    # model
    config.model = model = ml_collections.ConfigDict()
    model.name = 'MetaNeuralPredictor'
    model.ema_rate = 0.9999
    model.normalization = 'GroupNorm'
    model.nonlinearity = 'swish'
    model.nf = 128
    model.num_gnn_layers = 4
    model.size_cond = False
    model.embedding_type = 'positional'
    model.rw_depth = 16
    model.graph_layer = 'PosTransLayer'
    model.edge_th = -1.
    model.heads = 8
    model.attn_clamp = False

    # meta-predictor
    model.input_type = 'DA'
    model.hs = 32
    model.nz = 56
    model.num_sample = 20

    model.num_scales = 1000
    model.beta_min = 0.1
    model.beta_max = 5.0
    model.sigma_min = 0.1
    model.sigma_max = 5.0
    model.dropout = 0.1

    # graph encoder
    config.model.graph_encoder = graph_encoder = ml_collections.ConfigDict()
    graph_encoder.initial_hidden = 7
    graph_encoder.gcn_hidden = 144
    graph_encoder.gcn_layers = 4
    graph_encoder.linear_hidden = 128

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = 'Adam'
    optim.lr = 0.001
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 1000
    optim.grad_clip = 1.

    return config
