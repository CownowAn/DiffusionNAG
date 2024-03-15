"""Training Score Network"""

import ml_collections
import torch


def get_config():
    config = ml_collections.ConfigDict()

    # general
    config.folder_name = 'test'
    config.model_type = 'scorenet'
    config.task = 'tr_scorenet'
    config.exp_name = None
    config.seed = 42
    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    config.resume = False
    config.resume_ckpt_path = ''

    # training
    config.training = training = ml_collections.ConfigDict()
    training.sde = 'vesde'
    training.continuous = True
    training.reduce_mean = True

    training.batch_size = 256
    training.eval_batch_size = 1000
    training.n_iters = 250000
    training.snapshot_freq = 10000
    training.log_freq = 200
    training.eval_freq = 10000
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

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.batch_size = 1024
    evaluate.enable_sampling = True
    evaluate.num_samples = 1024

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
    data.label_list = None
    data.tg_dataset = None
    # aug_mask
    data.aug_mask_algo = 'floyd' # 'long_range' | 'floyd'

    # model
    config.model = model = ml_collections.ConfigDict()
    model.name = 'CATE'
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
    # for pos emb
    model.pos_enc_type = 2

    model.num_scales = 1000
    model.sigma_min = 0.1
    model.sigma_max = 5.0
    model.dropout = 0.1

    # graph encoder
    config.model.graph_encoder = graph_encoder = ml_collections.ConfigDict()
    graph_encoder.n_layers = 12
    graph_encoder.d_model = 64
    graph_encoder.n_head = 8
    graph_encoder.d_ff = 128
    graph_encoder.dropout = 0.1
    graph_encoder.n_vocab = 7

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = 'Adam'
    optim.lr = 2e-5
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 1000
    optim.grad_clip = 1.

    return config
