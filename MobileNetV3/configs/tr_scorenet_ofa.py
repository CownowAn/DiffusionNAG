"""Training PGSN on Community Small Dataset with GraphGDP"""

import ml_collections
import torch


def get_config():
    config = ml_collections.ConfigDict()
    
    # general
    config.resume = False
    config.resume_ckpt_path = './exp'
    config.folder_name = 'tr_scorenet'
    config.task = 'tr_scorenet'
    config.exp_name = None

    config.model_type = 'sde'

    # training
    config.training = training = ml_collections.ConfigDict()
    training.sde = 'vesde'
    training.continuous = True
    training.reduce_mean = True

    training.batch_size = 256
    training.eval_batch_size = 1000
    training.n_iters = 1000000
    training.snapshot_freq = 10000 
    training.log_freq = 200
    training.eval_freq = 10000
    ## store additional checkpoints for preemption
    training.snapshot_freq_for_preemption = 5000
    ## produce samples at each snapshot.
    training.snapshot_sampling = True
    training.likelihood_weighting = False

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.method = 'pc'
    sampling.predictor = 'euler_maruyama'
    sampling.corrector = 'none' 
    sampling.rtol = 1e-5
    sampling.atol = 1e-5
    sampling.ode_method = 'dopri5'  # 'rk4'
    sampling.ode_step = 0.01

    sampling.n_steps_each = 1
    sampling.noise_removal = True
    sampling.probability_flow = False
    sampling.snr = 0.16
    sampling.vis_row = 4
    sampling.vis_col = 4
    sampling.alpha = 0.5
    sampling.qtype = 'threshold'

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.begin_ckpt = 5
    evaluate.end_ckpt = 20
    evaluate.batch_size = 1024
    evaluate.enable_sampling = True
    evaluate.num_samples = 1024
    evaluate.mmd_distance = 'RBF'
    evaluate.max_subgraph = False
    evaluate.save_graph = False

    # data
    config.data = data = ml_collections.ConfigDict()
    data.centered = True
    data.dequantization = False

    data.root = './data/ofa/data_score_model/ofa_database_500000.pt'
    data.name = 'ofa'
    data.split_ratio = 0.9
    data.dataset_idx = 'random'
    data.max_node = 20
    data.n_vocab = 9 # 10 # 
    data.START_TYPE = 0
    data.END_TYPE = 1
    data.num_graphs = 100000
    data.num_channels = 1
    data.except_inout = False
    data.triu_adj = True
    data.connect_prev = False
    data.label_list = None
    data.tg_dataset = None
    data.node_rule_type = 2
    # aug_mask
    data.aug_mask_algo = 'none' 

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

    model.num_scales = 1000
    model.sigma_min = 0.1
    model.sigma_max = 1.0
    model.dropout = 0.1
    model.pos_enc_type = 2
    # graph encoder
    config.model.graph_encoder = graph_encoder = ml_collections.ConfigDict()
    graph_encoder.n_layers = 12
    graph_encoder.d_model = 64
    graph_encoder.n_head = 8
    graph_encoder.d_ff = 128
    graph_encoder.dropout = 0.1
    graph_encoder.n_vocab = 9 #10 # 30

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = 'Adam'
    optim.lr = 2e-5
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 1000
    optim.grad_clip = 1.

    config.seed = 42
    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    # log
    config.log = log = ml_collections.ConfigDict()
    log.use_wandb = True
    log.wandb_project_name = 'DiffusionNAG'
    log.log_valid_sample_prop = False
    log.num_graphs_to_visualize = 20

    return config
