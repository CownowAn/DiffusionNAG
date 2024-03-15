import ml_collections
import torch
from all_path import SCORE_MODEL_CKPT_PATH, SCORE_MODEL_DATA_PATH


def get_config():
    config = ml_collections.ConfigDict()
    
    config.search_space = None
    
    # genel
    config.resume = False
    config.folder_name = 'DiffusionNAG'
    config.task = 'tr_meta_predictor'
    config.exp_name = None
    config.model_type = 'meta_predictor'
    config.scorenet_ckpt_path = SCORE_MODEL_CKPT_PATH
    config.is_meta = True

    # training
    config.training = training = ml_collections.ConfigDict()
    training.sde = 'vesde'
    training.continuous = True
    training.reduce_mean = True
    training.noised = True

    training.batch_size = 128
    training.eval_batch_size = 512
    training.n_iters = 20000 
    training.snapshot_freq = 500
    training.log_freq = 500
    training.eval_freq = 500
    ## store additional checkpoints for preemption
    training.snapshot_freq_for_preemption = 1000
    ## produce samples at each snapshot.
    training.snapshot_sampling = True
    training.likelihood_weighting = False
    # training for perturbed data
    training.t_spot = 1.
    # training from pretrained score model
    training.load_pretrained = False
    training.pretrained_model_path = SCORE_MODEL_CKPT_PATH

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.method = 'pc'
    sampling.predictor = 'euler_maruyama'
    sampling.corrector = 'none' 
    # sampling.corrector = 'langevin' 
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

    # conditional
    sampling.classifier_scale = 1.0
    sampling.regress = True
    sampling.labels = 'max'
    sampling.weight_ratio = False
    sampling.weight_scheduling = False
    sampling.t_spot = 1.
    sampling.t_spot_end = 0.
    sampling.number_chain_steps = 50
    sampling.check_dataname = 'imagenet1k'

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.begin_ckpt = 5
    evaluate.end_ckpt = 20
    # evaluate.batch_size = 512
    evaluate.batch_size = 128
    evaluate.enable_sampling = True
    evaluate.num_samples = 1024
    evaluate.mmd_distance = 'RBF'
    evaluate.max_subgraph = False
    evaluate.save_graph = False
    
    # data
    config.data = data = ml_collections.ConfigDict()
    data.centered = True
    data.dequantization = False

    data.root = SCORE_MODEL_DATA_PATH
    data.name = 'ofa'
    data.split_ratio = 0.8
    data.dataset_idx = 'random'
    data.max_node = 20
    data.n_vocab = 9
    data.START_TYPE = 0
    data.END_TYPE = 1
    data.num_graphs = 100000
    data.num_channels = 1
    data.except_inout = False # ignore
    data.triu_adj = True
    data.connect_prev = False
    data.tg_dataset = None
    data.label_list = ['meta-acc']
    # aug_mask
    data.aug_mask_algo = 'none' # 'long_range' | 'floyd'
    # num_train
    data.num_train = 150

    # model
    config.model = model = ml_collections.ConfigDict()
    model.name = 'MetaPredictorCATE'
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
    #############################################################################
    # meta
    model.input_type = 'DA'
    model.hs = 512
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
    graph_encoder.n_layers = 2
    graph_encoder.d_model = 64
    graph_encoder.n_head = 2
    graph_encoder.d_ff = 32
    graph_encoder.dropout = 0.1
    graph_encoder.n_vocab = 9

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = 'Adam'
    optim.lr = 0.001
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
