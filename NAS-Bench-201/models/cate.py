# Most of this code is from https://github.com/AIoT-MLSys-Lab/CATE.git
# which was authored by Shen Yan, Kaiqiang Song, Fei Liu, Mi Zhang, 2021


import torch.nn as nn
import torch
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import utils
from .transformer import Encoder, SemanticEmbedding
from .set_encoder.setenc_models import SetPool


class MLP(torch.nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, use_bn=False, activate_func=F.relu):
        """
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            num_classes: the number of classes of input, to be treated with different gains and biases,
                    (see the definition of class `ConditionalLayer1d`)
        """

        super(MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.use_bn = use_bn
        self.activate_func = activate_func

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = torch.nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()

            self.linears.append(torch.nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(torch.nn.Linear(hidden_dim, output_dim))

            if self.use_bn:
                self.batch_norms = torch.nn.ModuleList()
                for layer in range(num_layers - 1):
                    self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))


    def forward(self, x):
        """
        :param x: [num_classes * batch_size, N, F_i], batch of node features
            note that in self.cond_layers[layer],
            `x` is splited into `num_classes` groups in dim=0,
            and then treated with different gains and biases
        """
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = self.linears[layer](h)
                if self.use_bn:
                    h = self.batch_norms[layer](h)
                h = self.activate_func(h)
            return self.linears[self.num_layers - 1](h)


""" Transformer Encoder """
class GraphEncoder(nn.Module):
    def __init__(self, config):
        super(GraphEncoder, self).__init__()
        # Forward Transformers
        self.encoder_f = Encoder(config)

    def forward(self, x, mask):
        h_f, hs_f, attns_f = self.encoder_f(x, mask)
        h = torch.cat(hs_f, dim=-1)
        return h
    
    @staticmethod
    def get_embeddings(h_x):
        h_x = h_x.cpu()
        return h_x[:, -1]


class CLSHead(nn.Module):
    def __init__(self, config, init_weights=None):
        super(CLSHead, self).__init__()
        self.layer_1 = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(p=config.dropout)
        self.layer_2 = nn.Linear(config.d_model, config.n_vocab)
        if init_weights is not None:
            self.layer_2.weight = init_weights

    def forward(self, x):
        x = self.dropout(torch.tanh(self.layer_1(x)))
        return F.log_softmax(self.layer_2(x), dim=-1)


@utils.register_model(name='CATE')
class CATE(nn.Module):
    def __init__(self, config):
        super(CATE, self).__init__()
        # Shared Embedding Layer
        self.opEmb = SemanticEmbedding(config.model.graph_encoder)
        self.dropout_op = nn.Dropout(p=config.model.dropout)
        self.d_model = config.model.graph_encoder.d_model
        self.act = act = get_act(config)
        # Time
        self.timeEmb1 = nn.Linear(self.d_model, self.d_model * 4)
        self.timeEmb2 = nn.Linear(self.d_model * 4, self.d_model)

        # 2 GraphEncoder for X and Y
        self.graph_encoder = GraphEncoder(config.model.graph_encoder)
        
        self.fdim = int(config.model.graph_encoder.n_layers * config.model.graph_encoder.d_model)
        self.final = MLP(num_layers=3, input_dim=self.fdim, hidden_dim=2*self.fdim, output_dim=config.data.n_vocab, 
                        use_bn=False, activate_func=F.elu)

        if 'pos_enc_type' in config.model:
            self.pos_enc_type = config.model.pos_enc_type
            if self.pos_enc_type == 1:
                raise NotImplementedError
            elif self.pos_enc_type == 2:
                if config.data.name == 'NASBench201':
                    self.pos_encoder = PositionalEncoding_Cell(d_model=self.d_model, max_len=config.data.max_node)
                else:
                    self.pos_encoder = PositionalEncoding_StageWise(d_model=self.d_model, max_len=config.data.max_node)
            elif self.pos_enc_type == 3:
                raise NotImplementedError
            else:
                self.pos_encoder = None
        else:
            self.pos_encoder = None


    def forward(self, X, time_cond, maskX):
        
        emb_x = self.dropout_op(self.opEmb(X)) 

        if self.pos_encoder is not None:
            emb_p = self.pos_encoder(emb_x)
            emb_x = emb_x + emb_p       
        
        # Time embedding
        timesteps = time_cond
        emb_t = get_timestep_embedding(timesteps, self.d_model)
        emb_t = self.timeEmb1(emb_t) # [32, 512]
        emb_t = self.timeEmb2(self.act(emb_t)) # [32, 64]
        emb_t = emb_t.unsqueeze(1)
        emb = emb_x + emb_t 

        h_x = self.graph_encoder(emb, maskX)
        h_x = self.final(h_x)
        
        return h_x


@utils.register_model(name='PredictorCATE')
class PredictorCATE(nn.Module):
    def __init__(self, config):
        super(PredictorCATE, self).__init__()
        # Shared Embedding Layer
        self.opEmb = SemanticEmbedding(config.model.graph_encoder)
        self.dropout_op = nn.Dropout(p=config.model.dropout)
        self.d_model = config.model.graph_encoder.d_model
        self.act = act = get_act(config)
        # Time
        self.timeEmb1 = nn.Linear(self.d_model, self.d_model * 4)
        self.timeEmb2 = nn.Linear(self.d_model * 4, self.d_model)

        # 2 GraphEncoder for X and Y
        self.graph_encoder = GraphEncoder(config.model.graph_encoder)
        
        self.fdim = int(config.model.graph_encoder.n_layers * config.model.graph_encoder.d_model)
        self.final = MLP(num_layers=3, input_dim=self.fdim, hidden_dim=2*self.fdim, output_dim=config.data.n_vocab, 
                        use_bn=False, activate_func=F.elu)
        
        self.rdim = int(config.data.max_node * config.data.n_vocab)
        self.regeress = MLP(num_layers=2, input_dim=self.rdim, hidden_dim=2*self.rdim, output_dim=1,
                            use_bn=False, activate_func=F.elu)

    def forward(self, X, time_cond, maskX):
        
        emb_x = self.dropout_op(self.opEmb(X))
        
        # Time embedding
        timesteps = time_cond
        emb_t = get_timestep_embedding(timesteps, self.d_model)
        emb_t = self.timeEmb1(emb_t) 
        emb_t = self.timeEmb2(self.act(emb_t)) 
        emb_t = emb_t.unsqueeze(1)
        emb = emb_x + emb_t 

        h_x = self.graph_encoder(emb, maskX)
        h_x = self.final(h_x)
        h_x = h_x.reshape(h_x.size(0), -1)
        h_x = self.regeress(h_x)
        
        return h_x


class PositionalEncoding_StageWise(nn.Module):
    
    def __init__(self, d_model, max_len):
        super(PositionalEncoding_StageWise, self).__init__() 
        NUM_STAGE = 5
        max_len = int(max_len / NUM_STAGE)
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False 
        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)
        _2i = torch.arange(0, d_model, step=2).float()
        self.encoding[:, ::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model))) 
        self.encoding = torch.cat([self.encoding] * NUM_STAGE, dim=0) 
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size() 
        return self.encoding[:seq_len, :].to(x.device)


class PositionalEncoding_Cell(nn.Module):
    
    def __init__(self, d_model, max_len):
        super(PositionalEncoding_Cell, self).__init__() 
        NUM_STAGE = 1
        max_len = int(max_len / NUM_STAGE)
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False 
        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)
        _2i = torch.arange(0, d_model, step=2).float()
        self.encoding[:, ::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model))) 
        self.encoding = torch.cat([self.encoding] * NUM_STAGE, dim=0) 
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size() 
        return self.encoding[:seq_len, :].to(x.device)


@utils.register_model(name='MetaPredictorCATE')
class MetaPredictorCATE(nn.Module):
    def __init__(self, config):
        super(MetaPredictorCATE, self).__init__()
        
        self.input_type= config.model.input_type
        self.hs = config.model.hs
        
        self.opEmb = SemanticEmbedding(config.model.graph_encoder)
        self.dropout_op = nn.Dropout(p=config.model.dropout)
        self.d_model = config.model.graph_encoder.d_model
        self.act = act = get_act(config)
        
        # Time
        self.timeEmb1 = nn.Linear(self.d_model, self.d_model * 4)
        self.timeEmb2 = nn.Linear(self.d_model * 4, self.d_model)

        self.graph_encoder = GraphEncoder(config.model.graph_encoder)
        
        self.fdim = int(config.model.graph_encoder.n_layers * config.model.graph_encoder.d_model)
        self.final = MLP(num_layers=3, input_dim=self.fdim, hidden_dim=2*self.fdim, output_dim=config.data.n_vocab, 
                        use_bn=False, activate_func=F.elu)
        
        self.rdim = int(config.data.max_node * config.data.n_vocab)
        self.regeress = MLP(num_layers=2, input_dim=self.rdim, hidden_dim=2*self.rdim, output_dim=2*self.rdim,
                            use_bn=False, activate_func=F.elu)
        
        # Set
        self.nz = config.model.nz
        self.num_sample = config.model.num_sample
        self.intra_setpool = SetPool(dim_input=512, 
                                    num_outputs=1, 
                                    dim_output=self.nz, 
                                    dim_hidden=self.nz, 
                                    mode='sabPF')
        self.inter_setpool = SetPool(dim_input=self.nz, 
                                 num_outputs=1, 
                                 dim_output=self.nz, 
                                 dim_hidden=self.nz, 
                                 mode='sabPF')
        self.set_fc = nn.Sequential(
            nn.Linear(512, self.nz),
            nn.ReLU())
        
        input_dim = 0
        if 'D' in self.input_type:
            input_dim += self.nz
        if 'A' in self.input_type:
            input_dim += 2*self.rdim
            
        self.pred_fc = nn.Sequential(
            nn.Linear(input_dim, self.hs),
            nn.Tanh(),
            nn.Linear(self.hs, 1)
            )
        
        self.sample_state = False
        self.D_mu = None

        
    def arch_encode(self, X, time_cond, maskX):
        emb_x = self.dropout_op(self.opEmb(X))

        # Time embedding
        timesteps = time_cond
        emb_t = get_timestep_embedding(timesteps, self.d_model)# time embedding
        emb_t = self.timeEmb1(emb_t) # [32, 512]
        emb_t = self.timeEmb2(self.act(emb_t)) # [32, 64]
        emb_t = emb_t.unsqueeze(1)
        emb = emb_x + emb_t 

        h_x = self.graph_encoder(emb, maskX)
        h_x = self.final(h_x)
        
        h_x = h_x.reshape(h_x.size(0), -1)
        h_x = self.regeress(h_x)
        return h_x


    def set_encode(self, task):
        proto_batch = []
        for x in task: 
            cls_protos = self.intra_setpool(
                x.view(-1, self.num_sample, 512)).squeeze(1)
            proto_batch.append(
                self.inter_setpool(cls_protos.unsqueeze(0)))
        v = torch.stack(proto_batch).squeeze()
        return v


    def predict(self, D_mu, A_mu):
        input_vec = []
        if 'D' in self.input_type:
            input_vec.append(D_mu)
        if 'A' in self.input_type:
            input_vec.append(A_mu)
        input_vec = torch.cat(input_vec, dim=1)
        return self.pred_fc(input_vec)


    def forward(self, X, time_cond, maskX, task):
        if self.sample_state:
            if self.D_mu is None:
                self.D_mu = self.set_encode(task)
            D_mu = self.D_mu
        else:
            D_mu = self.set_encode(task)
        A_mu = self.arch_encode(X, time_cond, maskX)
        y_pred = self.predict(D_mu, A_mu)
        return y_pred


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    # magic number 10000 is from transformers
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1: # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def get_act(config):
    """Get actiuvation functions from the config file."""

    if config.model.nonlinearity.lower() == 'elu':
        return nn.ELU()
    elif config.model.nonlinearity.lower() == 'relu':
        return nn.ReLU()
    elif config.model.nonlinearity.lower() == 'lrelu':
        return nn.LeakyReLU(negative_slope=0.2)
    elif config.model.nonlinearity.lower() == 'swish':
        return nn.SiLU()
    elif config.model.nonlinearity.lower() == 'tanh':
        return nn.Tanh()
    else:
        raise NotImplementedError('activation function does not exist!')
