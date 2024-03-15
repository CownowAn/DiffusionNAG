# Most of this code is from https://github.com/ultmaster/neuralpredictor.pytorch 
# which was authored by Yuge Zhang, 2020

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utils
from .set_encoder.setenc_models import SetPool

def normalize_adj(adj):
    # Row-normalize matrix
    last_dim = adj.size(-1)
    rowsum = adj.sum(2, keepdim=True).repeat(1, 1, last_dim)
    return torch.div(adj, rowsum)


def graph_pooling(inputs, num_vertices):
    num_vertices = num_vertices.to(inputs.device)
    out = inputs.sum(1)
    return torch.div(out, num_vertices.unsqueeze(-1).expand_as(out))


class DirectedGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight1 = nn.Parameter(torch.zeros((in_features, out_features)))
        self.weight2 = nn.Parameter(torch.zeros((in_features, out_features)))
        self.dropout = nn.Dropout(0.1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight1.data)
        nn.init.xavier_uniform_(self.weight2.data)

    def forward(self, inputs, adj):
        inputs = inputs.to(self.weight1.device)
        adj = adj.to(self.weight1.device)
        norm_adj = normalize_adj(adj)
        output1 = F.relu(torch.matmul(norm_adj, torch.matmul(inputs, self.weight1)))
        inv_norm_adj = normalize_adj(adj.transpose(1, 2))
        output2 = F.relu(torch.matmul(inv_norm_adj, torch.matmul(inputs, self.weight2)))
        out = (output1 + output2) / 2
        out = self.dropout(out)
        return out

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

# if nasbench-101: initial_hidden=5. if nasbench-201: initial_hidden=7
@utils.register_model(name='MetaNeuralPredictor')
class MetaeuralPredictor(nn.Module):
    # def __init__(self, initial_hidden=5, gcn_hidden=144, gcn_layers=4, linear_hidden=128):
    def __init__(self, config):
        super().__init__()
        # Arch
        self.gcn = [DirectedGraphConvolution(config.model.graph_encoder.initial_hidden if i == 0 else config.model.graph_encoder.gcn_hidden, 
                                             config.model.graph_encoder.gcn_hidden)
                    for i in range(config.model.graph_encoder.gcn_layers)]
        self.gcn = nn.ModuleList(self.gcn)
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(config.model.graph_encoder.gcn_hidden, config.model.graph_encoder.linear_hidden, bias=False)
        # self.fc2 = nn.Linear(config.model.graph_encoder.linear_hidden, 1, bias=False)
        
        # Time
        self.d_model  = config.model.graph_encoder.gcn_hidden
        self.timeEmb1 = nn.Linear(self.d_model, self.d_model * 4)
        self.timeEmb2 = nn.Linear(self.d_model * 4, self.d_model)
        
        self.act = act = get_act(config)
        self.input_type = config.model.input_type
        self.hs = config.model.hs
        
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
            input_dim += config.model.graph_encoder.linear_hidden
            
        self.pred_fc = nn.Sequential(
            nn.Linear(input_dim, self.hs),
            nn.Tanh(),
            nn.Linear(self.hs, 1)
            )
        
        self.sample_state = False
        self.D_mu = None

    def arch_encode(self, X, time_cond, maskX):
        # numv, adj, out = inputs["num_vertices"], inputs["adjacency"], inputs["operations"]
        out = X
        adj = maskX
        numv = torch.tensor([adj.size(1)] * adj.size(0)).to(out.device)
        gs = adj.size(1)  # graph node number
        
        timesteps = time_cond
        emb_t = get_timestep_embedding(timesteps, self.d_model)# time embedding
        emb_t = self.timeEmb1(emb_t)
        emb_t = self.timeEmb2(self.act(emb_t))

        adj_with_diag = normalize_adj(adj + torch.eye(gs, device=adj.device))  # assuming diagonal is not 1
        for layer in self.gcn:
            out = layer(out, adj_with_diag)
        out = graph_pooling(out, numv)
        # time
        out = out + emb_t
        out = self.fc1(out)
        out = self.dropout(out)
        
        # out = self.fc2(out).view(-1)
        # out = self.fc2(out)
        return out
    
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


import math
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