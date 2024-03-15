import torch
import torch.nn.functional as F

from models.GDSS.layers import DenseGCNConv, MLP
from .graph_utils import mask_x, pow_tensor
from .attention import AttentionLayer
from .. import utils

@utils.register_model(name='ScoreNetworkX')
class ScoreNetworkX(torch.nn.Module):

    # def __init__(self, max_feat_num, depth, nhid):
    def __init__(self, config):

        super(ScoreNetworkX, self).__init__()

        self.nfeat = config.data.n_vocab
        self.depth = config.model.depth
        self.nhid = config.model.nhid

        self.layers = torch.nn.ModuleList()
        for _ in range(self.depth):
            if _ == 0:
                self.layers.append(DenseGCNConv(self.nfeat, self.nhid))
            else:
                self.layers.append(DenseGCNConv(self.nhid, self.nhid))

        self.fdim = self.nfeat + self.depth * self.nhid
        self.final = MLP(num_layers=3, input_dim=self.fdim, hidden_dim=2*self.fdim, output_dim=self.nfeat, 
                            use_bn=False, activate_func=F.elu)

        self.activation = torch.tanh

    def forward(self, x, time_cond, maskX, flags=None):

        x_list = [x]
        for _ in range(self.depth):
            x = self.layers[_](x, maskX)
            x = self.activation(x)
            x_list.append(x)

        xs = torch.cat(x_list, dim=-1) # B x N x (F + num_layers x H)
        out_shape = (x.shape[0], x.shape[1], -1)
        x = self.final(xs).view(*out_shape)

        x = mask_x(x, flags)
        return x


@utils.register_model(name='ScoreNetworkX_GMH')
class ScoreNetworkX_GMH(torch.nn.Module):
    # def __init__(self, max_feat_num, depth, nhid, num_linears,
    #              c_init, c_hid, c_final, adim, num_heads=4, conv='GCN'):
    def __init__(self, config):
        super().__init__()
        
        self.max_feat_num = config.data.n_vocab
        self.depth = config.model.depth
        self.nhid = config.model.nhid
        self.c_init = config.model.c_init
        self.c_hid = config.model.c_hid
        self.c_final = config.model.c_final
        self.num_linears = config.model.num_linears
        self.num_heads = config.model.num_heads
        self.conv = config.model.conv
        self.adim = config.model.adim
        
        self.layers = torch.nn.ModuleList()
        for _ in range(self.depth):
            if _ == 0:
                self.layers.append(AttentionLayer(self.num_linears, self.max_feat_num, 
                                                  self.nhid, self.nhid, self.c_init, 
                                                  self.c_hid, self.num_heads, self.conv))
            elif _ == self.depth - 1:
                self.layers.append(AttentionLayer(self.num_linears, self.nhid, self.adim, 
                                                  self.nhid, self.c_hid, 
                                                  self.c_final, self.num_heads, self.conv))
            else:
                self.layers.append(AttentionLayer(self.num_linears, self.nhid, self.adim,
                                                  self.nhid, self.c_hid, 
                                                  self.c_hid, self.num_heads, self.conv))

        fdim = self.max_feat_num + self.depth * self.nhid
        self.final = MLP(num_layers=3, input_dim=fdim, hidden_dim=2*fdim, output_dim=self.max_feat_num, 
                         use_bn=False, activate_func=F.elu)

        self.activation = torch.tanh

    def forward(self, x, time_cond, maskX, flags=None):
        adjc = pow_tensor(maskX, self.c_init)

        x_list = [x]
        for _ in range(self.depth):
            x, adjc = self.layers[_](x, adjc, flags)
            x = self.activation(x)
            x_list.append(x)

        xs = torch.cat(x_list, dim=-1) # B x N x (F + num_layers x H)
        out_shape = (x.shape[0], x.shape[1], -1)
        x = self.final(xs).view(*out_shape)
        x = mask_x(x, flags)

        return x