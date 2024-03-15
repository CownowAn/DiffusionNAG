import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops.layers.torch import Rearrange
from einops import rearrange
import numpy as np

from . import utils


class SinusoidalPositionalEmbedding(nn.Embedding):

    def __init__(self, num_positions, embedding_dim):
        super().__init__(num_positions, embedding_dim) # torch.nn.Embedding(num_embeddings, embedding_dim)
        self.weight = self._init_weight(self.weight) # self.weight => nn.Embedding(num_positions, embedding_dim).weight
    
    @staticmethod
    def _init_weight(out: nn.Parameter):
        n_pos, embed_dim = out.shape
        pe = nn.Parameter(torch.zeros(out.shape))
        for pos in range(n_pos):
            for i in range(0, embed_dim, 2):
                pe[pos, i].data.copy_( torch.tensor( np.sin(pos / (10000 ** ( i / embed_dim)))) )
                pe[pos, i + 1].data.copy_( torch.tensor( np.cos(pos / (10000 ** ((i + 1) / embed_dim)))) )
        pe.detach_()
                
        return pe

    @torch.no_grad()
    def forward(self, input_ids):
        bsz, seq_len = input_ids.shape[:2] # for x, seq_len = max_node_num
        positions = torch.arange(seq_len, dtype=torch.long, device=self.weight.device)
        return super().forward(positions)


class MLP(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        *,
        expansion_factor = 2.,
        depth = 2,
        norm = False,
    ):
        super().__init__()
        hidden_dim = int(expansion_factor * dim_out)
        norm_fn = lambda: nn.LayerNorm(hidden_dim) if norm else nn.Identity()

        layers = [nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.SiLU(),
            norm_fn()
        )]

        for _ in range(depth - 1):
            layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                norm_fn()
            ))

        layers.append(nn.Linear(hidden_dim, dim_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.float())

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        dtype, device = x.dtype, x.device
        assert is_float_dtype(dtype), 'input to sinusoidal pos emb must be a float type'

        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device = device, dtype = dtype) * -emb)
        emb = rearrange(x, 'i -> i 1') * rearrange(emb, 'j -> 1 j')
        return torch.cat((emb.sin(), emb.cos()), dim = -1).type(dtype)

def is_float_dtype(dtype):
    return any([dtype == float_dtype for float_dtype in (torch.float64, torch.float32, torch.float16, torch.bfloat16)])


class PositionWiseFeedForward(nn.Module):

    def __init__(self, emb_dim: int, d_ff: int, dropout: float = 0.1):
        super(PositionWiseFeedForward, self).__init__()

        self.activation = nn.ReLU()
        self.w_1 = nn.Linear(emb_dim, d_ff)
        self.w_2 = nn.Linear(d_ff, emb_dim)
        self.dropout = dropout

    def forward(self, x):
        residual = x
        x = self.activation(self.w_1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.w_2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x + residual # residual connection for preventing gradient vanishing


class MultiHeadAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        emb_dim,
        num_heads,
        dropout=0.0,
        bias=False,
        encoder_decoder_attention=False,  # otherwise self_attention
        causal = True
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = emb_dim // num_heads
        assert self.head_dim * num_heads == self.emb_dim, "emb_dim must be divisible by num_heads"

        self.encoder_decoder_attention = encoder_decoder_attention
        self.causal = causal
        self.q_proj = nn.Linear(emb_dim, emb_dim, bias=bias)
        self.k_proj = nn.Linear(emb_dim, emb_dim, bias=bias)
        self.v_proj = nn.Linear(emb_dim, emb_dim, bias=bias)
        self.out_proj = nn.Linear(emb_dim, emb_dim, bias=bias)


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_heads,
            self.head_dim,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
        # This is equivalent to
        # return x.transpose(1,2)
    

    def scaled_dot_product(self, 
                            query: torch.Tensor, 
                            key: torch.Tensor, 
                            value: torch.Tensor,
                            attention_mask: torch.BoolTensor):

        attn_weights = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.emb_dim) # QK^T/sqrt(d)
        
        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(attention_mask.unsqueeze(1), float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)  # softmax(QK^T/sqrt(d))
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_probs, value) # softmax(QK^T/sqrt(d))V

        return attn_output, attn_probs
    
    
    def MultiHead_scaled_dot_product(self, 
                        query: torch.Tensor, 
                        key: torch.Tensor, 
                        value: torch.Tensor,
                        attention_mask: torch.BoolTensor):
        attention_mask = attention_mask.bool()

        attn_weights = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim) # QK^T/sqrt(d) # [6, 6]
        
        # Attention mask
        if attention_mask is not None:
            if self.causal:
                # (seq_len x seq_len)
                attn_weights = attn_weights.masked_fill(attention_mask.unsqueeze(0).unsqueeze(1), float("-inf"))
            else:
                # (batch_size x seq_len)
                attn_weights = attn_weights.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        
        attn_weights = F.softmax(attn_weights, dim=-1)  # softmax(QK^T/sqrt(d))
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_probs, value) # softmax(QK^T/sqrt(d))V
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        concat_attn_output_shape = attn_output.size()[:-2] + (self.emb_dim,)
        attn_output = attn_output.view(*concat_attn_output_shape)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights

        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        attention_mask: torch.Tensor = None,
        ):

        q = self.q_proj(query)
        # Enc-Dec attention
        if self.encoder_decoder_attention:
            k = self.k_proj(key)
            v = self.v_proj(key)
        # Self attention
        else:
            k = self.k_proj(query)
            v = self.v_proj(query)

        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)

        attn_output, attn_weights = self.MultiHead_scaled_dot_product(q,k,v,attention_mask)
        return attn_output, attn_weights

class EncoderLayer(nn.Module):
    def __init__(self, emb_dim, ffn_dim, attention_heads, 
                    attention_dropout, dropout):
        super().__init__()
        self.emb_dim = emb_dim
        self.ffn_dim = ffn_dim
        self.self_attn = MultiHeadAttention(            
            emb_dim=self.emb_dim,
            num_heads=attention_heads, 
            dropout=attention_dropout)
        self.self_attn_layer_norm = nn.LayerNorm(self.emb_dim)
        self.dropout = dropout
        self.activation_fn = nn.ReLU()
        self.PositionWiseFeedForward = PositionWiseFeedForward(self.emb_dim, self.ffn_dim, dropout)
        self.final_layer_norm = nn.LayerNorm(self.emb_dim)

    def forward(self, x, encoder_padding_mask):

        residual = x
        x, attn_weights = self.self_attn(query=x, key=x, attention_mask=encoder_padding_mask)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.self_attn_layer_norm(x)
        x = self.PositionWiseFeedForward(x)
        x = self.final_layer_norm(x)
        if torch.isinf(x).any() or torch.isnan(x).any():
            clamp_value = torch.finfo(x.dtype).max - 1000
            x = torch.clamp(x, min=-clamp_value, max=clamp_value)
        return x, attn_weights


@utils.register_model(name='DAGformer')
class DAGformer(torch.nn.Module):
    def __init__(self, config):
        # max_feat_num, 
        # max_node_num,
        # emb_dim,
        # ffn_dim, 
        # encoder_layers,
        # attention_heads, 
        # attention_dropout,
        # dropout,
        # hs,
        # time_dep=True,
        # num_timesteps=None,
        # return_attn=False,
        # except_inout=False,
        # connect_prev=True
        # ):
        super().__init__()
        
        self.dropout = config.model.dropout
        self.time_dep = config.model.time_dep
        self.return_attn = config.model.return_attn
        max_feat_num = config.data.n_vocab
        max_node_num = config.data.max_node
        emb_dim = config.model.emb_dim
        # num_timesteps = config.model.num_scales
        num_timesteps = None
        
        self.x_embedding = MLP(max_feat_num, emb_dim)
        # position embedding with topological order
        self.position_embedding = SinusoidalPositionalEmbedding(max_node_num, emb_dim)

        if self.time_dep:
            self.time_embedding = nn.Sequential(
            nn.Embedding(num_timesteps, emb_dim) if num_timesteps is not None 
            else nn.Sequential(SinusoidalPosEmb(emb_dim), MLP(emb_dim, emb_dim)), # also offer a continuous version of timestep embeddings, with a 2 layer MLP
            Rearrange('b (n d) -> b n d', n=1)
        )
        
        self.layers = nn.ModuleList([EncoderLayer(emb_dim, 
                                                  config.model.ffn_dim, 
                                                  config.model.attention_heads, 
                                                  config.model.attention_dropout, 
                                                  config.model.dropout) 
                                        for _ in range(config.model.encoder_layers)])
        
        self.pred_fc = nn.Sequential(
            nn.Linear(emb_dim, config.model.hs),
            nn.Tanh(),
            nn.Linear(config.model.hs, 1),
            # nn.Sigmoid()
            )
        
        # -------- Load Constant Adj Matrix (START) --------- #
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # from utils.graph_utils import get_const_adj
        # mat = get_const_adj(
        #     except_inout=except_inout, 
        #     shape_adj=(1, max_node_num, max_node_num), 
        #     device=torch.device('cpu'), 
        #     connect_prev=connect_prev)[0].cpu()
        # is_triu_ = is_triu(mat)
        # if is_triu_:
        #     self.adj_ = mat.T.to(self.device)
        # else:
            # self.adj_ = mat.to(self.device)
        # -------- Load Constant Adj Matrix (END) --------- #
        
    def forward(self, x, t, adj, flags=None):
        """
        :param x:  B x N x F_i
        :param adjs: B x C_i x N x N
        :return: x_o: B x N x F_o, new_adjs: B x C_o x N x N
        """

        assert len(x.shape) == 3
        
        self_attention_mask = torch.eye(adj.size(1)).to(self.device)
        # attention_mask = 1. - (self_attention_mask + self.adj_) 
        attention_mask = 1. - (self_attention_mask + adj[0]) 

        # -------- Generate input for DAGformer ------- #
        x_embed = self.x_embedding(x)
        # x_embed = x
        x_pos = self.position_embedding(x).unsqueeze(0)
        if self.time_dep:
            time_embed = self.time_embedding(t)

        x = x_embed + x_pos
        if self.time_dep:
            x = x + time_embed
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        self_attn_scores = []
        for encoder_layer in self.layers:
            x, attn = encoder_layer(x, attention_mask)
            self_attn_scores.append(attn.detach())

        x = self.pred_fc(x[:, -1, :]) # [256, 16]

        if self.return_attn:
            return x, self_attn_scores
        else:
            return x