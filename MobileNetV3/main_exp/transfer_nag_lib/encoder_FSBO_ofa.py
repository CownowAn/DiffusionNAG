######################################################################################
# Copyright (c) muhanzhang, D-VAE, NeurIPS 2019 [GitHub D-VAE]
# Modified by Hayeon Lee, Eunyoung Hyung, MetaD2A, ICLR2021, 2021. 03 [GitHub MetaD2A]
######################################################################################
# import math
# import random
import torch
import json
from torch import nn
import os
from torch.nn import functional as F
import datetime


## Our packages
import gpytorch
import logging

from transfer_nag_lib.DeepKernelGPHelpers import Metric
from transfer_nag_lib.DeepKernelGPModules import StandardDeepGP, ExactGPLayer
from transfer_nag_lib.MetaD2A_mobilenetV3.set_encoder.setenc_models import SetPool


class EncoderFSBO(nn.Module):
    def __init__(self, args, graph_config, dgp_arch):
        super(EncoderFSBO, self).__init__()

        ## GP parameters
        space="OFA_MBV3"
        c, D = 4230, 64
        dim = args.nz * 2
        rootdir = os.path.dirname(os.path.realpath(__file__))
        backbone_params = json.load(open(os.path.join(rootdir, "Setconfig90.json"), "rb"))
        backbone_params.update({"dim": dim})
        backbone_params.update({"fixed_context_size": dim})
        backbone_params.update({"minibatch_size": 256})
        backbone_params.update({"n_inner_steps": 1})
        backbone_params.update({"output_size_A": dgp_arch})

        checkpoint_path = os.path.join(rootdir, "checkpoints", "FSBO-metalearn", f"{space}",
                                       datetime.datetime.now().strftime('meta-%Y-%m-%d-%H-%M-%S-%f'))
        backbone_params.update({"checkpoint_path": checkpoint_path})
        self.fixed_context_size = backbone_params["fixed_context_size"]
        self.minibatch_size = backbone_params["minibatch_size"]
        self.n_inner_steps = backbone_params["n_inner_steps"]
        self.checkpoint_path = backbone_params["checkpoint_path"]
        os.makedirs(self.checkpoint_path, exist_ok=False)
        json.dump(backbone_params, open(os.path.join(self.checkpoint_path, "configuration.json"), "w"))
        # self.device = torch.device("cpu") # "cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        logging.basicConfig(filename=os.path.join(self.checkpoint_path, "log.txt"), level=logging.DEBUG)
        self.config = backbone_params
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.gp = ExactGPLayer(train_x=None, train_y=None, likelihood=self.likelihood, config=self.config,
                             dims=self.fixed_context_size)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp).to(self.device)
        self.gp.double()
        self.likelihood.double()
        self.mll.double()
        self.mse = nn.MSELoss()
        # self.mtrloader = get_meta_train_loader(
        #    args.batch_size, args.data_path, args.num_sample)
        # self.get_tasks()
        self.setup_writers()

        self.train_metrics = Metric()
        self.valid_metrics = Metric(prefix="valid: ")

        self.max_n = graph_config['max_n']  # maximum number of vertices
        self.nvt = graph_config['num_vertex_type'] if args.search_space == 'ofa' else args.nvt  # number of vertex types 
        self.START_TYPE = graph_config['START_TYPE']
        self.END_TYPE = graph_config['END_TYPE']
        self.hs = args.hs  # hidden state size of each vertex
        self.nz = args.nz  # size of latent representation z
        self.gs = args.hs  # size of graph state
        self.bidir = True  # whether to use bidirectional encoding
        self.vid = True
        self.input_type = 'DG'
        self.num_sample = args.num_sample

        if self.vid:
            self.vs = self.hs + self.max_n  # vertex state size = hidden state + vid
        else:
            self.vs = self.hs

        # 0. encoding-related
        self.grue_forward = nn.GRUCell(self.nvt, self.hs)  # encoder GRU
        self.grue_backward = nn.GRUCell(self.nvt, self.hs)  # backward encoder GRU
        self.fc1 = nn.Linear(self.gs, self.nz)  # latent mean
        self.fc2 = nn.Linear(self.gs, self.nz)  # latent logvar

        # 2. gate-related
        self.gate_forward = nn.Sequential(
            nn.Linear(self.vs, self.hs),
            nn.Sigmoid()
        )
        self.gate_backward = nn.Sequential(
            nn.Linear(self.vs, self.hs),
            nn.Sigmoid()
        )
        self.mapper_forward = nn.Sequential(
            nn.Linear(self.vs, self.hs, bias=False),
        ) # disable bias to ensure padded zeros also mapped to zeros
        self.mapper_backward = nn.Sequential(
            nn.Linear(self.vs, self.hs, bias=False),
        )

        # 3. bidir-related, to unify sizes
        if self.bidir:
            self.hv_unify = nn.Sequential(
                nn.Linear(self.hs * 2, self.hs),
            )
            self.hg_unify = nn.Sequential(
                nn.Linear(self.gs * 2, self.gs),
            )

        # 4. other
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.logsoftmax1 = nn.LogSoftmax(1)

        # 6. predictor
        np = self.gs
        self.intra_setpool = SetPool(dim_input=512,
                                     num_outputs=1,
                                     dim_output=self.nz,
                                     dim_hidden=self.nz,
                                     mode='sabPF').to(self.device)
        self.inter_setpool = SetPool(dim_input=self.nz,
                                     num_outputs=1,
                                     dim_output=self.nz,
                                     dim_hidden=self.nz,
                                     mode='sabPF').to(self.device)
        self.set_fc = nn.Sequential(
            nn.Linear(512, self.nz),
            nn.ReLU()).to(self.device)

        input_dim = 0
        if 'D' in self.input_type:
            input_dim += self.nz
        if 'G' in self.input_type:
            input_dim += self.nz

        self.pred_fc = StandardDeepGP(backbone_params)
        self.mseloss = nn.MSELoss(reduction='sum')
        # self.nasbench201 = torch.load(
        #     os.path.join(args.data_path, 'nasbench201.pt'))
        self.data_path = args.data_path
        self.pred_fc.to(self.device)
        self.inter_setpool.to(self.device)
        self.intra_setpool.to(self.device)
        self.grue_backward.to(self.device)
        self.grue_forward.to(self.device)
        self.gate_backward.to(self.device)
        self.gate_forward.to(self.device)
        self.mapper_backward.to(self.device)
        self.mapper_forward.to(self.device)
        self.hg_unify.to(self.device)
        self.hv_unify.to(self.device)
        self.fc1.to(self.device)
        self.fc2.to(self.device)

    # def get_topk_idx(self, topk=1):
    #     self.mtrloader.dataset.set_mode('train')
    #     if self.nasbench201 is None:
    #         self.nasbench201 = torch.load(
    #             os.path.join(self.data_path, 'nasbench201.pt'))
    #     z_repr = []
    #     g_repr = []
    #     acc_repr = []
    #     for x, g, acc in tqdm(self.mtrloader):
    #         str = decode_igraph_to_NAS_BENCH_201_string(g[0])
    #         arch_idx = -1
    #         for idx, arch_str in enumerate(self.nasbench201['arch']['str']):
    #             if arch_str == str:
    #                 arch_idx = idx
    #                 break
    #         g_repr.append(arch_idx)
    #         acc_repr.append(acc.detach().cpu().numpy()[0])
    #     best = np.argsort(-1 * np.array(acc_repr))[:topk]
    #     self.nasbench201 = None
    #     return np.array(g_repr)[best], np.array(acc_repr)[best]

    def randomly_init_deepgp(self, ):
        self.pred_fc = StandardDeepGP(self.config)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.gp = ExactGPLayer(train_x=None, train_y=None, likelihood=self.likelihood, config=self.config,
                             dims=self.fixed_context_size)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp).to(self.device)


    def setup_writers(self, ):
        train_log_dir = os.path.join(self.checkpoint_path, "train")
        os.makedirs(train_log_dir, exist_ok=True)
        # self.train_summary_writer = SummaryWriter(train_log_dir)

        valid_log_dir = os.path.join(self.checkpoint_path, "valid")
        os.makedirs(valid_log_dir, exist_ok=True)
        # self.valid_summary_writer = SummaryWriter(valid_log_dir)

    def get_mu_and_std(self, x_support, y_support, x_query, y_query):
        if x_support is not None:
            x_support.to(self.device)
            y_support.to(self.device)

            self.gp.set_train_data(inputs=x_support, targets=y_support, strict=False)
        self.gp.to(self.device)
        self.gp.eval()
        self.likelihood.eval()
        pred = self.likelihood(self.gp(x_query.to(self.device)))
        mu = pred.mean.detach().to("cpu").numpy().reshape(-1, )
        stddev = pred.stddev.detach().to("cpu").numpy().reshape(-1, )
        return mu, stddev

    def predict_finetune(self, z, labels=None, train=False):
        if len(labels) > 1:
            z = torch.squeeze(z)
        if train:
            self.gp.set_train_data(inputs=z, targets=labels, strict=False)
        y_dist = self.gp(z)
        predictions = self.likelihood(y_dist)
        return predictions.mean, y_dist

    def predict(self, D_mu, G_mu, labels=None, train=False):
        input_vec = []
        if 'D' in self.input_type:
            input_vec.append(D_mu)
        if 'G' in self.input_type:
            input_vec.append(G_mu)
        print(input_vec)
        input_vec = torch.cat(input_vec, dim=1)
        z = self.pred_fc(input_vec).double()
        if train:
            self.gp.set_train_data(inputs=z, targets=labels, strict=False)
        y_dist = self.gp(z.type(torch.DoubleTensor))
        predictions = self.likelihood(y_dist)
        return predictions.mean, y_dist

    def get_data_and_graph_repr(self, x, g, matrix=False):
        input_vec = []
        # self.pred_fc.to(self.device)
        self.pred_fc.eval()
        # self.inter_setpool.to(self.device)
        self.inter_setpool.eval()
        # self.intra_setpool.to(self.device)
        self.intra_setpool.eval()
        # self.grue_backward.to(self.device)
        self.grue_backward.eval()
        # self.grue_forward.to(self.device)
        self.grue_forward.eval()
        # self.gate_backward.to(self.device)
        self.gate_backward.eval()
        # self.gate_forward.to(self.device)
        self.gate_forward.eval()
        # self.mapper_backward.to(self.device)
        self.mapper_backward.eval()
        # self.mapper_forward.to(self.device)
        self.mapper_forward.eval()
        # self.hg_unify.to(self.device)
        self.hg_unify.eval()
        # self.hv_unify.to(self.device)
        self.hv_unify.eval()
        # self.fc1.to(self.device)
        self.fc1.eval()
        # self.fc2.to(self.device)
        self.fc2.eval()
        if 'D' in self.input_type:
            input_vec.append(self.set_encode([x for i in range(len(g))]).to(self.device))
        if 'G' in self.input_type:
            input_vec.append(self.graph_encode(g, matrix=matrix).squeeze())
        # print(input_vec)
        if len(g) == 1:
            input_vec = torch.cat(input_vec, dim=0)
            print(input_vec)
        else:
            input_vec = torch.cat(input_vec, dim=1)
        z = self.pred_fc(input_vec)
        return z.detach().cpu().numpy().tolist()

    def get_device(self):
        if self.device is None:
            self.device = next(self.parameters()).device
        return self.device

    def _get_zeros(self, n, length):
        return torch.zeros(n, length).to(self.get_device())  # get a zero hidden state

    def _get_zero_hidden(self, n=1):
        return self._get_zeros(n, self.hs)  # get a zero hidden state

    def _one_hot(self, idx, length):
        if type(idx) in [list, range]:
            if idx == []:
                return None
            idx = torch.LongTensor(idx).unsqueeze(0).t()
            x = torch.zeros((len(idx), length)).scatter_(1, idx, 1).to(self.get_device())
        else:
            idx = torch.LongTensor([idx]).unsqueeze(0)
            x = torch.zeros((1, length)).scatter_(1, idx, 1).to(self.get_device())
        return x

    def _gated(self, h, gate, mapper):
        return gate(h) * mapper(h)

    def _collate_fn(self, G):
        return [g.copy() for g in G]

    def _propagate_to(self, G, v, propagator, H=None, reverse=False, gate=None, mapper=None):
        # propagate messages to vertex index v for all graphs in G
        # return the new messages (states) at v
        G = [g for g in G if g.vcount() > v]
        if len(G) == 0:
            return
        if H is not None:
            idx = [i for i, g in enumerate(G) if g.vcount() > v]
            H = H[idx]
        v_types = [g.vs[v]['type'] for g in G]
        X = self._one_hot(v_types, self.nvt)
        if reverse:
            H_name = 'H_backward'  # name of the hidden states attribute
            H_pred = [[g.vs[x][H_name] for x in g.successors(v)] for g in G]
            if self.vid:
                vids = [self._one_hot(g.successors(v), self.max_n) for g in G]
            gate, mapper = self.gate_backward, self.mapper_backward
        else:
            H_name = 'H_forward'  # name of the hidden states attribute
            H_pred = [[g.vs[x][H_name] for x in g.predecessors(v)] for g in G]
            if self.vid:
                vids = [self._one_hot(g.predecessors(v), self.max_n) for g in G]
            if gate is None:
                gate, mapper = self.gate_forward, self.mapper_forward
        if self.vid:
            H_pred = [[torch.cat([x[i], y[i:i + 1]], 1) for i in range(len(x))] for x, y in zip(H_pred, vids)]
        # if h is not provided, use gated sum of v's predecessors' states as the input hidden state
        if H is None:
            max_n_pred = max([len(x) for x in H_pred])  # maximum number of predecessors
            if max_n_pred == 0:
                H = self._get_zero_hidden(len(G))
            else:
                H_pred = [torch.cat(h_pred +
                                    [self._get_zeros(max_n_pred - len(h_pred), self.vs)], 0).unsqueeze(0)
                          for h_pred in H_pred]  # pad all to same length
                H_pred = torch.cat(H_pred, 0)  # batch * max_n_pred * vs
                H = self._gated(H_pred, gate, mapper).sum(1)  # batch * hs
        Hv = propagator(X, H)
        for i, g in enumerate(G):
            g.vs[v][H_name] = Hv[i:i + 1]
        return Hv

    def _propagate_from(self, G, v, propagator, H0=None, reverse=False):
        # perform a series of propagation_to steps starting from v following a topo order
        # assume the original vertex indices are in a topological order
        if reverse:
            prop_order = range(v, -1, -1)
        else:
            prop_order = range(v, self.max_n)
        Hv = self._propagate_to(G, v, propagator, H0, reverse=reverse)  # the initial vertex
        for v_ in prop_order[1:]:
            self._propagate_to(G, v_, propagator, reverse=reverse)
        return Hv

    def _get_graph_state(self, G, decode=False):
        # get the graph states
        # when decoding, use the last generated vertex's state as the graph state
        # when encoding, use the ending vertex state or unify the starting and ending vertex states
        Hg = []
        for g in G:
            hg = g.vs[g.vcount() - 1]['H_forward']
            if self.bidir and not decode:  # decoding never uses backward propagation
                hg_b = g.vs[0]['H_backward']
                hg = torch.cat([hg, hg_b], 1)
            Hg.append(hg)
        Hg = torch.cat(Hg, 0)
        if self.bidir and not decode:
            Hg = self.hg_unify(Hg)
        return Hg

    def set_encode(self, X):
        proto_batch = []
        for x in X:
            cls_protos = self.intra_setpool(
                    x.view(-1, self.num_sample, 512)).squeeze(1)
            proto_batch.append(
                self.inter_setpool(cls_protos.unsqueeze(0)))
        v = torch.stack(proto_batch).squeeze()
        return v

    def graph_encode(self, G, matrix=False):
        # encode graphs G into latent vectors
        if matrix:
            mu = torch.Tensor([decode_igraph_to_NAS201_matrix(g).flatten() for g in G]).to(self.device)
        else:
            if type(G) != list:
                G = [G]
            self._propagate_from(G, 0, self.grue_forward, H0=self._get_zero_hidden(len(G)),
                                 reverse=False)
            if self.bidir:
                self._propagate_from(G, self.max_n - 1, self.grue_backward,
                                     H0=self._get_zero_hidden(len(G)), reverse=True)
            Hg = self._get_graph_state(G)
            mu = self.fc1(Hg)
            # logvar = self.fc2(Hg)
        return mu  # , logvar

    def reparameterize(self, mu, logvar, eps_scale=0.01):
        # return z ~ N(mu, std)
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu
