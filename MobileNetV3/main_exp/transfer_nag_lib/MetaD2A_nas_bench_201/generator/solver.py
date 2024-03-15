###########################################################################################
# Copyright (c) Hayeon Lee, Eunyoung Hyung [GitHub MetaD2A], 2021
# Rapid Neural Architecture Search by Learning to Generate Graphs from Datasets, ICLR 2021
###########################################################################################
from __future__ import print_function
import os
import sys
import random
from tqdm import tqdm
import numpy as np
import time

import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .generator_model import GeneratorModel
from transfer_nag_lib.MetaD2A_nas_bench_201.loader import get_meta_train_loader, get_meta_test_loader
from transfer_nag_lib.MetaD2A_nas_bench_201.metad2a_utils import load_model, save_model
from transfer_nag_lib.MetaD2A_nas_bench_201.metad2a_utils import Accumulator, Log
from transfer_nag_lib.MetaD2A_nas_bench_201.metad2a_utils import load_graph_config, decode_igraph_to_NAS_BENCH_201_string, decode_igraph_to_NAS201_matrix, \
    decode_NAS_BENCH_201_8_to_igraph


class Generator:
    def __init__(self, args):
        self.args = args
        self.batch_size = args.batch_size
        self.data_path = args.data_path
        self.num_sample = args.num_sample
        self.max_epoch = args.max_epoch
        self.save_epoch = args.save_epoch
        self.model_path = args.model_path
        self.save_path = args.save_path
        self.model_name = args.model_name
        self.test = args.test
        self.device = torch.device("cpu")

        graph_config = load_graph_config(
            args.graph_data_name, args.nvt, args.data_path)
        self.model = GeneratorModel(args, graph_config)
        self.nasbench201 = None
        self.model.to(self.device)

        if self.test:
            self.data_name = args.data_name
            self.num_class = args.num_class
            self.load_epoch = args.load_epoch
            self.num_gen_arch = 10  # args.num_gen_arch
            self.mtrloader = get_meta_train_loader(
                self.batch_size, self.data_path, self.num_sample, True)
            load_model(self.model, self.model_path, self.load_epoch)

        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
            self.scheduler = ReduceLROnPlateau(self.optimizer, 'min',
                                               factor=0.1, patience=10, verbose=True)
            self.mtrloader = get_meta_train_loader(
                self.batch_size, self.data_path, self.num_sample)
            self.mtrlog = Log(self.args, open(os.path.join(
                self.save_path, self.model_name, 'meta_train_generator.log'), 'w'))
            self.mtrlog.print_args()
            self.mtrlogger = Accumulator('loss', 'recon_loss', 'kld')
            self.mvallogger = Accumulator('loss', 'recon_loss', 'kld')

    def meta_train(self):
        sttime = time.time()
        for epoch in range(1, self.max_epoch + 1):
            self.mtrlog.ep_sttime = time.time()
            loss = self.meta_train_epoch(epoch)
            self.scheduler.step(loss)
            self.mtrlog.print(self.mtrlogger, epoch, tag='train')

            self.meta_validation()
            self.mtrlog.print(self.mvallogger, epoch, tag='valid')

            if epoch % self.save_epoch == 0:
                save_model(epoch, self.model, self.model_path)

        self.mtrlog.save_time_log()

    def meta_train_epoch(self, epoch):
        self.model.to(self.device)
        self.model.train()
        train_loss, recon_loss, kld_loss = 0, 0, 0

        self.mtrloader.dataset.set_mode('train')
        for x, g, acc in tqdm(self.mtrloader):
            self.optimizer.zero_grad()
            mu, logvar = self.model.set_encode(x.to(self.device))
            loss, recon, kld = self.model.loss(mu, logvar, g)
            loss.backward()
            self.optimizer.step()

            cnt = len(x)
            self.mtrlogger.accum([loss.item()/cnt,
                                  recon.item()/cnt,
                                  kld.item()/cnt])
        return self.mtrlogger.get('loss')

    def meta_validation(self):
        self.model.to(self.device)
        self.model.eval()
        train_loss, recon_loss, kld_loss = 0, 0, 0

        self.mtrloader.dataset.set_mode('valid')
        for x, g, acc in tqdm(self.mtrloader):
            with torch.no_grad():
                mu, logvar = self.model.set_encode(x.to(self.device))
                loss, recon, kld = self.model.loss(mu, logvar, g)

            cnt = len(x)
            self.mvallogger.accum([loss.item()/cnt,
                                   recon.item()/cnt,
                                   kld.item()/cnt])
        return self.mvallogger.get('loss')

    def meta_test(self):
        if self.data_name == 'all':
            for data_name in ['cifar100', 'cifar10', 'mnist', 'svhn', 'aircraft', 'pets']:
                self.meta_test_per_dataset(data_name)
        else:
            self.meta_test_per_dataset(self.data_name)

    def get_topk_idx(self, topk=1):
        self.mtrloader.dataset.set_mode('train')
        if self.nasbench201 is None:
            self.nasbench201 = torch.load(
                os.path.join(self.data_path, 'nasbench201.pt'))
        z_repr = []
        g_repr = []
        acc_repr = []
        for x, g, acc in tqdm(self.mtrloader):
            str = decode_igraph_to_NAS_BENCH_201_string(g[0])
            arch_idx = -1
            for idx, arch_str in enumerate(self.nasbench201['arch']['str']):
                if arch_str == str:
                    arch_idx = idx
                    break
            g_repr.append(arch_idx)
            acc_repr.append(acc.detach().cpu().numpy()[0])
        best = np.argsort(-1*np.array(acc_repr))[:topk]
        return np.array(g_repr)[best], np.array(acc_repr)[best]

    def topk_train(self, topk=1):
        self.mtrloader.dataset.set_mode('train')
        z_repr = []
        g_repr = []
        acc_repr = []
        for x, g, acc in tqdm(self.mtrloader):
            str = decode_igraph_to_NAS_BENCH_201_string(g[0])
            g_repr.append(str)
            acc_repr.append(acc.detach().cpu().numpy()[0])
        best = np.argsort(-1*np.array(acc_repr))[:topk]
        return np.array(g_repr)[best], np.array(acc_repr)[best]

    # def decode_igraph_to_NAS201(self, graph, encoding):
    #     str = decode_igraph_to_NAS_BENCH_201_string(graph)
    #     op_indices = convert_str_to_op_indices(str)
    #     naslib_object = NasBench201SearchSpace()
    #     convert_op_indices_to_naslib(op_indices=op_indices, naslib_object=naslib_object)
    #     enc = encode_201(arch=naslib_object, encoding_type=encoding)
    #     return enc

    # def decode_str_to_NAS201(self, str, encoding):
    #     op_indices = convert_str_to_op_indices(str)
    #     naslib_object = NasBench201SearchSpace()
    #     convert_op_indices_to_naslib(op_indices=op_indices, naslib_object=naslib_object)
    #     enc = encode_201(arch=naslib_object, encoding_type=encoding)
    #     return enc

    def train_dgp(self, encoding='path', encode=False):
        self.model.to(self.device)
        self.model.eval()

        self.mtrloader.dataset.set_mode('train')
        z_repr = []
        g_repr = []
        acc_repr = []
        for x, g, acc in tqdm(self.mtrloader):
            sys.stdout.flush()
            mu, logvar = self.model.set_encode(x.to(self.device))
            z = self.model.reparameterize(
                mu, logvar).cpu().detach().numpy().flatten()
            if encode:
                graph_matrix = self.decode_igraph_to_NAS201(
                    g[0], encoding=encoding)
            else:
                graph_matrix = decode_igraph_to_NAS201_matrix(g[0]).flatten()
            z_repr.append(np.concatenate((z, graph_matrix)))
            g_repr.append(graph_matrix)
            acc_repr.append(acc.detach().cpu().numpy()[0])
        return z_repr, g_repr, acc_repr

    def test_dgp(self, data_name='cifar10', encoding='path', encode=False):
        meta_test_path = os.path.join(
            self.save_path, 'meta_test', data_name, 'generated_arch')
        if not os.path.exists(meta_test_path):
            os.makedirs(meta_test_path)

        meta_test_loader = get_meta_test_loader(
            self.data_path, data_name, self.num_sample, self.num_class)

        print(f'==> generate architectures for {data_name}')
        inputs = []
        accs = []
        inputs_, accs_ = self.generate_architectures_dgp(
            meta_test_loader, data_name,
            meta_test_path, self.num_gen_arch, encoding=encoding, encode=encode)
        inputs.extend(inputs_)
        accs.extend(accs_)
        print(f'==> done\n')
        return np.array(inputs), np.array(accs)

    def generate_architectures_dgp(self,
                                   meta_test_loader, data_name, meta_test_path, num_gen_arch, encoding='path', encode=False):
        self.nasbench201 = torch.load(
            os.path.join(self.data_path, 'nasbench201.pt'))
        overall_arch_num = len(self.nasbench201['arch']['str'])
        self.model.eval()
        self.model.to(self.device)

        dataset_arch_repr = []
        acc_repr = []
        for x in meta_test_loader:
            mu, logvar = self.model.set_encode(x.to(self.device))
            z = self.model.reparameterize(mu, logvar).cpu().detach().numpy()[0]
            break
        with torch.no_grad():
            for i in range(overall_arch_num):
                if encode:
                    arch_str = self.nasbench201['arch']['str'][i]
                    arch = self.decode_str_to_NAS201(
                        arch_str, encoding=encoding)
                else:
                    arch_str = self.nasbench201['arch']['matrix'][i]
                    igraph, n = decode_NAS_BENCH_201_8_to_igraph(arch_str)
                    arch = decode_igraph_to_NAS201_matrix(igraph).flatten()
                dataset_arch_repr.append(np.concatenate((z, arch)))
                acc_repr.append(self.nasbench201['test-acc'][data_name][i])
                if i % 1000 == 0:
                    print(i)

        return dataset_arch_repr, acc_repr

    def get_items(self, full_target, full_source, source):
        return [full_target[full_source.index(_)] for _ in source]

    def meta_test_per_dataset(self, data_name):
        meta_test_path = os.path.join(
            self.save_path, 'meta_test', data_name, 'generated_arch')
        if not os.path.exists(meta_test_path):
            os.makedirs(meta_test_path)

        meta_test_loader = get_meta_test_loader(
            self.data_path, data_name, self.num_sample, self.num_class)

        print(f'==> generate architectures for {data_name}')
        runs = 10 if data_name in ['cifar10', 'cifar100'] else 1
        elasped_time = []
        for run in range(1, runs+1):
            print(f'==> run {run}/{runs}')
            elasped_time.append(self.generate_architectures(
                meta_test_loader, data_name,
                meta_test_path, run, self.num_gen_arch))
            print(f'==> done\n')

        time_path = os.path.join(
            self.save_path, 'meta_test', data_name, 'time.txt')
        with open(time_path, 'w') as f_time:
            msg = f'generator elasped time {np.mean(elasped_time):.2f}s'
            print(f'==> save time in {time_path}')
            f_time.write(msg+'\n')
            print(msg)

    def generate_architectures(self,
                               meta_test_loader, data_name, meta_test_path, run, num_gen_arch):
        self.model.eval()
        self.model.to(self.device)

        architecture_string_lst = []
        total_cnt, valid_cnt = 0, 0
        flag = False

        start = time.time()
        with torch.no_grad():
            for x in meta_test_loader:
                mu, logvar = self.model.set_encode(x.to(self.device))
                z = self.model.reparameterize(mu, logvar)
                generated_graph_lst = self.model.graph_decode(z)
                for g in generated_graph_lst:
                    architecture_string = decode_igraph_to_NAS_BENCH_201_string(
                        g)
                    total_cnt += 1
                    if architecture_string is not None:
                        if not architecture_string in architecture_string_lst:
                            valid_cnt += 1
                            architecture_string_lst.append(architecture_string)
                            if valid_cnt == num_gen_arch:
                                flag = True
                                break
                if flag:
                    break
        elapsed = time.time()-start

        spath = os.path.join(meta_test_path, f"run_{run}.txt")
        with open(spath, 'w') as f:
            print(f'==> save generated architectures in {spath}')
            msg = f'elapsed time: {elapsed:6.2f}s '
            print(msg)
            f.write(msg+'\n')
            for i, architecture_string in enumerate(architecture_string_lst):
                f.write(f"{architecture_string}\n")
        return elapsed
