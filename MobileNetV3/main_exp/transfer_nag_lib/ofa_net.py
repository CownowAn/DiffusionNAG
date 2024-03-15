import numpy as np
import copy
import itertools
import random
import sys
import os
import pickle
import torch
from torch.nn.functional import one_hot

# from ofa.imagenet_classification.run_manager import RunManager

# from naszilla.nas_bench_201.distances import *

# INPUT = 'input'
# OUTPUT = 'output'
# OPS = ['avg_pool_3x3', 'nor_conv_1x1', 'nor_conv_3x3', 'none', 'skip_connect']
# NUM_OPS = len(OPS)
# OP_SPOTS = 20
KS_LIST = [3, 5, 7] 
EXPAND_LIST = [3, 4, 6]
DEPTH_LIST = [2, 3, 4] 
NUM_STAGE = 5
MAX_LAYER_PER_STAGE = 4
MAX_N_BLOCK= NUM_STAGE * MAX_LAYER_PER_STAGE # 20
OPS = {
    '3-3': 0, '3-4': 1, '3-6': 2,
    '5-3': 3, '5-4': 4, '5-6': 5,
    '7-3': 6, '7-4': 7, '7-6': 8,
    }

OPS2STR = {
    0: '3-3', 1: '3-4', 2: '3-6',
    3: '5-3', 4: '5-4', 5: '5-6',
    6: '7-3', 7: '7-4', 8: '7-6',
    }
NUM_OPS = len(OPS)
LONGEST_PATH_LENGTH = 20
# OP_SPOTS = NUM_VERTICES - 2

# OPS = {
#     '3-3': 0, '3-4': 1, '3-6': 2,
#     '5-3': 3, '5-4': 4, '5-6': 5,
#     '7-3': 6, '7-4': 7, '7-6': 8,
#     }
# OFA evolution hyper-parameters
# self.arch_mutate_prob = kwargs.get("arch_mutate_prob", 0.1)
# self.resolution_mutate_prob = kwargs.get("resolution_mutate_prob", 0.5)
# self.population_size = kwargs.get("population_size", 100)
# self.max_time_budget = kwargs.get("max_time_budget", 500)
# self.parent_ratio = kwargs.get("parent_ratio", 0.25)
# self.mutation_ratio = kwargs.get("mutation_ratio", 0.5)


class OFASubNet:

    def __init__(self, string, accuracy_predictor=None):
        self.string = string
        self.accuracy_predictor = accuracy_predictor
        # self.run_config = run_config

    def get_string(self):
        return self.string

    def serialize(self):
        return {
            'string':self.string
        }

    @classmethod
    def random_cell(cls, 
                    nasbench, 
                    max_nodes=None, 
                    max_edges=None,
                    cutoff=None,
                    index_hash=None,
                    random_encoding=None):
        """
        OFA sample random subnet
        """
        # Randomly sample sub-networks from OFA network
        random_subnet_config = nasbench.sample_active_subnet()
        #{
        #     "ks": ks_setting,
        #     "e": expand_setting,
        #     "d": depth_setting,
        # }

        return {'string':cls.get_string_from_ops(random_subnet_config)}

    def encode(self, 
               predictor_encoding, 
               nasbench=None, 
               deterministic=True, 
               cutoff=None, 
               nasbench_ours=None,
               dataset=None):

        if predictor_encoding == 'adj':
            return self.encode_standard()
        elif predictor_encoding == 'path':
            raise NotImplementedError
            return self.encode_paths()
        elif predictor_encoding == 'trunc_path':
            if not cutoff:
                cutoff = 30
            dic = self.gcn_encoding(nasbench, 
                                     deterministic=deterministic,
                                     nasbench_ours=nasbench_ours,
                                     dataset=dataset)
            dic['trunc_path'] = self.encode_freq_paths(cutoff=cutoff)
            return dic
            # return self.encode_freq_paths(cutoff=cutoff)
        elif predictor_encoding == 'gcn':
            return self.gcn_encoding(nasbench, 
                                     deterministic=deterministic,
                                     nasbench_ours=nasbench_ours,
                                     dataset=dataset)
        else:
            print('{} is an invalid predictor encoding'.format(predictor_encoding))
            raise NotImplementedError()

    def get_ops_onehot(self):
        ops = self.get_op_dict()
        # ops = [INPUT, *ops, OUTPUT]
        node_types = torch.zeros(NUM_STAGE * MAX_LAYER_PER_STAGE).long() # w/o in / out
        num_vertices = len(OPS.values())
        num_nodes = NUM_STAGE * MAX_LAYER_PER_STAGE
        d_matrix = []
        # import pdb; pdb.set_trace()
        for i in range(NUM_STAGE):
            ds = ops['d'][i]
            for j in range(ds):
                d_matrix.append(ds)

            for j in range(MAX_LAYER_PER_STAGE - ds):
                d_matrix.append('none')

        for i, (ks, e, d) in enumerate(zip(
                ops['ks'], ops['e'], d_matrix)):
            if d == 'none':
                # node_types[i] = OPS[d]
                pass
            else:
                node_types[i] = OPS[f'{ks}-{e}']

        ops_onehot = one_hot(node_types, num_vertices).float()
        return ops_onehot


    def gcn_encoding(self, nasbench, deterministic, nasbench_ours=None, dataset=None):
        
        # op_map = [OUTPUT, INPUT, *OPS]
        ops = self.get_op_dict()
        # ops = [INPUT, *ops, OUTPUT]
        node_types = torch.zeros(NUM_STAGE * MAX_LAYER_PER_STAGE).long() # w/o in / out
        num_vertices = len(OPS.values())
        num_nodes = NUM_STAGE * MAX_LAYER_PER_STAGE
        d_matrix = []
        # import pdb; pdb.set_trace()
        for i in range(NUM_STAGE):
            ds = ops['d'][i]
            for j in range(ds):
                d_matrix.append(ds)

            for j in range(MAX_LAYER_PER_STAGE - ds):
                d_matrix.append('none')

        for i, (ks, e, d) in enumerate(zip(
                ops['ks'], ops['e'], d_matrix)):
            if d == 'none':
                # node_types[i] = OPS[d]
                pass
            else:
                node_types[i] = OPS[f'{ks}-{e}']

        ops_onehot = one_hot(node_types, num_vertices).float()
        val_loss = self.get_val_loss(nasbench, dataset=dataset) 
        test_loss = copy.deepcopy(val_loss)
        # (num node, ops types) --> (20, 28)
        def get_adj():
            adj = torch.zeros(num_nodes, num_nodes)
            for i in range(num_nodes-1):
                adj[i, i+1] = 1
            adj = np.array(adj)
            return adj
        
        matrix = get_adj()

        dic = {
            'num_vertices': num_vertices,
            'adjacency': matrix,
            'operations': ops_onehot,
            'mask': np.array([i < num_vertices for i in range(num_vertices)], dtype=np.float32),
            'val_acc': 1.0 - val_loss,
            'test_acc': 1.0 - test_loss,
            'x': ops_onehot
        }

        return dic

    def get_runtime(self, nasbench, dataset='cifar100'):
        return nasbench.query_by_index(index, dataset).get_eval('x-valid')['time']

    def get_val_loss(self, nasbench, deterministic=1, dataset='cifar100'):
        assert dataset == 'imagenet1k'
        # SuperNet version
        # ops = self.get_op_dict()
        # nasbench.set_active_subnet(ks=ops['ks'], e=ops['e'], d=ops['d']) 

        # subnet = nasbench.get_active_subnet(preserve_weight=True)
        # run_manager = RunManager(".tmp/eval_subnet", subnet, self.run_config, init=False)
        # # assign image size: 128, 132, ..., 224
        # self.run_config.data_provider.assign_active_img_size(224)
        # run_manager.reset_running_statistics(net=subnet)

        # loss, (top1, top5) = run_manager.validate(net=subnet)
        # # print("Results: loss=%.5f,\t top1=%.1f,\t top5=%.1f" % (loss, top1, top5))  
        # self.loss = loss
        # self.top1 = top1
        # self.top5 = top5
        
        # accuracy predictor version
        ops = self.get_op_dict()
        # resolutions = [160, 176, 192, 208, 224]
        # ops['r'] = [random.choice(resolutions)]
        ops['r'] = [224]
        acc = self.accuracy_predictor.predict_accuracy([ops])[0][0].item()
        return 1.0 - acc
    
    def get_test_loss(self, nasbench, dataset='cifar100', deterministic=1):
        ops = self.get_op_dict()
        # resolutions = [160, 176, 192, 208, 224]
        # ops['r'] = [random.choice(resolutions)]
        ops['r'] = [224]
        acc = self.accuracy_predictor.predict_accuracy([ops])[0][0].item()
        return 1.0 - acc
    
    def get_op_dict(self):
        # given a string, get the list of operations
        ops = { 
            "ks": [], "e": [], "d": []
            }
        tokens = self.string.split('_')
        for i, token in enumerate(tokens):
            d, ks, e = token.split('-')
            if i % MAX_LAYER_PER_STAGE == 0:
                ops['d'].append(int(d))
            ops['ks'].append(int(ks))
            ops['e'].append(int(e))
        return ops

    def get_num(self):
        # compute the unique number of the architecture, in [0, 15624]
        ops = self.get_op_dict()
        index = 0
        for i, op in enumerate(ops):
            index += OPS.index(op) * NUM_OPS ** i
        return index

    def get_random_hash(self):
        num = self.get_num()
        hashes = pickle.load(open('nas_bench_201/random_hash.pkl', 'rb'))
        return hashes[num]

    @classmethod
    def get_string_from_ops(cls, ops):
        string = ''
        for i, (ks, e) in enumerate(zip(ops['ks'], ops['e'])):
            d = ops['d'][int(i/MAX_LAYER_PER_STAGE)]
            string += f'{d}-{ks}-{e}_'
        return string[:-1]

    def perturb(self, 
                nasbench,
                mutation_rate=1):
        # deterministic version of mutate
        ops = self.get_op_dict()
        new_ops = []
        num = np.random.choice(len(ops))
        for i, op in enumerate(ops):
            if i == num:
                available = [o for o in OPS if o != op]
                new_ops.append(np.random.choice(available))
            else:
                new_ops.append(op)
        return {'string':self.get_string_from_ops(new_ops)}

    def mutate(self, 
               nasbench, 
               mutation_rate=0.1, 
               mutate_encoding='adj',
               index_hash=None,
               cutoff=30,
               patience=5000):
        p = 0
        mutation_rate = mutation_rate / 10 # OFA rate: 0.1
        arch_dict = self.get_op_dict()

        if mutate_encoding == 'adj':
            # OFA version mutation 
            # https://github.com/mit-han-lab/once-for-all/blob/master/ofa/nas/search_algorithm/evolution.py
            for i in range(MAX_N_BLOCK):
                if random.random() < mutation_rate:
                    available_ks = [ks for ks in KS_LIST if ks != arch_dict["ks"][i]]
                    available_e = [e for e in EXPAND_LIST if e != arch_dict["e"][i]]
                    arch_dict["ks"][i] = random.choice(available_ks)
                    arch_dict["e"][i] = random.choice(available_e)

            for i in range(NUM_STAGE):
                if random.random() < mutation_rate:
                    available_d = [d for d in DEPTH_LIST if d != arch_dict["d"][i]]
                    arch_dict["d"][i] = random.choice(available_d)
            return {'string':self.get_string_from_ops(arch_dict)}

        elif mutate_encoding in ['path', 'trunc_path']:
            raise NotImplementedError()
        else:
            print('{} is an invalid mutate encoding'.format(mutate_encoding))
            raise NotImplementedError()

    def encode_standard(self):
        """ 
        compute the standard encoding
        """
        ops = self.get_op_dict()
        encoding = []
        for i, (ks, e) in enumerate(zip(ops['ks'], ops['e'])):
            string = f'{ks}-{e}'
            encoding.append(OPS[string])
        return encoding

    def encode_one_hot(self):
        """
        compute the one-hot encoding
        """
        encoding = self.encode_standard()
        one_hot = []
        for num in encoding:
            for i in range(len(OPS)):
                if i == num:
                    one_hot.append(1)
                else:
                    one_hot.append(0)
        return one_hot

    def get_num_params(self, nasbench):
        # todo: add this method
        return 100

    def get_paths(self):
        """ 
        return all paths from input to output
        """
        path_blueprints = [[3], [0,4], [1,5], [0,2,5]]
        ops = self.get_op_dict()
        paths = []
        for blueprint in path_blueprints:
            paths.append([ops[node] for node in blueprint])
        return paths

    def get_path_indices(self):
        """
        compute the index of each path
        """
        paths = self.get_paths()
        path_indices = []

        for i, path in enumerate(paths):
            if i == 0:
                index = 0
            elif i in [1, 2]:
                index = NUM_OPS
            else:
                index = NUM_OPS + NUM_OPS ** 2
            import pdb; pdb.set_trace()
            for j, op in enumerate(path):
                index += OPS.index(op) * NUM_OPS ** j
            path_indices.append(index)

        return tuple(path_indices)

    def encode_paths(self):
        """ output one-hot encoding of paths """
        num_paths = sum([NUM_OPS ** i for i in range(1, LONGEST_PATH_LENGTH + 1)])
        path_indices = self.get_path_indices()
        encoding = np.zeros(num_paths)
        for index in path_indices:
            encoding[index] = 1
        return encoding

    def encode_freq_paths(self, cutoff=30):
        # natural cutoffs 5, 30, 155 (last)
        num_paths = sum([NUM_OPS ** i for i in range(1, LONGEST_PATH_LENGTH + 1)])
        path_indices = self.get_path_indices()
        encoding = np.zeros(cutoff)
        for index in range(min(num_paths, cutoff)):
            if index in path_indices:
                encoding[index] = 1
        return encoding

    def distance(self, other, dist_type, cutoff=30):
        if dist_type == 'adj':
            distance = adj_distance(self, other)
        elif dist_type == 'path':
            distance = path_distance(self, other)        
        elif dist_type == 'trunc_path':
            distance = path_distance(self, other, cutoff=cutoff)
        elif dist_type == 'nasbot':
            distance = nasbot_distance(self, other)
        else:
            print('{} is an invalid distance'.format(distance))
            raise NotImplementedError()
        return distance


    def get_neighborhood(self, 
                         nasbench, 
                         mutate_encoding,
                         shuffle=True):
        nbhd = []
        ops = self.get_op_dict()

        if mutate_encoding == 'adj':
            # OFA version mutation variation
            # https://github.com/mit-han-lab/once-for-all/blob/master/ofa/nas/search_algorithm/evolution.py
            for i in range(MAX_N_BLOCK):
                available_ks = [ks for ks in KS_LIST if ks != ops["ks"][i]]
                for ks in available_ks:
                    new_ops = ops.copy()
                    new_ops["ks"][i] = ks
                    new_arch = {'string':self.get_string_from_ops(new_ops)}
                    nbhd.append(new_arch)

                available_e = [e for e in EXPAND_LIST if e != ops["e"][i]]
                for e in available_e:
                    new_ops = ops.copy()
                    new_ops["e"][i] = e
                    new_arch = {'string':self.get_string_from_ops(new_ops)}
                    nbhd.append(new_arch)
            # for i in range(MAX_N_BLOCK):
            #     available_ks = [ks for ks in KS_LIST if ks != ops["ks"][i]]
            #     available_e = [e for e in EXPAND_LIST if e != ops["e"][i]]
            #     for ks, e in zip(available_ks, available_e):
            #         new_ops = ops.copy()
            #         new_ops["ks"][i] = ks
            #         new_ops["e"][i] = e
            #         new_arch = {'string':self.get_string_from_ops(new_ops)}
            #         nbhd.append(new_arch)

            for i in range(NUM_STAGE):
                available_d = [d for d in DEPTH_LIST if d != ops["d"][i]]
                for d in available_d:
                    new_ops = ops.copy()
                    new_ops["d"][i] = d
                    new_arch = {'string':self.get_string_from_ops(new_ops)}
                    nbhd.append(new_arch)

        # if mutate_encoding == 'adj':
        #     for i in range(len(ops)):
        #         import pdb; pdb.set_trace()
        #         available = [op for op in OPS.keys() if op != ops[i]]
        #         for op in available:
        #             new_ops = ops.copy()
        #             new_ops[i] = op
        #             new_arch = {'string':self.get_string_from_ops(new_ops)}
        #             nbhd.append(new_arch)

        elif mutate_encoding in ['path', 'trunc_path']:

            if mutate_encoding == 'trunc_path':
                path_blueprints = [[3], [0,4], [1,5]]
            else:
                path_blueprints = [[3], [0,4], [1,5], [0,2,5]]
            ops = self.get_op_dict()

            for blueprint in path_blueprints:
                for new_path in itertools.product(OPS, repeat=len(blueprint)):
                    new_ops = ops.copy()

                    for i, op in enumerate(new_path):
                        new_ops[blueprint[i]] = op

                        # check if it's the same
                        same = True
                        for j in range(len(ops)):
                            if ops[j] != new_ops[j]:
                                same = False
                        if not same:
                            new_arch = {'string':self.get_string_from_ops(new_ops)}
                            nbhd.append(new_arch)
        else:
            print('{} is an invalid mutate encoding'.format(mutate_encoding))
            raise NotImplementedError()

        if shuffle:
            random.shuffle(nbhd)                
        return nbhd


    def get_unique_string(self):
        ops = self.get_op_dict()
        d_matrix = []
        for i in range(NUM_STAGE):
            ds = ops['d'][i]
            for j in range(ds):
                d_matrix.append(ds)

            for j in range(MAX_LAYER_PER_STAGE - ds):
                d_matrix.append('none')

        string = ''
        for i, (ks, e, d) in enumerate(zip(ops['ks'], ops['e'], d_matrix)):
            if d == 'none':
                string += f'0-0-0_'
            else:
                string += f'{d}-{ks}-{e}_'
        return string[:-1]


