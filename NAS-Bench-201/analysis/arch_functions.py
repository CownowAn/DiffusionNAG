import numpy as np
import torch
from all_path import *


class BasicArchMetrics(object):
    def __init__(self, train_ds=None, train_arch_str_list=None):
        if train_ds is None:
            self.ops_decoder = ['input', 'output', 'none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
        else:
            self.ops_decoder = train_ds.ops_decoder
            self.nasbench201 = torch.load(NASBENCH201_INFO)
            self.train_arch_str_list = train_arch_str_list


    def compute_validity(self, generated):
        START_TYPE = self.ops_decoder.index('input')
        END_TYPE = self.ops_decoder.index('output')
        
        valid = []
        valid_arch_str = []
        all_arch_str = []
        for x in generated:
            is_valid, error_types = is_valid_NAS201_x(x, START_TYPE, END_TYPE)
            if is_valid:
                valid.append(x)
                arch_str = decode_x_to_NAS_BENCH_201_string(x, self.ops_decoder)
                valid_arch_str.append(arch_str)
            else:
                arch_str = None
            all_arch_str.append(arch_str)
        validity = 0 if len(generated) == 0 else (len(valid)/len(generated))
        return valid, validity, valid_arch_str, all_arch_str


    def compute_uniqueness(self, valid_arch_str):
        return list(set(valid_arch_str)), len(set(valid_arch_str)) / len(valid_arch_str)


    def compute_novelty(self, unique):
        num_novel = 0
        novel = []
        if self.train_arch_str_list is None:
            print("Dataset arch_str is None, novelty computation skipped")
            return 1, 1
        for arch_str in unique:
            if arch_str not in self.train_arch_str_list:
                novel.append(arch_str)
                num_novel += 1
        return novel, num_novel / len(unique)


    def evaluate(self, generated, check_dataname='cifar10'):
        valid, validity, valid_arch_str, all_arch_str = self.compute_validity(generated)

        if validity > 0:
            unique, uniqueness = self.compute_uniqueness(valid_arch_str)
            if self.train_arch_str_list is not None:
                _, novelty = self.compute_novelty(unique)
            else:
                novelty = -1.0
        else:
            novelty = -1.0
            uniqueness = 0.0
            unique = []
            
        if uniqueness > 0.:
            arch_idx_list, flops_list, params_list, latency_list = list(), list(), list(), list()
            for arch in unique:
                arch_index, flops, params, latency = \
                    get_arch_acc_info(self.nasbench201, arch=arch, dataname=check_dataname)
                arch_idx_list.append(arch_index)
                flops_list.append(flops)
                params_list.append(params)
                latency_list.append(latency)
        else:
            arch_idx_list, flops_list, params_list, latency_list = [-1], [0], [0], [0]
        
        return ([validity, uniqueness, novelty], 
                unique,
                dict(arch_idx_list=arch_idx_list, flops_list=flops_list, params_list=params_list, latency_list=latency_list), 
                all_arch_str)


class BasicArchMetricsMeta(object):
    def __init__(self, train_ds=None, train_arch_str_list=None):
        if train_ds is None:
            self.ops_decoder = ['input', 'output', 'none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
        else:
            self.ops_decoder = train_ds.ops_decoder
            self.nasbench201 = torch.load(NASBENCH201_INFO)
            self.train_arch_str_list = train_arch_str_list


    def compute_validity(self, generated):
        START_TYPE = self.ops_decoder.index('input')
        END_TYPE = self.ops_decoder.index('output')

        valid = []
        valid_arch_str = []
        all_arch_str = []
        error_types = []

        for x in generated:
            is_valid, error_type = is_valid_NAS201_x(x, START_TYPE, END_TYPE)
            if is_valid:
                valid.append(x)
                arch_str = decode_x_to_NAS_BENCH_201_string(x, self.ops_decoder)
                valid_arch_str.append(arch_str)
            else:
                arch_str = None
                error_types.append(error_type)
            all_arch_str.append(arch_str)

        # exceptional case
        validity = 0 if len(generated) == 0 else (len(valid)/len(generated))
        if len(valid) == 0:
            validity = 0
            valid_arch_str = []

        return valid, validity, valid_arch_str, all_arch_str


    def compute_uniqueness(self, valid_arch_str):
        return list(set(valid_arch_str)), len(set(valid_arch_str)) / len(valid_arch_str)


    def compute_novelty(self, unique):
        num_novel = 0
        novel = []
        if self.train_arch_str_list is None:
            print("Dataset arch_str is None, novelty computation skipped")
            return 1, 1
        for arch_str in unique:
            if arch_str not in self.train_arch_str_list:
                novel.append(arch_str)
                num_novel += 1
        return novel, num_novel / len(unique)


    def evaluate(self, generated, check_dataname='cifar10'):
        valid, validity, valid_arch_str, all_arch_str = self.compute_validity(generated)
        
        if validity > 0:
            unique, uniqueness = self.compute_uniqueness(valid_arch_str)
            if self.train_arch_str_list is not None:
                _, novelty = self.compute_novelty(unique)
            else:
                novelty = -1.0
        else:
            novelty = -1.0
            uniqueness = 0.0
            unique = []
            
        if uniqueness > 0.:
            arch_idx_list, flops_list, params_list, latency_list = list(), list(), list(), list()
            for arch in unique:
                arch_index, flops, params, latency = \
                    get_arch_acc_info_meta(self.nasbench201, arch=arch, dataname=check_dataname)
                arch_idx_list.append(arch_index)
                flops_list.append(flops)
                params_list.append(params)
                latency_list.append(latency)
        else:
            arch_idx_list, flops_list, params_list, latency_list = [-1], [0], [0], [0]
        
        return ([validity, uniqueness, novelty], 
                unique,
                dict(arch_idx_list=arch_idx_list, flops_list=flops_list, params_list=params_list, latency_list=latency_list), 
                all_arch_str)


def get_arch_acc_info(nasbench201, arch, dataname='cifar10'):
    arch_index = nasbench201['str'].index(arch)
    flops = nasbench201['flops'][dataname][arch_index]
    params = nasbench201['params'][dataname][arch_index]
    latency = nasbench201['latency'][dataname][arch_index]
    return arch_index, flops, params, latency


def get_arch_acc_info_meta(nasbench201, arch, dataname='cifar10'):
    arch_index = nasbench201['str'].index(arch)
    flops = nasbench201['flops'][dataname][arch_index]
    params = nasbench201['params'][dataname][arch_index]
    latency = nasbench201['latency'][dataname][arch_index]
    return arch_index, flops, params, latency


def decode_igraph_to_NAS_BENCH_201_string(g):
    if not is_valid_NAS201(g):
        return None
    m = decode_igraph_to_NAS201_matrix(g)
    types = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
    return '|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|'.\
        format(types[int(m[1][0])],
                types[int(m[2][0])], types[int(m[2][1])],
                types[int(m[3][0])], types[int(m[3][1])], types[int(m[3][2])])


def decode_igraph_to_NAS201_matrix(g):
    m = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    xys = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)]
    for i, xy in enumerate(xys):
        m[xy[0]][xy[1]] = float(g.vs[i + 1]['type']) - 2
    import numpy
    return numpy.array(m)


def decode_x_to_NAS_BENCH_201_matrix(x):
    m = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    xys = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)]
    for i, xy in enumerate(xys):
        # m[xy[0]][xy[1]] = int(torch.argmax(torch.tensor(x[i+1])).item()) - 2
        m[xy[0]][xy[1]] = int(torch.argmax(torch.tensor(x[i+1])).item())
    import numpy
    return numpy.array(m)


def decode_x_to_NAS_BENCH_201_string(x, ops_decoder):
    """_summary_

    Args:
        x (torch.Tensor): x_elem [8, 7]

    Returns:
        arch_str
    """
    is_valid, error_type = is_valid_NAS201_x(x)
    if not is_valid:
        return None
    m = decode_x_to_NAS_BENCH_201_matrix(x)
    types = ops_decoder
    arch_str = '|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|'.\
        format(types[int(m[1][0])],
                types[int(m[2][0])], types[int(m[2][1])],
                types[int(m[3][0])], types[int(m[3][1])], types[int(m[3][2])])
    return arch_str


def decode_x_to_NAS_BENCH_201_string(x, ops_decoder):
    """_summary_
    Args:
        x (torch.Tensor): x_elem [8, 7]
    Returns:
        arch_str
    """

    if not is_valid_NAS201_x(x)[0]:
        return None
    m = decode_x_to_NAS_BENCH_201_matrix(x)
    types = ops_decoder
    arch_str = '|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|'.\
        format(types[int(m[1][0])],
                types[int(m[2][0])], types[int(m[2][1])],
                types[int(m[3][0])], types[int(m[3][1])], types[int(m[3][2])])
    return arch_str


def is_valid_DAG(g, START_TYPE=0, END_TYPE=1):
    res = g.is_dag()
    n_start, n_end = 0, 0
    for v in g.vs:
        if v['type'] == START_TYPE:
            n_start += 1
        elif v['type'] == END_TYPE:
            n_end += 1
        if v.indegree() == 0 and v['type'] != START_TYPE:
            return False
        if v.outdegree() == 0 and v['type'] != END_TYPE:
            return False
    return res and n_start == 1 and n_end == 1


def is_valid_NAS201(g, START_TYPE=0, END_TYPE=1):
    # first need to be a valid DAG computation graph
    res = is_valid_DAG(g, START_TYPE, END_TYPE)
    # in addition, node i must connect to node i+1
    res = res and len(g.vs['type']) == 8
    res = res and not (START_TYPE in g.vs['type'][1:-1])
    res = res and not (END_TYPE in g.vs['type'][1:-1])
    return res


def check_single_node_type(x):
    for x_elem in x:
        if int(np.sum(x_elem)) != 1:
            return False
    return True


def check_start_end_nodes(x, START_TYPE, END_TYPE):
    if x[0][START_TYPE] != 1:
        return False
    if x[-1][END_TYPE] != 1:
        return False
    return True


def check_interm_node_types(x, START_TYPE, END_TYPE):
    for x_elem in x[1:-1]:
        if x_elem[START_TYPE] == 1:
            return False
        if x_elem[END_TYPE] == 1:
            return False
    return True


ERORR_NB201 = {
    'MULTIPLE_NODE_TYPES': 1,
    'No_START_END': 2,
    'INTERM_START_END': 3,
    'NO_ERROR': -1
}


def is_valid_NAS201_x(x, START_TYPE=0, END_TYPE=1):
    # first need to be a valid DAG computation graph
    assert len(x.shape) == 2

    if not check_single_node_type(x):
        return False, ERORR_NB201['MULTIPLE_NODE_TYPES']
    
    if not check_start_end_nodes(x, START_TYPE, END_TYPE):
        return False, ERORR_NB201['No_START_END']

    if not check_interm_node_types(x, START_TYPE, END_TYPE):
        return False, ERORR_NB201['INTERM_START_END']
    
    return True, ERORR_NB201['NO_ERROR']


def compute_arch_metrics(arch_list,
                         train_arch_str_list, 
                         train_ds,
                         check_dataname='cifar10'):
    metrics = BasicArchMetrics(train_ds, train_arch_str_list)
    arch_metrics = metrics.evaluate(arch_list, check_dataname=check_dataname)
    all_arch_str = arch_metrics[-1]
    return arch_metrics, all_arch_str

def compute_arch_metrics_meta(arch_list,
                              train_arch_str_list,
                              train_ds,
                              check_dataname='cifar10'):
    metrics = BasicArchMetricsMeta(train_ds, train_arch_str_list)
    arch_metrics = metrics.evaluate(arch_list, check_dataname=check_dataname)
    return arch_metrics
