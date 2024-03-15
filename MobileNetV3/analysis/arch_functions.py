import numpy as np
import torch
import wandb
import igraph
from torch.nn.functional import one_hot


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


class BasicArchMetricsOFA(object):
    def __init__(self, train_ds=None, train_arch_str_list=None, except_inout=False, data_root=None):
        if data_root is not None:
            self.ofa = torch.load(data_root)
            self.train_arch_list = self.ofa['x']
        else:
            self.ofa = None
            self.train_arch_list = None
        # self.ofa = torch.load(data_root)
        self.ops_decoder = OPS
        self.except_inout = except_inout
        
    def get_string_from_onehot_x(self, x):
        # node_types = torch.nonzero(torch.tensor(x).long(), as_tuple=True)[1]
        x = torch.tensor(x)
        ds = torch.sum(x.view(NUM_STAGE, -1), dim=1)
        string = ''
        for i, _ in enumerate(x):
            if sum(_) == 0:
                string += '0-0-0_'
            else:
                string += f'{int(ds[int(i/MAX_LAYER_PER_STAGE)])}-' + OPS2STR[torch.nonzero(torch.tensor(_)).item()] + '_'
        return string[:-1]


    def compute_validity(self, generated, adj=None, mask=None):
        """ generated: list of couples (positions, node_types)"""       
        valid = []
        error_types = []
        valid_str = []
        for x in generated:
            is_valid, error_type = is_valid_OFA_x(x)
            if is_valid:
                valid.append(torch.tensor(x).long())
                valid_str.append(self.get_string_from_onehot_x(x))
            else:
                error_types.append(error_type)

        return valid, len(valid) / len(generated), valid_str, None, error_types

    def compute_uniqueness(self, valid_arch):
        unique = []
        for x in valid_arch:
            if not any([torch.equal(x, tr_m) for tr_m in unique]):
                unique.append(x)
        return unique, len(unique) / len(valid_arch)

    def compute_novelty(self, unique):
        num_novel = 0
        novel = []
        if self.train_arch_list is None:
            print("Dataset arch_str is None, novelty computation skipped")
            return 1, 1
        for arch in unique:
            if not any([torch.equal(arch, tr_m) for tr_m in self.train_arch_list]):
            # if arch not in self.train_arch_list[1:]:
                novel.append(arch)
                num_novel += 1
        return novel, num_novel / len(unique)

    def evaluate(self, generated, adj, mask, check_dataname='cifar10'):
        """ generated: list of pairs """
        valid_arch, validity, _, _, error_types = self.compute_validity(generated, adj, mask)
        
        print(f"Validity over {len(generated)} archs: {validity * 100 :.2f}%")
        error_1 = torch.sum(torch.tensor(error_types) == 1) / len(generated)
        error_2 = torch.sum(torch.tensor(error_types) == 2) / len(generated)
        error_3 = torch.sum(torch.tensor(error_types) == 3) / len(generated)
        print(f"Unvalid-Multi_Node_Type over {len(generated)} archs: {error_1 * 100 :.2f}%")
        print(f"INVALID_1OR2 over {len(generated)} archs: {error_2 * 100 :.2f}%")
        print(f"INVALID_3AND4 over {len(generated)} archs: {error_3 * 100 :.2f}%")
        # print(f"Number of connected components of {len(generated)} molecules: min:{nc_min:.2f} mean:{nc_mu:.2f} max:{nc_max:.2f}")

        if validity > 0:
            unique, uniqueness = self.compute_uniqueness(valid_arch)
            print(f"Uniqueness over {len(valid_arch)} valid archs: {uniqueness * 100 :.2f}%")

            if self.train_arch_list is not None:
                _, novelty = self.compute_novelty(unique)
                print(f"Novelty over {len(unique)} unique valid archs: {novelty * 100 :.2f}%")
            else:
                novelty = -1.0
            
        else:
            novelty = -1.0
            uniqueness = 0.0
            unique = []
            
        test_acc_list, flops_list, params_list, latency_list = [0], [0], [0], [0]
        all_arch_str = None
        return ([validity, uniqueness, novelty, error_1, error_2, error_3], 
                unique,
                dict(test_acc_list=test_acc_list, flops_list=flops_list, params_list=params_list, latency_list=latency_list), 
                all_arch_str)


class BasicArchMetricsMetaOFA(object):
    def __init__(self, train_ds=None, train_arch_str_list=None, except_inout=False, data_root=None):
        if data_root is not None:
            self.ofa = torch.load(data_root)
            self.train_arch_list = self.ofa['x']
        else:
            self.ofa = None
            self.train_arch_list = None
        self.ops_decoder = OPS

    def get_string_from_onehot_x(self, x):
        x = torch.tensor(x)
        ds = torch.sum(x.view(NUM_STAGE, -1), dim=1)
        string = ''
        for i, _ in enumerate(x):
            if sum(_) == 0:
                string += '0-0-0_'
            else:
                string += f'{int(ds[int(i/MAX_LAYER_PER_STAGE)])}-' + OPS2STR[torch.nonzero(torch.tensor(_)).item()] + '_'
        return string[:-1]

    def compute_validity(self, generated, adj=None, mask=None):
        """ generated: list of couples (positions, node_types)"""
        valid = []
        valid_arch_str = []
        all_arch_str = []
        error_types = []
        for x in generated:
            is_valid, error_type = is_valid_OFA_x(x)
            if is_valid:
                valid.append(torch.tensor(x).long())
                arch_str = self.get_string_from_onehot_x(x)
                valid_arch_str.append(arch_str)
            else:
                arch_str = None
                error_types.append(error_type)
            all_arch_str.append(arch_str)
        validity = 0 if len(generated) == 0 else (len(valid)/len(generated))
        return valid, validity, valid_arch_str, all_arch_str, error_types

    def compute_uniqueness(self, valid_arch):
        unique = []
        for x in valid_arch:
            if not any([torch.equal(x, tr_m) for tr_m in unique]):
                unique.append(x)
        return unique, len(unique) / len(valid_arch)

    def compute_novelty(self, unique):
        num_novel = 0
        novel = []
        if self.train_arch_list is None:
            print("Dataset arch_str is None, novelty computation skipped")
            return 1, 1
        for arch in unique:
            if not any([torch.equal(arch, tr_m) for tr_m in self.train_arch_list]):
                novel.append(arch)
                num_novel += 1
        return novel, num_novel / len(unique)

    def evaluate(self, generated, adj, mask, check_dataname='imagenet1k'):
        """ generated: list of pairs """
        valid_arch, validity, _, _, error_types = self.compute_validity(generated, adj, mask)
        
        print(f"Validity over {len(generated)} archs: {validity * 100 :.2f}%")
        error_1 = torch.sum(torch.tensor(error_types) == 1) / len(generated)
        error_2 = torch.sum(torch.tensor(error_types) == 2) / len(generated)
        error_3 = torch.sum(torch.tensor(error_types) == 3) / len(generated)
        print(f"Unvalid-Multi_Node_Type over {len(generated)} archs: {error_1 * 100 :.2f}%")
        print(f"INVALID_1OR2 over {len(generated)} archs: {error_2 * 100 :.2f}%")
        print(f"INVALID_3AND4 over {len(generated)} archs: {error_3 * 100 :.2f}%")

        if validity > 0:
            unique, uniqueness = self.compute_uniqueness(valid_arch)
            print(f"Uniqueness over {len(valid_arch)} valid archs: {uniqueness * 100 :.2f}%")

            if self.train_arch_list is not None:
                _, novelty = self.compute_novelty(unique)
                print(f"Novelty over {len(unique)} unique valid archs: {novelty * 100 :.2f}%")
            else:
                novelty = -1.0
            
        else:
            novelty = -1.0
            uniqueness = 0.0
            unique = []
            
        test_acc_list, flops_list, params_list, latency_list = [0], [0], [0], [0]
        all_arch_str = None
        return ([validity, uniqueness, novelty, error_1, error_2, error_3], 
                unique,
                dict(test_acc_list=test_acc_list, flops_list=flops_list, params_list=params_list, latency_list=latency_list), 
                all_arch_str)


def get_arch_acc_info(nasbench201, arch, dataname='cifar10'):
    arch_index = nasbench201['str'].index(arch)
    test_acc = nasbench201['test-acc'][dataname][arch_index]
    flops = nasbench201['flops'][dataname][arch_index]
    params = nasbench201['params'][dataname][arch_index]
    latency = nasbench201['latency'][dataname][arch_index]
    return test_acc, flops, params, latency


def get_arch_acc_info_meta(nasbench201, arch, dataname='cifar10'):
    arch_index = nasbench201['str'].index(arch)
    flops = nasbench201['flops'][dataname][arch_index]
    params = nasbench201['params'][dataname][arch_index]
    latency = nasbench201['latency'][dataname][arch_index]
    if 'cifar' in dataname:
        test_acc = nasbench201['test-acc'][dataname][arch_index]
    else:
        # TODO
        test_acc = None
    return arch_index, test_acc, flops, params, latency


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


def construct_igraph(node_type, edge_type, ops_decoder, except_inout=True):
    assert node_type.shape[0] == edge_type.shape[0]
    
    START_TYPE = ops_decoder.index('input')
    END_TYPE = ops_decoder.index('output')
    
    g = igraph.Graph(directed=True)
    for i, node in enumerate(node_type):
        new_type = node.item()
        g.add_vertex(type=new_type)
        if new_type == END_TYPE:
            end_vertices = set([v.index for v in g.vs.select(_outdegree_eq=0) if v.index != g.vcount()-1])
            for v in end_vertices:
                g.add_edge(v, i)
        elif i > 0:
            for ek in range(i):
                ek_score = edge_type[ek][i].item()
                if ek_score >= 0.5:
                    g.add_edge(ek, i)
    
    return g


def compute_arch_metrics(arch_list, adj, mask, train_arch_str_list, 
                         train_ds, timestep=None, name=None, except_inout=False, data_root=None):
    """ arch_list: (dict) """
    metrics = BasicArchMetricsOFA(data_root=data_root)
    arch_metrics = metrics.evaluate(arch_list, adj, mask, check_dataname='cifar10')
    all_arch_str = arch_metrics[-1]
    
    if wandb.run:
        arch_prop = arch_metrics[2]
        test_acc_list = arch_prop['test_acc_list']
        flops_list = arch_prop['flops_list']
        params_list = arch_prop['params_list']
        latency_list = arch_prop['latency_list']
        if arch_metrics[0][1] > 0.: # uniquness > 0.
            dic = {
                'Validity': arch_metrics[0][0], 'Uniqueness': arch_metrics[0][1], 'Novelty': arch_metrics[0][2],
                'test_acc_max': np.max(test_acc_list), 'test_acc_min': np.min(test_acc_list), 'test_acc_mean': np.mean(test_acc_list), 'test_acc_std': np.std(test_acc_list),
                'flops_max': np.max(flops_list), 'flops_min': np.min(flops_list), 'flops_mean': np.mean(flops_list), 'flops_std': np.std(flops_list),
                'params_max': np.max(params_list), 'params_min': np.min(params_list), 'params_mean': np.mean(params_list), 'params_std': np.std(params_list),
                'latency_max': np.max(latency_list), 'latency_min': np.min(latency_list), 'latency_mean': np.mean(latency_list), 'latency_std': np.std(latency_list),
                }
        else:
            dic = {
                'Validity': arch_metrics[0][0], 'Uniqueness': arch_metrics[0][1], 'Novelty': arch_metrics[0][2],
                'test_acc_max': -1, 'test_acc_min': -1, 'test_acc_mean': -1, 'test_acc_std': 0,
                'flops_max': -1, 'flops_min': -1, 'flops_mean': -1, 'flops_std': 0,
                'params_max': -1, 'params_min': -1, 'params_mean': -1, 'params_std': 0,
                'latency_max': -1, 'latency_min': -1, 'latency_mean': -1, 'latency_std': 0,
                }
        if timestep is not None:
            dic.update({'step': timestep})

        wandb.log(dic)

    return arch_metrics, all_arch_str

def compute_arch_metrics_meta(
        arch_list, adj, mask, train_arch_str_list, train_ds, 
        timestep=None, check_dataname='cifar10', name=None):
    """ arch_list: (dict) """

    metrics = BasicArchMetricsMetaOFA(train_ds, train_arch_str_list)
    arch_metrics = metrics.evaluate(arch_list, adj, mask, check_dataname=check_dataname)
    if wandb.run:
        arch_prop = arch_metrics[2]
        if name != 'ofa':
            arch_idx_list = arch_prop['arch_idx_list']
        test_acc_list = arch_prop['test_acc_list']
        flops_list = arch_prop['flops_list']
        params_list = arch_prop['params_list']
        latency_list = arch_prop['latency_list']
        if arch_metrics[0][1] > 0.: # uniquness > 0.
            dic = {
                'Validity': arch_metrics[0][0], 'Uniqueness': arch_metrics[0][1], 'Novelty': arch_metrics[0][2],
                'test_acc_max': np.max(test_acc_list), 'test_acc_min': np.min(test_acc_list), 'test_acc_mean': np.mean(test_acc_list), 'test_acc_std': np.std(test_acc_list),
                'flops_max': np.max(flops_list), 'flops_min': np.min(flops_list), 'flops_mean': np.mean(flops_list), 'flops_std': np.std(flops_list),
                'params_max': np.max(params_list), 'params_min': np.min(params_list), 'params_mean': np.mean(params_list), 'params_std': np.std(params_list),
                'latency_max': np.max(latency_list), 'latency_min': np.min(latency_list), 'latency_mean': np.mean(latency_list), 'latency_std': np.std(latency_list),
                }
        else:
            dic = {
                'Validity': arch_metrics[0][0], 'Uniqueness': arch_metrics[0][1], 'Novelty': arch_metrics[0][2],
                'test_acc_max': -1, 'test_acc_min': -1, 'test_acc_mean': -1, 'test_acc_std': 0,
                'flops_max': -1, 'flops_min': -1, 'flops_mean': -1, 'flops_std': 0,
                'params_max': -1, 'params_min': -1, 'params_mean': -1, 'params_std': 0,
                'latency_max': -1, 'latency_min': -1, 'latency_mean': -1, 'latency_std': 0,
                }
        if timestep is not None:
            dic.update({'step': timestep})

    return arch_metrics


def check_multiple_nodes(x):
    assert len(x.shape) == 2
    for x_elem in x:
        x_elem = np.array(x_elem)
        if int(np.sum(x_elem)) > 1:
            return False
    return True

def check_inout_node(x, START_TYPE=0, END_TYPE=1):
    assert len(x.shape) == 2
    return x[0][START_TYPE] == 1 and x[-1][END_TYPE] == 1

def check_none_in_1_and_2_layers(x, NONE_TYPE=None):
    assert len(x.shape) == 2
    first_and_second_layers = [0, 1, 4, 5, 8, 9, 12, 13, 16, 17]
    for layer in first_and_second_layers:
        if int(np.sum(x[layer])) == 0:
            return False
    return True

def check_none_in_3_and_4_layers(x, NONE_TYPE=None):
    assert len(x.shape) == 2
    third_layers = [2, 6, 10, 14, 18]
    
    for layer in third_layers:
        if int(np.sum(x[layer])) == 0:
            if int(np.sum(x[layer+1])) != 0:
                return False
    return True


def check_interm_inout_node(x, START_TYPE, END_TYPE):
    for x_elem in x[1:-1]:
        if x_elem[START_TYPE] == 1: 
            return False 
        if x_elem[END_TYPE] == 1: 
            return False


def is_valid_OFA_x(x):
    ERORR = {
        'MULIPLE_NODES': 1,
        'INVALID_1OR2_LAYERS': 2,
        'INVALID_3AND4_LAYERS': 3,
        'NO_ERROR': -1
    }
    if not check_multiple_nodes(x):
        return False, ERORR['MULIPLE_NODES']

    if not check_none_in_1_and_2_layers(x):
        return False, ERORR['INVALID_1OR2_LAYERS']

    if not check_none_in_3_and_4_layers(x):
        return False, ERORR['INVALID_3AND4_LAYERS']

    return True, ERORR['NO_ERROR']


def get_x_adj_from_opsdict_ofa(ops):
    node_types = torch.zeros(NUM_STAGE * MAX_LAYER_PER_STAGE).long() # w/o in / out
    num_vertices = len(OPS.values())
    num_nodes = NUM_STAGE * MAX_LAYER_PER_STAGE
    d_matrix = []

    for i in range(NUM_STAGE):
        ds = ops['d'][i]
        for j in range(ds):
            d_matrix.append(ds)

        for j in range(MAX_LAYER_PER_STAGE - ds):
            d_matrix.append('none')

    for i, (ks, e, d) in enumerate(zip(
            ops['ks'], ops['e'], d_matrix)):
        if d == 'none':
            pass
        else:
            node_types[i] = OPS[f'{ks}-{e}']

    x = one_hot(node_types, num_vertices).float()

    def get_adj():
        adj = torch.zeros(num_nodes, num_nodes)
        for i in range(num_nodes-1):
            adj[i, i+1] = 1
        adj = np.array(adj)
        return adj
    
    adj = get_adj()
    return x, adj


def get_string_from_onehot_x(x):
    x = torch.tensor(x)
    ds = torch.sum(x.view(NUM_STAGE, -1), dim=1)
    string = ''
    for i, _ in enumerate(x):
        if sum(_) == 0:
            string += '0-0-0_'
        else:
            string += f'{int(ds[int(i/MAX_LAYER_PER_STAGE)])}-' + OPS2STR[torch.nonzero(torch.tensor(_)).item()] + '_'
    return string[:-1]