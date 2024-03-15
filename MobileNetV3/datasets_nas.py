from __future__ import print_function
import torch
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset

from torch_geometric.utils import to_networkx

from analysis.arch_functions import get_x_adj_from_opsdict_ofa, get_string_from_onehot_x
from all_path import PROCESSED_DATA_PATH, SCORE_MODEL_DATA_IDX_PATH
from analysis.arch_functions import OPS


def get_data_scaler(config):
    """Data normalizer. Assume data are always in [0, 1]."""

    if config.data.centered:
        # Rescale to [-1, 1]
        return lambda x: x * 2. - 1.
    else:
        return lambda x: x


def get_data_inverse_scaler(config):
    """Inverse data normalizer."""

    if config.data.centered:
        # Rescale [-1, 1] to [0, 1]
        return lambda x: (x + 1.) / 2.
    else:
        return lambda x: x


def networkx_graphs(dataset):
    return [to_networkx(dataset[i], to_undirected=False, remove_self_loops=True) for i in range(len(dataset))]


def get_dataloader(config, train_dataset, eval_dataset, test_dataset):
    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=config.training.batch_size,
                            shuffle=True,
                            collate_fn=collate_fn_ofa if config.model_type == 'meta_predictor' else None)
    eval_loader = DataLoader(dataset=eval_dataset,
                            batch_size=config.training.batch_size,
                            shuffle=False,
                            collate_fn=collate_fn_ofa if config.model_type == 'meta_predictor' else None)
    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=config.training.batch_size,
                            shuffle=False,
                            collate_fn=collate_fn_ofa if config.model_type == 'meta_predictor' else None)

    return train_loader, eval_loader, test_loader


def get_dataloader_iter(config, train_dataset, eval_dataset, test_dataset):
    
    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=config.training.batch_size if len(train_dataset) > config.training.batch_size else len(train_dataset),
                            # batch_size=8,
                            shuffle=True,)
    eval_loader = DataLoader(dataset=eval_dataset,
                            batch_size=config.training.batch_size if len(eval_dataset) > config.training.batch_size else len(eval_dataset),
                            # batch_size=8,
                            shuffle=False,)
    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=config.training.batch_size if len(test_dataset) > config.training.batch_size else len(test_dataset),
                            # batch_size=8,
                            shuffle=False,)

    return train_loader, eval_loader, test_loader


def is_triu(mat):
    is_triu_ = np.allclose(mat, np.triu(mat))
    return is_triu_


def collate_fn_ofa(batch):
    # x, adj, label_dict, task
    x = torch.stack([item[0] for item in batch])
    adj = torch.stack([item[1] for item in batch])
    label_dict = {}
    for item in batch:
        for k, v in item[2].items():
            if not k in label_dict.keys():
                 label_dict[k] = []
            label_dict[k].append(v)
    for k, v in label_dict.items():
         label_dict[k] = torch.tensor(v)
    task = [item[3] for item in batch]
    return x, adj, label_dict, task


def get_dataset(config):
    """Create data loaders for training and evaluation.

    Args:
        config: A ml_collection.ConfigDict parsed from config files.

    Returns:
        train_ds, eval_ds, test_ds
    """
    num_train = config.data.num_train if 'num_train' in config.data else None
    NASDataset = OFADataset
        
    train_dataset  = NASDataset(
        config.data.root,
        config.data.split_ratio, 
        config.data.except_inout, 
        config.data.triu_adj, 
        config.data.connect_prev,
        'train',
        config.data.label_list,
        config.data.tg_dataset,
        config.data.dataset_idx,
        num_train,
        node_rule_type=config.data.node_rule_type)
    eval_dataset  = NASDataset(
        config.data.root,
        config.data.split_ratio, 
        config.data.except_inout, 
        config.data.triu_adj, 
        config.data.connect_prev,
        'eval',
        config.data.label_list,
        config.data.tg_dataset,
        config.data.dataset_idx,
        num_train,
        node_rule_type=config.data.node_rule_type)

    test_dataset  = NASDataset(
        config.data.root,
        config.data.split_ratio, 
        config.data.except_inout, 
        config.data.triu_adj, 
        config.data.connect_prev,
        'test',
        config.data.label_list,
        config.data.tg_dataset,
        config.data.dataset_idx,
        num_train,
        node_rule_type=config.data.node_rule_type)


    return train_dataset, eval_dataset, test_dataset


def get_meta_dataset(config):
    database = MetaTrainDatabaseOFA
    data_path = PROCESSED_DATA_PATH

    train_dataset = database(
		data_path,
		config.model.num_sample,
		config.data.label_list,
		True,
		config.data.except_inout, 
        config.data.triu_adj, 
        config.data.connect_prev,
        'train')
    eval_dataset = database(
		data_path,
		config.model.num_sample,
		config.data.label_list,
		True,
		config.data.except_inout, 
        config.data.triu_adj, 
        config.data.connect_prev,
        'val')
    # test_dataset = MetaTestDataset()
    test_dataset = None
    return train_dataset, eval_dataset, test_dataset

def get_meta_dataloader(config ,train_dataset, eval_dataset, test_dataset):
    if config.data.name == 'ofa':
        train_loader = DataLoader(dataset=train_dataset,
                                batch_size=config.training.batch_size,
                                shuffle=True,)
                                # collate_fn=collate_fn_ofa)
        eval_loader = DataLoader(dataset=eval_dataset,
                                batch_size=config.training.batch_size,)
                                # collate_fn=collate_fn_ofa)
    else:
        train_loader = DataLoader(dataset=train_dataset,
                                batch_size=config.training.batch_size,
                                shuffle=True)
        eval_loader = DataLoader(dataset=eval_dataset,
                                batch_size=config.training.batch_size,
                                shuffle=False)
    # test_loader = DataLoader(dataset=test_dataset,
    #                          batch_size=config.training.batch_size,
    #                          shuffle=False)
    test_loader = None
    return train_loader, eval_loader, test_loader 


class MetaTestDataset(Dataset):
	def __init__(self, data_path, data_name, num_sample, num_class=None):
		self.num_sample = num_sample
		self.data_name = data_name

		num_class_dict = {
		'cifar100': 100,
		'cifar10':  10,
		'mnist':    10,
		'svhn':     10,
		'aircraft': 30,
		'pets':     37
		}

		if num_class is not None:
			self.num_class = num_class
		else:
			self.num_class = num_class_dict[data_name]
		self.x = torch.load(os.path.join(data_path, f'aircraft100bylabel.pt' if 'ofa' in data_path and data_name == 'aircraft' else f'{data_name}bylabel.pt' ))

	def __len__(self):
		return 1000000

	def __getitem__(self, index):
		data = []
		classes = list(range(self.num_class))
		for cls in classes:
			cx = self.x[cls][0]
			ridx = torch.randperm(len(cx))
			data.append(cx[ridx[:self.num_sample]])
		x = torch.cat(data)
		return x


class MetaTrainDatabaseOFA(Dataset):
    # def __init__(self, data_path, num_sample, is_pred=False):
    def __init__(
        self,
        data_path, 
        num_sample, 
        label_list,
        is_pred=True,
        except_inout=False, 
        triu_adj=True, 
        connect_prev=False,
        mode='train'):

        self.ops_decoder = list(OPS.keys())
        self.mode = mode
        self.acc_norm = True
        self.num_sample = num_sample
        self.x = torch.load(os.path.join(data_path, 'imgnet32bylabel.pt'))
		
        if is_pred:  
            self.dpath = f'{data_path}/predictor/processed/'
        else:
            raise NotImplementedError
        
        self.dname = 'database_219152_14.0K'
        data = torch.load(self.dpath + f'{self.dname}_{self.mode}.pt')
        self.net = data['net']
        self.x_list = []
        self.adj_list = []
        self.arch_str_list = []
        for net in self.net:
            x, adj = get_x_adj_from_opsdict_ofa(net)
            # ---------- matrix ---------- #
            self.x_list.append(x)
            self.adj_list.append(torch.tensor(adj))
            # ---------- arch_str ---------- #
            self.arch_str_list.append(get_string_from_onehot_x(x))
        # ---------- labels ---------- #
        self.label_list = label_list
        if self.label_list is not None:
            self.flops_list = data['flops']
            self.params_list = None
            self.latency_list = None

        self.acc_list = data['acc']
        self.mean = data['mean']
        self.std = data['std']
        self.task_lst = data['class']

    def __len__(self):
        return len(self.acc_list)
	
    def __getitem__(self, index):
        data = []
        classes = self.task_lst[index]
        acc = self.acc_list[index]
        graph = self.net[index]

        # ---------- x -----------
        x = self.x_list[index]
        # ---------- adj ----------
        adj = self.adj_list[index]
        acc = self.acc_list[index]

        for i, cls in enumerate(classes):
            cx = self.x[cls.item()][0]
            ridx = torch.randperm(len(cx))
            data.append(cx[ridx[:self.num_sample]])
        task = torch.cat(data)
        if self.acc_norm:
            acc = ((acc - self.mean) / self.std) / 100.0
        else:
            acc = acc / 100.0

        label_dict = {}
        if self.label_list is not None:
            assert type(self.label_list) == list
            for label in self.label_list:
                if label == 'meta-acc':
                    label_dict[f"{label}"] = acc
                else:
                    raise ValueError
        return x, adj, label_dict, task


class OFADataset(Dataset):
    def __init__(
        self, 
        data_path,
        split_ratio=0.8, 
        except_inout=False, 
        triu_adj=True, 
        connect_prev=False,
        mode='train',
        label_list=None,
        tg_dataset=None,
        dataset_idx='random',
        num_train=None,
        node_rule_type=None):
        
        # ---------- entire dataset ---------- #
        self.data = torch.load(data_path)
        self.except_inout = except_inout
        self.triu_adj = triu_adj
        self.connect_prev = connect_prev
        self.node_rule_type = node_rule_type

        # ---------- x ---------- #
        self.x_list = self.data['x_none2zero']
        
        # ---------- adj ---------- #
        assert self.connect_prev == False
        self.n_adj = len(self.data['node_type'][0])
        const_adj = self.get_not_connect_prev_adj()
        self.adj_list = [const_adj] * len(self.x_list)

        # ---------- arch_str ---------- #
        self.arch_str_list = self.data['net_setting']
        # ---------- labels ---------- #
        self.label_list = label_list 
        if self.label_list is not None:
            raise NotImplementedError
        
        # ----------- split dataset ---------- #
        self.ds_idx = list(torch.load(SCORE_MODEL_DATA_IDX_PATH))

        self.split_ratio = split_ratio
        if num_train is None:
            num_train = int(len(self.x_list) * self.split_ratio)
            num_test = len(self.x_list) - num_train
        else:
            num_train = num_train
            num_test = len(self.x_list) - num_train
        # ----------- compute mean and std w/ training dataset ---------- #
        if self.label_list is not None:
            self.train_idx_list = self.ds_idx[:num_train]
            print('Computing mean and std of the training set...')
            from collections import defaultdict
            LABEL_TO_MEAN_STD = defaultdict(dict)
            assert type(self.label_list) == list
            for label in self.label_list:
                if label == 'test-acc':
                    self.test_acc_list_tr = [self.test_acc_list[i] for i in self.train_idx_list]
                    LABEL_TO_MEAN_STD[label]['std'], LABEL_TO_MEAN_STD[label]['mean'] = torch.std_mean(torch.tensor(self.test_acc_list_tr))
                elif label == 'flops':
                    self.flops_list_tr = [self.flops_list[i] for i in self.train_idx_list]
                    LABEL_TO_MEAN_STD[label]['std'], LABEL_TO_MEAN_STD[label]['mean'] = torch.std_mean(torch.tensor(self.flops_list_tr))
                elif label == 'params':
                    self.params_list_tr = [self.params_list[i] for i in self.train_idx_list]
                    LABEL_TO_MEAN_STD[label]['std'], LABEL_TO_MEAN_STD[label]['mean'] = torch.std_mean(torch.tensor(self.params_list_tr))
                elif label == 'latency':
                    self.latency_list_tr = [self.latency_list[i] for i in self.train_idx_list]
                    LABEL_TO_MEAN_STD[label]['std'], LABEL_TO_MEAN_STD[label]['mean'] = torch.std_mean(torch.tensor(self.latency_list_tr))
                else:
                    raise ValueError
        
        self.mode = mode
        if self.mode in ['train']:
            self.idx_list = self.ds_idx[:num_train]
        elif self.mode in ['eval']:
            self.idx_list = self.ds_idx[:num_test]
        elif self.mode in ['test']:
            self.idx_list = self.ds_idx[num_train:]
        
        self.x_list_ = [self.x_list[i] for i in self.idx_list]
        self.adj_list_ = [self.adj_list[i] for i in self.idx_list]
        self.arch_str_list_ = [self.arch_str_list[i] for i in self.idx_list]

        if self.label_list is not None:
            assert type(self.label_list) == list
            for label in self.label_list:
                if label == 'test-acc':
                    self.test_acc_list_ = [self.test_acc_list[i] for i in self.idx_list]
                    self.test_acc_list_ = self.normalize(self.test_acc_list_, LABEL_TO_MEAN_STD[label]['mean'], LABEL_TO_MEAN_STD[label]['std'])
                elif label == 'flops':
                    self.flops_list_ = [self.flops_list[i] for i in self.idx_list]
                    self.flops_list_ = self.normalize(self.flops_list_, LABEL_TO_MEAN_STD[label]['mean'], LABEL_TO_MEAN_STD[label]['std'])
                elif label == 'params':
                    self.params_list_ = [self.params_list[i] for i in self.idx_list]
                    self.params_list_ = self.normalize(self.params_list_, LABEL_TO_MEAN_STD[label]['mean'], LABEL_TO_MEAN_STD[label]['std'])
                elif label == 'latency':
                    self.latency_list_ = [self.latency_list[i] for i in self.idx_list]
                    self.latency_list_ = self.normalize(self.latency_list_, LABEL_TO_MEAN_STD[label]['mean'], LABEL_TO_MEAN_STD[label]['std'])
                else:
                    raise ValueError

    def normalize(self, original, mean, std):
        return [(i-mean)/std for i in original]
    
    def get_not_connect_prev_adj(self):
        _adj = torch.zeros(self.n_adj, self.n_adj)
        for i in range(self.n_adj-1):
            _adj[i, i+1] = 1
        _adj = _adj.to(torch.float32).to('cpu') # torch.tensor(_adj, dtype=torch.float32, device=torch.device('cpu'))
        # if self.except_inout:
        #     _adj = _adj[1:-1, 1:-1]
        return _adj

    @property
    def adj(self):
        return self.adj_list_[0]
    
    # @property
    def mask(self, algo='floyd', data='ofa'):
        from utils import aug_mask
        return aug_mask(self.adj, algo=algo, data=data)[0]
    
    def get_unnoramlized_entire_data(self, label, tg_dataset):
        entire_test_acc_list = self.data['test-acc'][tg_dataset]
        entire_flops_list = self.data['flops'][tg_dataset]
        entire_params_list = self.data['params'][tg_dataset]
        entire_latency_list = self.data['latency'][tg_dataset]
        
        if label == 'test-acc':
            return entire_test_acc_list
        elif label == 'flops':
            return entire_flops_list
        elif label == 'params':
            return entire_params_list
        elif label == 'latency':
            return entire_latency_list
        else:
            raise ValueError
    
    
    def get_unnoramlized_data(self, label, tg_dataset):
        entire_test_acc_list = self.data['test-acc'][tg_dataset]
        entire_flops_list = self.data['flops'][tg_dataset]
        entire_params_list = self.data['params'][tg_dataset]
        entire_latency_list = self.data['latency'][tg_dataset]
        
        if label == 'test-acc':
            return [entire_test_acc_list[i] for i in self.idx_list]
        elif label == 'flops':
            return [entire_flops_list[i] for i in self.idx_list]
        elif label == 'params':
            return [entire_params_list[i] for i in self.idx_list]
        elif label == 'latency':
            return [entire_latency_list[i] for i in self.idx_list]
        else:
            raise ValueError
    
    def __len__(self):
        return len(self.x_list_)

    def __getitem__(self, index):
        
        label_dict = {}
        if self.label_list is not None:
            assert type(self.label_list) == list
            for label in self.label_list:
                if label == 'test-acc':
                    label_dict[f"{label}"] = self.test_acc_list_[index]
                elif label == 'flops':
                    label_dict[f"{label}"] = self.flops_list_[index]
                elif label == 'params':
                    label_dict[f"{label}"] = self.params_list_[index]
                elif label == 'latency':
                    label_dict[f"{label}"] = self.latency_list_[index]
                else:
                    raise ValueError
        
        return self.x_list_[index], self.adj_list_[index], label_dict