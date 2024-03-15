from __future__ import print_function
import torch
import os
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from analysis.arch_functions import decode_x_to_NAS_BENCH_201_matrix, decode_x_to_NAS_BENCH_201_string
from all_path import *


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


def is_triu(mat):
	is_triu_ = np.allclose(mat, np.triu(mat))
	return is_triu_


def get_dataset(config):
	train_dataset  = NASBench201Dataset(
		data_path=NASBENCH201_INFO,
		mode='train')

	eval_dataset  = NASBench201Dataset(
		data_path=NASBENCH201_INFO,
		mode='eval')

	test_dataset  = NASBench201Dataset(
		data_path=NASBENCH201_INFO,
		mode='test')

	return train_dataset, eval_dataset, test_dataset


def get_dataloader(config, train_dataset, eval_dataset, test_dataset):
	train_loader = DataLoader(dataset=train_dataset,
							batch_size=config.training.batch_size,
							shuffle=True,
							collate_fn=None)

	eval_loader = DataLoader(dataset=eval_dataset,
							batch_size=config.training.batch_size,
							shuffle=False,
							collate_fn=None)

	test_loader = DataLoader(dataset=test_dataset,
							batch_size=config.training.batch_size,
							shuffle=False,
							collate_fn=None)

	return train_loader, eval_loader, test_loader


class NASBench201Dataset(Dataset):
	def __init__(
		self, 
		data_path,
		split_ratio=1.0,
		mode='train',
		label_list=None,
		tg_dataset=None):

		self.ops_decoder = ['input', 'output', 'none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']

		# ---------- entire dataset ---------- #
		self.data = torch.load(data_path)
		# ---------- igraph ---------- #
		self.igraph_list = self.data['g']
		# ---------- x ---------- #
		self.x_list = self.data['x']
		# ---------- adj ---------- #
		adj = self.get_adj()
		self.adj_list = [adj] * len(self.igraph_list)
		# ---------- matrix ---------- #
		self.matrix_list = self.data['matrix']
		# ---------- arch_str ---------- #
		self.arch_str_list = self.data['str']
		# ---------- labels ---------- #
		self.label_list = label_list
		if self.label_list is not None:
			self.val_acc_list = self.data['val-acc'][tg_dataset]
			self.test_acc_list = self.data['test-acc'][tg_dataset]
			self.flops_list = self.data['flops'][tg_dataset]
			self.params_list = self.data['params'][tg_dataset]
			self.latency_list = self.data['latency'][tg_dataset]

		# ----------- split dataset ---------- #
		self.ds_idx = list(torch.load(DATA_PATH + '/ridx.pt'))
		self.split_ratio = split_ratio
		num_train = int(len(self.x_list) * self.split_ratio)
		num_test = len(self.x_list) - num_train

		# ----------- compute mean and std w/ training dataset ---------- #
		if self.label_list is not None:
			self.train_idx_list = self.ds_idx[:num_train]
			print('>>> Computing mean and std of the training set...')
			LABEL_TO_MEAN_STD = defaultdict(dict)
			assert type(self.label_list) == list, f"self.label_list is {type(self.label_list)}"
			for label in self.label_list:
				if label == 'val-acc':
					self.val_acc_list_tr = [self.val_acc_list[i] for i in self.train_idx_list]
					LABEL_TO_MEAN_STD[label]['std'], LABEL_TO_MEAN_STD[label]['mean'] = torch.std_mean(torch.tensor(self.val_acc_list_tr))
				elif label == 'test-acc':
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
			if num_test == 0:
				self.idx_list = self.ds_idx[:100]
			else:
				self.idx_list = self.ds_idx[:num_test]
		elif self.mode in ['test']:
			if num_test == 0:
				self.idx_list = self.ds_idx[15000:]
			else:
				self.idx_list = self.ds_idx[num_train:]

		self.igraph_list_ = [self.igraph_list[i] for i in self.idx_list]
		self.x_list_ = [self.x_list[i] for i in self.idx_list]
		self.adj_list_ = [self.adj_list[i] for i in self.idx_list]
		self.matrix_list_ = [self.matrix_list[i] for i in self.idx_list]
		self.arch_str_list_ = [self.arch_str_list[i] for i in self.idx_list]

		if self.label_list is not None:
			assert type(self.label_list) == list
			for label in self.label_list:
				if label == 'val-acc':
					self.val_acc_list_ = [self.val_acc_list[i] for i in self.idx_list]
					self.val_acc_list_ = self.normalize(self.val_acc_list_, LABEL_TO_MEAN_STD[label]['mean'], LABEL_TO_MEAN_STD[label]['std'])
				elif label == 'test-acc':
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


	# def get_not_connect_prev_adj(self):
	def get_adj(self):
		adj = np.asarray(
				[[0, 1, 1, 1, 0, 0, 0, 0],
				[0, 0, 0, 0, 1, 1, 0, 0],
				[0, 0, 0, 0, 0, 0, 1, 0],
				[0, 0, 0, 0, 0, 0, 0, 1],
				[0, 0, 0, 0, 0, 0, 1, 0],
				[0, 0, 0, 0, 0, 0, 0, 1],
				[0, 0, 0, 0, 0, 0, 0, 1],
				[0, 0, 0, 0, 0, 0, 0, 0]]
			)
		adj = torch.tensor(adj, dtype=torch.float32, device=torch.device('cpu'))
		return adj


	@property
	def adj(self):
		return self.adj_list_[0]


	def mask(self, algo='floyd'):
		from utils import aug_mask
		return aug_mask(self.adj, algo=algo)[0]


	def get_unnoramlized_entire_data(self, label, tg_dataset):
		entire_val_acc_list = self.data['val-acc'][tg_dataset]
		entire_test_acc_list = self.data['test-acc'][tg_dataset]
		entire_flops_list = self.data['flops'][tg_dataset]
		entire_params_list = self.data['params'][tg_dataset]
		entire_latency_list = self.data['latency'][tg_dataset]
		
		if label == 'val-acc':
			return entire_val_acc_list
		elif label == 'test-acc':
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
		entire_val_acc_list = self.data['val-acc'][tg_dataset]
		entire_test_acc_list = self.data['test-acc'][tg_dataset]
		entire_flops_list = self.data['flops'][tg_dataset]
		entire_params_list = self.data['params'][tg_dataset]
		entire_latency_list = self.data['latency'][tg_dataset]
		
		if label == 'val-acc':
			return [entire_val_acc_list[i] for i in self.idx_list]
		elif label == 'test-acc':
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
				if label == 'val-acc':
					label_dict[f"{label}"] = self.val_acc_list_[index]
				elif label == 'test-acc':
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


# ---------- Meta-Dataset ---------- #
def get_meta_dataset(config):
	train_dataset = MetaTrainDatabase(
		data_path=DATA_PATH,
		num_sample=config.model.num_sample,
		label_list=config.data.label_list,
		mode='train')

	eval_dataset = MetaTrainDatabase(
		data_path=DATA_PATH,
		num_sample=config.model.num_sample,
		label_list=config.data.label_list,
		mode='eval')

	test_dataset = None

	return train_dataset, eval_dataset, test_dataset


def get_meta_dataloader(config ,train_dataset, eval_dataset, test_dataset):
	train_loader = DataLoader(dataset=train_dataset,
							batch_size=config.training.batch_size,
							shuffle=True)

	eval_loader = DataLoader(dataset=eval_dataset,
							batch_size=config.training.batch_size,
							shuffle=False)

	test_loader = None

	return train_loader, eval_loader, test_loader 


class MetaTrainDatabase(Dataset):
	def __init__(
		self, 
		data_path, 
		num_sample, 
		label_list,
		mode='train'):

		self.ops_decoder = ['input', 'output', 'none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']

		self.mode = mode
		self.acc_norm = True
		self.num_sample = num_sample
		self.x = torch.load(os.path.join(data_path, 'imgnet32bylabel.pt'))

		mtr_data_path = os.path.join(data_path, 'meta_train_tasks_predictor.pt')
		idx_path = os.path.join(data_path, 'meta_train_tasks_predictor_idx.pt')
		data = torch.load(mtr_data_path)

		self.acc_list = data['acc']
		self.task = data['task']

		# ---------- igraph ---------- #
		self.igraph_list = data['g']
		# ---------- x ---------- #
		self.x_list = data['x']
		# ---------- adj ---------- #
		adj = self.get_adj()
		self.adj_list = [adj] * len(self.igraph_list)
		# ---------- matrix ----------- #
		if 'matrix' in data:
			self.matrix_list = data['matrix']
		else:
			self.matrix_list = [decode_x_to_NAS_BENCH_201_matrix(i) for i in self.x_list]
		# ---------- arch_str ---------- #
		if 'str' in data:
			self.arch_str_list = data['str']
		else:
			self.arch_str_list = [decode_x_to_NAS_BENCH_201_string(i, self.ops_decoder) for i in self.x_list]
		# ---------- label ---------- #
		self.label_list = label_list
		if self.label_list is not None:
			self.flops_list = torch.tensor(data['flops'])
			self.params_list = torch.tensor(data['params'])
			self.latency_list = torch.tensor(data['latency'])

		random_idx_lst = torch.load(idx_path)
		self.idx_lst = {}
		self.idx_lst['eval'] = random_idx_lst[:400]
		self.idx_lst['train'] = random_idx_lst[400:]
		self.acc_list = torch.tensor(self.acc_list)
		self.mean = torch.mean(self.acc_list[self.idx_lst['train']]).item()
		self.std = torch.std(self.acc_list[self.idx_lst['train']]).item()
		self.task_lst = torch.load(os.path.join(data_path, 'meta_train_task_lst.pt'))


	def get_adj(self):
		adj = np.asarray(
				[[0, 1, 1, 1, 0, 0, 0, 0],
				[0, 0, 0, 0, 1, 1, 0, 0],
				[0, 0, 0, 0, 0, 0, 1, 0],
				[0, 0, 0, 0, 0, 0, 0, 1],
				[0, 0, 0, 0, 0, 0, 1, 0],
				[0, 0, 0, 0, 0, 0, 0, 1],
				[0, 0, 0, 0, 0, 0, 0, 1],
				[0, 0, 0, 0, 0, 0, 0, 0]]
			)
		adj = torch.tensor(adj, dtype=torch.float32, device=torch.device('cpu'))
		return adj


	@property
	def adj(self):
		return self.adj_list[0]


	def mask(self, algo='floyd'):
		from utils import aug_mask
		return aug_mask(self.adj, algo=algo)[0]


	def set_mode(self, mode):
		self.mode = mode


	def __len__(self):
		return len(self.idx_lst[self.mode])


	def __getitem__(self, index):
		data = []
		ridx = self.idx_lst[self.mode]
		tidx = self.task[ridx[index]]
		classes = self.task_lst[tidx]

		# ---------- igraph -----------
		graph = self.igraph_list[ridx[index]]
		# ---------- x -----------
		x = self.x_list[ridx[index]]
		# ---------- adj ----------
		adj = self.adj_list[ridx[index]]

		acc = self.acc_list[ridx[index]]
		for cls in classes:
			cx = self.x[cls-1][0]
			ridx = torch.randperm(len(cx))
			data.append(cx[ridx[:self.num_sample]])
		task = torch.cat(data)
		if self.acc_norm:
			acc = ((acc- self.mean) / self.std) / 100.0
		else:
			acc = acc / 100.0

		label_dict = {}
		if self.label_list is not None:
			assert type(self.label_list) == list
			for label in self.label_list:
				if label == 'meta-acc':
					label_dict[f"{label}"] = acc
				elif label == 'flops':
					label_dict[f"{label}"] = self.flops_list[ridx[index]]
				elif label == 'params':
					label_dict[f"{label}"] = self.params_list[ridx[index]]
				elif label == 'latency':
					label_dict[f"{label}"] = self.latency_list[ridx[index]]
				else:
					raise ValueError

		return x, adj, label_dict, task


class MetaTestDataset(Dataset):
	def __init__(self, data_path, data_name, num_sample, num_class=None):
		self.num_sample = num_sample
		self.data_name = data_name

		num_class_dict = {
		'cifar100': 100,
		'cifar10':  10,
		'aircraft': 30,
		'pets':     37
		}

		if num_class is not None:
			self.num_class = num_class
		else:
			self.num_class = num_class_dict[data_name]

		self.x = torch.load(os.path.join(data_path, f'{data_name}bylabel.pt'))


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