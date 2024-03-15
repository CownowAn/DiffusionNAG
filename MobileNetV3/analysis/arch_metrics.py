from analysis.arch_functions import compute_arch_metrics, compute_arch_metrics_meta
from torch import Tensor
import wandb
import torch.nn as nn


class SamplingArchMetrics(nn.Module):
    def __init__(self, config, train_ds, exp_name):
        super().__init__()
        
        self.exp_name = exp_name
        self.train_ds = train_ds
        if config.data.name == 'ofa':
            self.train_arch_str_list = train_ds.x_list_
        else:
            self.train_arch_str_list = train_ds.arch_str_list_
        self.name = config.data.name
        self.except_inout = config.data.except_inout
        self.data_root = config.data.root


    def forward(self, arch_list: list, adj, mask, this_sample_dir, test=False, timestep=None):
        """_summary_
        :params arch_list: list of archs
        :params adj: [batch_size, num_nodes, num_nodes]
        :params mask: [batch_size, num_nodes, num_nodes]
        """
        arch_metrics, all_arch_str = compute_arch_metrics(
            arch_list, adj, mask, self.train_arch_str_list, self.train_ds, timestep=timestep,
            name=self.name, except_inout=self.except_inout, data_root=self.data_root)
        # arch_metrics 
        # ([validity, uniqueness, novelty], 
            # unique,
            # dict(test_acc_list=test_acc_list, flops_list=flops_list, params_list=params_list, latency_list=latency_list), 
            # all_arch_str)

        if test and self.name != 'ofa':
            with open(r'final_.txt', 'w') as fp:
                for arch_str in all_arch_str:
                    # write each item on a new line
                    fp.write("%s\n" % arch_str)
                print('All archs saved')

        if self.name != 'ofa':
            valid_unique_arch = arch_metrics[1]
            valid_unique_arch_prop_dict = arch_metrics[2] # test_acc, flops, params, latency
            # textfile = open(f'{this_sample_dir}/archs/{name}/valid_unique_arch_step-{current_step}.txt', "w")
            textfile = open(f'{this_sample_dir}/valid_unique_archs.txt', "w")
            for i in range(len(valid_unique_arch)):
                textfile.write(f"Arch: {valid_unique_arch[i]} \n")
                textfile.write(f"Test Acc: {valid_unique_arch_prop_dict['test_acc_list'][i]} \n")
                textfile.write(f"FLOPs: {valid_unique_arch_prop_dict['flops_list'][i]} \n ")
                textfile.write(f"#Params: {valid_unique_arch_prop_dict['params_list'][i]} \n")
                textfile.write(f"Latency: {valid_unique_arch_prop_dict['latency_list'][i]} \n \n")
            textfile.writelines(valid_unique_arch)
            textfile.close()
            
        # res_dic = {
        #         'Validity': arch_metrics[0][0], 'Uniqueness': arch_metrics[0][1], 'Novelty': arch_metrics[0][2],
        #         'test_acc_max': -1, 'test_acc_min':-1, 'test_acc_mean': -1, 'test_acc_std': 0,
        #         'flops_max': -1, 'flops_min':-1, 'flops_mean': -1, 'flops_std': 0,
        #         'params_max': -1, 'params_min':-1, 'params_mean': -1, 'params_std': 0,
        #         'latency_max': -1, 'latency_min':-1, 'latency_mean': -1, 'latency_std': 0,
        #         }

        return arch_metrics

class SamplingArchMetricsMeta(nn.Module):
    def __init__(self, config, train_ds, exp_name, train_index=None, nasbench=None):
        super().__init__()
        
        self.exp_name = exp_name
        self.train_ds = train_ds
        self.search_space = config.data.name 
        if self.search_space == 'ofa':
            self.train_arch_str_list = None
        else:
            self.train_arch_str_list = [train_ds.arch_str_list[i] for i in train_ds.idx_lst['train']]

    def forward(self, arch_list: list, adj, mask, this_sample_dir, test=False, 
                timestep=None, check_dataname='cifar10'):
        """_summary_
        :params arch_list: list of archs
        :params adj: [batch_size, num_nodes, num_nodes]
        :params mask: [batch_size, num_nodes, num_nodes]
        """
        arch_metrics = compute_arch_metrics_meta(arch_list, adj, mask, self.train_arch_str_list, 
                                            self.train_ds, timestep=timestep, check_dataname=check_dataname,
                                            name=self.search_space)
        all_arch_str = arch_metrics[-1]

        if test:
            with open(r'final_.txt', 'w') as fp:
                for arch_str in all_arch_str:
                    # write each item on a new line
                    fp.write("%s\n" % arch_str)
                print('All archs saved')

        valid_unique_arch = arch_metrics[1] # arch_str
        valid_unique_arch_prop_dict = arch_metrics[2] # test_acc, flops, params, latency
        # textfile = open(f'{this_sample_dir}/archs/{name}/valid_unique_arch_step-{current_step}.txt', "w")
        if self.search_space != 'ofa':
            textfile = open(f'{this_sample_dir}/valid_unique_archs.txt', "w")
            for i in range(len(valid_unique_arch)):
                textfile.write(f"Arch: {valid_unique_arch[i]} \n")
                textfile.write(f"Arch Index: {valid_unique_arch_prop_dict['arch_idx_list'][i]} \n")
                textfile.write(f"Test Acc: {valid_unique_arch_prop_dict['test_acc_list'][i]} \n")
                textfile.write(f"FLOPs: {valid_unique_arch_prop_dict['flops_list'][i]} \n ")
                textfile.write(f"#Params: {valid_unique_arch_prop_dict['params_list'][i]} \n")
                textfile.write(f"Latency: {valid_unique_arch_prop_dict['latency_list'][i]} \n \n")
            textfile.writelines(valid_unique_arch)
            textfile.close()
        
        return arch_metrics