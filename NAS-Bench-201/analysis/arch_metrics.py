from analysis.arch_functions import compute_arch_metrics, compute_arch_metrics_meta
import torch.nn as nn


class SamplingArchMetrics(nn.Module):
    def __init__(self, 
                 config, 
                 train_ds, 
                 exp_name,):

        super().__init__()
        self.exp_name = exp_name
        self.train_ds = train_ds
        self.train_arch_str_list = train_ds.arch_str_list_


    def forward(self, 
                arch_list: list,
                this_sample_dir,
                check_dataname='cifar10'):

        arch_metrics, all_arch_str = compute_arch_metrics(arch_list=arch_list,
                                                          train_arch_str_list=self.train_arch_str_list, 
                                                          train_ds=self.train_ds, 
                                                          check_dataname=check_dataname)

        valid_unique_arch = arch_metrics[1] # arch_str
        valid_unique_arch_prop_dict = arch_metrics[2] # flops, params, latency
        textfile = open(f'{this_sample_dir}/valid_unique_archs.txt', "w")
        for i in range(len(valid_unique_arch)):
            textfile.write(f"Arch: {valid_unique_arch[i]} \n")
            textfile.write(f"Arch Index: {valid_unique_arch_prop_dict['arch_idx_list'][i]} \n")
            textfile.write(f"FLOPs: {valid_unique_arch_prop_dict['flops_list'][i]} \n")
            textfile.write(f"#Params: {valid_unique_arch_prop_dict['params_list'][i]} \n")
            textfile.write(f"Latency: {valid_unique_arch_prop_dict['latency_list'][i]} \n\n")
        textfile.writelines(valid_unique_arch)
        textfile.close()

        return arch_metrics


class SamplingArchMetricsMeta(nn.Module):
    def __init__(self, 
                 config, 
                 train_ds, 
                 exp_name):

        super().__init__()
        self.exp_name = exp_name
        self.train_ds = train_ds
        self.search_space = config.data.name 
        self.train_arch_str_list = [train_ds.arch_str_list[i] for i in train_ds.idx_lst['train']]


    def forward(self,
                arch_list: list,
                this_sample_dir,
                check_dataname='cifar10'):
        
        arch_metrics = compute_arch_metrics_meta(arch_list=arch_list,
                                                 train_arch_str_list=self.train_arch_str_list,
                                                 train_ds=self.train_ds,
                                                 check_dataname=check_dataname)

        valid_unique_arch = arch_metrics[1] # arch_str
        valid_unique_arch_prop_dict = arch_metrics[2] # flops, params, latency
        textfile = open(f'{this_sample_dir}/valid_unique_archs.txt', "w")
        for i in range(len(valid_unique_arch)):
            textfile.write(f"Arch: {valid_unique_arch[i]} \n")
            textfile.write(f"Arch Index: {valid_unique_arch_prop_dict['arch_idx_list'][i]} \n")
            textfile.write(f"FLOPs: {valid_unique_arch_prop_dict['flops_list'][i]} \n")
            textfile.write(f"#Params: {valid_unique_arch_prop_dict['params_list'][i]} \n")
            textfile.write(f"Latency: {valid_unique_arch_prop_dict['latency_list'][i]} \n\n")
        textfile.writelines(valid_unique_arch)
        textfile.close()

        return arch_metrics