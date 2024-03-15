from __future__ import print_function
import torch
import os
import gc
import sys
import numpy as np
import os
import subprocess

from nag_utils import mean_confidence_interval
from nag_utils import restore_checkpoint
from nag_utils import load_graph_config
from nag_utils import load_model

sys.path.append(os.path.join(os.getcwd(), 'main_exp'))
from nas_bench_201 import train_single_model
from unnoised_model import MetaSurrogateUnnoisedModel
from diffusion.run_lib import generate_archs_meta
from diffusion.run_lib import get_sampling_fn_meta
from diffusion.run_lib import get_score_model
from diffusion.run_lib import get_surrogate 
from loader import MetaTestDataset
from logger import Logger
from all_path import *


class NAG:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        ## Target dataset information
        self.raw_data_path = RAW_DATA_PATH
        self.data_path = DATA_PATH
        self.data_name = args.data_name
        self.num_class = args.num_class
        self.num_sample = args.num_sample

        graph_config = load_graph_config(args.graph_data_name, args.nvt, NASBENCH201)
        self.meta_surrogate_unnoised_model = MetaSurrogateUnnoisedModel(args, graph_config)
        load_model(model=self.meta_surrogate_unnoised_model, 
                   ckpt_path=META_SURROGATE_UNNOISED_CKPT_PATH)
        self.meta_surrogate_unnoised_model.to(self.device)

        ## Load pre-trained meta-surrogate model
        self.meta_surrogate_ckpt_path = META_SURROGATE_CKPT_PATH

        ## Load score network model (base diffusion model)
        self.load_diffusion_model(args=args)

        ## Check config
        self.check_config()

        ## Set logger
        self.logger = Logger(
            log_dir=args.exp_name,
            write_textfile=True
        )
        self.logger.update_config(args, is_args=True)
        self.logger.write_str(str(vars(args)))
        self.logger.write_str('-' * 100)


    def check_config(self):
        """
        Check if the configuration of the pre-trained score network model matches that of the meta surrogate model.
        """
        scorenet_config = torch.load(self.config.scorenet_ckpt_path)['config']
        meta_surrogate_config = torch.load(self.meta_surrogate_ckpt_path)['config']
        assert scorenet_config.model.sigma_min == meta_surrogate_config.model.sigma_min
        assert scorenet_config.model.sigma_max == meta_surrogate_config.model.sigma_max
        assert scorenet_config.training.sde == meta_surrogate_config.training.sde
        assert scorenet_config.training.continuous == meta_surrogate_config.training.continuous
        assert scorenet_config.data.centered == meta_surrogate_config.data.centered
        assert scorenet_config.data.max_node == meta_surrogate_config.data.max_node
        assert scorenet_config.data.n_vocab == meta_surrogate_config.data.n_vocab


    def forward(self, x, arch):
        D_mu = self.meta_surrogate_unnoised_model.set_encode(x.to(self.device))
        G_mu = self.meta_surrogate_unnoised_model.graph_encode(arch)
        y_pred = self.meta_surrogate_unnoised_model.predict(D_mu, G_mu)
        return y_pred


    def meta_test(self):
        if self.data_name == 'all':
            for data_name in ['cifar10', 'cifar100', 'aircraft', 'pets']:
                self.meta_test_per_dataset(data_name)
        else:
            self.meta_test_per_dataset(self.data_name)


    def meta_test_per_dataset(self, data_name):
        ## Load NASBench201
        self.nasbench201 = torch.load(NASBENCH201)
        all_arch_str = np.array(self.nasbench201['arch']['str'])

        ## Load meta-test dataset
        self.test_dataset = MetaTestDataset(self.data_path, data_name, self.num_sample, self.num_class)

        ## Set save path
        meta_test_path = os.path.join(META_TEST_PATH, data_name)
        os.makedirs(meta_test_path, exist_ok=True)
        f_arch_str = open(os.path.join(self.args.exp_name, 'architecture.txt'), 'w')
        f_arch_acc = open(os.path.join(self.args.exp_name, 'accuracy.txt'), 'w')

        ## Generate architectures
        gen_arch_str = self.get_gen_arch_str()
        gen_arch_igraph = self.get_items(
            full_target=self.nasbench201['arch']['igraph'],
            full_source=self.nasbench201['arch']['str'],
            source=gen_arch_str)

        ## Sort with unnoised meta-surrogate model
        y_pred_all = []
        self.meta_surrogate_unnoised_model.eval()
        self.meta_surrogate_unnoised_model.to(self.device)
        with torch.no_grad():
            for arch_igraph in gen_arch_igraph:
                x, g = self.collect_data(arch_igraph)
                y_pred = self.forward(x, g)
                y_pred = torch.mean(y_pred)
                y_pred_all.append(y_pred.cpu().detach().item())
        sorted_arch_lst = self.sort_arch(data_name, torch.tensor(y_pred_all), gen_arch_str)

        ## Record the information of the architecture generated in sorted order
        for _, arch_str in enumerate(sorted_arch_lst):
            f_arch_str.write(f'{arch_str}\n')
        arch_idx_lst = [self.nasbench201['arch']['str'].index(i) for i in sorted_arch_lst]
        arch_str_lst = []
        arch_acc_lst = []

        ## Get the accuracy of the architecture
        if 'cifar' in data_name:
            sorted_acc_lst = self.get_items(
                full_target=self.nasbench201['test-acc'][data_name],
                full_source=self.nasbench201['arch']['str'],
                source=sorted_arch_lst)
            arch_str_lst += sorted_arch_lst
            arch_acc_lst += sorted_acc_lst
            for arch_idx, acc in zip(arch_idx_lst, sorted_acc_lst):
                msg = f'Avg {acc:4f} (%)'
                f_arch_acc.write(msg + '\n')
        else:
            if self.args.multi_proc:
                ## Run multiple processes in parallel
                run_file = os.path.join(os.getcwd(), 'main_exp', 'transfer_nag', 'run_multi_proc.py')
                MAX_CAP = 5 # hard-coded for available GPUs
                if not len(arch_idx_lst) > MAX_CAP:
                    arch_idx_lst_ = [arch_idx for arch_idx in arch_idx_lst if not os.path.exists(os.path.join(meta_test_path, str(arch_idx)))]
                    support_ = ','.join([str(i) for i in arch_idx_lst_])
                    num_split = int(3 * len(arch_idx_lst_)) # why 3? => running for 3 seeds
                    cmd = f"python {run_file} --num_split {num_split} --arch_idx_lst {support_} --meta_test_path {meta_test_path} --data_name {data_name} --raw_data_path {self.raw_data_path}"
                    subprocess.run([cmd], shell=True)
                else:
                    arch_idx_lst_ = []
                    for j, arch_idx in enumerate(arch_idx_lst):
                        if not os.path.exists(os.path.join(meta_test_path, str(arch_idx))):
                            arch_idx_lst_.append(arch_idx)
                        if (len(arch_idx_lst_) == MAX_CAP) or (j == len(arch_idx_lst) - 1):
                            support_ = ','.join([str(i) for i in arch_idx_lst_])
                            num_split = int(3 * len(arch_idx_lst_))
                            cmd = f"python {run_file} --num_split {num_split} --arch_idx_lst {support_} --meta_test_path {meta_test_path} --data_name {data_name} --raw_data_path {self.raw_data_path}"
                            subprocess.run([cmd], shell=True)
                            arch_idx_lst_ = []

                while True:
                    try:
                        acc_runs_lst = []
                        epoch = 199
                        seeds = (777, 888, 999)
                        for arch_idx in arch_idx_lst:
                            acc_runs = []
                            save_path_ = os.path.join(meta_test_path, str(arch_idx))
                            for seed in seeds:
                                result = torch.load(os.path.join(save_path_, f'seed-0{seed}.pth'))
                                acc_runs.append(result[data_name]['valid_acc1es'][f'x-test@{epoch}'])
                            acc_runs_lst.append(acc_runs)
                        break
                    except:
                        pass
                    for i in acc_runs_lst:print(np.mean(i))
                for arch_idx, acc_runs in zip(arch_idx_lst, acc_runs_lst):
                    for r, acc in enumerate(acc_runs):
                        msg = f'run {r+1} {acc:.2f} (%)'
                        f_arch_acc.write(msg + '\n')
                    m, h = mean_confidence_interval(acc_runs)
                    msg = f'Avg {m:.2f}+-{h.item():.2f} (%)'
                    f_arch_acc.write(msg + '\n')
                    arch_acc_lst.append(np.mean(acc_runs))
                    arch_str_lst.append(all_arch_str[arch_idx])

            else:
                for arch_idx in arch_idx_lst:
                    acc_runs = self.train_single_arch(
                        data_name, self.nasbench201['str'][arch_idx], meta_test_path)
                    for r, acc in enumerate(acc_runs):
                        msg = f'run {r+1} {acc:.2f} (%)'
                        f_arch_acc.write(msg + '\n')
                    m, h = mean_confidence_interval(acc_runs)
                    msg = f'Avg {m:.2f}+-{h.item():.2f} (%)'
                    f_arch_acc.write(msg + '\n')
                    arch_acc_lst.append(np.mean(acc_runs))
                    arch_str_lst.append(all_arch_str[arch_idx])

        # Save results
        results_path = os.path.join(self.args.exp_name, 'results.pt')
        torch.save({
            'arch_idx_lst': arch_idx_lst,
            'arch_str_lst': arch_str_lst,
            'arch_acc_lst': arch_acc_lst
        }, results_path)
        print(f">>> Save the results at {results_path}...")


    def train_single_arch(self, data_name, arch_str, meta_test_path):
        save_path = os.path.join(meta_test_path, arch_str)
        seeds = (777, 888, 999)
        train_single_model(save_dir=save_path,
                           workers=24,
                           datasets=[data_name],
                           xpaths=[f'{self.raw_data_path}/{data_name}'],
                           splits=[0],
                           use_less=False,
                           seeds=seeds,
                           model_str=arch_str,
                           arch_config={'channel': 16, 'num_cells': 5})
        epoch = 199
        test_acc_lst = []
        for seed in seeds:
            result = torch.load(os.path.join(save_path, f'seed-0{seed}.pth'))
            test_acc_lst.append(result[data_name]['valid_acc1es'][f'x-test@{epoch}'])
        return test_acc_lst


    def sort_arch(self, data_name, y_pred_all, gen_arch_str):
        _, sorted_idx = torch.sort(y_pred_all, descending=True)
        sotred_gen_arch_str = [gen_arch_str[_] for _ in sorted_idx]
        return sotred_gen_arch_str


    def collect_data_only(self):
        x_batch = []
        x_batch.append(self.test_dataset[0])
        return torch.stack(x_batch).to(self.device)


    def collect_data(self, arch_igraph):
        x_batch, g_batch = [], []
        for _ in range(10):
            x_batch.append(self.test_dataset[0])
            g_batch.append(arch_igraph)
        return torch.stack(x_batch).to(self.device), g_batch


    def get_items(self, full_target, full_source, source):
        return [full_target[full_source.index(_)] for _ in source]


    def load_diffusion_model(self, args):
        self.config = torch.load('./configs/transfer_nag_config.pt')
        self.config.device = torch.device('cuda')
        self.config.data.label_list = ['meta-acc']
        self.config.scorenet_ckpt_path = SCORENET_CKPT_PATH
        self.config.sampling.classifier_scale = args.classifier_scale
        self.config.eval.batch_size = args.eval_batch_size
        self.config.sampling.predictor = args.predictor
        self.config.sampling.corrector = args.corrector
        self.config.sampling.check_dataname = self.data_name
        self.sampling_fn, self.sde = get_sampling_fn_meta(self.config)
        self.score_model, self.score_ema, self.score_config = get_score_model(self.config)


    def get_gen_arch_str(self):
        ## Load meta-surrogate model
        meta_surrogate_config = torch.load(self.meta_surrogate_ckpt_path)['config']
        meta_surrogate_model = get_surrogate(meta_surrogate_config)
        meta_surrogate_state = dict(model=meta_surrogate_model, step=0, config=meta_surrogate_config)
        meta_surrogate_state = restore_checkpoint(
            self.meta_surrogate_ckpt_path, 
            meta_surrogate_state,
            device=self.config.device, 
            resume=True)

        ## Get dataset embedding, x
        with torch.no_grad():
            x = self.collect_data_only()

        ## Generate architectures
        generated_arch_str = generate_archs_meta(
            config=self.config,
            sampling_fn=self.sampling_fn,
            score_model=self.score_model, 
            score_ema=self.score_ema,
            meta_surrogate_model=meta_surrogate_model,
            num_samples=self.args.n_gen_samples,
            args=self.args,
            task=x)

        ## Clean up
        meta_surrogate_model = None
        gc.collect()

        return generated_arch_str