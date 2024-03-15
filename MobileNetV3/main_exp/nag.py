from __future__ import print_function
import torch
import os
import gc
import sys
from tqdm import tqdm
import numpy as np
import time
import os

from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.stats import pearsonr

from transfer_nag_lib.MetaD2A_mobilenetV3.metad2a_utils import load_graph_config, decode_ofa_mbv3_str_to_igraph
from transfer_nag_lib.MetaD2A_mobilenetV3.metad2a_utils import get_log
from transfer_nag_lib.MetaD2A_mobilenetV3.metad2a_utils import save_model, mean_confidence_interval

from transfer_nag_lib.MetaD2A_mobilenetV3.loader import get_meta_train_loader, MetaTestDataset

from transfer_nag_lib.encoder_FSBO_ofa import EncoderFSBO as PredictorModel
from transfer_nag_lib.MetaD2A_mobilenetV3.predictor import Predictor as MetaD2APredictor
from transfer_nag_lib.MetaD2A_mobilenetV3.evaluation.train import train_single_model

from diffusion.run_lib import generate_archs 
from diffusion.run_lib import get_sampling_fn_meta
from diffusion.run_lib import get_score_model
from diffusion.run_lib import get_predictor 

sys.path.append(os.path.join(os.getcwd()))
from all_path import *
from utils import restore_checkpoint


class NAG:
    def __init__(self, args, dgp_arch=[99, 50, 179, 194], bohb=False):
        self.args = args
        self.batch_size = args.batch_size
        self.num_sample = args.num_sample
        self.max_epoch = args.max_epoch
        self.save_epoch = args.save_epoch
        self.save_path = args.save_path
        self.search_space = args.search_space
        self.model_name = 'predictor'
        self.test = args.test
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.max_corr_dict = {'corr': -1, 'epoch': -1}
        self.train_arch = args.train_arch
        self.use_metad2a_predictor_selec = args.use_metad2a_predictor_selec

        self.raw_data_path = RAW_DATA_PATH
        self.model_path = UNNOISE_META_PREDICTOR_CKPT_PATH
        self.data_path = PROCESSED_DATA_PATH   
        self.classifier_ckpt_path = NOISE_META_PREDICTOR_CKPT_PATH
        self.load_diffusion_model(self.args.n_training_samples, args.pos_enc_type)

        graph_config = load_graph_config(
            args.graph_data_name, args.nvt, self.data_path)

        self.model = PredictorModel(args, graph_config, dgp_arch=dgp_arch)
        self.metad2a_model = MetaD2APredictor(args).model
        
        if self.test:
            self.data_name = args.data_name
            self.num_class = args.num_class
            self.load_epoch = args.load_epoch
            self.n_training_samples = self.args.n_training_samples
            self.n_gen_samples = args.n_gen_samples
            self.folder_name = args.folder_name
            self.unique = args.unique
            
            model_state_dict = self.model.state_dict()
            load_max_pt = 'ckpt_max_corr.pt'
            ckpt_path = os.path.join(self.model_path, load_max_pt)
            ckpt = torch.load(ckpt_path)
            for k, v in ckpt.items():
                if k in model_state_dict.keys():
                    model_state_dict[k] = v
            self.model.cpu()
            self.model.load_state_dict(model_state_dict)
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min',
                      factor=0.1, patience=1000, verbose=True)
        self.mtrloader = get_meta_train_loader(
            self.batch_size, self.data_path, self.num_sample, is_pred=True)

        self.acc_mean = self.mtrloader.dataset.mean
        self.acc_std = self.mtrloader.dataset.std


    def forward(self, x, arch, labels=None, train=False, matrix=False, metad2a=False):
        if metad2a: 
            D_mu = self.metad2a_model.set_encode(x.to(self.device))
            G_mu = self.metad2a_model.graph_encode(arch)
            y_pred = self.metad2a_model.predict(D_mu, G_mu)
            return y_pred
        else:
            D_mu = self.model.set_encode(x.to(self.device))
            G_mu = self.model.graph_encode(arch, matrix=matrix)
            y_pred, y_dist = self.model.predict(D_mu, G_mu, labels=labels, train=train)
            return y_pred, y_dist
    
    def meta_train(self):
        sttime = time.time()
        for epoch in range(1, self.max_epoch + 1):
            self.mtrlog.ep_sttime = time.time()
            loss, corr = self.meta_train_epoch(epoch)
            self.scheduler.step(loss)
            self.mtrlog.print_pred_log(loss, corr, 'train', epoch)
            valoss, vacorr = self.meta_validation(epoch)
            if self.max_corr_dict['corr'] < vacorr or epoch==1:
                self.max_corr_dict['corr'] = vacorr
                self.max_corr_dict['epoch'] = epoch
                self.max_corr_dict['loss'] = valoss
                save_model(epoch, self.model, self.model_path, max_corr=True)

            self.mtrlog.print_pred_log(
                valoss, vacorr, 'valid', max_corr_dict=self.max_corr_dict)

            if epoch % self.save_epoch == 0:
                save_model(epoch, self.model, self.model_path)

        self.mtrlog.save_time_log()
        self.mtrlog.max_corr_log(self.max_corr_dict)

    def meta_train_epoch(self, epoch):
        self.model.to(self.device)
        self.model.train()

        self.mtrloader.dataset.set_mode('train')

        dlen = len(self.mtrloader.dataset)
        trloss = 0
        y_all, y_pred_all = [], []
        pbar = tqdm(self.mtrloader)

        for x, g, acc in pbar:
            self.optimizer.zero_grad()
            y_pred, y_dist = self.forward(x, g, labels=acc, train=True, matrix=False)
            y = acc.to(self.device).double()
            print(y.double())
            print(y_dist)
            loss = -self.model.mll(y_dist, y)
            loss.backward()
            self.optimizer.step()

            y = y.tolist()
            y_pred = y_pred.squeeze().tolist()
            y_all += y
            y_pred_all += y_pred
            pbar.set_description(get_log(
                epoch, loss, y_pred, y, self.acc_std, self.acc_mean))
            trloss += float(loss)

        return trloss / dlen, pearsonr(np.array(y_all),
                                       np.array(y_pred_all))[0]
    
    def meta_validation(self, epoch):
        self.model.to(self.device)
        self.model.eval()

        valoss = 0
        self.mtrloader.dataset.set_mode('valid')
        dlen = len(self.mtrloader.dataset)
        y_all, y_pred_all = [], []
        pbar = tqdm(self.mtrloader)

        with torch.no_grad():
            for x, g, acc in pbar:
                y_pred, y_dist = self.forward(x, g, labels=acc, train=False, matrix=False)
                y = acc.to(self.device)
                loss = -self.model.mll(y_dist, y)

                y = y.tolist()
                y_pred = y_pred.squeeze().tolist()
                y_all += y
                y_pred_all += y_pred
                pbar.set_description(get_log(
                    epoch, loss, y_pred, y, self.acc_std, self.acc_mean, tag='val'))
                valoss += float(loss)
                try:
                    pearson_corr = pearsonr(np.array(y_all), np.array(y_pred_all))[0]
                except Exception as e:
                    pearson_corr = 0

        return valoss / dlen, pearson_corr

    def meta_test(self):
        if self.data_name == 'all':
            for data_name in ['cifar10', 'cifar100', 'aircraft', 'pets']:
                acc = self.meta_test_per_dataset(data_name)
        else:
            acc = self.meta_test_per_dataset(self.data_name)
        return acc
    
    
    def meta_test_per_dataset(self, data_name):        
        self.test_dataset = MetaTestDataset(
            self.data_path, data_name, self.num_sample, self.num_class)
        
        meta_test_path = self.args.exp_name
        os.makedirs(meta_test_path, exist_ok=True)
        f_arch_str = open(os.path.join(meta_test_path, 'architecture.txt'), 'w')
        f = open(os.path.join(meta_test_path, 'accuracy.txt'), 'w')
        
        elasped_time = []

        print(f'==> select top architectures for {data_name} by meta-predictor...')
        
        gen_arch_str = self.get_gen_arch_str()            
        
        gen_arch_igraph = [decode_ofa_mbv3_str_to_igraph(_) for _ in gen_arch_str]
        
        y_pred_all = []
        self.metad2a_model.eval()
        self.metad2a_model.to(self.device)
        
        # MetaD2A ver. prediction
        sttime = time.time()
        with torch.no_grad():
            for i, arch_igraph in enumerate(gen_arch_igraph):
                x, g = self.collect_data(arch_igraph)
                y_pred = self.forward(x, g, metad2a=True)
                y_pred = torch.mean(y_pred)
                y_pred_all.append(y_pred.cpu().detach().item())
        
        if self.use_metad2a_predictor_selec:
            top_arch_lst = self.select_top_arch(
                data_name, torch.tensor(y_pred_all), gen_arch_str, self.n_training_samples)
        else:
            top_arch_lst = gen_arch_str[:self.n_training_samples]
        
        elasped = time.time() - sttime
        elasped_time.append(elasped)
        
        for _, arch_str in enumerate(top_arch_lst):
            f_arch_str.write(f'{arch_str}\n'); print(f'neural architecture config: {arch_str}')
        
        support = top_arch_lst
        x_support = []
        y_support = []
        seeds = [777, 888, 999]
        y_support_per_seed = {
            _: [] for _ in seeds
        }
        net_info = {
            'params': [],
            'flops': [],
        }
        best_acc = 0.0
        best_sampe_num = 0

        print("Data name: %s" % data_name)
        for i, arch_str in enumerate(support):
            save_path = os.path.join(meta_test_path, arch_str)
            os.makedirs(save_path, exist_ok=True)
            acc_runs = []
            for seed in seeds:
                print(f'==> train for {data_name} {arch_str} ({seed})')
                valid_acc, max_valid_acc, params, flops = train_single_model(save_path=save_path,
                                workers=8,
                                datasets=data_name,
                                xpaths=f'{self.raw_data_path}/{data_name}',
                                splits=[0],
                                use_less=False,
                                seed=seed,
                                model_str=arch_str,
                                device='cuda',
                                lr=0.01,
                                momentum=0.9,
                                weight_decay=4e-5,
                                report_freq=50,
                                epochs=20,
                                grad_clip=5,
                                cutout=True,
                                cutout_length=16,
                                autoaugment=True,
                                drop=0.2,
                                drop_path=0.2,
                                img_size=224)
                acc_runs.append(valid_acc)
                y_support_per_seed[seed].append(valid_acc)
                
            for r, acc in enumerate(acc_runs):
                msg = f'run {r + 1} {acc:.2f} (%)'
                f.write(msg + '\n')
                f.flush()
                print(msg)
            m, h = mean_confidence_interval(acc_runs)
            
            if m > best_acc:
                best_acc = m
                best_sampe_num = i
            msg = f'Avg {m:.3f}+-{h.item():.2f} (%) (best acc {best_acc:.3f} - #{i})'
            f.write(msg + '\n')
            print(msg)
            y_support.append(np.mean(acc_runs))
            x_support.append(arch_str)
            net_info['params'].append(params)
            net_info['flops'].append(flops)
        torch.save({'y_support': y_support, 'x_support': x_support, 
                    'y_support_per_seed': y_support_per_seed, 
                    'net_info': net_info,
                    'best_acc': best_acc,
                    'best_sample_num': best_sampe_num}, 
                                            meta_test_path+'/result.pt')
                                    

        return None
    
    
    def train_single_arch(self, data_name, arch_str, meta_test_path):
        save_path = os.path.join(meta_test_path, arch_str)
        seeds = (777, 888, 999)
        train_single_model(save_path=save_path,
                           workers=24,
                           datasets=[data_name],
                           xpaths=[f'{self.raw_data_path}/{data_name}'],
                           splits=[0],
                           use_less=False,
                           seeds=seeds,
                           model_str=arch_str,
                           arch_config={'channel': 16, 'num_cells': 5})
        # Changed training time from 49/199
        epoch = 49 if data_name == 'mnist' else 199
        test_acc_lst = []
        for seed in seeds:
            result = torch.load(os.path.join(save_path, f'seed-0{seed}.pth'))
            test_acc_lst.append(result[data_name]['valid_acc1es'][f'x-test@{epoch}'])
        return test_acc_lst


    def select_top_arch(
            self, data_name, y_pred_all, gen_arch_str, N):
        _, sorted_idx = torch.sort(y_pred_all, descending=True)
        sotred_gen_arch_str = [gen_arch_str[_] for _ in sorted_idx]
        final_str = sotred_gen_arch_str[:N]
        return final_str

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

    def load_diffusion_model(self, n_training_samples, pos_enc_type):
        self.config = torch.load(CONFIG_PATH)
        self.config.data.root = SCORE_MODEL_DATA_PATH
        self.config.scorenet_ckpt_path = SCORE_MODEL_CKPT_PATH
        torch.save(self.config, CONFIG_PATH)
        
        self.sampling_fn, self.sde = get_sampling_fn_meta(self.config)
        self.sampling_fn_training_samples, _ = get_sampling_fn_meta(self.config, init=True, n_init=n_training_samples)
        self.score_model, self.score_ema, self.score_config \
            = get_score_model(self.config, pos_enc_type=pos_enc_type)
    
    def get_gen_arch_str(self):
        classifier_config = torch.load(self.classifier_ckpt_path)['config']
        # Load meta-predictor
        classifier_model = get_predictor(classifier_config)
        classifier_state = dict(model=classifier_model, step=0, config=classifier_config)
        classifier_state = restore_checkpoint(self.classifier_ckpt_path, 
                                              classifier_state, device=self.config.device, resume=True)
        print(f'==> load checkpoint for our predictor: {self.classifier_ckpt_path}...')
                
        with torch.no_grad():
            x = self.collect_data_only()
        
        generated_arch_str = generate_archs(
            self.config, 
            self.sampling_fn,
            self.score_model, 
            self.score_ema, 
            classifier_model,
            num_samples=self.n_gen_samples, 
            patient_factor=self.args.patient_factor,
            batch_size=self.args.eval_batch_size,
            classifier_scale=self.args.classifier_scale,
            task=x if self.args.fix_task else None)
        
        gc.collect()
        return generated_arch_str
