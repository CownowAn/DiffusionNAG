import os
import wandb
import torch
import numpy as np


class Logger:
    def __init__(
        self,
        exp_name,
        log_dir=None,
        exp_suffix="",
        write_textfile=True,
        use_wandb=False,
        wandb_project_name=None,
        entity='hysh',
        config=None
    ):

        self.log_dir = log_dir
        self.write_textfile = write_textfile
        self.use_wandb = use_wandb

        self.logs_for_save = {}
        self.logs = {}

        if self.write_textfile:
            self.f = open(os.path.join(log_dir, 'logs.txt'), 'w')

        if self.use_wandb:
            exp_suffix = "_".join(exp_suffix.split("/")[:-1])
            wandb.init(
                config=config if config is not None else wandb.config,
                entity=entity,
                project=wandb_project_name, 
                name=exp_name + "_" + exp_suffix, 
                group=exp_name,
                reinit=True)

    def write_str(self, log_str):
        self.f.write(log_str+'\n')
        self.f.flush()

    def update_config(self, v, is_args=False):
        if is_args:
            self.logs_for_save.update({'args': v})
        else:
            self.logs_for_save.update(v)
        if self.use_wandb:
            wandb.config.update(v, allow_val_change=True)

    def write_log_nohead(self, element, step):
        log_str = f"{step} | "
        log_dict = {}
        for key, val in element.items():
            if not key in self.logs_for_save:
                self.logs_for_save[key] =  []
            self.logs_for_save[key].append(val)
            log_str += f'{key} {val} | '
            log_dict[f'{key}'] = val
        
        if self.write_textfile:
            self.f.write(log_str+'\n')
            self.f.flush()

        if self.use_wandb:
            wandb.log(log_dict, step=step)

    def write_log(self, element, step, return_log_dict=False):
        log_str = f"{step} | "
        log_dict = {}
        for head, keys  in element.items():
            for k in keys:
                if k in self.logs:
                    v = self.logs[k].avg
                if not k in self.logs_for_save:
                    self.logs_for_save[k] = []
                self.logs_for_save[k].append(v)
                log_str += f'{k} {v}| '
                log_dict[f'{head}/{k}'] = v

        if self.write_textfile:
            self.f.write(log_str+'\n')
            self.f.flush()

        if return_log_dict:
            return log_dict
        
        if self.use_wandb:
            wandb.log(log_dict, step=step)

    def log_sample(self, sample_x):
        wandb.log({"sampled_x": [wandb.Image(x.unsqueeze(-1).cpu().numpy()) for x in sample_x]})
    
    def log_valid_sample_prop(self, arch_metric, x_axis, y_axis):
        assert x_axis in ['test_acc', 'flops', 'params', 'latency']
        assert y_axis in ['test_acc', 'flops', 'params', 'latency']
        
        data = [[x, y] for (x, y) in zip(arch_metric[2][f'{x_axis}_list'], arch_metric[2][f'{y_axis}_list'])]
        table = wandb.Table(data=data, columns = [x_axis, y_axis])
        wandb.log({f"valid_sample ({x_axis}-{y_axis})" : wandb.plot.scatter(table, x_axis, y_axis)})
    
    def save_log(self, name=None):
        name = 'logs.pt' if name is None else name
        torch.save(self.logs_for_save, os.path.join(self.log_dir, name))

    def update(self, key, v, n=1):
        if not key in self.logs:
            self.logs[key] = AverageMeter()
        self.logs[key].update(v, n)

    def reset(self, keys=None, except_keys=[]):
        if keys is not None:
            if isinstance(keys, list):
                for key in keys:
                    self.logs[key] =  AverageMeter()
            else:
                self.logs[keys] = AverageMeter()
        else:
            for key in self.logs.keys():
                if not key in except_keys:
                    self.logs[key] = AverageMeter()

    def avg(self, keys=None, except_keys=[]):
        if keys is not None:
            if isinstance(keys, list):
                return {key: self.logs[key].avg for key in keys if key in self.logs.keys()}
            else:
                return self.logs[keys].avg
        else:
            avg_dict = {}
            for key in self.logs.keys():
                if not key in except_keys:
                    avg_dict[key] =  self.logs[key].avg
            return avg_dict 


class AverageMeter(object):
	"""
	Computes and stores the average and current value
	Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
	"""

	def __init__(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def get_metrics(g_embeds, x_embeds, logit_scale, prefix='train'):
    metrics = {}
    logits_per_g = (logit_scale * g_embeds @ x_embeds.t()).detach().cpu()
    logits_per_x = logits_per_g.t().detach().cpu()

    logits = {"g_to_x": logits_per_g, "x_to_g": logits_per_x}
    ground_truth = torch.arange(len(x_embeds)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{prefix}_{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{prefix}_{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{prefix}_{name}_R@{k}"] = np.mean(preds < k)

    return metrics