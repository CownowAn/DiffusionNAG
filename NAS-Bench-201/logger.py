import os
import wandb
import torch
import numpy as np


class Logger:
    def __init__(
        self,
        log_dir=None,
        write_textfile=True
        ):

        self.log_dir = log_dir
        self.write_textfile = write_textfile

        self.logs_for_save = {}
        self.logs = {}

        if self.write_textfile:
            self.f = open(os.path.join(log_dir, 'logs.txt'), 'w')


    def write_str(self, log_str):
        self.f.write(log_str+'\n')
        self.f.flush()


    def update_config(self, v, is_args=False):
        if is_args:
            self.logs_for_save.update({'args': v})
        else:
            self.logs_for_save.update(v)


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