#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 14:02:53 2021

@author: hsjomaa
"""
import numpy as np
from scipy.stats import norm
import pandas as pd
from torch import autograd as ag
import torch
from sklearn.preprocessing import PowerTransformer


def regret(output,response):
    incumbent   = output[0]
    best_output = []
    for _ in output:
        incumbent = _ if _ > incumbent else incumbent
        best_output.append(incumbent)
    opt       = max(response)
    orde      = list(np.sort(np.unique(response))[::-1])
    tmp       = pd.DataFrame(best_output,columns=['regret_validation'])
    
    tmp['rank_valid']        = tmp['regret_validation'].map(lambda x : orde.index(x))
    tmp['regret_validation'] = opt - tmp['regret_validation']
    return tmp

def EI(incumbent, model_fn,support,queries,return_variance, return_score=False):
    mu, stddev     = model_fn(queries)
    mu             = mu.reshape(-1,)
    stddev         = stddev.reshape(-1,)
    if return_variance:
        stddev         = np.sqrt(stddev)
    with np.errstate(divide='warn'):
        imp = mu - incumbent
        Z = imp / stddev
        score = imp * norm.cdf(Z) + stddev * norm.pdf(Z)
    if not return_score:
        score[support] = 0
        return np.argmax(score)
    else:
        return score
    
    
class Metric(object):
    def __init__(self,prefix='train: '):
        self.reset()
        self.message=prefix + "loss: {loss:.2f} - noise: {log_var:.2f} - mse: {mse:.2f}"
        
    def update(self,loss,noise,mse):
        self.loss.append(np.asscalar(loss))
        self.noise.append(np.asscalar(noise))
        self.mse.append(np.asscalar(mse))
    
    def reset(self,):
        self.loss = []
        self.noise = []
        self.mse = []
    
    def report(self):
        return self.message.format(loss=np.mean(self.loss),
                            log_var=np.mean(self.noise),
                            mse=np.mean(self.mse))
    
    def get(self):
        return {"loss":np.mean(self.loss),
                "noise":np.mean(self.noise),
                "mse":np.mean(self.mse)}
    
def totorch(x,device):
    if type(x) is tuple:
        return tuple([ag.Variable(torch.Tensor(e)).to(device) for e in x])
    return torch.Tensor(x).to(device)


def prepare_data(indexes, support, Lambda, response, metafeatures=None, output_transform=False):
    # Generate indexes of the batch
    X,E,Z,y,r = [],[],[],[],[]
    #### get support data
    for dim in indexes:
        if metafeatures is not None:
            Z.append(metafeatures)
        E.append(Lambda[support])
        X.append(Lambda[dim])
        r_ = response[support,np.newaxis]
        y_ = response[dim]
        if output_transform:
            power = PowerTransformer(method="yeo-johnson")
            r_ = power.fit_transform(r_)
            y_ = power.transform(y_.reshape(-1,1)).reshape(-1,)
        r.append(r_)
        y.append(y_)
    X = np.array(X)
    E = np.array(E)
    Z = np.array(Z)
    y = np.array(y)
    r = np.array(r)
    return (np.expand_dims(E, axis=-1), r, np.expand_dims(X, axis=-1), Z), y
