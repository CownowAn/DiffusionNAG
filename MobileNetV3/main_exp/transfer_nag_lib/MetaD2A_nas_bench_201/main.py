###########################################################################################
# Copyright (c) Hayeon Lee, Eunyoung Hyung [GitHub MetaD2A], 2021
# Rapid Neural Architecture Search by Learning to Generate Graphs from Datasets, ICLR 2021
###########################################################################################
import os
import sys
import random
import numpy as np
import argparse
import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# from parser import get_parser
from .generator import Generator
from .predictor import Predictor
sys.path.append(os.getcwd())


def str2bool(v):
    return v.lower() in ['t', 'true', True]


def get_parser():
    parser = argparse.ArgumentParser()
    # general settings
    parser.add_argument('--seed', type=int, default=333)
    parser.add_argument('--gpu', type=str, default='0',
                        help='set visible gpus')
    parser.add_argument('--model_name', type=str, default='generator',
                        help='select model [generator|predictor]')
    parser.add_argument('--save-path', type=str,
                        default='C:\\Users\\gress\\OneDrive\\Documents\\Gresa\\DeepKernelGP\\MetaD2A_nas_bench_201\\results', help='the path of save directory')
    parser.add_argument('--data-path', type=str,
                        default='C:\\Users\\gress\\OneDrive\\Documents\\Gresa\\DeepKernelGP\\MetaD2A_nas_bench_201\\data', help='the path of save directory')
    parser.add_argument('--save-epoch', type=int, default=400,
                        help='how many epochs to wait each time to save model states')
    parser.add_argument('--max-epoch', type=int, default=400,
                        help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size for generator')
    parser.add_argument('--graph-data-name',
                        default='nasbench201', help='graph dataset name')
    parser.add_argument('--nvt', type=int, default=7,
                        help='number of different node types, 7: NAS-Bench-201 including in/out node')
    # set encoder
    parser.add_argument('--num-sample', type=int, default=20,
                        help='the number of images as input for set encoder')
    # graph encoder
    parser.add_argument('--hs', type=int, default=56,
                        help='hidden size of GRUs')
    parser.add_argument('--nz', type=int, default=56,
                        help='the number of dimensions of latent vectors z')
    # test
    parser.add_argument('--test', action='store_true',
                        default=True, help='turn on test mode')
    parser.add_argument('--load-epoch', type=int, default=400,
                        help='checkpoint epoch loaded for meta-test')
    parser.add_argument('--data-name', type=str,
                        default=None, help='meta-test dataset name')
    parser.add_argument('--num-class', type=int, default=None,
                        help='the number of class of dataset')
    parser.add_argument('--num-gen-arch', type=int, default=800,
                        help='the number of candidate architectures generated by the generator')
    parser.add_argument('--train-arch', type=str2bool, default=True,
                        help='whether to train the searched architecture')

    args = parser.parse_args()

    return args


def main():
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0")
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    args.model_path = os.path.join(args.save_path, args.model_name, 'model')
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    if args.model_name == 'generator':
        g = Generator(args)
        if args.test:
            g.meta_test()
        else:
            g.meta_train()
    elif args.model_name == 'predictor':
        p = Predictor(args)
        if args.test:
            p.meta_test()
        else:
            p.meta_train()
    else:
        raise ValueError('You should select generator|predictor|train_arch')


if __name__ == '__main__':
    main()