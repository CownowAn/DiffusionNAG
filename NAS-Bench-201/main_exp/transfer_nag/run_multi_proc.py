from torch.multiprocessing import Process
import os
from absl import app, flags
import sys
import torch

sys.path.append(os.path.join(os.getcwd(), 'main_exp'))
from nas_bench_201 import train_single_model
from all_path import NASBENCH201

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_split", 15, "The number of splits")
flags.DEFINE_list("arch_idx_lst", None, "arch index list")
flags.DEFINE_list("arch_str_lst", None, "arch str list")
flags.DEFINE_string("meta_test_path", None, "meta test path")
flags.DEFINE_string("data_name", None, "data_name")
flags.DEFINE_string("raw_data_path", None, "raw_data_path")


def run_single_process(rank, seed, arch_idx, meta_test_path, data_name, 
                       raw_data_path, num_split=15, backend="nccl"):
    # 8 GPUs 
    device = ['0', '1', '2', '3', '4', '5', '6', '7', '0', '1', '2', '3', '4', '5', '6', '7',
              '0', '1', '2', '3', '4', '5', '6', '7', '0', '1', '2', '3', '4', '5', '6', '7'][rank]
    os.environ["CUDA_VISIBLE_DEVICES"] = device

    save_path = os.path.join(meta_test_path, str(arch_idx))
    if type(seed) == int:
        seeds = [seed]
    elif type(seed) in [list, tuple]:
        seeds = seed
    
    nasbench201 = torch.load(NASBENCH201)
    arch_str = nasbench201['arch']['str'][arch_idx]
    os.makedirs(save_path, exist_ok=True)
    train_single_model(save_dir=save_path,
                        workers=24,
                        datasets=[data_name],
                        xpaths=[f'{raw_data_path}/{data_name}'],
                        splits=[0],
                        use_less=False,
                        seeds=seeds,
                        model_str=arch_str,
                        arch_config={'channel': 16, 'num_cells': 5})


def run_multi_process(argv):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "1234"
    os.environ["WANDB_SILENT"] = "true"
    processes = []

    arch_idx_lst = [int(i) for i in FLAGS.arch_idx_lst]
    seeds = [777, 888, 999] * len(arch_idx_lst)
    arch_idx_lst_ = []
    for i in arch_idx_lst:
        arch_idx_lst_ += [i] * 3
        
    for arch_idx in arch_idx_lst:
        os.makedirs(os.path.join(FLAGS.meta_test_path, str(arch_idx)), exist_ok=True)
    
    for rank in range(FLAGS.num_split):
        arch_idx = arch_idx_lst_[rank]
        seed = seeds[rank]
        p = Process(target=run_single_process, args=(rank,
                                                    seed, 
                                                    arch_idx,
                                                    FLAGS.meta_test_path, 
                                                    FLAGS.data_name, 
                                                    FLAGS.raw_data_path))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    while any(p.is_alive() for p in processes):
        continue
    print("All processes have completed.")


if __name__ == "__main__":
    app.run(run_multi_process)