import os
import sys
import json
import logging
import numpy as np
import copy
import torch
import torch.nn as nn
import random
import torch.optim as optim

from transfer_nag_lib.MetaD2A_mobilenetV3.evaluation.evaluator import OFAEvaluator
from torchprofile import profile_macs
from transfer_nag_lib.MetaD2A_mobilenetV3.evaluation.codebase.networks import NSGANetV2
from transfer_nag_lib.MetaD2A_mobilenetV3.evaluation.parser import get_parse
from transfer_nag_lib.MetaD2A_mobilenetV3.evaluation.eval_utils import get_dataset
from transfer_nag_lib.MetaD2A_nas_bench_201.metad2a_utils import reset_seed
from transfer_nag_lib.ofa_net import OFASubNet


# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
# device_list = [int(_) for _ in args.gpu.split(',')]
# args.n_gpus = len(device_list)
# args.device = torch.device("cuda:0")

# if args.seed is None or args.seed < 0: args.seed = random.randint(1, 100000)
# torch.cuda.manual_seed(args.seed)
# torch.manual_seed(args.seed)
# np.random.seed(args.seed)
# random.seed(args.seed)



# args.save_path = os.path.join(args.save_path, f'evaluation/{args.data_name}')
# if args.model_config.startswith('flops@'):
# 	args.save_path += f'-nsganetV2-{args.model_config}-{args.seed}'
# else:
# 	args.save_path += f'-metaD2A-{args.bound}-{args.seed}'
# if not os.path.exists(args.save_path):
# 	os.makedirs(args.save_path)

# args.data_path = os.path.join(args.data_path, args.data_name)

# log_format = '%(asctime)s %(message)s'
# logging.basicConfig(stream=sys.stdout, level=print,
#                     format=log_format, datefmt='%m/%d %I:%M:%S %p')
# fh = logging.FileHandler(os.path.join(args.save_path, 'log.txt'))
# fh.setFormatter(logging.Formatter(log_format))
# logging.getLogger().addHandler(fh)
# if not torch.cuda.is_available():
# 	print('no gpu self.args.device available')
# 	sys.exit(1)
# print("args = %s", args)



def set_architecture(n_cls, evaluator, drop_path, drop, img_size, n_gpus, device, save_path, model_str):
	# g, acc = evaluator.get_architecture(model_str)
	g = OFASubNet(model_str).get_op_dict()
	subnet, config = evaluator.sample(g)
	net = NSGANetV2.build_from_config(subnet.config, drop_connect_rate=drop_path)
	net.load_state_dict(subnet.state_dict())
	
	NSGANetV2.reset_classifier(
		net, last_channel=net.classifier.in_features,
		n_classes=n_cls, dropout_rate=drop)
	# calculate #Paramaters and #FLOPS
	inputs = torch.randn(1, 3, img_size, img_size)
	flops = profile_macs(copy.deepcopy(net), inputs) / 1e6
	params = sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6
	net_name = "net_flops@{:.0f}".format(flops)
	print('#params {:.2f}M, #flops {:.0f}M'.format(params, flops))
	# OFAEvaluator.save_net_config(save_path, net, net_name + '.config')
	if torch.cuda.device_count() > 1:
		print("Let's use", torch.cuda.device_count(), "GPUs!")
		net = nn.DataParallel(net)
	net = net.to(device)
	
	return net, net_name, params, flops


def train(train_queue, net, criterion, optimizer, grad_clip, device, report_freq):
	net.train()
	train_loss, correct, total = 0, 0, 0
	for step, (inputs, targets) in enumerate(train_queue):
		# upsample by bicubic to match imagenet training size
		inputs, targets = inputs.to(device), targets.to(device)
		optimizer.zero_grad()
		outputs = net(inputs)
		loss = criterion(outputs, targets)
		loss.backward()
		nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
		optimizer.step()
		train_loss += loss.item()
		_, predicted = outputs.max(1)
		total += targets.size(0)
		correct += predicted.eq(targets).sum().item()
		if step % report_freq == 0:
			print(f'train step {step:03d} loss {train_loss / total:.4f} train acc {100. * correct / total:.4f}')
	print(f'train acc {100. * correct / total:.4f}')
	return train_loss / total, 100. * correct / total


def infer(valid_queue, net, criterion, device, report_freq, early_stop=False):
	net.eval()
	test_loss, correct, total = 0, 0, 0
	with torch.no_grad():
		for step, (inputs, targets) in enumerate(valid_queue):
			inputs, targets = inputs.to(device), targets.to(device)
			outputs = net(inputs)
			loss = criterion(outputs, targets)
			test_loss += loss.item()
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()
			if step % report_freq == 0:
				print(f'valid {step:03d} {test_loss / total:.4f} {100. * correct / total:.4f}')
			if early_stop and step == 10:
				break
	acc = 100. * correct / total
	print('valid acc {:.4f}'.format(100. * correct / total))
	
	return test_loss / total, acc


def train_single_model(save_path, workers, datasets, xpaths, splits, use_less,
                       seed, model_str, device,
                    	lr=0.01,
                        momentum=0.9,
                        weight_decay=4e-5,
                        report_freq=50,
                        epochs=150,
                        grad_clip=5,
                        cutout=True,
                        cutout_length=16,
                        autoaugment=True,
                        drop=0.2,
                        drop_path=0.2,
                        img_size=224,
                        batch_size=96,
                        ):
	assert torch.cuda.is_available(), 'CUDA is not available.'
	torch.backends.cudnn.enabled = True
	torch.backends.cudnn.deterministic = True
	reset_seed(seed)
	# save_dir = Path(save_dir)
	# logger = Logger(str(save_dir), 0, False)
	os.makedirs(save_path, exist_ok=True)
	to_save_name = save_path + '/seed-{:04d}.pth'.format(seed)
	print(to_save_name)
	# args = get_parse()
	num_gen_arch = None
	evaluator = OFAEvaluator(num_gen_arch, img_size, drop_path,
		model_path='/home/data/GTAD/checkpoints/ofa/ofa_net/ofa_mbv3_d234_e346_k357_w1.0')

	train_queue, valid_queue, n_cls = get_dataset(datasets, batch_size, 
        xpaths, workers, img_size, autoaugment, cutout, cutout_length)
	net, net_name, params, flops = set_architecture(n_cls, evaluator, 
        drop_path, drop, img_size, n_gpus=1, device=device, save_path=save_path, model_str=model_str)


	# net.to(device)

	parameters = filter(lambda p: p.requires_grad, net.parameters())
	optimizer = optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
	criterion = nn.CrossEntropyLoss().to(device)
	scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

	# assert epochs == 1
	max_valid_acc = 0
	max_epoch = 0
	for epoch in range(epochs):
		print('epoch {:d} lr {:.4f}'.format(epoch, scheduler.get_lr()[0]))
		
		train(train_queue, net, criterion, optimizer, grad_clip, device, report_freq)
		_, valid_acc = infer(valid_queue, net, criterion, device, report_freq)
		torch.save(valid_acc, to_save_name)
		print(f'seed {seed:04d} last acc {valid_acc:.4f} max acc {max_valid_acc:.4f}')
		if max_valid_acc < valid_acc:
			max_valid_acc = valid_acc
			max_epoch = epoch
		# parent_path = os.path.abspath(os.path.join(save_path, os.pardir))
		# with open(parent_path + '/accuracy.txt', 'a+') as f:
		# 	f.write(f'{model_str} seed {seed:04d} {valid_acc:.4f}\n')
	
	return valid_acc, max_valid_acc, params, flops
	
 
################ NAS BENCH 201 #####################
# def train_single_model(save_dir, workers, datasets, xpaths, splits, use_less,
#                        seeds, model_str, arch_config):
#     assert torch.cuda.is_available(), 'CUDA is not available.'
#     torch.backends.cudnn.enabled = True
#     torch.backends.cudnn.deterministic = True
#     torch.set_num_threads(workers)

#     save_dir = Path(save_dir)
#     logger = Logger(str(save_dir), 0, False)

#     if model_str in CellArchitectures:
#         arch = CellArchitectures[model_str]
#         logger.log(
#             'The model string is found in pre-defined architecture dict : {:}'.format(model_str))
#     else:
#         try:
#             arch = CellStructure.str2structure(model_str)
#         except:
#             raise ValueError(
#                 'Invalid model string : {:}. It can not be found or parsed.'.format(model_str))

#     assert arch.check_valid_op(get_search_spaces(
#         'cell', 'nas-bench-201')), '{:} has the invalid op.'.format(arch)
#     # assert arch.check_valid_op(get_search_spaces('cell', 'full')), '{:} has the invalid op.'.format(arch)
#     logger.log('Start train-evaluate {:}'.format(arch.tostr()))
#     logger.log('arch_config : {:}'.format(arch_config))

#     start_time, seed_time = time.time(), AverageMeter()
#     for _is, seed in enumerate(seeds):
#         logger.log(
#             '\nThe {:02d}/{:02d}-th seed is {:} ----------------------<.>----------------------'.format(_is, len(seeds),
#                                                                                                         seed))
#         to_save_name = save_dir / 'seed-{:04d}.pth'.format(seed)
#         if to_save_name.exists():
#             logger.log(
#                 'Find the existing file {:}, directly load!'.format(to_save_name))
#             checkpoint = torch.load(to_save_name)
#         else:
#             logger.log(
#                 'Does not find the existing file {:}, train and evaluate!'.format(to_save_name))
#             checkpoint = evaluate_all_datasets(arch, datasets, xpaths, splits, use_less,
#                                                seed, arch_config, workers, logger)
#             torch.save(checkpoint, to_save_name)
#         # log information
#         logger.log('{:}'.format(checkpoint['info']))
#         all_dataset_keys = checkpoint['all_dataset_keys']
#         for dataset_key in all_dataset_keys:
#             logger.log('\n{:} dataset : {:} {:}'.format(
#                 '-' * 15, dataset_key, '-' * 15))
#             dataset_info = checkpoint[dataset_key]
#             # logger.log('Network ==>\n{:}'.format( dataset_info['net_string'] ))
#             logger.log('Flops = {:} MB, Params = {:} MB'.format(
#                 dataset_info['flop'], dataset_info['param']))
#             logger.log('config : {:}'.format(dataset_info['config']))
#             logger.log('Training State (finish) = {:}'.format(
#                 dataset_info['finish-train']))
#             last_epoch = dataset_info['total_epoch'] - 1
#             train_acc1es, train_acc5es = dataset_info['train_acc1es'], dataset_info['train_acc5es']
#             valid_acc1es, valid_acc5es = dataset_info['valid_acc1es'], dataset_info['valid_acc5es']
#         # measure elapsed time
#         seed_time.update(time.time() - start_time)
#         start_time = time.time()
#         need_time = 'Time Left: {:}'.format(convert_secs2time(
#             seed_time.avg * (len(seeds) - _is - 1), True))
#         logger.log(
#             '\n<<<***>>> The {:02d}/{:02d}-th seed is {:} <finish> other procedures need {:}'.format(_is, len(seeds), seed,
#                                                                                                      need_time))
#     logger.close()
# ###################

if __name__ == '__main__':
	train_single_model()
