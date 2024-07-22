# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

#import models_mae
import models_mae

from engine_pretrain import train_one_epoch_nopoisson

# from util.EMA import create_ema, update

from util.captioning_data import get_captioning400m_data
from common.utils import bool_flag
import functorch
import pdb
from private_transformers import PrivacyEngine
# from private_transformers import private_transformers

# captioning loader
from opacus.data_loader import DPDataLoader
from opacus.optimizers.optimizer import DPOptimizer
from opacus.optimizers.ddpoptimizer import DistributedDPOptimizer

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    #Batch size related
    parser.add_argument('--batch_size', default=64, type=int, help='Total Logical Batch size used for DP-SGD. If TAN simulation is performed, this corresponds to the effective batch size size outiside of simulation')
    parser.add_argument('--TAN_batch_size', default=None, type=int,
                        help='Logical batch size to use to simulate large B training. Sigma will be divided to have constant sigma/B. cf Sander et al., 2023')
    parser.add_argument('--target_txt_len', default=77, type=int,help='Maximum number of tokens for the image captions')    
    parser.add_argument('--accum_iter', default=-0, type=int,help='Accumulate gradient iterations. It will be set tautomatically')
    parser.add_argument('--max_physical_B', default=128, type=int, help='max nb of examples per GPU, then accum is performed')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16_autoregressive_nobias', type=str, metavar='MODEL',
                        help='Name of model to train.')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--mask_ratio', default=0, type=float,
                        help='Masking ratio (percentage of removed patches). We do not use this for our trainings.')
    parser.add_argument('--resume', default='',help='resume from checkpoint')
    parser.add_argument('--init', default=False,help='If set to True and args.resume is provided, then the model will be loaded but not optimizer etc. used for syntheticaly pretrained models')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--epochs', default=50, type=int, help="this variable is not really used, only to fill some DP libraries compulsory args. See the arguments related to the number of iterations.")
    parser.add_argument('--warmup_epochs', type=float, default=1, metavar='N',
                        help='epochs to warmup LR. this variable is not really used, only to fill some DP libraries compulsory args. See the arguments related to the number of iterations')
    parser.add_argument('--warmup_iterations', type=float, default=1000, metavar='N',
                        help='iterations to warmup LR')  
    parser.add_argument('--overall_iterations', type=float, default=20000, metavar='N',
                        help='iterations to warmup LR')  
    
    # Privacy parameters
    parser.add_argument('--DP', default="", type=str,help="use '' for non-DP training, and 'ghost' otherwise", choices = ["", "ghost"])
    parser.add_argument('--amp', default=False, type=bool_flag,help="using AMP with ghost norm. Careful use!")
    parser.add_argument('--amp_base', default=8191, type=float,help="AMP scaler initialization")
    parser.add_argument('--amp_growth_factor', default=1.1, type=float,help="AMP_growth_factor")
    parser.add_argument('--amp_backoff_factor', default=0.901, type=float,help="AMP backoff factor")
    parser.add_argument('--amp_set_nans_to_zero', default=False, type=bool_flag,help="After the first backward of ghost norm, wether or not to set the nan values to 0 and still perform the backprop on the second loss.")
    parser.add_argument('--load_if_possible', default=True, type=bool_flag, help="Set to False to start training from scratch even though there are available checkoints in the directory.")
    parser.add_argument('--sigma', default=0, type=float,help='noise std')
    parser.add_argument('--max_grad_norm', default=100000000, type=float,help='Clipping value')

    parser.add_argument('--data_ratio', default=1.0, type=float,help='data ratio (what proportion of the training data to use for training).')

    # Other parameters
    parser.add_argument('--output_dir', default='./output_dir', help='path where to save, empty for no saving')
    parser.add_argument('--dataset_path', default=None, help='Path for captioning dataset Dir')
    parser.add_argument('--nb_samples', default=None, type = int, help='Number of samples in the dataset')

    parser.add_argument('--log_dir', default='./output_dir',help='path where to log')
    parser.add_argument('--device', default='cuda',help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--save_freq',type=int, default=5000,
                        help='step frequence at which to save the model')
    parser.add_argument('--target_step',type=int, default=5708,
                        help='Force to save the model at this step')
    parser.add_argument('--print_freq',type=int, default=20,
                        help='step frequence at which to print')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help="start epoch. Not used. See 'privacy_step' arg")
    parser.add_argument('--privacy_step', default=0, type=int, metavar='S',
                        help='current gradient step')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    #debug
    parser.add_argument('--debug_DP', default=False, type=bool_flag,
                        help='check if functorch computes good grads')
    # distributed training parameters
    parser.add_argument('--distributed', default=True, type=bool_flag,
                        help='Wheter or not to use distributed process')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    print("Entering main")
    ## Initialize training
    args.privacy_step=0 #Actual gradient step. If it is restarting, it will reload the privacy step from previous training.
    if args.distributed:
        misc.init_distributed_mode(args)
    world_size = misc.get_world_size()
    num_tasks = world_size
    global_rank = misc.get_rank()
    print(f"args.accum_iter:{args.accum_iter}")
    print(f"world_size:{world_size}")
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    print('=========== Start Training ===========')
    device = torch.device(args.device)
    # fix the seed for reproducibility; not sure if this is necessary?
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # simple augmentation
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    

    #Set up learning rate. The base learning rate corresponds to B=256, and we follow a linear scale.
    #We set to learning rate as a function of the "reference batch size"  (cf Sander et al., 2023). If we simulate training using TAN with a smaller batch size, we keep the same learning rate.
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * args.batch_size / 256
    print("base lr: %.2e" % args.blr)
    print("actual lr: %.2e" % args.lr)

    #To save compute, we can use TAN simulation (Sander et al., 2023) to train with the same number of steps and same SNR but smaller batch size "TAN_batch_size"
    sigma_eff = args.sigma/args.batch_size #Effective noise in DP-SGD training
    if args.TAN_batch_size:
        print(f"We are simulating training of B={args.batch_size} and sigma={args.sigma} with B_TAN={args.TAN_batch_size}")
        args.sigma /=(args.batch_size/args.TAN_batch_size)
        args.batch_size = args.TAN_batch_size
    else:
        print(f"This is NOT simulating a TAN simulation.\n Training with logical batch size of {args.batch_size} and physical batch size of {args.max_physical_B}")
    args.accum_iter=(args.batch_size//world_size)//args.max_physical_B
    print(f"the new value for sigma is {args.sigma} and for B is {args.batch_size}")
    print(f"world size is {world_size}, so the number grad accumulation is {args.accum_iter}")

    # captioning dataset loading
    assert args.dataset_path is not None, "args.dataset_path is empty"
    print("Training WITHOUT poisson sampling. There cannot be formal DP guarantees in this setting")
    DATASET_PATH = args.dataset_path
    DataInfo = get_captioning400m_data(batch_size=(args.batch_size //args.accum_iter) // world_size,
                                    input_shards=DATASET_PATH,
                                    preprocess_img=transform_train,
                                    world_size=num_tasks,
                                    num_workers=args.num_workers,
                                    return_txt=True,
                                    nb_samples=args.nb_samples)
    data_loader_train = DataInfo.dataloader


    # Logging
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # Define the model
    model = models_mae.__dict__[args.model](text_max_length=args.target_txt_len)
    model.to(device)
    nb_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters with grad:", nb_param)
    if args.DP=="ghost":
        print(f"grad norm should be higher than {np.sqrt(nb_param)*sigma_eff}")
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    # Wrapper for distributed training
    if args.DP=="ghost" and args.distributed:
        from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
        model=DPDDP(model)
        model_without_ddp = model.module
    elif args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)#find_unused_parameters might not be necessary
        model_without_ddp = model.module
    
    # Define optimizer and loss scaler for AMP
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    print(optimizer)
    loss_scaler = NativeScaler()
    #scaler for AMP speed up
    if args.amp:
        print(f"Using AMP with base factor: {args.amp_base}, growth_factor={args.amp_growth_factor}, backoff_factor: {args.amp_backoff_factor}")
        scaler = torch.cuda.amp.GradScaler(args.amp_base, growth_factor=args.amp_growth_factor, backoff_factor=args.amp_backoff_factor)
    else:
        print("Not using AMP! training can be slow")
        scaler= None 
    misc.reload(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, scaler=scaler)
    if args.DP == "ghost":
        #Privacy Engine wrapper. Important to set "batch_size=args.batch_size // world_size"
        privacy_engine = PrivacyEngine(model,distributed=args.distributed, batch_size=args.batch_size // world_size,max_grad_norm=args.max_grad_norm,rank=global_rank,clipping_mode="ghost",noise_multiplier=args.sigma,epochs=args.epochs,scaler = scaler, amp_set_nans_to_zero=args.amp_set_nans_to_zero)
        privacy_engine.attach(optimizer)
    #Start training
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        DataInfo.set_epoch(epoch)
        train_stats = train_one_epoch_nopoisson(
            model, model_without_ddp, None, data_loader_train,
            optimizer, device, epoch, loss_scaler,None,
            log_writer=log_writer,args=args
        )
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},'epoch': epoch,}
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)