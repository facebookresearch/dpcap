# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch
import torch.nn.functional as F
import util.misc as misc
import util.lr_sched as lr_sched
from transformers import BertTokenizer
import torch.nn as nn

import functorch

from util.misc import set_grad_to_vec
import pdb

import torch.distributed as dist
from util.opacus_batchmemory import BatchMemoryManager
# from util.EMA import create_ema, update

def generate_square_subsequent_mask(sz: int):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def train_one_epoch(model: torch.nn.Module,
                    model_without_ddp: torch.nn.Module,
                    dp_optimizer_dummy: torch.optim.Optimizer,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, compute_grad_and_loss,
                    log_writer=None,
                    args=None):
    model.train(True)

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if args.DP=="ghost":
        metric_logger.add_meter('grad_norm', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        if args.amp:
            #metric_logger.add_meter('nb_nans_amp_mean', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            metric_logger.add_meter('amp_scale', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            metric_logger.add_meter('prop_nans', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = args.print_freq

    accum_iter = args.accum_iter
    print('accum_iter: ', accum_iter)

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # Load the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    norm_ghost = 0#trick for the metric logger
    lr_sched.adjust_learning_rate_iter(optimizer, args.privacy_step, args)
    with BatchMemoryManager(
        data_loader=data_loader, 
        num_splits=accum_iter, 
        optimizer=dp_optimizer_dummy
    ) as memory_safe_data_loader:
        nb_batches = len(memory_safe_data_loader)
        for data_iter_step, samples in enumerate(metric_logger.log_every(memory_safe_data_loader, print_freq,  args,header)):

            # get image and text inputs
            imgs_ = samples[0]
            texts_ = samples[1]

            dp_step_skip = dp_optimizer_dummy._step_skip_queue.pop(0)
            # dp_step_skip = (data_iter_step+1)%accum_iter==0

            if data_iter_step==0:
                print(f"starting a new epoch. physical batch size per GPU is:{len(imgs_)}")
                print(f"the first sentence in the batch is: {texts_[0]}")

            # tokenize the text inputs (with padding). carefull we are also masking the padding tokens
            texts_tokenized_input = tokenizer.batch_encode_plus(
                texts_,
                add_special_tokens=True,
                max_length=args.target_txt_len, #We can do shorter. see https://pypi.org/project/open-clip-torch/#files 
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids_ = texts_tokenized_input['input_ids']
                
                
            
            # samples = samples.to(device, non_blocking=True)
            imgs_ = imgs_.to(device, non_blocking=True)
            input_ids_ = input_ids_.to(device, non_blocking=True)
            if args.DP=="ghost":
                assert "autoregressive" in args.model, "Training pipeline for autoregressive decoder only"
                from contextlib import nullcontext
                if args.debug_DP:
                    assert (not(args.distributed) or args.world_size==1) and accum_iter==1
                    from tests.test_ghost import test_functorch
                    import copy
                    model_functorch = copy.deepcopy(model)
                    torch.backends.cuda.enable_mem_efficient_sdp(False)
                    grads, loss_functorch = test_functorch(model_functorch,  imgs_, input_ids_, args, device, "mean")
                    grad_tensor = torch.cat([grad.view(grad.size(0), -1) for grad in grads], dim=1)
                    grad_norm = grad_tensor.norm(2, 1)
                    multiplier = grad_norm.new(grad_norm.size()).fill_(1)
                    multiplier[grad_norm.gt(args.max_grad_norm)] = args.max_grad_norm / grad_norm[grad_norm.gt(args.max_grad_norm)]
                    grad_tensor *= multiplier.unsqueeze(1)
                    set_grad_to_vec(model_functorch, grad_tensor.mean(0))
                    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
                    ok2 = torch.cat([param.grad.flatten() for param in model_functorch.parameters() if param.requires_grad])
                    del model_functorch, grads, loss_functorch, grad_tensor, grad_norm
                with torch.cuda.amp.autocast() if args.amp else nullcontext():
                    #loss = optimizer.privacy_engine.forward(imgs_, input_ids_, mask_ratio=args.mask_ratio, prob_parallel_decoding=0, target_text_length=args.target_txt_len, device=device)
                    loss = model.forward(imgs_, input_ids_, mask_ratio=args.mask_ratio, prob_parallel_decoding=0, target_text_length=args.target_txt_len, device=device)
                    loss_value = loss.mean().item()
                    # loss /= accum_iter
                if not dp_step_skip and args.distributed:
                    # we use a per iteration (instead of per epoch) lr scheduler
                    dist.barrier()
                loss_scaler(loss, optimizer, parameters=model.parameters(),update_grad=not(dp_step_skip), DP=args.DP)
                if not(dp_step_skip):
                    lr_sched.adjust_learning_rate_iter(optimizer, args.privacy_step, args)
                    #We clook at the persample gradients before noise addition.
                    ok = torch.cat([param.grad.flatten() for param in model.parameters() if param.requires_grad])
                    norm_ghost = ok.norm(2)
                    if args.debug_DP:
                        assert cos(ok,ok2).item() ==1, f"cosine sim {cos(ok,ok2).item():.5}"
                    # TODO check this zero grad
                    optimizer.zero_grad()
                    args.privacy_step+=1
                    # update(model, ema, args.privacy_step, decay=args.ema)
                metric_logger.update(grad_norm=norm_ghost)
                if args.amp:
                    #metric_logger.update(nb_nans_amp_mean= (optimizer.privacy_engine.nb_nans / (data_iter_step*len(imgs_)*221)) if data_iter_step!=0 else 0)
                    #metric_logger.update(prop_nans= optimizer.privacy_engine.prop_nans)
                    metric_logger.update(prop_nans= optimizer.privacy_engine.nb_nans / (data_iter_step*len(imgs_)) if data_iter_step!=0 else 0)
                    metric_logger.update(amp_scale=optimizer.privacy_engine.scaler.get_scale())
            else:
                with torch.cuda.amp.autocast():
                    assert "autoregressive" in args.model, "Training pipeline for autoregressive decoder only"
                    loss = model.forward(imgs_, input_ids_, mask_ratio=args.mask_ratio, prob_parallel_decoding=0, target_text_length=args.target_txt_len, device=device, reduction="mean")
                    #loss, _, _ = model.forward_loss(imgs_, input_ids_, mask_ratio=args.mask_ratio)
                    loss_value = loss.item()
                    # loss /= accum_iter
                    loss_scaler(loss, optimizer, parameters=model.parameters(),
                                update_grad=(data_iter_step + 1) % accum_iter == 0)
                    if (data_iter_step + 1) % accum_iter == 0:
                        lr_sched.adjust_learning_rate_iter(optimizer, args.privacy_step, args)
                        optimizer.zero_grad()
                        args.privacy_step+=1

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            torch.cuda.synchronize()

            metric_logger.update(loss=loss_value)

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

            # save model
            if (data_iter_step % args.save_freq == 0) and not(epoch==0 and data_iter_step==0) :
                print('save model ckpt at iteration ', data_iter_step)
                misc.save_model(
                    # args=args, model=model, model_without_ddp=model_without_ddp,ema= ema,optimizer=optimizer,
                    args=args, model=model, model_without_ddp=model_without_ddp,optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, end_epoch=data_iter_step==0, scaler=optimizer.privacy_engine.scaler)
            if args.privacy_step == args.target_step and args.rank ==0:
                print('save model ckpt at privacy step ', args.privacy_step)
                misc.save_model(
                    # args=args, model=model, model_without_ddp=model_without_ddp, ema= ema,optimizer=optimizer,
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, end_epoch=data_iter_step==0, scaler=optimizer.privacy_engine.scaler, target_step=True)
            loss_value_reduce = misc.all_reduce_mean(loss_value)
            if log_writer is not None and not(dp_step_skip):
                """ We use epoch_1000x as the x-axis in tensorboard.
                This calibrates different curves when batch size changes.
                """
                # epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                nb_batches = len(memory_safe_data_loader)
                epoch_1000x = int((data_iter_step / nb_batches + epoch) * 1000)
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_nopoisson(model: torch.nn.Module,
                    model_without_ddp: torch.nn.Module,
                    dp_optimizer_dummy: torch.optim.Optimizer,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, compute_grad_and_loss,
                    log_writer=None,
                    args=None):
    model.train(True)

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if args.DP=="ghost":
        metric_logger.add_meter('grad_norm', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        if args.amp:
            #metric_logger.add_meter('nb_nans_amp_mean', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            metric_logger.add_meter('amp_scale', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            metric_logger.add_meter('prop_nans', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = args.print_freq

    accum_iter = args.accum_iter
    print('accum_iter: ', accum_iter)

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # Load the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    norm_ghost = 0#trick for the metric logger
    nb_batches = data_loader.num_batches
    lr_sched.adjust_learning_rate_iter(optimizer, args.privacy_step, args)
    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq,  args,header)):

        # get image and text inputs
        imgs_ = samples[0]
        texts_ = samples[1][0]

        dp_step_skip = (data_iter_step + 1) % accum_iter != 0
        # dp_step_skip = (data_iter_step+1)%accum_iter==0

        if data_iter_step==0:
            print(f"starting a new epoch. physical batch size per GPU is:{len(imgs_)}")
            print(f"the first sentence in the batch is: {texts_[0]}")

        # tokenize the text inputs (with padding). carefull we are also masking the padding tokens
        texts_tokenized_input = tokenizer.batch_encode_plus(
            texts_,
            add_special_tokens=True,
            max_length=args.target_txt_len, #We can do shorter. see https://pypi.org/project/open-clip-torch/#files 
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids_ = texts_tokenized_input['input_ids']
            
        
        # samples = samples.to(device, non_blocking=True)
        imgs_ = imgs_.to(device, non_blocking=True)
        input_ids_ = input_ids_.to(device, non_blocking=True)
        if args.DP=="ghost":
            assert "autoregressive" in args.model, "Training pipeline for autoregressive decoder only"
            from contextlib import nullcontext
            if args.debug_DP:
                assert (not(args.distributed) or args.world_size==1) and accum_iter==1
                from tests.test_ghost import test_functorch
                import copy
                model_functorch = copy.deepcopy(model)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                grads, loss_functorch = test_functorch(model_functorch,  imgs_, input_ids_, args, device, "mean")
                grad_tensor = torch.cat([grad.view(grad.size(0), -1) for grad in grads], dim=1)
                grad_norm = grad_tensor.norm(2, 1)
                multiplier = grad_norm.new(grad_norm.size()).fill_(1)
                multiplier[grad_norm.gt(args.max_grad_norm)] = args.max_grad_norm / grad_norm[grad_norm.gt(args.max_grad_norm)]
                grad_tensor *= multiplier.unsqueeze(1)
                set_grad_to_vec(model_functorch, grad_tensor.mean(0))
                cos = nn.CosineSimilarity(dim=0, eps=1e-6)
                ok2 = torch.cat([param.grad.flatten() for param in model_functorch.parameters() if param.requires_grad])
                del model_functorch, grads, loss_functorch, grad_tensor, grad_norm
            with torch.cuda.amp.autocast() if args.amp else nullcontext():
                #loss = optimizer.privacy_engine.forward(imgs_, input_ids_, mask_ratio=args.mask_ratio, prob_parallel_decoding=0, target_text_length=args.target_txt_len, device=device)
                loss = model.forward(imgs_, input_ids_, mask_ratio=args.mask_ratio, prob_parallel_decoding=0, target_text_length=args.target_txt_len, device=device)
                loss_value = loss.mean().item()
            loss_scaler(loss, optimizer, parameters=model.parameters(),update_grad=not(dp_step_skip), DP=args.DP)
            if not(dp_step_skip):
                lr_sched.adjust_learning_rate_iter(optimizer, args.privacy_step, args)
                ok = torch.cat([param.grad.flatten() for param in model.parameters() if param.requires_grad])
                norm_ghost = ok.norm(2)
                if args.debug_DP:
                    assert cos(ok,ok2).item() ==1, f"cosine sim {cos(ok,ok2).item():.5}"
                # TODO check this zero grad
                optimizer.zero_grad()
                args.privacy_step+=1
                # update(model, ema, args.privacy_step, decay=args.ema)
            metric_logger.update(grad_norm=norm_ghost)
            if args.amp:
                #metric_logger.update(nb_nans_amp_mean= (optimizer.privacy_engine.nb_nans / (data_iter_step*len(imgs_)*221)) if data_iter_step!=0 else 0)
                #metric_logger.update(prop_nans= optimizer.privacy_engine.prop_nans)
                metric_logger.update(prop_nans= optimizer.privacy_engine.nb_nans / (data_iter_step*len(imgs_)) if data_iter_step!=0 else 0)
                metric_logger.update(amp_scale=optimizer.privacy_engine.scaler.get_scale())
        else:
            with torch.cuda.amp.autocast():
                assert "autoregressive" in args.model, "Training pipeline for autoregressive decoder only"
                loss = model.forward(imgs_, input_ids_, mask_ratio=args.mask_ratio, prob_parallel_decoding=0, target_text_length=args.target_txt_len, device=device, reduction="mean")
                #loss, _, _ = model.forward_loss(imgs_, input_ids_, mask_ratio=args.mask_ratio)
                loss_value = loss.item()
                # loss /= accum_iter
                loss_scaler(loss, optimizer, parameters=model.parameters(),
                            update_grad=(data_iter_step + 1) % accum_iter == 0)
                if (data_iter_step + 1) % accum_iter == 0:
                    optimizer.zero_grad()
                    args.privacy_step+=1

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        # save model
        if (data_iter_step % args.save_freq == 0) and not(epoch==0 and data_iter_step==0) :
            print('save model ckpt at iteration ', data_iter_step)
            misc.save_model(
                # args=args, model=model, model_without_ddp=model_without_ddp,ema= ema,optimizer=optimizer,
                args=args, model=model, model_without_ddp=model_without_ddp,optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, end_epoch=data_iter_step==0, scaler=optimizer.privacy_engine.scaler)
        if args.privacy_step == args.target_step and args.rank ==0:
            print('save model ckpt at privacy step ', args.privacy_step)
            misc.save_model(
                # args=args, model=model, model_without_ddp=model_without_ddp, ema= ema,optimizer=optimizer,
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, end_epoch=data_iter_step==0, scaler=optimizer.privacy_engine.scaler, target_step=True)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and not(dp_step_skip):
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            # epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            nb_batches = data_loader.num_batches
            epoch_1000x = int((data_iter_step / nb_batches + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}