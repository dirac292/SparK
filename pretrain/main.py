# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import math
import sys
import time
from functools import partial
from typing import List

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

import dist
import encoder
from decoder import LightDecoder
from models import build_sparse_encoder
from sampler import DistInfiniteBatchSampler, worker_init_fn
from spark import SparK
from utils import arg_util, misc, lamb
from utils.imagenet import build_dataset_to_pretrain
from utils.lr_control import lr_wd_annealing, get_param_groups

import wandb
import os


class LocalDDP(torch.nn.Module):
    def __init__(self, module):
        super(LocalDDP, self).__init__()
        self.module = module
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def main_pt():
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    print(f'initial args:\n{str(args)}')
    args.log_epoch()
    
    # build data
    print(f'[build data for pre-training] ...\n')
    dataset_train = build_dataset_to_pretrain(args.data_path, args.input_size)
    data_loader_train = DataLoader(
        dataset=dataset_train, num_workers=args.dataloader_workers, pin_memory=True,
        batch_sampler=DistInfiniteBatchSampler(
            dataset_len=len(dataset_train), glb_batch_size=args.glb_batch_size,
            shuffle=True, filling=True, rank=dist.get_rank(), world_size=dist.get_world_size(),
        ), worker_init_fn=worker_init_fn
    )
    itrt_train, iters_train = iter(data_loader_train), len(data_loader_train)
    print(f'[dataloader] gbs={args.glb_batch_size}, lbs={args.batch_size_per_gpu}, iters_train={iters_train}')
    
    # build encoder and decoder
    enc: encoder.SparseEncoder = build_sparse_encoder(args.model, input_size=args.input_size, sbn=args.sbn, drop_path_rate=args.dp, verbose=False)
    dec = LightDecoder(enc.downsample_raito, sbn=args.sbn)
    model_without_ddp = SparK(
        sparse_encoder=enc, dense_decoder=dec, mask_ratio=args.mask,
        densify_norm=args.densify_norm, sbn=args.sbn,
    ).to(args.device)
    print(f'[PT model] model = {model_without_ddp}\n')


    if not args.resume_from: # Only load if not resuming from a checkpoint
        print(f'=> HACK: Loading ImageNet-1K Supervised Weights into ResNetEncoder (sp_cnn)...')
        import torchvision.models as models
        
        # 1. Load standard ResNet50 weights from torchvision (Dense weights)
        # Note: 'weights="IMAGENET1K_V1"' is the modern equivalent of 'pretrained=True'
        resnet_pretrained = models.resnet50(pretrained=True) 
        pretrained_dict = resnet_pretrained.state_dict()
        
        # 2. Get the SparK backbone's state dict (Sparse weights container)
        # The backbone is stored in 'sparse_encoder.sp_cnn'
        target_model = model_without_ddp.sparse_encoder.sp_cnn
        model_dict = target_model.state_dict()
        
        # 3. Filter and Remap Keys
        # We exclude 'fc' (classification head) as SparK is an encoder-decoder
        # We also ensure keys match the target model's expectations
        new_state_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict:
                # Standard case: key names match exactly (e.g., conv1.weight)
                if v.shape == model_dict[k].shape:
                    new_state_dict[k] = v
                else:
                    print(f"   [Skipping] Shape mismatch for {k}: src {v.shape} vs dst {model_dict[k].shape}")
            elif k.replace('module.', '') in model_dict:
                 # Handle potential DDP prefix differences if source was saved differently
                 clean_k = k.replace('module.', '')
                 new_state_dict[clean_k] = v
            else:
                 # Typically 'fc.weight' and 'fc.bias' end up here
                 pass

        # 4. Load the filtered weights
        # strict=False allows loading even if the head is missing (which is expected)
        msg = target_model.load_state_dict(new_state_dict, strict=False)
        print(f'=> HACK: ImageNet Weights Loaded. Missing keys (expected for head): {msg.missing_keys}')
    # ==============================================================================

    # the model has been randomly initialized in their construction time
    # now try to load some checkpoint as model weight initialization; this ONLY loads the model weights
    misc.initialize_weight(args.init_weight, model_without_ddp)
    
    if dist.initialized():
        model: DistributedDataParallel = DistributedDataParallel(model_without_ddp, device_ids=[dist.get_local_rank()], find_unused_parameters=False, broadcast_buffers=False)
    else:
        model = LocalDDP(model_without_ddp)
    
    # build optimizer and lr_scheduler
    param_groups: List[dict] = get_param_groups(model_without_ddp, nowd_keys={'cls_token', 'pos_embed', 'mask_token', 'gamma'})
    opt_clz = {
        'sgd': partial(torch.optim.SGD, momentum=0.9, nesterov=True),
        'adamw': partial(torch.optim.AdamW, betas=(0.9, args.ada)),
        'lamb': partial(lamb.TheSameAsTimmLAMB, betas=(0.9, args.ada), max_grad_norm=5.0),
    }[args.opt]
    optimizer = opt_clz(params=param_groups, lr=args.lr, weight_decay=0.0)
    print(f'[optimizer] optimizer({opt_clz}) ={optimizer}\n')
    
    # try to resume the experiment from some checkpoint.pth; this will load model weights, optimizer states, and last epoch (ep_start)
    # if loaded, ep_start will be greater than 0
    ep_start, performance_desc = misc.load_checkpoint(args.resume_from, model_without_ddp, optimizer)
    if ep_start >= args.ep: # load from a complete checkpoint file
        print(f'  [*] [PT already done]    Min/Last Recon Loss: {performance_desc}')
    else:   # perform pre-training
        tb_lg = misc.TensorboardLogger(args.tb_lg_dir, is_master=dist.is_master(), prefix='pt')
        if dist.is_master():

            if not hasattr(sys.stderr, 'isatty'):
                sys.stderr.isatty = lambda: False
            if not hasattr(sys.stdout, 'isatty'):
                sys.stdout.isatty = lambda: False
            wandb.init(
                project="ssl-pretraining",  # Change this to your project name
                name=f"spark_{args.model}_10ep",
                entity="critical-ml-dg",
                config=vars(args),
                mode = "online"
            )
        min_loss = 1e9
        print(f'[PT start] from ep{ep_start}')
        
        pt_start_time = time.time()
        for ep in range(ep_start, args.ep):
            ep_start_time = time.time()
            tb_lg.set_step(ep * iters_train)
            if hasattr(itrt_train, 'set_epoch'):
                itrt_train.set_epoch(ep)
            
            stats = pre_train_one_ep(ep, args, tb_lg, itrt_train, iters_train, model, optimizer)
            last_loss = stats['last_loss']
            min_loss = min(min_loss, last_loss)
            performance_desc = f'{min_loss:.4f} {last_loss:.4f}'
            misc.save_checkpoint_with_meta_info_and_opt_state(f'{args.model}_withdecoder_1kpretrained_spark_style.pth', args, ep, performance_desc, model_without_ddp.state_dict(with_config=True), optimizer.state_dict())
            misc.save_checkpoint_model_weights_only(f'{args.model}_1kpretrained_timm_style.pth', args, model_without_ddp.sparse_encoder.sp_cnn.state_dict())
            
            ep_cost = round(time.time() - ep_start_time, 2) + 1    # +1s: approximate the following logging cost
            remain_secs = (args.ep-1 - ep) * ep_cost
            remain_time = datetime.timedelta(seconds=round(remain_secs))
            finish_time = time.strftime("%m-%d %H:%M", time.localtime(time.time() + remain_secs))
            print(f'  [*] [ep{ep}/{args.ep}]    Min/Last Recon Loss: {performance_desc},    Cost: {ep_cost}s,    Remain: {remain_time},    Finish @ {finish_time}')
            
            args.cur_ep = f'{ep + 1}/{args.ep}'
            args.remain_time, args.finish_time = str(remain_time), str(finish_time)
            args.last_loss = last_loss
            args.log_epoch()
            
            tb_lg.update(min_loss=min_loss, head='train', step=ep)
            tb_lg.update(rest_hours=round(remain_secs/60/60, 2), head='z_burnout', step=ep)
            tb_lg.flush()

            # [INSERT THIS BLOCK] WandB Epoch Logging
            if dist.is_master():
                wandb.log({
                    'epoch_min_loss': min_loss,
                    'epoch_last_loss': last_loss,
                    'hours_remaining': round(remain_secs/60/60, 2),
                })
        
        # finish pre-training
        tb_lg.update(min_loss=min_loss, head='result', step=ep_start)
        tb_lg.update(min_loss=min_loss, head='result', step=args.ep)
        tb_lg.flush()
        print(f'final args:\n{str(args)}')
        print('\n\n')
        print(f'  [*] [PT finished]    Min/Last Recon Loss: {performance_desc},    Total Cost: {(time.time() - pt_start_time) / 60 / 60:.1f}h\n')
        print('\n\n')
        if dist.is_master():
            wandb.finish()
        tb_lg.close()
        time.sleep(10)
    
    args.remain_time, args.finish_time = '-', time.strftime("%m-%d %H:%M", time.localtime(time.time()))
    args.log_epoch()


def pre_train_one_ep(ep, args: arg_util.Args, tb_lg: misc.TensorboardLogger, itrt_train, iters_train, model: DistributedDataParallel, optimizer):
    model.train()
    me = misc.MetricLogger(delimiter='  ')
    me.add_meter('max_lr', misc.SmoothedValue(window_size=1, fmt='{value:.5f}'))
    header = f'[PT] Epoch {ep}:'
    
    optimizer.zero_grad()
    early_clipping = args.clip > 0 and not hasattr(optimizer, 'global_grad_norm')
    late_clipping = hasattr(optimizer, 'global_grad_norm')
    if early_clipping:
        params_req_grad = [p for p in model.parameters() if p.requires_grad]
    
    for it, inp in enumerate(me.log_every(iters_train, itrt_train, 3, header)):
        # adjust lr and wd
        min_lr, max_lr, min_wd, max_wd = lr_wd_annealing(optimizer, args.lr, args.wd, args.wde, it + ep * iters_train, args.wp_ep * iters_train, args.ep * iters_train)
        
        # forward and backward
        inp = inp.to(args.device, non_blocking=True)
        SparK.forward
        loss = model(inp, active_b1ff=None, vis=False)
        optimizer.zero_grad()
        loss.backward()
        loss = loss.item()
        if not math.isfinite(loss):
            print(f'[rk{dist.get_rank():02d}] Loss is {loss}, stopping training!', force=True, flush=True)
            sys.exit(-1)
        
        # optimize
        grad_norm = None
        if early_clipping: grad_norm = torch.nn.utils.clip_grad_norm_(params_req_grad, args.clip).item()
        optimizer.step()
        if late_clipping: grad_norm = optimizer.global_grad_norm
        torch.cuda.synchronize()
        
        # log
        me.update(last_loss=loss)
        me.update(max_lr=max_lr)
        tb_lg.update(loss=me.meters['last_loss'].global_avg, head='train_loss')
        tb_lg.update(sche_lr=max_lr, head='train_hp/lr_max')
        tb_lg.update(sche_lr=min_lr, head='train_hp/lr_min')
        tb_lg.update(sche_wd=max_wd, head='train_hp/wd_max')
        tb_lg.update(sche_wd=min_wd, head='train_hp/wd_min')
        if dist.is_master():
            wandb.log({
                'train_loss': me.meters['last_loss'].global_avg,
                'lr_max': max_lr,
                'lr_min': min_lr,
                'wd_max': max_wd,
                'grad_norm': grad_norm if grad_norm is not None else 0.0,
                'epoch': ep,
                'iter': it + ep * iters_train
            })
        
        if grad_norm is not None:
            me.update(orig_norm=grad_norm)
            tb_lg.update(orig_norm=grad_norm, head='train_hp')
        tb_lg.set_step()
    
    me.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in me.meters.items()}


if __name__ == '__main__':
    main_pt()
