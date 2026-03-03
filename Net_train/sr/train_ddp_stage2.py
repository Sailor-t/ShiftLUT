# from model import TinyLUT_S, TinyLUT
import logging
import math
import sys
from PIL import Image

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import os
import time
import torch.nn.functional as F
import numpy as np
from data_ddp import Provider, SRBenchmark
sys.path.insert(0, "../../")  # run under the project directory
from common.option import TrainOptions
from common.utils import PSNR, logger_info, _rgb2ycbcr, modcrop
from common.Writer import Logger
import model_2 as Model

torch.backends.cudnn.benchmark = True

def SaveCheckpoint(model_G, opt, i, best=False):
    if best:
        torch.save(model_G.state_dict(), os.path.join(
            opt.expDir, 'Model_best.pth'))
        logger.info("Checkpoint saved {}".format('best'))
    else:
        torch.save(model_G.state_dict(), os.path.join(
            opt.expDir, 'Model_{:06d}.pth'.format(i)))
        logger.info("Checkpoint saved {}".format(str(i)))

def valid_steps_ddp(model_G, valid, opt, iter, rank, world_size, logger, writer, full=False):
    datasets = ['Set5', 'Set14', 'BSD100', 'Manga109', 'Urban100']
    if not full:
        datasets = datasets[:2]
    model_G.eval()

    with torch.no_grad():
        for dataset_name in datasets:
            files = valid.files[dataset_name]
            total_files = len(files)

            # 每个 GPU 处理一部分数据
            files_per_rank = files[rank::world_size]

            local_psnrs = []

            for file_name in files_per_rank:
                key = f"{dataset_name}_{file_name[:-4]}"
                val_H = np.asarray(valid.ims[key]).astype(np.float32)
                val_L = np.asarray(valid.ims[key + f'x{opt.scale}']).astype(np.float32) - 128.0
                val_L = np.transpose(val_L, [2, 0, 1])[np.newaxis, ...]
                val_L = torch.from_numpy(val_L).to(rank)

                output = model_G(val_L, 'valid') + 128.0
                output = torch.clamp(output, 0, 255).cpu().numpy()[0]
                output = np.transpose(output, [1, 2, 0]).round().astype(np.uint8)

                img_gt = val_H.astype(np.uint8)
                psnr_val = PSNR(_rgb2ycbcr(output)[:, :, 0], _rgb2ycbcr(img_gt)[:, :, 0], shave_border=4)
                local_psnrs.append(psnr_val)

            # 将本地 psnr 转为 tensor
            local_tensor = torch.tensor(local_psnrs, dtype=torch.float32, device=rank)
            length = local_tensor.shape[0]

            # 收集所有进程的长度
            sizes = [torch.zeros(1, dtype=torch.int32, device=rank) for _ in range(world_size)]
            dist.all_gather(sizes, torch.tensor([length], dtype=torch.int32, device=rank))

            # 填充为相同长度
            max_len = max([s.item() for s in sizes])
            if len(local_psnrs) < max_len:
                pad = torch.zeros(max_len - len(local_psnrs), device=rank)
                local_tensor = torch.cat([local_tensor, pad], dim=0)

            # 收集所有进程的 PSNR
            all_psnrs = [torch.zeros(max_len, dtype=torch.float32, device=rank) for _ in range(world_size)]
            dist.all_gather(all_psnrs, local_tensor)
            # print(all_psnrs)
            
            # 只在 rank 0 汇总和记录
            if rank == 0:
                valid_psnrs = [all_psnrs[i][:sizes[i]] for i in range(world_size)]
                all_psnrs_flat = torch.cat(valid_psnrs)
                avg_psnr = all_psnrs_flat.mean().item()

                logger.info(f'Iter {iter} | Dataset {dataset_name} | AVG Val PSNR: {avg_psnr:.2f}')
                writer.scalar_summary(f'PSNR_valid/{dataset_name}', avg_psnr, iter)
                global best_psnr
                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    SaveCheckpoint(model_G.module, opt, 0, True)

if __name__ == "__main__":
    # 初始化分布式环境
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 获取训练配置
    opt_inst = TrainOptions()
    if rank == 0:
        opt = opt_inst.parse()
        best_psnr = 0
    dist.barrier()
    if rank != 0:
        opt = opt_inst.parse()

    # 设置设备
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    model = getattr(Model, opt.model)

    model_G = model(scale=opt.scale, stacks=opt.stack, use_shift=opt.use_shift, msb_base=opt.msb_base, cnum=opt.cnum).cuda()

    # Optimizers
    params_G = list(filter(lambda p: p.requires_grad, model_G.parameters()))
    opt_G = optim.Adam(params_G, lr=opt.lr0, betas=(0.9, 0.999), eps=1e-8, weight_decay=opt.weightDecay, amsgrad=False)

    # LR
    warmup_iters = 0
    base_lr = 1e-3

    if opt.lr1 < 0:
        def lf(x):
            if x < warmup_iters:
                return base_lr / opt.lr0  # 因为LambdaLR是乘到初始lr0上的比例
            else:
                progress = (x - warmup_iters) / (opt.totalIter - warmup_iters)
                return (((1 + math.cos(progress * math.pi)) / 2) * 0.8 + 0.2)
    else:
        lr_b = opt.lr1 / opt.lr0
        lr_a = 1 - lr_b
        def lf(x):
            if x < warmup_iters:
                temp_lr = (opt.lr0 - base_lr) / warmup_iters * x + base_lr
                return temp_lr / opt.lr0
            else:
                progress = (x - warmup_iters) / (opt.totalIter - warmup_iters)
                return (((1 + math.cos(progress * math.pi)) / 2) * lr_a + lr_b)

    scheduler = optim.lr_scheduler.LambdaLR(opt_G, lr_lambda=lf)

    # Load saved params
    if opt.startIter > 0:
        lm = torch.load(
            os.path.join(opt.expDir[:-2], 'Model_{:06d}.pth'.format(opt.startIter)), weights_only=True)
        model_G.load_state_dict(lm, strict=True)
        scheduler._step_count = opt.startIter
    else:
        load_dict = {
            'ShiftLUT_F': 200000,
        }
        
        load_model = os.path.basename(opt.expDir)[:-4]
        load_iter = load_dict[load_model]

        lm = torch.load(
            os.path.join(f'../models/{load_model}', 'Model_{:06d}.pth'.format(load_iter)), weights_only=True)
        model_G.load_state_dict(lm, strict=False)
        load_offset_model = load_model
        model_G.load_constant(os.path.join('../offset_stat', f'{load_offset_model}'))
        model_G = model_G.to(rank)

    # 包装模型为DDP
    model_G = DDP(model_G, device_ids=[rank], find_unused_parameters=False)

    # Training dataset
    train_iter = Provider(opt.batchSize, opt.workerNum, opt.scale, opt.trainDir, opt.cropSize)
    train_iter.build(rank, world_size)


    # Valid dataset
    valid = SRBenchmark(opt.valDir, scale=opt.scale, use_sets=2)

    if rank == 0:
        l_accum = [0., 0., 0.]
        dT = 0.
        rT = 0.
        accum_samples = 0
        
        # Tensorboard for monitoring
        writer = Logger(log_dir=opt.logDir, scrip=os.path.basename(opt.expDir), log_hist=False)

        logger_name = 'train'
        logger_info(logger_name, os.path.join(opt.expDir, logger_name + '.log'))
        logger = logging.getLogger(logger_name)
        logger.info(opt_inst.print_options(opt))
    else:
        logger = None
        writer = None
        
    valid_steps_ddp(model_G.module, valid, opt, opt.startIter, rank, world_size, logger, writer)
    # TRAINING

    for i in range(opt.startIter + 1, opt.totalIter + 1):
        model_G.train()

        # Data preparing
        st = time.time()
        im, lb = train_iter.next()
        im = im.to(device)
        lb = lb.to(device)
        
        if rank == 0:
            dT += time.time() - st
            st = time.time()

        # TRAIN G
        opt_G.zero_grad()
        pred = model_G(im,'train')+128.0
        loss_G = F.mse_loss(pred/255.0, lb)
        l2_loss = torch.tensor([0], device=loss_G.device)
        loss_total = loss_G + l2_loss * (1e-4)
        loss_total.backward()
        opt_G.step()
        scheduler.step()


        # Validation
        if i % opt.valStep == 0:
            # validation during multi GPU training
            # valid_steps(model_G.module, valid, opt, i)
            valid_steps_ddp(model_G.module, valid, opt, i, rank, world_size, logger, writer)

        if rank == 0:
            rT += time.time() - st
            # For monitoring
            accum_samples += opt.batchSize
            l_accum[0] += loss_G.item()
            l_accum[1] += l2_loss.item()
            # Show informationa
            if i % opt.displayStep == 0:
                writer.scalar_summary('loss_Pixel', l_accum[0] / opt.displayStep, i)

                logger.info("{} | Iter:{:6d}, Sample:{:6d}, Loss:{:.2e}, Loss_alpha:{:.2e}, dT:{:.4f}, rT:{:.4f}".format(
                    opt.expDir, i, accum_samples, l_accum[0] / opt.displayStep, l_accum[1] / opt.displayStep, dT / opt.displayStep,
                                                rT / opt.displayStep))
                l_accum = [0., 0., 0.]
                dT = 0.
                rT = 0.
            # Save models
            if i % opt.saveStep == 0:
                SaveCheckpoint(model_G.module, opt, i)


    # valid_steps_ddp(model_G.module, valid, opt, opt.totalIter, rank, world_size, logger, writer, True)

    if rank == 0:
        logger.info("Complete")

    dist.destroy_process_group()