import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import sys
import numpy as np
from data_ddp_deblocking import DBBenchmark
from model_LUT import LUT, inference
sys.path.insert(0, "../../")  # run under the project directory
from common.utils import PSNR, logger_info, _rgb2ycbcr, modcrop, cal_ssim, bPSNR
from tqdm import tqdm

qf = 10
val = DBBenchmark('/user/val', qf, 2)
datasets = ['Classic5', 'LIVE1']

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def cal(rank, world_size, model_G, valid, model_name):
    setup(rank, world_size)
    model_G = model_G.to(rank)
    # model_G = DDP(model_G, device_ids=[rank])

    batch_size = 1

    # if rank == 0:
    #     psnr_datasets = {}

    with torch.no_grad():
        model_G.eval()

        for i in range(len(datasets)):
            files = valid.files[datasets[i]]
            # files = files[:1]
            # if rank == 0:
            #     print(f'Begin {datasets[i]}...')
            # psnrs = []
            psnrs = torch.zeros(len(files), device=rank)
            bpsnrs = torch.zeros(len(files), device=rank)
            # ssims = torch.zeros(len(files), device=rank)
            for j in range(rank, len(files), world_size * batch_size):  # 按 batch_size 取数据
                # print(rank, j)
                batch_files = files[j:j + batch_size]  # 取 batch_size 张图片
                batch_in, batch_gt = [], []
                for file in batch_files:
                    key = datasets[i] + '_' + file[:-4]

                    val_H = valid.ims[key]
                    val_L = valid.ims[key + 'x%d' % qf]
                    val_L = np.transpose(val_L, [2, 0, 1])[np.newaxis, ...]
                    val_L = torch.from_numpy(val_L).to(rank)
                    val_L = torch.round(val_L*255)

                    batch_S = inference(model_G, val_L-128.0) + val_L
                    batch_S_hat = torch.clamp(batch_S, 0, 255)
                    image_out = batch_S_hat.cpu().numpy()[0]
                    image_out = np.transpose(image_out, [1, 2, 0]).round().astype(np.uint8)  # HxWxC

                    img_gt = val_H.astype(np.uint8)
                    
                    p = PSNR(image_out[:, :, 0], img_gt[:, :, 0], 0)
                    psnrs[j] = torch.tensor(p, device=rank)
                    b = bPSNR(image_out, img_gt, 0)
                    bpsnrs[j] = torch.tensor(b, device=rank)
            # dist.barrier()  # 同步所有进程
            # print(rank, psnrs)
            dist.all_reduce(psnrs)
            dist.all_reduce(bpsnrs)
            dist.barrier()
            psnrs = psnrs.cpu()
            bpsnrs = bpsnrs.cpu()
            # ssims = ssims.cpu()
            # Only rank 0 prints the results
            if rank == 0:
                # print(psnrs)
                # print(psnrs.mean().item())
                # print(f'In {datasets[i]}:')
                print(f'{psnrs.mean().item():.2f}/{bpsnrs.mean().item():.2f}', end='\n')
                # psnr_datasets[datasets[i]] = psnsrs.numpy()
                # torch.save(psnrs, f'results/{model_name}_{datasets[i]}.pt')

    cleanup()

import signal
import sys

def signal_handler(sig, frame):
    print(f"[Rank {os.getpid()}] Caught Ctrl+C, cleaning up...")
    cleanup()  # 销毁进程组
    sys.exit(0)

def main(rank, world_size, model_name):
    signal.signal(signal.SIGINT, signal_handler)
    LUT_path, stacks, scale = '../LUTs/ShiftLUT_ultra_qf=10_dncnn_v3_int4', 7, 1
    decompose_factor = 1
    model_G = LUT(LUT_path, stacks, scale, decompose_factor).cuda()
    cal(rank, world_size, model_G, val, model_name)

if __name__ == "__main__":
    world_size = 4

    model_name = 'ShiftLUT'

    mp.spawn(main,
                args=(world_size, model_name),
                nprocs=world_size,
                join=True)