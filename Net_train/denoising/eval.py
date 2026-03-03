import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import numpy as np
import sys
from tqdm import tqdm
from PIL import Image
from data_ddp_denoising import DNBenchmark

sys.path.insert(0, "../")  # run under the project directory
import model_2 as Model
from common.utils import PSNR, logger_info, _rgb2ycbcr, modcrop, cal_ssim

sigma = 25
val = DNBenchmark('/user/val', sigma, 2)
datasets = ['Set12', 'BSD68']

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def cal(rank, world_size, model_G, valid, model_name):
    setup(rank, world_size)
    model_G = model_G.to(rank)
    model_G = DDP(model_G, device_ids=[rank])

    batch_size = 1

    # if rank == 0:
    #     psnr_datasets = {}

    with torch.no_grad():
        model_G.eval()

        for i in range(len(datasets)):
            files = valid.files[datasets[i]]
            psnrs = torch.zeros(len(files), device=rank)
            # ssims = torch.zeros(len(files), device=rank)
            for j in range(rank, len(files), world_size * batch_size):  # 按 batch_size 取数据
                # print(rank, j)
                batch_files = files[j:j + batch_size]  # 取 batch_size 张图片
                batch_in, batch_gt = [], []

                for file in batch_files:
                    key = datasets[i] + '_' + file[:-4]

                    val_H = valid.ims[key]
                    val_L = valid.ims[key + 'x%d' % sigma]
                    # if j == 0:
                    #     print(val_L.shape)
                    #     Image.fromarray(np.round(val_L*255)[:,:,0].astype(np.uint8)).save(f'Set12_{j}_input.png')
                    val_L = np.transpose(val_L, [2, 0, 1])[np.newaxis, ...]
                    val_L = torch.from_numpy(val_L).to(rank)
                    val_L = torch.round(val_L*255)

                    batch_S = model_G(val_L-128.0, 'valid') + val_L
                    batch_S_hat = torch.clamp(batch_S, 0, 255)
                    image_out = batch_S_hat.cpu().numpy()[0]
                    image_out = np.transpose(image_out, [1, 2, 0]).astype(np.uint8)  # HxWxC

                    # print(torch.nn.functional.mse_loss(torch.from_numpy(val_H/255.0).to(rank).permute(2,0,1)[None], batch_S_hat/255.0))
                    img_gt = val_H.astype(np.uint8)
                    # if j == 0:
                    #     print(image_out.shape)
                    #     Image.fromarray(image_out[:,:,0]).save(f'Set12_{j}.png')
                    
                    p = PSNR(img_gt[:, :, 0], image_out[:, :, 0], 0)
                    psnrs[j] = torch.tensor(p, device=rank)
                    # print(file, p)
            # dist.barrier()
            # print(rank, psnrs)
            dist.all_reduce(psnrs)
            # dist.all_reduce(ssims)
            dist.barrier()
            psnrs = psnrs.cpu()
            # ssims = ssims.cpu()
            # Only rank 0 prints the results
            if rank == 0:
                print(f'{psnrs.mean().item():.2f}', end=' ')

    cleanup()

import signal
import sys

def signal_handler(sig, frame):
    print(f"[Rank {os.getpid()}] Caught Ctrl+C, cleaning up...")
    cleanup()  # 销毁进程组
    sys.exit(0)


def main(rank, world_size, model_name, stacks, msb_base, cnum, step):
    signal.signal(signal.SIGINT, signal_handler)
    model = getattr(Model, 'TinyLUTRE')
    model_G = model(scale=1, stacks=stacks, use_shift=True, msb_base=msb_base, cnum=cnum)
    model_G = model_G.to(rank)

    lm = torch.load(os.path.join('../../models', f'{model_name}', 'Model_{:06d}.pth'.format(step)), weights_only=True)
    model_G.load_state_dict(lm, strict=True)

    cal(rank, world_size, model_G, val, model_name)

if __name__ == "__main__":
    world_size = 1

    model_list = [
        # ('ShiftLUT_F_denoising_50_int', 7, 6, 16, 43000),
        ('ShiftLUT_F_denoising_25_int', 7, 6, 16, 50000),
    ]

    for model_name, stacks, msb_base, cnum, step in model_list:
        print(f'\n Model is {model_name} with step={step} now...')
        mp.spawn(main,
                    args=(world_size, model_name, stacks, msb_base, cnum, step),
                    nprocs=world_size,
                    join=True)
            #     break