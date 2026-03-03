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
import model_1 as Model
from common.utils import PSNR, logger_info, _rgb2ycbcr, modcrop, cal_ssim

sigma = 50
val = DNBenchmark('/user/val', sigma, 2)
datasets = ['Set12', 'BSD68']
# datasets = ['Set12']


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def cal(rank, world_size, model_G, valid, model_name):
    setup(rank, world_size)
    model_G = model_G.to(rank)
    # model_G = DDP(model_G, device_ids=[rank])

    batch_size = 1

    with torch.no_grad():
        model_G.eval()

        for i in range(len(datasets)):
            files = valid.files[datasets[i]]
            psnrs = torch.zeros(len(files), device=rank)
            ssims = torch.zeros(len(files), device=rank)
            for j in range(rank, len(files), world_size * batch_size):  # 按 batch_size 取数据
                # if j > 0:
                #     break
                # print(rank, j)
                batch_files = files[j:j + batch_size]  # 取 batch_size 张图片
                batch_in, batch_gt = [], []

                for file in batch_files:
                    key = datasets[i] + '_' + file[:-4]

                    val_H = valid.ims[key]
                    val_L = valid.ims[key + 'x%d' % sigma]
                    val_L = np.transpose(val_L, [2, 0, 1])[np.newaxis, ...]
                    val_L = torch.from_numpy(val_L).to(rank)
                    val_L = torch.round(val_L*255)

                    batch_S = model_G(val_L-128.0, 'valid')
                    batch_S = batch_S + val_L
                    batch_S_hat = torch.clamp(batch_S, 0, 255)
                    image_out = batch_S_hat.cpu().numpy()[0]
                    image_out = np.transpose(image_out, [1, 2, 0])  # HxWxC
            
                    img_gt = val_H.astype(np.uint8)
                    
                    p = PSNR(img_gt[:, :, 0], image_out[:, :, 0], 0)
                    psnrs[j] = torch.tensor(p, device=rank)
                    s = cal_ssim(img_gt[:,:,0], image_out[:,:,0])
                    ssims[j] = torch.tensor(s, device=rank)

            # dist.barrier()  # 同步所有进程
            # print(rank, psnrs)
            dist.all_reduce(psnrs)
            # dist.all_reduce(ssims)
            dist.barrier()
            psnrs = psnrs.cpu()
            # ssims = ssims.cpu()
            # Only rank 0 prints the results
            if rank == 0:
                # print(psnrs, end=' ')
                # print(psnrs.mean().item())
                # print(f'In {datasets[i]}:')
                print(f'{psnrs.mean().item():.2f}/{ssims.mean().item():.4f}', end='\n')
                # psnr_datasets[datasets[i]] = psnsrs.numpy()
                # torch.save(psnrs, f'PSNRs/model_o{orientation}_{datasets[i]}.pt')
        if rank == 0:
            os.makedirs(f"../offset_stat/{model_name}", exist_ok=True)
            for cnt in range(model_G.module.stacks+1):
                np.save(f"../offset_stat/{model_name}/msb_{cnt}.npy", model_G.module.msb[f"scs{cnt}"].accum) 
            # for cnt in range(Model.lsb_clip):
            #     np.save(f"../offset_stat/{model_name}/lsb_{cnt}.npy", model_G.module.lsb[f"scs{cnt}"].accum) 

    cleanup()

import signal
import sys

def signal_handler(sig, frame):
    print(f"[Rank {os.getpid()}] Caught Ctrl+C, cleaning up...")
    cleanup()  # 销毁进程组
    sys.exit(0)

def main(rank, world_size, model_name, stacks, msb_base, cnum, step):
    model = getattr(Model, 'TinyLUTRE')
    model_G = model(scale=1, stacks=stacks, use_shift=True, msb_base=msb_base, cnum=cnum).to(rank)
    lm = torch.load(os.path.join('../../models', model_name, 'Model_{:06d}.pth'.format(step)), weights_only=True)
    model_G.load_state_dict(lm, strict=True)
    for cnt in range(stacks+1):
        model_G.module.msb[f"scs{cnt}"].pre_accum()
    cal(rank, world_size, model_G, val, model_name)

if __name__ == "__main__":
    world_size = 1
    models_list = [
        # ('DSSLUT_S', 0, 6, 16, 187000),
        (f'ShiftLUT_F_denoising_{sigma}', 7, 6, 16, 164000)
        ]

    for model_name, stacks, msb_base, cnum, step in models_list:
        print(f'\n Model is {model_name} now...')
        mp.spawn(main,
                    args=(world_size, model_name, stacks, msb_base, cnum, step),
                    nprocs=world_size,
                    join=True)
            #     break」