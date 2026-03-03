import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import numpy as np
import sys
from tqdm import tqdm
from PIL import Image
from data_ddp_deblocking import DBBenchmark


import model_1 as Model

sys.path.insert(0, "../../")  # run under the project directory
from common.utils import PSNR, logger_info, _rgb2ycbcr, modcrop, cal_ssim, bPSNR

qf = 30
val = DBBenchmark('/user/val', qf, 2)
datasets = ['Classic5', 'LIVE1']

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def cal(rank, world_size, model_G, valid, model_name):
    setup(rank, world_size)
    model_G = model_G.to(rank)

    batch_size = 1

    # if rank == 0:
    #     psnr_datasets = {}

    with torch.no_grad():
        model_G.eval()

        for i in range(len(datasets)):
            files = valid.files[datasets[i]]
            psnrs = torch.zeros(len(files), device=rank)
            psnrbs = torch.zeros(len(files), device=rank)
            # ssims = torch.zeros(len(files), device=rank)
            for j in range(rank, len(files), world_size * batch_size):  # 按 batch_size 取数据
                # if j > 0:
                #     break
                # print(rank, j)
                batch_files = files[j:j + batch_size]  # 取 batch_size 张图片
                batch_in, batch_gt = [], []

                for file in batch_files:
                    # print(file)
                    # if file[:-4] not in ['monarch', 'peppers']:
                    #     continue

                    key = datasets[i] + '_' + file[:-4]

                    val_H = valid.ims[key]
                    val_L = valid.ims[key + 'x%d' % qf]
                    val_L = np.transpose(val_L, [2, 0, 1])[np.newaxis, ...]
                    val_L = torch.from_numpy(val_L).to(rank)
                    val_L = torch.round(val_L*255)

                    batch_S = model_G(val_L-128.0, 'valid')
                    batch_S = batch_S + val_L
                    batch_S_hat = torch.clamp(batch_S, 0, 255)

                    image_out = batch_S_hat.cpu().numpy()[0]
                    image_out = np.transpose(image_out, [1, 2, 0]).round().astype(np.uint8)  # HxWxC
                    # if datasets[i] == 'Set12' and file[:-4] == '01':
                    #     print(image_out[:8,:8,0])
                    #     print((val_L-128.0)[0,0,:8,:8])
                    #     Image.fromarray(image_out[:,:,0]).save('net_result.png')

                    img_gt = val_H.astype(np.uint8)
                    
                    p = PSNR(image_out[:, :, 0], img_gt[:, :, 0], 0)
                    psnrs[j] = torch.tensor(p, device=rank)
                    pb = bPSNR(image_out[:, :], img_gt[:, :], 0);
                    psnrbs[j] = torch.tensor(pb, device=rank)
                    # print(f'{file}: {pb:.2f}')
                    # Image.fromarray(image_out[:,:,0]).save(f'plot_results/{file[:-4]}_Ours.png')
            # dist.barrier()  # 同步所有进程
            # print(rank, psnrs)
            dist.all_reduce(psnrs)
            dist.all_reduce(psnrbs)
            dist.barrier()
            psnrs = psnrs.cpu()
            prnrbs = psnrbs.cpu()
            # ssims = ssims.cpu()
            # Only rank 0 prints the results
            if rank == 0:
                # print(psnrs)
                # print(psnrs.mean().item())
                # print(f'In {datasets[i]}:')
                # if i == 0:
                #     print(model_G.module.module.lsb_alpha.flatten())
                #     print(model_G.module.module.msb_alpha.flatten())
                #     print(f'{model_G.module.module.lsb_alpha.mean().item():.5f}')
                #     print(f'{model_G.module.module.msb_alpha.mean().item():.5f}')
                print(f'{psnrs.mean().item():.2f}/{psnrbs.mean().item():.2f}', end=' ')
                # psnr_datasets[datasets[i]] = psnsrs.numpy()
                # torch.save(psnrs, f'PSNRs/model_o{orientation}_{datasets[i]}.pt')
                # torch.save(psnrs, f'results/{model_name}_{datasets[i]}.pt')
        if rank == 0:
            os.makedirs(f"../../offset_stat/{model_name}", exist_ok=True)
            for cnt in range(model_G.module.stacks+1):
                np.save(f"../../offset_stat/{model_name}/msb_{cnt}.npy", model_G.module.msb[f"scs{cnt}"].accum) 

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
        # ('DSSLUT_S_qf=10', 0, 6, 16, 139000),
        # ('ShiftLUT_F_deblocking_20', 7, 6, 16, 55000),
        # ('ShiftLUT_F_deblocking_30', 7, 6, 16, 39000),
        ('ShiftLUT_F_deblocking_40', 7, 6, 16, 119000),
        ]

    for model_name, stacks, msb_base, cnum, step in models_list:
        print(f'\n Model is {model_name} now...')
        mp.spawn(main,
                    args=(world_size, model_name, stacks, msb_base, cnum, step),
                    nprocs=world_size,
                    join=True)
            #     break」