import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import numpy as np
from data_ddp import SRBenchmark
import model_1 as Model
sys.path.insert(0, "../../")  # run under the project directory
from common.utils import PSNR, logger_info, _rgb2ycbcr, modcrop, cal_ssim
from tqdm import tqdm

val = SRBenchmark('../../TinyLUT/val', 4, 5)
datasets = ['Set5', 'Set14', 'Urban100', 'BSD100', 'Manga109']
# datasets = ['Set5']
# datasets = ['Set5', 'Set14', 'BSD100', 'Manga109']
# datasets = ['Urban100']

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
            # if rank == 0:
            #     print(f'Begin {datasets[i]}...')
            # psnrs = []
            psnrs = torch.zeros(len(files), device=rank)
            ssims = torch.zeros(len(files), device=rank)
            for j in range(rank, len(files), world_size * batch_size):  # 按 batch_size 取数据
                # print(rank, j)
                batch_files = files[j:j + batch_size]  # 取 batch_size 张图片
                batch_in, batch_gt = [], []

                for file in batch_files:
                    key = datasets[i] + '_' + file[:-4]

                    tmp_L = valid.ims[key + 'x%d' % 4]
                    val_L = np.asarray(tmp_L).astype(np.float32) - 128.0  # HxWxC
                    val_L = np.transpose(val_L, [2, 0, 1])  # CxHxW
                    val_L = val_L[np.newaxis, ...]
                    val_L = torch.from_numpy(val_L).to(rank)
                    # f'{datasets[i]}_{j}'
                    batch_S = model_G(val_L, 'valid') + 128.0
                    batch_S_hat = torch.clamp(batch_S, 0, 255)
                    image_out = batch_S_hat.cpu().numpy()
                    image_out = np.transpose(image_out[0, :, :, :], [1, 2, 0])  # HxWxC

                    tmp_H = valid.ims[key]
                    img_gt = np.asarray(tmp_H).astype(np.uint8)  # HxWxC
                    img_gt = modcrop(img_gt[:, :], 4)

                    if img_gt.shape[0] < image_out.shape[0]:
                        image_out = image_out[:img_gt.shape[0]]
                    if img_gt.shape[1] < image_out.shape[1]:
                        image_out = image_out[:, :img_gt.shape[1]]

                    if img_gt.shape[0] > image_out.shape[0]:
                        image_out = np.pad(image_out, ((0, img_gt.shape[0] - image_out.shape[0]), (0, 0), (0, 0)))
                    if img_gt.shape[1] > image_out.shape[1]:
                        image_out = np.pad(image_out, ((0, 0), (0, img_gt.shape[-1] - image_out.shape[1]), (0, 0)))

                    CROP_S = 4
                    p = PSNR(_rgb2ycbcr(img_gt)[:, :, 0], _rgb2ycbcr(image_out)[:, :, 0], CROP_S)
                    psnrs[j] = torch.tensor(p, device=rank)
                    s = cal_ssim(_rgb2ycbcr(img_gt)[:,:,0], _rgb2ycbcr(image_out)[:,:,0])
                    ssims[j] = torch.tensor(s, device=rank)
            # dist.barrier()  # 同步所有进程
            # print(rank, psnrs)
            dist.all_reduce(psnrs)
            dist.all_reduce(ssims)
            dist.barrier()
            psnrs = psnrs.cpu()
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

    cleanup()


import signal
import sys

def signal_handler(sig, frame):
    print(f"[Rank {os.getpid()}] Caught Ctrl+C, cleaning up...")
    cleanup()  # 销毁进程组
    sys.exit(0)

def main(rank, world_size, model_name, stacks, msb_base, cnum, step):
    model = getattr(Model, 'TinyLUTRE')
    model_G = model(scale=4, stacks=stacks, use_shift=True, msb_base=msb_base, cnum=cnum).to(rank)
    lm = torch.load(os.path.join('../models', model_name, 'Model_{:06d}.pth'.format(step)), weights_only=True)
    model_G.load_state_dict(lm, strict=True)
    for cnt in range(stacks+1):
        model_G.module.msb[f"scs{cnt}"].pre_accum()
    cal(rank, world_size, model_G, val, model_name)

if __name__ == "__main__":
    world_size = 1
    models_list = [
        # ('ShiftLUT_ultra_s1', 199000),
        # ('ShiftLUT_ultra_s1_5', 196000),
        # ('ShiftLUT_ultra_s1_5', 196000),
        # ('ShiftLUT_ultra5_2', 194000),
        # ('ShiftLUT_ultra5_4', 200000),
        # ('ShiftLUT_ultra7_6', 193000)
        # ('ShiftLUT_ultra7_4', 184000)
        # ('ShiftLUT_ultra_msb4', 190000),
        # ('ShiftLUT_ultra_msb5', 183000),
        # ('ShiftLUT_ultra_msb7', 190000),
        # ('ShiftLUT_ultra_msb8', 188000)

        # ('ShiftLUT_ultra_msb6_s4', 4, 6, 16, 195000),
        # ('ShiftLUT_ultra_msb6_s5', 5, 6, 16, 194000),
        # ('ShiftLUT_ultra_msb6_s6', 6, 6, 16, 196000),
        # ('ShiftLUT_ultra_msb6_s0', 0, 6, 16, 166000),
        # ('ShiftLUT_ultra_msb6_s1', 1, 6, 16, 194000),
        # ('DSSLUT_L_stage1_fl0', 7, 6, 16, 196000),
        # ('DSSLUT_L_stage1_fl0_reset', 7, 6, 16, 193000),
        # ('DSSLUT_L_stage1_fl1', 7, 6, 16, 192000),
        # ('DSSLUT_L_stage1_fl1_reset', 7, 6, 16, 185000)
        # ('DSSLUT_L_stage1_fl0_reset', 7, 6, 16, 58000)
        # ('DSSLUT_S_mobilenet', 0, 6, 16, 200000),
        # ('DSSLUT_S_mobilenet_cp64', 0, 6, 16, 103000)

        # ('ShiftLUT_ultra_msb6_s7_c12_offset', 7, 6, 12, 196000),
        # ('ShiftLUT_ultra_msb6_s7_c8_offset', 7, 6, 8, 193000),
        # ('ShiftLUT_ultra_msb6_s7_c4_offset', 7, 6, 4, 190000),
        ]

    for model_name, stacks, msb_base, cnum, step in models_list:
        print(f'\n Model is {model_name} now...')
        mp.spawn(main,
                    args=(world_size, model_name, stacks, msb_base, cnum, step),
                    nprocs=world_size,
                    join=True)
            #     break」