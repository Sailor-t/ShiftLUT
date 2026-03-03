import numpy as np
import sys

sys.path.insert(0, "../../")  # run under the project directory
from common.option import TestOptions
from common.utils import PSNR, cal_ssim, bPSNR
from PIL import Image
import os
import torch
from model_2 import TinyLUTRE
from data_ddp_deblocking import DBBenchmark

model = 'TinyLUTRE'

def valid_steps(model_G, valid, model_name):
    datasets = ['Classic5', 'LIVE1']
    _im_num = [5, 29]

    with torch.no_grad():
        model_G.eval()

        for i in range(len(datasets)):
            psnrs = []
            ssims = []
            bpsnrs = []
            files = valid.files[datasets[i]]

            # result_path = os.path.join(opt.resultRoot, exp_name, datasets[i], "qf{}".format(qf))
            # if not os.path.isdir(result_path):
            #     os.makedirs(result_path)

            for j in range(len(files)):
                key = datasets[i] + '_' + files[j][:-4]

                lb = valid.ims[key]
                input_im = valid.ims[key + 'x%d' % qf]

                # no need to divide 255.0
                # input_im = input_im.astype(np.float32) / 255.0
                im = torch.Tensor(np.expand_dims(
                    np.transpose(input_im, [2, 0, 1]), axis=0)).cuda()
                im = torch.round(im*255)
                pred = model_G(im-128.0, 'valid')+im
                pred = torch.clamp(pred, 0, 255).cpu().numpy()[0]
                pred = np.transpose(pred, [1, 2, 0]).astype(np.uint8)

                psnrs.append(
                    PSNR(pred[:, :, 0], lb[:, :, 0], 0))  # single channel, no scale change
                ssims.append(cal_ssim(pred[:, :, 0], lb[:, :, 0]))
                bpsnrs.append(bPSNR(pred, lb, 0))

                # input_img = np.round(np.clip(input_im * 255.0, 0, 255)).astype(np.uint8)
                # Image.fromarray(input_img[:, :, 0]).save(
                #     os.path.join(result_path, '{}_input.png'.format(key.split('_')[-1])))
                # Image.fromarray(lb[:, :, 0].astype(np.uint8)).save(
                #     os.path.join(result_path, '{}_gt.png'.format(key.split('_')[-1])))

                # Image.fromarray(pred[:, :, 0]).save(
                #     os.path.join(result_path, '{}_net.png'.format(key.split('_')[-1])))
            # print(psnrs)
            # print(bpsnrs)
            print('Dataset {} | AVG Val PSNR: {:02f} SSIM: {:02f} bPSNR: {:02f}'.format(datasets[i],
                                                                                        np.mean(np.asarray(psnrs)),
                                                                                        np.mean(np.asarray(ssims)),
                                                                                        np.mean(np.asarray(bpsnrs))))

if __name__ == '__main__':
    # model_name, step = 'ShiftLUT_qf=10_dncnn_real_offset2', 161000
    # model_name, step = 'ShiftLUT_qf=10_dncnn_real_offset3', 32000
    # model_name, step = 'ShiftLUT_qf=10_dncnn_real_offset3', 151000
    # model_name, step = 'ShiftLUT_qf=10_warmup', 61000
    # model_name, step = 'ShiftLUT_qf=10_warmup', 193000
    model_list = [
        ('ShiftLUT_F_qf=40_int', 7, 6, 16, 181000, 40),
        ('ShiftLUT_F_qf=30_int', 7, 6, 16, 120000, 30),
        ('ShiftLUT_F_qf=20_int', 7, 6, 16, 88000, 20),
    ]

    for model_name, stacks, msb_base, cnum, step, qf in model_list:
        if model in ['TinyLUTRE']:
            model_G = TinyLUTRE(scale=1, stacks=stacks, use_shift=True, msb_base=msb_base, cnum=cnum).cuda()
            lm = torch.load(f'../../models/{model_name}/Model_{step:06d}.pth', weights_only=True)
            # lm = torch.load(f'../models/debug/Model_{50:06d}.pth', weights_only=True)
            model_G.load_state_dict(lm, strict=True)

        # Valid dataset
        valid = DBBenchmark('/user/val', qf=qf, use_sets=2)

        valid_steps(model_G, valid, f'{model_name}_{step}')