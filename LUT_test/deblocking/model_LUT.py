import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
from torch.autograd import Function
debug_oo = False
import time
import json

def LUTclip(x):
    return x.clamp(-128, 127)

def Query(w, x, sample_d):
    if sample_d == 1:
        return w[x]

    a1 = x//sample_d
    a2 = a1+1

    b = x%sample_d
    weight = b/sample_d

    # print(a1.shape, weight.shape, w[a1].shape)
    if len(w.shape) > 1:
        weight = weight[...,None]
        
    ans = w[a1]*(1-weight) + w[a2]*weight
    return ans
    # return w[a1*sample_d]

class dwLUT(nn.Module):
    def __init__(self, path):
        super().__init__()
        arr = np.load(path+'.npz')
        self.arr = nn.ParameterDict()
        with open(path+'.json', "r") as f:
            self.meta = json.load(f)
        for i in range(self.meta['IN']):
            for o in range(self.meta['OUT']):
                self.arr[f'i{i}o{o}'] = nn.Parameter(torch.from_numpy(arr[f'i{i}o{o}']).to(torch.int32), requires_grad=False)
        self.w = nn.Parameter(torch.from_numpy(np.load(path+'.npy')), requires_grad=False)
        self.steps = {}
        for info in self.meta["LUTs"]:
            self.steps[f'i{info["in"]}o{info["out"]}'] = info["step"]

    def forward(self, x, debug):
        N,C,H,W = x.shape
        # x = F.pad(x, (1, 1, 1, 1), mode='reflect')
        x = F.pad(x, (1, 1, 1, 1), mode='replicate')
        unfolded = F.unfold(x.to(torch.float32), kernel_size=(3, 3), padding=0).to(torch.uint8) # (N, C*K*K, L)
        unfolded = unfolded.view(N, C, 9, -1).int()  # (N, C, K*K, L)

        if C == 1:
            out = []
            for c in range(16):
                out_c = sum([Query(self.arr[f'i{i}o{c}'], unfolded[:,0,i,:], self.steps[f'i{i}o{c}']) for i in range(9)])
                out_c = torch.round(out_c/9.0)
                out.append(out_c)
            out = torch.stack(out, 1)
            out = out.reshape(N, out.shape[1], H, W)
            return out

        elif C == 16:
            out = []
            for c in range(16):
                out_c = sum([Query(self.arr[f'i{i}o{c}'], unfolded[:,c,i,:], self.steps[f'i{i}o{c}']) for i in range(9)])
                out_c = torch.round(out_c/9.0)
                out.append(out_c)
            out = torch.stack(out, 1)
            out = out.reshape(N, out.shape[1], H, W)
            return out
        else:
            raise AttributeError(f"GG with channel_num{C}")
        return x

class pwLUT(nn.Module):
    def __init__(self, path):
        super().__init__()
        arr = np.load(path+'.npz')
        self.arr = nn.ParameterDict()
        with open(path+'.json', "r") as f:
            self.meta = json.load(f)
        for i in range(self.meta['IN']):
            for o in range(self.meta['OUT']):
                self.arr[f'i{i}o{o}'] = nn.Parameter(torch.from_numpy(arr[f'i{i}o{o}']).to(torch.int32), requires_grad=False)
        self.w = nn.Parameter(torch.from_numpy(np.load(path+'.npy')), requires_grad=False).to(torch.int32)
        self.steps = {}
        for info in self.meta["LUTs"]:
            self.steps[f'i{info["in"]}o{info["out"]}'] = info["step"]


    def forward(self, x, debug):
        x = x.int()
        N, C, H, W = x.shape
        out = torch.zeros(N, self.meta['OUT'], H, W, device=x.device)
        for outc in range(self.meta['OUT']):
            out[:, outc, ...] = torch.round(sum([Query(self.arr[f'i{c}o{outc}'], x[:,c,:,:], self.steps[f'i{c}o{outc}']) for c in range(16)])/16.0)
        return out

class LUT(nn.Module):
    def __init__(self, lut_path, stack, scale, DW_split_level=1):
        super().__init__()

        self.msb = nn.ModuleList()
        self.lsb = nn.ModuleList()
        self.offset_constant = np.load(os.path.join(lut_path, 'offset.npy'))

        for i in range(stack+1):
            # if DW_split_level == 2:
            #     self.msb.append(dwLUTv2(np.load(os.path.join(lut_path, f'DW{i}_MSB_m.npy'))
            #         , np.load(os.path.join(lut_path, f'DW{i}_MSB_l.npy'))))
            # elif DW_split_level == 3:
            #     self.msb.append(dwLUTv3(np.load(os.path.join(lut_path, f'DW{i}_MSB_1.npy'))
            #         , np.load(os.path.join(lut_path, f'DW{i}_MSB_2'))
            #         , np.load(os.path.join(lut_path, f'DW{i}_MSB_3'))
            #         , np.load(os.path.join(lut_path, f'DW{i}_MSB'))
            #         ))
            # else:
            self.msb.append(dwLUT(os.path.join(lut_path, f'DW{i}_MSB')))

            self.msb.append(pwLUT(os.path.join(lut_path, f'PW{i}_MSB')))
        
        self.lsb.append(dwLUT(os.path.join(lut_path, f'DW{0}_LSB')))
        self.msb.append(pwLUT(os.path.join(lut_path, f'UP_MSB')))

        self.pixel_shuffle = nn.PixelShuffle(scale)
        if os.path.exists(os.path.join(lut_path, 'lsb_alpha.npy')):
            self.lsb_alpha = torch.from_numpy(np.load(os.path.join(lut_path, 'lsb_alpha.npy')))
        if os.path.exists(os.path.join(lut_path, 'msb_alpha.npy')):
            self.msb_alpha = torch.from_numpy(np.load(os.path.join(lut_path, 'msb_alpha.npy')))

    @staticmethod
    def low_high(image):
        xl = torch.remainder(image, 4)
        xh = torch.div(image, 4, rounding_mode='floor')
        return xl.to(torch.int16), xh.to(torch.int16)

    def shift(self, x, constant_offset):
        N, C, H, W = x.shape
        shifted = x.clone()

        for c in range(16):
            dy = constant_offset[0,c].item()
            dx = constant_offset[1,c].item()
            x_pad = F.pad(x[:,c], pad=(abs(dx), abs(dx), abs(dy), abs(dy)), mode='replicate')
            shifted[:,c] = x_pad[:, abs(dy)+dy:abs(dy)+dy+H, abs(dx)+dx:abs(dx)+dx+W]

        return shifted

    def forward(self, x, debug=False):
        x_l, x_h = self.low_high(x)
        cnt = 0
        for i in range(len(self.msb)-1):

            if isinstance(self.msb[i], pwLUT):
                if cnt < 8:
                    x_h = self.shift(x_h, self.offset_constant[cnt])
                cnt+=1

            x_h = self.msb[i](x_h+32, False)+x_h

            if i == 0:
                x_l = self.lsb[i](x_l, False)+x_l
                x_l.clamp_(0, 3)
                x_h += x_l

            x_h.clamp_(-32, 31)

        x = self.msb[-1](x_h+32, False)

        x = self.pixel_shuffle(x)
        x = x.to(torch.int16)
        return x

def inference(module, x):
    N,C,H,W = x.shape
    x = x.reshape(N*C,1,H,W)
    x = LUTclip(x)

    batch_S1 = module(x, True)
    batch_S2 = module(torch.rot90(x, 1, [2,3]), True)
    batch_S2 = torch.rot90(batch_S2, 3, [2,3])

    batch_S3 = module(torch.rot90(x, 2, [2,3]), True)
    batch_S3 = torch.rot90(batch_S3, 2, [2,3])

    batch_S4 = module(torch.rot90(x, 3, [2,3]), True)
    batch_S4 = torch.rot90(batch_S4, 1, [2,3])

    batch_S = sum(LUTclip(batch) for batch in [batch_S1, batch_S2, batch_S3, batch_S4])
    # print(batch_S.shape, N, C, H, W)
    batch_S = batch_S.reshape(N, C, batch_S.shape[-2], batch_S.shape[-1])
    return batch_S

if __name__ == '__main__':
    print('Hello World')