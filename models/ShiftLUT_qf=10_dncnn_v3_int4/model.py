import torch
from torch.autograd import Function
import torch.nn.functional as F
import torch.nn as nn

default_cnum = 16 # follow the paper
nf = 28
debug = False
# nf = 64
# dense = False
class XQuantize(Function):
    @staticmethod
    def forward(ctx, x, quantize=True):
        if quantize:
            x = torch.round(x)
        return x
    
    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs, None

def LUTclip(x):
    return x.clamp(-128, 127)

############### Basic Convolutional Layers ###############
class Conv(nn.Module):
    """ 2D convolution w/ MSRA init. """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=bias)
        nn.init.kaiming_normal_(self.conv.weight)
        if bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)

class ActConv(nn.Module):
    """ Conv. with activation. """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(ActConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.act = nn.ReLU(inplace=True)
        nn.init.kaiming_normal_(self.conv.weight)
        if bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.act(self.conv(x))
        
class DenseConv(nn.Module):
    """ Dense connected Conv. with activation. """

    def __init__(self, in_nf, nf):
        super(DenseConv, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_nf, nf, 1, stride=1, padding=0, dilation=1)

    def forward(self, x):
        feat = self.act(self.conv1(x))
        out = torch.cat([x, feat], dim=1)
        return out
class PointOneChannel(torch.nn.Module):
    def __init__(self):
        super(PointOneChannel, self).__init__()
        self.conv1 = ActConv(1, nf, 1)
        self.conv2 = DenseConv(nf, nf)
        self.conv3 = Conv(nf+nf, default_cnum, 1)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return XQuantize.apply(x)

class PointConv(torch.nn.Module):
    def __init__(self):
        super(PointConv, self).__init__()
        self.conv = nn.ModuleList([PointOneChannel() for _ in range(default_cnum)])
        
    def forward(self,x):
        ori = [LUTclip(self.conv[i](x[:,i:i+1,:,:])) for i in range(default_cnum)]
        return XQuantize.apply(sum(ori)/default_cnum)

class UpOneChannel(torch.nn.Module):
    def __init__(self, scale=1):
        super(UpOneChannel, self).__init__()
        self.conv1 = ActConv(1, nf, 1)
        self.conv2 = DenseConv(nf, nf)
        self.conv3 = DenseConv(nf + nf * 1, nf)
        self.conv4 = DenseConv(nf + nf * 2, nf)
        self.conv5 = Conv(nf + nf * 3, scale**2, 1)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return XQuantize.apply(x)
        
class UpConv(torch.nn.Module):
    def __init__(self, scale=1):
        super(UpConv, self).__init__()
        self.Conv = nn.ModuleList([UpOneChannel(scale) for _ in range(default_cnum)])

    def forward(self,x):
        ori = [LUTclip(self.Conv[i](x[:,i:i+1,:,:])) for i in range(default_cnum)]
        return XQuantize.apply(sum(ori) / default_cnum)

class DepthWise(torch.nn.Module): # 实际上也可以当作1通道的StandardConv用
    def __init__(self, in_c=default_cnum, out_c=default_cnum, stack=1):
        super().__init__()
        assert in_c == 1 or in_c == out_c
        self.in_c = in_c
        self.out_c = out_c
        self.act = nn.ReLU(inplace=True)
        kernels = []
        biass = []
        self.stack = stack
        for i in range(self.stack):
            k = nn.Parameter(torch.zeros(1, self.out_c, 9, 1))
            nn.init.kaiming_normal_(k)
            kernels.append(k)
            b = nn.Parameter(torch.zeros(1, self.out_c, 9, 1))
            biass.append(b)

        self.kernels = nn.ParameterList(kernels)
        self.biass = nn.ParameterList(biass)

    def forward(self, x):
        # x = F.pad(x, (2, 0, 2, 0), mode='reflect')
        x = F.pad(x, (1, 1, 1, 1), mode='replicate')
        # x = F.pad(x, (1, 1, 1, 1), mode='reflect')
        N,C,H,W = x.shape
        assert C == self.in_c
        res = []
        
        # 展开输入图像的每个3x3区域
        unfolded = F.unfold(x, kernel_size=(3, 3), padding=0)  # (N, C*K*K, L)
        unfolded = unfolded.view(N, C, 9, -1)  # (N, C, K*K, L)
        
        output = unfolded
        # 应用卷积核
        for i in range(self.stack):
            output = output * self.kernels[i] + self.biass[i]  # (N, C, K*K, L)
            if i == self.stack-1:
                output = XQuantize.apply(LUTclip(output))
            else:
                output = self.act(output)

        # 重塑结果
        output = output.permute(0, 1, 3, 2)  # (N, C, L, K*K)
        output = output.view(N, self.out_c, H-2, W-2, 9)  # (N, C, H, W, 9)
        output = output.mean(dim=-1)
        output = XQuantize.apply(output)
        return output

class StConv(torch.nn.Module): # 为了支持多通道，实现了这个
    def __init__(self, in_c):
        super().__init__()
        self.kernel = nn.Parameter(torch.zeros(in_c, default_cnum, 9, 1))
        nn.init.kaiming_normal_(self.kernel)
        self.bias = nn.Parameter(torch.zeros(in_c, default_cnum, 9, 1))        
        self.in_c = in_c
        
    def forward(self, x):
        N,C,H,W = x.shape
        assert C == self.in_c
        res = []

        # 展开输入图像的每个3x3区域
        unfolded = F.unfold(x, kernel_size=(3, 3), padding=0)  # (N, C*K*K, L)
        unfolded = unfolded.view(N, C, 9, -1).unsqueeze(2)  # (N, C_in, 1, K*K, L)

        # 应用卷积核
        output = unfolded * self.kernel + self.bias  # (N, C_in, C_out, K*K, L)
        output = XQuantize.apply(LUTclip(output))

        # 重塑结果
        output = output.permute(0, 2, 4, 1, 3)  # (N, C_out, L, C_in, K*K)
        output = output.reshape(N, default_cnum, H-2, W-2, C*9)  # (N, C_out, H-2, W-2, C_in * 9)
        output = XQuantize.apply(LUTclip(output.mean(dim=-1)))
        return output

import numpy as np

class OffsetShiftModule_basic(nn.Module):
    #default offset shift module
    #contiguous offset_x and offset_y
    def __init__(self, in_channels):
        super().__init__()
        self.offset = nn.Parameter(torch.zeros(2, default_cnum, dtype=torch.int), requires_grad=False)
    
    def forward(self, x):
        N, C, H, W = x.shape
        shifted = torch.zeros_like(x)
        for c in range(16):
            dy = self.offset[0,c].item()
            dx = self.offset[1,c].item()
            x_pad = F.pad(x[:,c], pad=(abs(dx), abs(dx), abs(dy), abs(dy)), mode='replicate')
            shifted[:,c] = x_pad[:, abs(dy)+dy:abs(dy)+dy+H, abs(dx)+dx:abs(dx)+dx+W]

        return shifted

    def load_constant(self, file_path):
        offset_np = np.load(file_path)
        offset_np = offset_np.mean(axis=-1)
        offset = torch.from_numpy(offset_np).round().int()
        self.offset.data = offset
        print(self.offset, self.offset.requires_grad)

lsb_clip = 0

class TinyLUT(torch.nn.Module):
    @staticmethod
    def low_high(image):
        xl = torch.remainder(image, 4)
        xh = torch.div(image, 4, rounding_mode='floor')
        xl_ = image.clone()
        xh_ = image.clone()
        xl_.data = xl.data
        xh_.data = xh.data
        return xl_.type(torch.float32), xh_.type(torch.float32)

    def __init__(self, scale, stacks, sp_shift_c, sp_shift_p, sp_shift_d):
        super().__init__()
        self.msb = nn.ModuleDict()
        self.lsb = nn.ModuleDict()
        self._init_branch(self.msb, scale, stacks, sp_shift_c, sp_shift_p, sp_shift_d, True, 'msb')
        self._init_branch(self.lsb, scale, stacks, sp_shift_c, sp_shift_p, sp_shift_d, True, 'lsb')
        self.scale = scale
        self.stacks = stacks
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.sp_shift_c = sp_shift_c
        self.sp_shift_p = sp_shift_p

    def load_constant(self, path):
        for i in range(self.stacks+1):
            self.msb[f"scs{i}"].load_constant(f'{path}/msb_{i}.npy')
        for i in range(lsb_clip):
            self.lsb[f"scs{i}"].load_constant(f'{path}/lsb_{i}.npy')

    def _init_branch(self, branch: nn.ModuleDict, scale, stacks, sp_shift_c, sp_shift_p, sp_shift_d, use_shift, branch_name):
        dsnets = nn.ModuleList()

        dsnets.append(DepthWise(1, default_cnum))
        if branch_name == 'msb' or lsb_clip > 0:
            dsnets.append(PointConv())

        for i in range(1, 1+stacks):
            if branch_name == 'lsb' and i >= lsb_clip:
                continue

            dsnets.append(DepthWise(default_cnum, default_cnum))
            dsnets.append(PointConv())

        branch["dsnets"] = dsnets

        if branch_name == 'msb':
            branch["up"] = UpConv(scale)

        if use_shift and sp_shift_p > 0:
            cs = stacks+1 if branch_name == 'msb' else lsb_clip
            for i in range(cs):
                branch[f"scs{i}"] = OffsetShiftModule_basic(default_cnum)

    def forward(self, x, phase='train', use_lsb=True, x_pth=None, accum=None):
        N,C,H,W = x.shape
        x = x.reshape(N*C,1,H,W)
        x_l, x_h = self.low_high(x)
        c, p = self.sp_shift_c, self.sp_shift_p

        # DSLUTS
        for i in range(len(self.msb["dsnets"])):
            msb_module = self.msb["dsnets"][i]
            # Spatial Channel Shift
            if isinstance(msb_module, PointConv) and p > 0:
                x_h = self.msb[f"scs{i//2}"](x_h)

            if i >= len(self.lsb["dsnets"]):
                x_h = (msb_module(x_h)+x_h).clamp(-32, 31)
            else:
                lsb_module = self.lsb["dsnets"][i]
                if isinstance(lsb_module, PointConv) and p > 0:
                    x_l = self.lsb[f"scs{i//2}"](x_l)
                x_l = (x_l+lsb_module(x_l)).clamp(0,3)
                x_h = (msb_module(x_h)+x_h+x_l).clamp(-32, 31)

        # UP
        x = self.msb["up"](x_h)
        x = self.pixel_shuffle(x)
        x = x.reshape(N, C, x.shape[-2], x.shape[-1])
        return x

class TinyLUTRE(torch.nn.Module): 
    # TinyLUT + Rotational Ensemble
    def __init__(self, scale, stacks, sp_shift_c, sp_shift_p, sp_shift_d):
        super().__init__()
        module_type = TinyLUT
        self.module = module_type(scale, stacks, sp_shift_c, sp_shift_p, sp_shift_d)
        # self.pad_tp = (2*(1+stacks), 0, 2*(1+stacks), 0)
        self.pad_tp = (0, 0, 0, 0)
    
    def load_constant(self, path):
        self.module.load_constant(path)

    def forward(self, x, phase='train', use_lsb=True, x_id=None, accum=None):
        if x_id is not None:
            x_id = os.path.join('offset_vis', x_id)
            if not os.path.exists(x_id):
                os.mkdir(x_id)

        N,C,H,W = x.shape
        x = x.reshape(N*C,1,H,W)
        if phase == 'train':
            # 1. 对输入旋转，并填充
            x_batch = torch.cat([F.pad(torch.rot90(x, i, [2,3]), self.pad_tp, mode='reflect') for i in range(4)], dim=0)

            # 2. 一次性处理所有旋转后的图像，并进行裁剪
            batch_S_all = LUTclip(self.module(x_batch, phase, use_lsb))

            # 3. 将结果分割为 4 部分
            batch_S_all = torch.chunk(batch_S_all, 4, dim=0)

            # 4. 对每部分进行反向旋转并求和
            batch_S = sum([torch.rot90(batch_S_all[i], (-i)%4, [2,3]) for i in range(4)])
            
            # 5. 计算 L2 正则化损失
            # print(self.module.msb.clip_alpha)
            # l2_loss = torch.tensor(0)
            # l2_loss = (torch.sum(self.module.lsb.clip_alpha**2)+torch.sum(self.module.msb.clip_alpha**2)) / (torch.numel(self.module.lsb.clip_alpha) + torch.numel(self.module.msb.clip_alpha))
            # print(l2_loss)
        else:
            # 测试的时候图片可能不规则，就不这样并行了
            batch_S1 = self.module(F.pad(x, self.pad_tp, mode='reflect'), phase, use_lsb, x_id, accum)

            batch_S2 = self.module(F.pad(torch.rot90(x, 1, [2,3]), self.pad_tp, mode='reflect'), phase, use_lsb, x_id, accum)
            batch_S2 = torch.rot90(batch_S2, 3, [2,3])
        
            batch_S3 = self.module(F.pad(torch.rot90(x, 2, [2,3]), self.pad_tp, mode='reflect'), phase, use_lsb, x_id, accum)
            batch_S3 = torch.rot90(batch_S3, 2, [2,3])
        
            batch_S4 = self.module(F.pad(torch.rot90(x, 3, [2,3]), self.pad_tp, mode='reflect'), phase, use_lsb, x_id, accum)
            batch_S4 = torch.rot90(batch_S4, 1, [2,3])
        
            batch_S = sum(LUTclip(batch) for batch in [batch_S1, batch_S2, batch_S3, batch_S4])
        
        # 注意：这句话很关键
        # batch_S = XQuantize.apply(batch_S/4.0)
        batch_S = batch_S.reshape(N, C, batch_S.shape[-2], batch_S.shape[-1])
        return batch_S

    def forward_with_lh(self, x_l, x_h):
        # x_l, x_h = TinyLUT.low_high(x)
        N,C,H,W = x_l.shape
        x_l = x_l.reshape(N*C,1,H,W)
        x_h = x_h.reshape(N*C,1,H,W)
        # x = x.reshape(N*C,1,H,W)

        # 测试的时候图片可能不规则，就不这样并行了
        # batch_S1 = self.module.forward(x)
        # batch_S2 = self.module.forward(torch.rot90(x, 1, [2,3]))
        # batch_S3 = self.module.forward(torch.rot90(x, 2, [2,3]))
        # batch_S4 = self.module.forward(torch.rot90(x, 3, [2,3]))


        batch_S1 = self.module.forward_with_lh(x_l, x_h)
        batch_S2 = self.module.forward_with_lh(torch.rot90(x_l, 1, [2,3]),torch.rot90(x_h, 1, [2,3]))
        batch_S3 = self.module.forward_with_lh(torch.rot90(x_l, 2, [2,3]),torch.rot90(x_h, 2, [2,3]))
        batch_S4 = self.module.forward_with_lh(torch.rot90(x_l, 3, [2,3]),torch.rot90(x_h, 3, [2,3]))
        # print(batch_S1.shape, batch_S2.shape)
        batch_S2 = torch.rot90(batch_S2, 3, [2,3])
        batch_S3 = torch.rot90(batch_S3, 2, [2,3])
        batch_S4 = torch.rot90(batch_S4, 1, [2,3])
    
        batch_S = sum(LUTclip(batch) for batch in [batch_S1, batch_S2, batch_S3, batch_S4])
    
        batch_S = batch_S.reshape(N, C, batch_S.shape[-2], batch_S.shape[-1])
        return batch_S