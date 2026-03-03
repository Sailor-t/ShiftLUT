import torch
from torch.autograd import Function
import torch.nn.functional as F
import torch.nn as nn

nf = 28
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
    def __init__(self, default_cnum):
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
    def __init__(self, default_cnum):
        super(PointConv, self).__init__()
        self.conv = nn.ModuleList([PointOneChannel(default_cnum) for _ in range(default_cnum)])
        self.c = default_cnum
        
    def forward(self, x, use_shift):
        ori = [LUTclip(self.conv[i](x[:,i:i+1,:,:])) for i in range(self.c)]
        return XQuantize.apply(sum(ori)/self.c)

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
    def __init__(self, scale, cnum):
        super(UpConv, self).__init__()
        self.c = cnum
        self.Conv = nn.ModuleList([UpOneChannel(scale) for _ in range(self.c)])

    def forward(self,x):
        ori = [LUTclip(self.Conv[i](x[:,i:i+1,:,:])) for i in range(self.c)]
        return XQuantize.apply(sum(ori) / self.c)

class DepthWise(torch.nn.Module): # 实际上也可以当作1通道的StandardConv用
    def __init__(self, in_c, out_c, stack=1):
        super().__init__()
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

    def forward(self, x, use_shift):
        # x = F.pad(x, (2, 0, 2, 0), mode='reflect')
        # x = F.pad(x, (1, 1, 1, 1), mode='replicate')
        if use_shift:
            x = F.pad(x, (1, 1, 1, 1), mode='reflect')
        else:
            x = F.pad(x, (2, 0, 2, 0), mode='reflect')

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
    def __init__(self, cnum):
        super().__init__()
        self.cnum = cnum
        self.offset = nn.Parameter(torch.zeros(2, cnum, dtype=torch.int), requires_grad=False)
    
    def forward(self, x):
        N, C, H, W = x.shape
        shifted = torch.zeros_like(x)
        for c in range(self.cnum):
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

class TinyLUT(torch.nn.Module):
    @staticmethod
    def low_high(image, lsb_base):
        xl = torch.remainder(image, 2**lsb_base)
        xh = torch.div(image, 2**lsb_base, rounding_mode='floor')
        xl_ = image.clone()
        xh_ = image.clone()
        xl_.data = xl.data
        xh_.data = xh.data
        return xl_.type(torch.float32), xh_.type(torch.float32)

    def __init__(self, scale, stacks, use_shift, msb_base, cnum):
        super().__init__()
        self.cnum = cnum
        self.msb = nn.ModuleDict()
        self.lsb = nn.ModuleDict()
        self._init_branch(self.msb, scale, stacks, use_shift, 'msb')
        self._init_branch(self.lsb, scale, stacks, use_shift, 'lsb')
        self.scale = scale
        self.stacks = stacks
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.use_shift = use_shift
        self.clip_msb = lambda x: torch.clamp(x, -(2**(msb_base-1)), 2**(msb_base-1)-1)
        self.clip_lsb = lambda x: torch.clamp(x, 0, 2**(8-msb_base)-1)
        print(f'msb_range: {-(2**(msb_base-1))}~{2**(msb_base-1)-1}')
        print(f'lsb_range: {0}~{2**(8-msb_base)-1}')
        self.lsb_base = 8-msb_base
        
    def load_constant(self, path):
        for i in range(self.stacks+1):
            self.msb[f"scs{i}"].load_constant(f'{path}/msb_{i}.npy')

    def _init_branch(self, branch: nn.ModuleDict, scale, stacks, use_shift, branch_name):
        dsnets = nn.ModuleList()
        dsnets.append(DepthWise(1, self.cnum))
        
        if branch_name == 'msb':
            dsnets.append(PointConv(self.cnum))
            for i in range(1, 1+stacks):
                dsnets.append(DepthWise(self.cnum, self.cnum))
                dsnets.append(PointConv(self.cnum))
            branch["up"] = UpConv(scale, self.cnum)
            if use_shift:
                for i in range(stacks+1):
                    branch[f"scs{i}"] = OffsetShiftModule_basic(self.cnum)

        branch["dsnets"] = dsnets

    def forward(self, x, phase='train'):
        N,C,H,W = x.shape
        x = x.reshape(N*C,1,H,W)
        x_l, x_h = self.low_high(x, self.lsb_base)

        # DSLUTS
        for i in range(len(self.msb["dsnets"])):
            msb_module = self.msb["dsnets"][i]
            # Spatial Channel Shift
            if isinstance(msb_module, PointConv) and self.use_shift:
                x_h = self.msb[f"scs{i//2}"](x_h)

            if i >= len(self.lsb["dsnets"]):
                x_h = self.clip_msb(msb_module(x_h, self.use_shift)+x_h)
            else:
                lsb_module = self.lsb["dsnets"][i]
                x_l = self.clip_lsb(x_l+lsb_module(x_l, self.use_shift))
                x_h = self.clip_msb(msb_module(x_h, self.use_shift)+x_h+x_l)

        # UP
        if self.cnum == self.scale**2:
            x = LUTclip(self.msb["up"](x_h)+x_h)
        else:
            x = self.msb["up"](x_h)
        x = self.pixel_shuffle(x)
        x = x.reshape(N, C, x.shape[-2], x.shape[-1])
        return x

class TinyLUTRE(torch.nn.Module): 
    # TinyLUT + Rotational Ensemble
    def __init__(self, **kwargs):
        super().__init__()
        self.module = TinyLUT(**kwargs)
    
    def load_constant(self, path):
        self.module.load_constant(path)

    def forward(self, x, phase='train'):
        N,C,H,W = x.shape
        x = x.reshape(N*C,1,H,W)
        if phase == 'train':
            # 1. 对输入旋转，并填充
            x_batch = torch.cat([torch.rot90(x, i, [2,3]) for i in range(4)], dim=0)

            # 2. 一次性处理所有旋转后的图像，并进行裁剪
            batch_S_all = LUTclip(self.module(x_batch, phase))

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
            batch_S1 = self.module(x, phase)

            batch_S2 = self.module(torch.rot90(x, 1, [2,3]), phase)
            batch_S2 = torch.rot90(batch_S2, 3, [2,3])
        
            batch_S3 = self.module(torch.rot90(x, 2, [2,3]), phase)
            batch_S3 = torch.rot90(batch_S3, 2, [2,3])
        
            batch_S4 = self.module(torch.rot90(x, 3, [2,3]), phase)
            batch_S4 = torch.rot90(batch_S4, 1, [2,3])
        
            batch_S = sum(LUTclip(batch) for batch in [batch_S1, batch_S2, batch_S3, batch_S4])
        
        # 注意：这句话很关键
        # batch_S = XQuantize.apply(batch_S/4.0)
        batch_S = batch_S.reshape(N, C, batch_S.shape[-2], batch_S.shape[-1])
        return batch_S