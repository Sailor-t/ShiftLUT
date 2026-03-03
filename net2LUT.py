import torch
import numpy as np
import os
from torch.autograd import Function
import importlib
import json

LUT_path_root = os.path.join('LUT_test', 'LUTs')


# scale, stacks, cnum, model_name, step, EPS = 1, 7, 16, 'ShiftLUT_sigma=15_int', 180000, 0.55 # 60KB
# scale, stacks, cnum, model_name, step, EPS = 1, 7, 16, 'ShiftLUT_qf=10_dncnn_v3_int4', 128000, 0.62 # 60KB 29.12/28.96, 69KB 29.13/28.97
# scale, stacks, msb_base, cnum, model_name, step, EPS = (4, 0, 6, 16, 'ShiftLUT_sr_s0_int', 194000, 0.50) # 24KB
scale, stacks, msb_base, cnum, model_name, step, EPS = (4, 1, 6, 16, 'ShiftLUT_sr_s1_int', 195000, 0.45)
# scale, stacks, msb_base, cnum, model_name, step, EPS = (4, 7, 6, 16, 'ShiftLUT_sr_s7_int', 189000, 0.38) # K_Max ~ 64


LUT_path = os.path.join(LUT_path_root, model_name)
os.makedirs(LUT_path, exist_ok=True)

module_name = f'models.{model_name}.model'
Model = importlib.import_module(module_name)

TinyLUTRE = Model.TinyLUTRE
default_cnum = cnum
DepthWise = Model.DepthWise
PointConv = Model.PointConv
UpConv = Model.UpConv


# model_G = TinyLUTRE(scale, stacks, 0, 1, -1)
model_G = TinyLUTRE(scale=scale, stacks=stacks, use_shift=True, msb_base=msb_base, cnum=cnum)

lm = torch.load(
    os.path.join('models', model_name, 'Model_{:06d}.pth'.format(step)), weights_only=True)
model_G.load_state_dict(lm, strict=True)
model_G = model_G.cuda()

def save_constant(self):
    combined_tensors = []
    
    for i in range(stacks+1):
        # lsb_tensor = self.lsb[f"scs{i}"].offset.view(2,16)
        msb_tensor = self.msb[f"scs{i}"].offset.view(2,16)
        combined_tensors.append(msb_tensor)
    
    # 将所有 (2, 2, 16) 的张量堆叠成一个 (8, 2, 2, 16) 的张量
    result_tensor = torch.stack(combined_tensors)
    print(result_tensor.shape)
    result_tensor = result_tensor.to(torch.int)

    np.save(os.path.join(LUT_path, 'offset.npy'), result_tensor.cpu().numpy())

class Round(Function):
    @staticmethod
    def forward(ctx, x):
        out = torch.round(x)
        return out
    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs

def LUTclip(x):
    return x.clamp(-128, 127)
    
def DW2LUT(m, input_tensor): # SCLUT and DWLUT
    assert isinstance(m, DepthWise)
    assert (m.stack == 1)
    k, b = m.kernels[0], m.biass[0]
    # print(b)
    # print(k.shape, input_tensor.shape)
    out = Round.apply(LUTclip(k*input_tensor+b))
    out = out.permute(0, 2, 1, 3)
    out = out[..., 0]
    return out # shape: [data_range, kernel_size, out_c]

def PW2LUT(m, input_tensor):
    assert isinstance(m, PointConv)
    ori = [LUTclip(m.conv[i](input_tensor))[:,:,0,0].cpu().numpy() for i in range(default_cnum)]
    return np.stack(ori, 1).astype(np.int8) # shape: [data_range, in_c, out_c]

def UP2LUT(m, input_tensor):
    assert isinstance(m, UpConv)
    ori = [LUTclip(m.Conv[i](input_tensor))[:,:,0,0].cpu().numpy() for i in range(default_cnum)]
    return np.stack(ori, 1).astype(np.int8) # shape: [data_range, in_c, scale**2]

def generate_input(base):
    # 2D input
    first_ = base.cuda().unsqueeze(1)
    first__ = torch.cat([first_, first_], 1)
    first___ = torch.cat([first__, first__], 1)
    first_8 = torch.cat([first___, first___], 1)
    first_9 = torch.cat([first_8, first_], 1)
    
    # Rearange input: [N, 4] -> [N, C=1, H=2, W=2]
    input_tensor_dw = first_9.unsqueeze(1).unsqueeze(1).reshape(-1,1,9,1).float()
    # 由于对不同大小的patch进行推理时的底层实现不同，所以这里要repeat
    input_tensor_pw = first_.unsqueeze(1).unsqueeze(1).reshape(-1,1,1,1).repeat(1,1,5,5).float()
    return input_tensor_dw, input_tensor_pw

def Query(w, x, sample_d):
    if sample_d == 1:
        return w[x]

    a1 = x//sample_d
    a2 = a1+1

    b = x%sample_d
    weight = b/sample_d
        
    ans = w[a1*sample_d]*(1-weight) + w[a2*sample_d]*weight
    return ans

def cal(x, step_a, step_b):
    ans = [np.abs(Query(x, i, step_a) - Query(x, i, step_b)).mean() for i in range(64)]
    return np.mean(ans)

def save_LUT(path, LUT, sample_d):
    # print(LUT.shape, sample_d, LUT[::sample_d].shape)
    N, IN, OUT = LUT.shape
    # np.save(path, final)

    arrays = {}
    meta = {"IN": IN, "OUT": OUT, "LUTs":[]}

    sum = 0
    for in_ch in range(IN):  # 遍历输入通道
        for out_ch in range(OUT):
            data = LUT[:, in_ch, out_ch]
            step = sample_d
            
            if N == 65:
                # while step < 16 and cal(data, step, step*2)*step < EPS:
                #     step *= 2
                # N/2s
                # N-N/s= N* (s-1)/s
                # print(f'Step A: {step}')
                step = 1
                for candidate_step in [2, 4, 8, 16]:
                    if cal(data, 1, candidate_step) < EPS*(1-1/candidate_step):
                        step = candidate_step
                
                # print(f'Step B: {step}')

            if step == 1:
                data = data[:64]
            else:
                data = data[::step]
            # np.save(f'{path}_{in_ch}_{out_ch}', data)
            arrays[f'i{in_ch}o{out_ch}'] = data
            meta["LUTs"].append({"in": in_ch, "out": out_ch, "step": step})
            sum += data.size

    np.savez(path + ".npz", **arrays)
    with open(path + ".json", "w") as f:
        json.dump(meta, f, indent=2)

    return sum

def Branch2LUT(branch, base, branch_name, stacks, sample_d):
    with torch.no_grad():
        input_tensor_dw, input_tensor_pw = generate_input(base)
        tot = 0
        for i in range(stacks+1):

            res = DW2LUT(branch.dsnets[i*2], input_tensor_dw)
            res = res.cpu().numpy()
            print(f"For {i} in {branch_name}, Resulting DWLUT size: ", res.shape)
            tot += save_LUT(LUT_path+f"/DW{i}_{branch_name}", res, sample_d[i*2])

            if branch_name == 'MSB':
                res = PW2LUT(branch.dsnets[i*2+1], input_tensor_pw)
                print(f"For {i} in {branch_name}, Resulting PWLUT size: ", res.shape)
                tot += save_LUT(LUT_path+f"/PW{i}_{branch_name}", res, sample_d[i*2+1])

        if branch_name == 'MSB':
            res = UP2LUT(branch.up, input_tensor_pw)
            print(f"For {i} in {branch_name}, Resulting PWLUT size: ", res.shape)
            tot += save_LUT(LUT_path+f"/UP_{branch_name}", res, sample_d[-1])

        print(f"For {branch_name}, storage = {tot/1024}KB.")

save_constant(model_G.module)

# msb_steps = [8, 2] * 8 + [2]
msb_steps = [1, 1] * (stacks+1) + [1]

# msb_steps[-1] = 4
# msb_steps[-2] = 4
# msb_steps[-4] = 4
# msb_steps[-6] = 4

assert (stacks+1)*2+1 == len(msb_steps)
    

# np.save(os.path.join(LUT_path, 'steps.npy'), np.array(msb_steps))

Branch2LUT(model_G.module.msb, torch.arange(-32, 33, 1), 'MSB', stacks, msb_steps)
Branch2LUT(model_G.module.lsb, torch.arange(0, 4, 1), 'LSB', 0, [1])