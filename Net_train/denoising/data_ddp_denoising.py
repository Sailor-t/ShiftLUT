import os
import random
import sys

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm

sys.path.insert(0, "../../")  # run under the project directory
from common.utils import modcrop

class Provider(object):
    def __init__(self, batch_size, num_workers, sigma, path, patch_size):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.path = path
        self.patch_size = patch_size

        self.data = DIV2K(sigma, path, patch_size)
        self.rank = None
        self.world_size = None
        self.data_iter = None
        self.iteration = 0
        self.epoch = 1

    def build(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size

        self.sampler = DistributedSampler(
            self.data,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=self.epoch  # 每轮换种子
        )

        self.data_loader = DataLoader(
            dataset=self.data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.sampler,
            drop_last=False,
            pin_memory=True
        )

        self.data_iter = iter(self.data_loader)

    def next(self):
        if self.data_iter is None:
            raise RuntimeError("Data iterator not initialized. Call build(rank, world_size) first.")

        try:
            batch = next(self.data_iter)
            self.iteration += 1
            return batch[0].cuda(), batch[1].cuda()
        except StopIteration:
            self.epoch += 1
            self.build(self.rank, self.world_size)  # 用新 epoch 更新随机种子
            self.iteration += 1
            batch = next(self.data_iter)
            return batch[0].cuda(), batch[1].cuda()

class DIV2K(Dataset):
    def __init__(self, sigma, path, patch_size, rigid_aug=True):
        super(DIV2K, self).__init__()
        self.sigma = sigma
        self.sz = patch_size
        self.rigid_aug = rigid_aug
        self.path = path
        self.file_list = [str(i).zfill(4) for i in range(1, 901)]

        self.hr_cache = os.path.join(path, "cache_hr.npy")
        if not os.path.exists(self.hr_cache):
            self.cache_hr()
        self.hr_ims = np.load(self.hr_cache, allow_pickle=True).item()

    def cache_hr(self):
        hr_dict = dict()
        dataHR = os.path.join(self.path, "HR")
        for f in self.file_list:
            hr_dict[f] = np.array(Image.open(os.path.join(dataHR, f + ".png")))
        np.save(self.hr_cache, hr_dict, allow_pickle=True)

    def __getitem__(self, idx):
        key = self.file_list[idx % len(self.file_list)]  # 按 index 获取图像
        lb = self.hr_ims[key]

        shape = lb.shape
        i = random.randint(0, shape[0] - self.sz)
        j = random.randint(0, shape[1] - self.sz)
        c = random.choice([0, 1, 2])

        lb = lb[i:i+self.sz, j:j+self.sz, c]

        if self.rigid_aug:
            if random.random() < 0.5:
                lb = np.fliplr(lb)
            if random.random() < 0.5:
                lb = np.flipud(lb)
            k = random.randint(0, 3)
            lb = np.rot90(lb, k)

        lb = np.expand_dims(lb.astype(np.float32) / 255.0, axis=0)
        im = lb + np.random.normal(0, self.sigma/255.0, lb.shape).astype(np.float32)
        im = np.round(im*255.0)

        return im, lb

    def __len__(self):
        return len(self.file_list)

class DNBenchmark(Dataset):
    def __init__(self, path, sigma=5, use_sets=1):
        super(DNBenchmark, self).__init__()
        datasets = ['Set12', 'BSD68', 'CBSD68', 'Kodak24', 'McMaster']
        self.ims = dict()
        self.files = dict()
        pic_num = [12, 68, 68, 24, 18]
        _ims_all = np.sum(pic_num[:use_sets])*2

        for dataset in datasets[:use_sets]:
            folder = os.path.join(path, dataset)
            files = os.listdir(folder)
            files.sort()
            self.files[dataset] = files

            for i in range(len(files)):
                im_hr = np.array(Image.open(
                    os.path.join(path, dataset, files[i])))
                if dataset in ["Set12", "BSD68"]:
                    im_hr = im_hr[:, :, np.newaxis]

                key = dataset + '_' + files[i][:-4]

                self.ims[key] = im_hr

                np.random.seed(seed=0)  # set seed for random noise
                im_lr = im_hr / 255.0 + \
                    np.random.normal(0, sigma/255.0, im_hr.shape)

                key = dataset + '_' + files[i][:-4] + 'x%d' % sigma
                self.ims[key] = im_lr.astype(np.float32)

                # assert (im_lr.shape[2] == im_hr.shape[2] == 1)

        assert (len(self.ims.keys()) == _ims_all)
