import os

from data import common

import numpy as np
import imageio

import torch
import torch.utils.data as data


class Demo(data.Dataset):
    def __init__(self, args, name='Demo', train=False, benchmark=False):
        self.args = args
        self.name = name
        self.train = False
        self.n_frames = args.n_frames

        self.filelist = []
        for f in os.listdir(args.dir_demo):
            if f.find('.png') >= 0 or f.find('.bmp') >= 0:
                self.filelist.append(os.path.join(args.dir_demo, f))
        self.filelist.sort()

        self.filelist_GT = []
        for f in os.listdir(args.dir_demo_GT):
            if f.find('.png') >= 0 or f.find('.bmp') >= 0:
                self.filelist_GT.append(os.path.join(args.dir_demo_GT, f))
        self.filelist_GT.sort()

    def __getitem__(self, idx):
        index = idx + 1
        filename = os.path.splitext(os.path.basename(self.filelist[index]))[0]

        frames_lr = []
        for i in range(index - self.n_frames // 2, index + self.n_frames // 2 + 1):
            if i < 0:
                frames_lr.append(imageio.imread(self.filelist[index + self.n_frames // 2 - i]))
            elif i >= len(self.filelist):
                frames_lr.append(imageio.imread(self.filelist[index - self.n_frames // 2 - i + len(self.filelist) - 1]))
            else:
                frames_lr.append(imageio.imread(self.filelist[i]))
        frame_hr = imageio.imread(self.filelist_GT[index])

        frames_lr = np.array(frames_lr)

        frames_lr = frames_lr / 255.0

        frames_lr = np.ascontiguousarray(frames_lr.transpose((0, 3, 1, 2)))
        frame_hr = np.ascontiguousarray(frame_hr.transpose((2, 0, 1)))
        frames_lr = torch.from_numpy(frames_lr).float()
        frame_hr = torch.from_numpy(frame_hr).float()

        return frames_lr, frame_hr, filename

    def __len__(self):
        return len(self.filelist) - 2

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale
