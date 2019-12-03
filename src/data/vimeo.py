import os
import glob
import random
import pickle

from data import common

import numpy as np
import imageio
import torch
import torch.utils.data as data


class Vimeo(data.Dataset):
    def __init__(self, args, name='', train=True, benchmark=False):
        self.args = args
        self.name = name
        self.train = train
        self.scale = args.scale
        self.n_frames = args.n_frames
        
        self._set_filesystem(args.dir_data)

    def _set_filesystem(self, dir_data):
        if self.train:
            self.apath = os.path.join(dir_data, 'vimeo_super_resolution_train')
            self.dir_hr = os.path.join(self.apath, 'HR')
            self.dir_lr = os.path.join(self.apath, 'bicubic_LR')
            self.train_pathlist = self.loadpath(os.path.join(self.apath, 'sep_trainlist.txt'))
        else:
            self.apath = os.path.join(dir_data, 'vimeo_super_resolution_test')
            self.dir_hr = os.path.join(self.apath, 'HR')
            self.dir_lr = os.path.join(self.apath, 'bicubic_LR')
            self.test_pathlist = self.loadpath(os.path.join(self.apath, 'sep_testlist.txt'))
            
    def __getitem__(self, idx):
        if self.train:
            path_code = self.train_pathlist[idx]
        else:
            path_code = self.test_pathlist[idx]

        frames_lr = []
        for i in range(4 - self.n_frames // 2, 5 + self.n_frames // 2):
            frames_lr.append(imageio.imread(os.path.join(self.dir_lr, path_code, 'im%d.png' % i)))
        frame_hr = imageio.imread(os.path.join(self.dir_hr, path_code, 'im4.png'))
        frames_lr, frame_hr = self.get_patch(frames_lr, frame_hr)

        frames_lr = np.array(frames_lr)

        frames_lr = frames_lr / 255.0

        frames_lr = np.ascontiguousarray(frames_lr.transpose((0, 3, 1, 2)))
        frame_hr = np.ascontiguousarray(frame_hr.transpose((2, 0, 1)))
        frames_lr = torch.from_numpy(frames_lr).float()
        frame_hr = torch.from_numpy(frame_hr).float()

        return frames_lr, frame_hr, path_code.replace('/', '_')

    def __len__(self):
        if self.train:
            return len(self.train_pathlist)
        else:
            return len(self.test_pathlist)

    def get_patch(self, lr, hr):
        scale = self.scale[0]
        if self.train:
            lr, hr = common.get_patch(
                lr, hr,
                patch_size=self.args.patch_size,
                scale=scale,
                multi=(len(self.scale) > 1),
            )
            if not self.args.no_augment:
                lr, hr = common.augment(lr, hr)
        return lr, hr

    def loadpath(self, pathlistfile):
        fp = open(pathlistfile)
        pathlist = fp.read().splitlines()
        fp.close()
        return pathlist
    
    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale
