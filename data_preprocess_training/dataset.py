from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm
import json
from plyfile import PlyData, PlyElement


class AirplaneDataset(data.Dataset):
    def __init__(self,
                 data_root='data',
                 train_folder='train',
                 n_points=4500,
                 data_augmentation=True):
        self.data_root = data_root
        self.train_folder = train_folder
        self.n_points = n_points
        self.data_augmentation = data_augmentation

        # global_segmentation, a dict of { (plane seg, corresponding class num) }
        self.global_segmentation = {}
        with open(os.path.join(self.data_root, 'airplaneCategory.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.global_segmentation[ls[0]] = ls[1]
        # print(self.global_segmentation)

        self.file_list = os.listdir(os.path.join(self.data_root, self.train_folder))

    def __getitem__(self, index):
        # read data from npy file
        raw_data = np.load(os.path.join(self.data_root, self.train_folder, self.file_list[index]))

        point_set = raw_data[:, 0:3].astype(np.float32)
        point_seg = raw_data[:, 3].astype(np.int64)
        # print(point_set.shape, seg.shape)

        # deal with raw data, reordering, centering, scaling and augmenting
        choice = np.random.choice(len(point_seg), self.n_points, replace=True)
        # resample
        point_set = point_set[choice, :]
        point_seg = point_seg[choice]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augmentation:
            point_set += np.random.normal(0, 0.001, size=point_set.shape)  # random jitter

        point_set = torch.from_numpy(point_set)
        point_seg = torch.from_numpy(point_seg)

        return point_set, point_seg

    def __len__(self):
        return len(self.file_list)


if __name__ == '__main__':
    d = AirplaneDataset(data_root='data')
    print('there are ', len(d), ' files in path')
    ps, seg = d[0]
    print(ps.size(), ps.type(), seg.size(), seg.type())
