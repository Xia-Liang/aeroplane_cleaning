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
                 root,
                 n_points=1500,
                 # split='train',
                 data_augmentation=True):
        self.n_points = n_points
        self.root = 'data'
        self.data_augmentation = data_augmentation

        # global_segmentation, a dict of { (plane seg, corresponding class num) }
        self.global_segmentation = {}
        with open(os.path.join(self.root, 'airplaneCategory.txt'),
                  'r') as f:
            for line in f:
                ls = line.strip().split()
                self.global_segmentation[ls[0]] = ls[1]
        # print(len(self.segmentation), ' classes')
        # print(self.segmentation)

        self.data_path = []  # a list with (ply file , seg file)
        for point_file in os.listdir(os.path.join(self.root, 'point')):
            self.data_path.append((os.path.join(self.root, 'point', point_file),
                                  os.path.join(self.root, 'label', point_file + '.seg')))

    def __getitem__(self, index):
        fn = self.data_path[index]
        # point_set, point_seg: points and labels in same-name file
        point_set = np.loadtxt(fn[0]).astype(np.float32)
        point_seg = np.loadtxt(fn[1]).astype(np.int64)
        # print(point_set.shape, seg.shape)

        # deal with raw data, reordering, centering, scaling and augmenting
        choice = np.random.choice(len(point_seg), self.n_points, replace=True)
        # resample
        point_set = point_set[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        point_seg = point_seg[choice]
        point_set = torch.from_numpy(point_set)
        point_seg = torch.from_numpy(point_seg)

        return point_set, point_seg

    def __len__(self):
        return len(self.data_path)



if __name__ == '__main__':
    dataset = 'shapenet'
    data_path = 'data'

    if dataset == 'shapenet':
        d = AirplaneDataset(root=data_path)
        print(len(d))
        ps, seg = d[0]
        print(ps.size(), ps.type(), seg.size(), seg.type())
