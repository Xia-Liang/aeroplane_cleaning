from __future__ import print_function
import argparse
import os
import random
import torch
import torch.utils.data
from dataset import AirplaneDataset
from model import PointNetDenseCls, feature_transform_regularizer
from tqdm import tqdm
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='data', help="dataset path")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--npoints', type=int, default=6000, help='num of points of each file')
parser.add_argument('--outf', type=str, default='seg', help='model output folder')

# basic info of parser
opt = parser.parse_args()
opt.manualSeed = random.randint(1, 10000)  # fix seed
print(opt)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# load dataset for test
dataset = AirplaneDataset(root=opt.dataset, n_points=opt.npoints)
train_size = int(len(dataset) * 0.8)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batchSize,
    shuffle=True,  # in order or not
    num_workers=int(opt.workers))  # mulit-process

num_classes = len(dataset.global_segmentation)

models = os.listdir(opt.outf)
models.sort()
print('load model: ', models[-1])

classifier = PointNetDenseCls(k=num_classes, feature_transform=opt.feature_transform)
classifier.load_state_dict(torch.load(os.path.join(opt.outf, models[-1])))
classifier.cuda()

'''
benchmark mIOU, Mean Intersection over Union，平均交并比, mIOU=TP/(FP+FN+TP), is better when approaches 1
global_ious check for global seg, that means tags 0~12
airplane_ious only check for airplane seg, that means tags 1~11

in data_preprocess.py define, tags 0 for others, tags 12 for vehicles, tags 1~11 for airplane
'''

global_ious = []
airplane_ious = []
for i, data in tqdm(enumerate(test_loader, 0)):
    points, target = data
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    pred_choice = pred.data.max(2)[1]

    pred_np = pred_choice.cpu().data.numpy()
    target_np = target.cpu().data.numpy()

    for shape_idx in range(target_np.shape[0]):
        global_tags = range(num_classes)  # 0~12
        airplane_tags = range(1, num_classes - 1)  # 1~11
        part_global_ious = []
        part_airplane_ious = []
        for tag in global_tags:
            I = np.sum(np.logical_and(pred_np[shape_idx] == tag, target_np[shape_idx] == tag))  # true positive
            U = np.sum(np.logical_or(pred_np[shape_idx] == tag, target_np[shape_idx] == tag))  # others sum
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_global_ious.append(iou)

        for tag in airplane_tags:
            I = np.sum(np.logical_and(pred_np[shape_idx] == tag, target_np[shape_idx] == tag))  # true positive
            U = np.sum(np.logical_or(pred_np[shape_idx] == tag, target_np[shape_idx] == tag))  # others sum
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_airplane_ious.append(iou)

        global_ious.append(np.mean(part_global_ious))
        airplane_ious.append(np.mean(part_airplane_ious))

print("mIOU for global: {}".format(np.mean(global_ious)))
print("mIOU for airplane: {}".format(np.mean(airplane_ious)))
