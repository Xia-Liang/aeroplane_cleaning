from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from dataset import AirplaneDataset
from model import PointNetDenseCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# --------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='data', help="dataset path")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--nepoch', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--npoints', type=int, default=6000, help='num of points of each file')
parser.add_argument('--outf', type=str, default='seg', help='model output folder')
# basic info of parser
opt = parser.parse_args()
opt.manualSeed = random.randint(1, 10000)  # fix seed
print(opt)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# make output file
try:
    if not os.path.exists(opt.outf):
        os.makedirs(opt.outf)
except OSError:
    raise OSError('fail to make dir')

# --------------------------------------------------------------------------------------------------------------

# load dataset into cuda, print basic info
dataset = AirplaneDataset(root=opt.dataset, n_points=opt.npoints)
train_size = int(len(dataset) * 0.8)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=opt.batchSize,
    shuffle=True,  # in order or not
    num_workers=int(opt.workers))  # mulit-process
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batchSize,
    shuffle=True,  # in order or not
    num_workers=int(opt.workers))  # mulit-process
print('train size: ', len(train_dataset), ' test size: ', len(test_dataset))
num_batch = len(train_dataset) / opt.batchSize

# --------------------------------------------------------------------------------------------------------------

# print basic info, set classifier, optimizer and sheduler
blue = lambda x: '\033[94m' + x + '\033[0m'  # for test output highlight
num_classes = len(dataset.global_segmentation)
print('classes', num_classes)
print(dataset.global_segmentation)
classifier = PointNetDenseCls(k=num_classes, feature_transform=opt.feature_transform)
optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))  # Stochastic Optimization
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)  # adjust the learning rate
classifier.cuda()

# write out events and summaries to the event file
writer = SummaryWriter('run')

# --------------------------------------------------------------------------------------------------------------

# training
max_acc_train = 0
max_acc_test = 0
for epoch in range(opt.nepoch):
    total_train_loss = 0
    total_test_loss = 0
    total_train_correct = 0
    total_test_correct = 0

    optimizer.step()
    scheduler.step()

    # train
    for i, data in enumerate(train_loader, 0):
        # points (batch size, num of points, (xyz)); target (batch size, num of points, target)
        points, target = data
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()  # Sets the gradients of all optimized torch.Tensor s to zero.
        classifier = classifier.train()

        # pred (batch size, num of points, target class); trans (batch size, 3, 3)
        pred, trans, trans_feat = classifier(points)
        # extend ( batch size * num of points, num of target class)
        pred = pred.view(-1, num_classes)

        target = target.view(-1, 1)[:, 0]  # no need to -1, since ground truth is 0~12
        # print(pred.size(), target.size())
        loss = F.nll_loss(pred, target)
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()

        # pred.data (batch size * num of points, 4 class of plane)
        # pred.data.max(1) (1, batch size * num of points) return max of each row
        # pred.data.max(1)[1] return index of max value, which represents class
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        if i % 10 == 0:
            print('[%d epoch: %d-th batch/%d total] train loss: %f accuracy: %f' %
                  (epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize * opt.npoints)))
        total_train_loss += loss.item()
        total_train_correct += correct.item()

    # test
    for j, data in enumerate(test_loader, 0):
        points, target = data
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = classifier.eval()
        pred, _, _ = classifier(points)
        pred = pred.view(-1, num_classes)
        target = target.view(-1, 1)[:, 0]
        loss = F.nll_loss(pred, target)
        pred_choice = pred.data.max(1)[1]

        correct = pred_choice.eq(target.data).cpu().sum()
        total_test_loss += loss.item()
        total_test_correct += correct.item()
    print('[%d epoch %s]  loss: %f accuracy: %f' %
          (epoch, blue('test'), total_test_loss, total_test_correct / len(test_dataset) / opt.npoints))

    acc_train = total_train_correct / opt.npoints / len(train_dataset)
    acc_test = total_test_correct / opt.npoints / len(test_dataset)

    writer.add_scalar('Loss/train', total_train_loss, epoch)
    writer.add_scalar('Loss/test', total_test_loss, epoch)
    writer.add_scalar('Acc/train', acc_train, epoch)
    writer.add_scalar('Acc/test', acc_test, epoch)

    if epoch > 75:
        if epoch % 5 == 0 and acc_train > 0.9 and acc_test > 0.85:
            torch.save(classifier.state_dict(), '%s/seg_model_%d.pth' % (opt.outf, epoch))
        elif acc_train >= max_acc_train and acc_test >= max_acc_test:
            max_acc_train = acc_train
            max_acc_test = acc_test
            torch.save(classifier.state_dict(), '%s/seg_model_%d_best.pth' % (opt.outf, epoch))

# --------------------------------------------------------------------------------------------------------------
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
