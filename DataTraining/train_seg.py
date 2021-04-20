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

# print(torch.__version__ )
#
# print(torch.version.cuda)
#
# print(torch.cuda.is_available())


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=40, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='seg', help='output folder')
parser.add_argument('--npoints', type=int, default=1500, help='num of points of each file')
parser.add_argument('--dataset', type=str, default='/home/xial/code/umProject/PointNet/data', help="dataset path")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = AirplaneDataset(root=opt.dataset)
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


print(len(train_dataset), len(test_dataset))
num_classes = len(dataset.global_segmentation)
print('classes', num_classes)
print(dataset.global_segmentation)
try:
    os.makedirs(opt.outf)
except OSError:
    pass

blue = lambda x: '\033[94m' + x + '\033[0m'  # for test output highlight

classifier = PointNetDenseCls(k=num_classes, feature_transform=opt.feature_transform)

# if opt.model != '':
#     classifier.load_state_dict(torch.load(opt.model))

# Stochastic Optimization
optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
# adjust the learning rate
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()

num_batch = len(train_dataset) / opt.batchSize

for epoch in range(opt.nepoch):
    optimizer.step()
    scheduler.step()
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

        target = target.view(-1, 1)[:, 0]  # no need to -1, since ground truth is 0~11
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
        print('[%d epoch: %d batch/%d total] train loss: %f accuracy: %f' %
              (epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize * opt.npoints)))

        if i % 10 == 0:
            j, data = next(enumerate(test_loader, 0))
            points, target = data
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()
            pred, _, _ = classifier(points)
            pred = pred.view(-1, num_classes)
            target = target.view(-1, 1)[:, 0]
            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]

            # # for debuging
            # cpu_pred = pred_choice.cpu().numpy()
            # cpu_tar = target.data.cpu().numpy()
            # with open('data/debug/pred_tar.txt','w') as f:
            #     for tt in range(len(cpu_pred)):
            #         f.write(str(cpu_pred[tt]) + '  ' + str(cpu_tar[tt]) + '\n')
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d epoch: %d batch/%d total] %s loss: %f accuracy: %f' %
                  (epoch, i, num_batch, blue('test'), loss.item(), correct.item() / float(opt.batchSize * opt.npoints)))

    torch.save(classifier.state_dict(), '%s/seg_model_%d.pth' % (opt.outf, epoch))

# benchmark mIOU, Mean Intersection over Union，平均交并比, mIOU=TP/(FP+FN+TP), is better when approaches 1
shape_ious = []
for i, data in tqdm(enumerate(test_loader, 0)):
    points, target = data
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    pred_choice = pred.data.max(2)[1]

    pred_np = pred_choice.cpu().data.numpy()
    target_np = target.cpu().data.numpy() - 1

    for shape_idx in range(target_np.shape[0]):
        parts = range(num_classes)  # np.unique(target_np[shape_idx])
        part_ious = []
        for part in parts:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))  # true positive
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))  # others sum
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))

print("mIOU : {}".format(np.mean(shape_ious)))
