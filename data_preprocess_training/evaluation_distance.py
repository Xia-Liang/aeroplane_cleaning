import os
import numpy as np
import torch
from model import PointNetDenseCls

testdir = os.path.join('data', 'testset')

file_list = os.listdir(os.path.join(testdir, 'lidar'))
print(len(file_list))
# sem_lidar_dir = os.listdir(os.path.join(testdir, 'semLidar'))

classifier = PointNetDenseCls(k=7, feature_transform=False)
# =====================================================================================================
# load model
# ======================================================================================================
classifier.load_state_dict(torch.load(os.path.join('seg', 'class6_points4500_3.pth')))
classifier.eval()

tag_list = ['None', 'FrontCabin', 'RearCabin', 'Tail', 'Wing', 'Engine', 'Wheel']

for file in file_list:
    # -----------------------------------------------------------
    # process semLidar
    ground_truth = np.load(os.path.join(testdir, 'semLidar', file))
    ground_truth_dist = np.sqrt(np.sum(ground_truth[:, 0:3] ** 2, axis=1))

    if len(ground_truth_dist) < 5000:
        continue
    # -----------------------------------------------------------
    # process lidar
    lidar_array = np.load(os.path.join(testdir, 'lidar', file))
    while lidar_array.shape[0] < 4500:
        lidar_array = np.concatenate([lidar_array, lidar_array], axis=0)
    choice = np.random.choice(lidar_array.shape[0], 4500, replace=True)
    lidar_array = lidar_array[choice, :]
    dist = np.sqrt(np.sum(lidar_array ** 2, axis=1))  # real distance, shape = (length, )
    # process the lidar data into cuda
    lidar_array = lidar_array - np.expand_dims(np.mean(lidar_array, axis=0), 0)  # centering
    max_dist = np.max(np.sqrt(np.sum(lidar_array ** 2, axis=1)), 0)  # max distance after centering
    cuda_array = lidar_array / max_dist  # scale
    # print(cuda_array.shape)
    cuda_array = cuda_array.reshape((1, 4500, 3))
    # gpu
    cuda_array = torch.from_numpy(cuda_array).float().transpose(2, 1)
    pred, _, _ = classifier(cuda_array)
    pred = pred.view(-1, 7)
    pred_choice = pred.data.max(1)[1].cpu().numpy()  # return indices of max val, shape = (length, )
    # print(pred_choice.shape)
    result = np.concatenate((dist.reshape(4500, 1), pred_choice.reshape(4500, 1)), axis=1)
    # print(result.shape)  # shape = (length, 2)
    # -----------------------------------------------------------
    # print result
    print(file)
    print('lidar points %4.0f predict as None, %4.0f predict as NotNone'
          % (len(dist[np.where(pred_choice == 0)]), len(dist[np.where(pred_choice != 0)])), end=', ')
    print('%6.0f sem lidar points' % len(ground_truth_dist))
    print('lidar: min distance from other object %4.1fm' % min(dist))
    print(' ')

    print('%10s %12s %12s %15s %15s' % ('tag', 'lidar(m)', 'semLidar(m)', 'lidarPoints', 'semLidarPoints'))

    # tag 0
    print('%10s' % tag_list[0], end=' ')
    print('%11.1f ' % np.min(dist[np.where(pred_choice == 0)], initial='inf '), end='')
    print('%11.1f ' % np.min(ground_truth_dist[np.where(ground_truth[:, 3] == 0)], initial='inf '), end='')
    print('%15.0f ' % max(0, len(dist[np.where(pred_choice == 0)])), end='')
    print('%15.0f ' % max(0, len(ground_truth_dist[np.where(ground_truth[:, 3] == 0)])))
    # tag 1~6
    for tag in range(1, 7):
        print('%10s' % tag_list[tag], end=' ')
        print('%11.1f ' % np.min(dist[np.where(pred_choice == tag)], initial='inf '), end='')
        print('%11.1f ' % np.min(ground_truth_dist[np.where(ground_truth[:, 3] == (tag+33))], initial='inf'), end='')
        print('%15.0f ' % max(0, len(dist[np.where(pred_choice == tag)])), end='')
        print('%15.0f ' % max(0, len(ground_truth_dist[np.where(ground_truth[:, 3] == (tag+33))])))
    print('-------------------------------------')
