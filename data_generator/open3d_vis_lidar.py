'''
vis lidar data

vis lidar data classification by pointnet
'''
import numpy as np
import open3d
import os
import time
try:
    from model.model import PointNetDenseCls
    import torch
except ImportError:
    raise ImportError('cannot import model')

path = os.path.join('D:\\mb95541\\aeroplane\\data\\new')

# # # --------------------------testset ------------------------------------------------------------------------------

# # ----------------------------------------------------------------------
# # vis file for each file in [partly_vis]
#
# for i in range(1, 6):
#     print('------------------------', str(i))
#     file_list = os.listdir(os.path.join(path, 'checkpoint9_test', str(i)))
#     for file in file_list:
#         points = np.loadtxt(os.path.join(path, 'checkpoint9_test', str(i), file))[:, 0:3]
#         point_cloud = open3d.geometry.PointCloud()
#         point_cloud.points = open3d.utility.Vector3dVector(points)
#         open3d.visualization.draw_geometries([point_cloud])
#         print(file)
#         # esc to close current and vis next


# # ----------------------------------------------------------------------
# # vis 6 (checkpoints) file together, lidar, coordinate system transformer, in [checkpoint9_test]
# while True:
#     global_point = [[0, 0, 0]]
#     for i in range(1, 7):
#         file_list = os.listdir(os.path.join(path, 'testset_global_scan', str(i)))
#         file = np.random.choice(file_list)
#         points = np.loadtxt(os.path.join(path, 'testset_global_scan', str(i), file))[:, 0:3]
#         global_point = np.concatenate((global_point, points), axis=0)
#     # random choose 6000 points
#     index = np.random.choice(global_point.shape[0], 4500)
#     global_point = global_point[index, :]
#
#     global_point = global_point - np.expand_dims(np.mean(global_point, axis=0), 0)
#     max_dist = np.max(np.sqrt(np.sum(global_point ** 2, axis=1)), 0)  # max distance after centering
#     cuda_array = global_point / max_dist
#
#     point_cloud = open3d.geometry.PointCloud()
#     point_cloud.points = open3d.utility.Vector3dVector(global_point)
#
#     open3d.visualization.draw_geometries([point_cloud])
#     time.sleep(0.5)


# ----------------------------------------------------------------------
# vis lidar classification in [checkpoint9_test]
classifier = PointNetDenseCls(k=7)
classifier.load_state_dict(torch.load('model\\seg_model_194_best.pth'))
classifier.eval()

while True:
    global_point = [[0, 0, 0]]
    for i in range(1, 7):
        file_list = os.listdir(os.path.join(path, 'testset_global_scan', str(i)))
        file = np.random.choice(file_list)
        points = np.loadtxt(os.path.join(path, 'testset_global_scan', str(i), file))[:, 0:3]
        global_point = np.concatenate((global_point, points), axis=0)
    # random choose 6000 points
    index = np.random.choice(global_point.shape[0], 4500)
    global_point = global_point[index, :]

    global_point = global_point - np.expand_dims(np.mean(global_point, axis=0), 0)
    max_dist = np.max(np.sqrt(np.sum(global_point ** 2, axis=1)), 0)  # max distance after centering
    cuda_array = global_point / max_dist

    # ----------------------------get tags by point net---------------------------------------------------------
    # print(cuda_array.shape)
    cuda_array = cuda_array.reshape((1, 4500, 3))
    # gpu
    cuda_array = torch.from_numpy(cuda_array).float().transpose(2, 1)
    pred, _, _ = classifier(cuda_array)
    pred = pred.view(-1, 7)
    pred_choice = pred.data.max(1)[1].cpu().numpy()  # return indices of max val, shape = (length, )
    pred_choice = np.reshape(pred_choice, (-1, 1))

    # ----------------------------set color array---------------------------------------------------------
    rgb = [[0, 0, 0],
           [132, 94, 194],
           [214, 93, 177],
           [255, 111, 145],
           [255, 150, 113],
           [255, 199, 95],
           [249, 248, 113]]
    rgb_array = np.zeros((4500, 3))

    for i in range(7):
        rgb_array[np.where(pred_choice[:, 0] == i), :] = rgb[i]
    rgb_array = rgb_array.astype(np.float) / 255.0

    # ----------------------------plot ---------------------------------------------------------

    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(global_point)
    point_cloud.colors = open3d.utility.Vector3dVector(rgb_array)

    open3d.visualization.draw_geometries([point_cloud])
    time.sleep(0.5)