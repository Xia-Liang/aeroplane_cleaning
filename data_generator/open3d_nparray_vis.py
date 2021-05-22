import numpy as np
import open3d
import os
import time

path = os.path.join('D:\\mb95541\\aeroplane\\data\\new')

# # ----------------------------------------------------------------------
# # vis file for each file
# file_list = os.listdir(os.path.join(path, 'vis'))
# for file in file_list:
#     points = np.load(os.path.join(path, file))[:, 0:3]
#     point_cloud = open3d.geometry.PointCloud()
#     point_cloud.points = open3d.utility.Vector3dVector(points)
#     open3d.visualization.draw_geometries([point_cloud])
#     print(file)
#     # esc to close current and vis next

# # ----------------------------------------------------------------------
# # vis 6 (checkpoints) file together 1-6
# while True:
#     global_point = [[0, 0, 0]]
#     for i in range(1, 7):
#         file_list = os.listdir(os.path.join(path, 'checkpoint' + str(i)))
#         file = np.random.choice(file_list)
#         points = np.loadtxt(os.path.join(path, 'checkpoint' + str(i), file))[:, 0:3]
#         global_point = np.concatenate((global_point, points), axis=0)
#         # random choose 6000 points
#         index = np.random.choice(global_point.shape[0], 6000)
#         global_point = global_point[index, :]
#     point_cloud = open3d.geometry.PointCloud()
#     point_cloud.points = open3d.utility.Vector3dVector(global_point)
#     open3d.visualization.draw_geometries([point_cloud])
#     time.sleep(0.5)

# ----------------------------------------------------------------------
# vis 2 (checkpoints) file together 7-8, go straight, different location, no turning around
while True:
    global_point = [[0, 0, 0]]
    for i in range(7, 9):
        file_list = os.listdir(os.path.join(path, 'checkpoint' + str(i)))
        # choose 3 file
        file = np.random.choice(file_list)
        points = np.loadtxt(os.path.join(path, 'checkpoint' + str(i), file))[:, 0:3]
        global_point = np.concatenate((global_point, points), axis=0)

        file = np.random.choice(file_list)
        points = np.loadtxt(os.path.join(path, 'checkpoint' + str(i), file))[:, 0:3]
        global_point = np.concatenate((global_point, points), axis=0)

        file = np.random.choice(file_list)
        points = np.loadtxt(os.path.join(path, 'checkpoint' + str(i), file))[:, 0:3]
        global_point = np.concatenate((global_point, points), axis=0)

    # random choose 6000 points
    index = np.random.choice(global_point.shape[0], 4000)
    global_point = global_point[index, :]
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(global_point)
    open3d.visualization.draw_geometries([point_cloud])
    time.sleep(0.5)

