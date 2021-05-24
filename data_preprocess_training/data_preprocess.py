"""
preprocess the lidar data, 12 classes in total

1. read npy file
2. turn class num
3. generate global points and random drop points
4. at least 4500 points
5. do scaling by tags and rotation for whole

in ObjectLabel.h, user defined tags
    enum class CityObjectLabel : uint8_t {
        None         =   0u,
        Vehicle = 10u,
        CPcutCockpit = 23u,
        CPcutDome = 24u,
        CPcutEmpennage = 25u,
        CPcutEngineLeft = 26u,
        CPcutEngineRight = 27u,
        CPcutGearFront = 28u,
        CPcutGearLeft = 29u,
        CPcutGearRight = 30u,
        CPcutMainBody = 31u,
        CPcutWingLeft = 32u,
        CPcutWingRight = 33u,
        AirplaneFrontCabin = 34u,
        AirplaneRearCabin = 35u,
        AirplaneTail = 36u,
        AirplaneWing = 37u,
        AirplaneEngine = 38u,
        AirplaneWheel = 39u,
    };

write xyz data into points folder

write tags data into labels folder
    turn tags to 0 for other object
    1~6 for airplane segments

write tags data, 7 class

    None, vehicle, others = 0
    AirplaneFrontCabin        = 1
    AirplaneRearCabin           = 2
    AirplaneTail      = 3
    AirplaneWing     = 4
    AirplaneEngine    = 5
    AirplaneWheel      = 6
"""

import os
import glob
import re
import numpy as np
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--min_points', type=int, default=4500, help='min number of points per file')
parser.add_argument('--basic_data_size', type=int, default=500, help='basic file of ground truth scanning')
parser.add_argument('--train_data_size', type=int, default=10000, help='num of training data for pointnet')
parser.add_argument('--num_tags', type=int, default=7, help='num of tags in training set')

opt = parser.parse_args()
print(opt)

root_path = os.path.join('data')
raw_data_path = os.path.join(root_path, 'raw_data')
cleaned_data_path = os.path.join(root_path, 'cleaned_data')
train_data_path = os.path.join(root_path, 'train')

folder_list = [item for item in os.listdir(raw_data_path) if 'checkpoint' in item]
for i in range(opt.basic_data_size):
    global_point_set = [[0, 0, 0, 0]]
    for folder in folder_list:
        # random choose one file in each checkpoint file, concatenate them and generate global pointset
        file = random.choice(os.listdir(os.path.join(raw_data_path, folder)))
        points = np.loadtxt(os.path.join(raw_data_path, folder, file))
        # turn tags 0 for others, 1-6 for plane
        points[:, 3] = points[:, 3] - 33
        points[points[:, 3] < 0] = 0
        global_point_set = np.concatenate((global_point_set, points), axis=0)
    # print(global_point_set.shape)
    index = np.random.choice(global_point_set.shape[0], opt.min_points)
    global_point_set = global_point_set[index, :]
    np.save(os.path.join(cleaned_data_path, str(i)), global_point_set)

file_list = os.listdir(cleaned_data_path)

for i in range(opt.train_data_size):
    file = random.choice(file_list)
    data = np.load(os.path.join(cleaned_data_path, file))
    # scaling
    scaling_index = [random.randrange(80, 120, 1) / 100 for i in range(opt.num_tags)]  # 0.8 ~ 1.2 scaling
    for tag in range(0, opt.num_tags):  # 7 tags in total
        data[data[:, 3] == tag][:, 0:3] *= scaling_index[tag]  # for each tag, scalling xyz for all points
    # rotation,
    rotation_theta = [np.random.uniform(0, np.pi / 12),
                             np.random.uniform(0, np.pi / 12),
                             np.random.uniform(0, np.pi * 2)]  # x,y for 15, z for 360
    rotation_x = np.array([[1, 0, 0],
                           [0, np.cos(rotation_theta[0]), -np.sin(rotation_theta[0])],
                           [0, np.sin(rotation_theta[0]), np.cos(rotation_theta[0])]])
    rotation_y = np.array([[np.cos(rotation_theta[1]), 0, -np.sin(rotation_theta[1])],
                           [0, 1, 0],
                           [np.sin(rotation_theta[1]), 0, np.cos(rotation_theta[1])]])
    rotation_z = np.array([[np.cos(rotation_theta[2]), -np.sin(rotation_theta[2]), 0],
                           [np.sin(rotation_theta[2]), np.cos(rotation_theta[2]), 0],
                           [0, 0, 1]])
    data[:, [0, 1, 2]] = data[:, [0, 1, 2]].dot(rotation_x).dot(rotation_y).dot(rotation_z)
    np.save(os.path.join(train_data_path, str(i)), data)


# for i, file in enumerate(file_list):
#     raw_data = np.load(os.path.join(raw_data_path, file))  # xyz, tag
#
#     # generate airplane tags from 1~11
#     raw_data[:, 3] = raw_data[:, 3] - 33
#     raw_data[raw_data[:, 3] < 0] = 0
#
#     # if data.shape[0] < 4500, multi itself until 4500
#     while raw_data.shape[0] < opt.min_points:
#         raw_data = np.concatenate((raw_data, raw_data), axis=0)
#
#     # save data as xyzTag, shape = (n points, 4)
#     np.save(os.path.join(train_data_path, file), raw_data)

# original file
# # read train data
# with open(os.path.join(raw_data_path, file)) as f:
#     lines = f.read().splitlines()[10:]
#     # print(len(list(filter(lambda x: x[-2:] == '10', lines))))
#
#     # collect all the valid data (only airplane)
#     data = list(filter(lambda x: (x[-2:] != '10' and x[-2:] != ' 0'), lines))
#     # continue if valid data is too small
#     if len(data) < opt.min_valid_data:
#         continue
#
#     # collect all scene and vehicle data, random drop
#     data_scene = list(filter(lambda x: x[-2:] == ' 0', lines))
#     data_vehicle = list(filter(lambda x: x[-2:] == '10', lines))
#     data.extend(random.choices(data_scene,
#                                k=int(len(data_scene) * (1 - opt.scene_drop_percent))))
#     data.extend(random.choices(data_vehicle,
#                                k=int(len(data_vehicle) * (1 - opt.vehicle_drop_percent))))
#     # print(len(data))
#
#     # if data is still too small, add scene points
#     while len(data) < opt.min_data_lines:
#         data.extend(random.choices(data_scene, k=100))
#     # print('         ', len(data))
#
#     # write xyz data
#     with open(os.path.join(root_path, 'point', file), 'wb') as points:
#         reg = r'^(\S*\s){3}'  # return 'x y z '
#         for line in data:
#             points.write((re.match(reg, line).group() + '\n').encode())
#     # write label data
#     with open(os.path.join(root_path, 'label', file + '.seg'), 'wb') as labels:
#         for line in data:
#             labels.write((airplane_label(line.split(' ')[5]) + '\n').encode())  # return 'tags'
