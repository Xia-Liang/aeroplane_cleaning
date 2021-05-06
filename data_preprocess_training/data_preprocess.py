"""
preprocess the lidar data, 12 classes in total

1. read npy file
2. turn class num
3. random drop mainbody points
4. at least 4500 points

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
    };

write xyz data into points folder

write tags data into labels folder
    turn tags to 0 for other object
    1~11 for airplane segments

write tags data, 12 class

    None, vehicle, others = 0
    CPcutCockpit        = 1
    CPcutDome           = 2
    CPcutEmpennage      = 3
    CPcutEngineLeft     = 4
    CPcutEngineRight    = 5
    CPcutGearFront      = 6
    CPcutGearLeft       = 7
    CPcutGearRight      = 8
    CPcutMainBody       = 9
    CPcutWingLeft       = 10
    CPcutWingRight      = 11
"""

import os
import glob
import re
import numpy as np
import random
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('--mainbody_drop_rate', type=float, default=0.4)
parser.add_argument('--min_points', type=int, default=4500, help='min number of points per file')
parser.add_argument('--debug_mode', type=bool, default=False)

opt = parser.parse_args()
print(opt)


if opt.debug_mode:
    root_path = os.path.join('data', 'debug')
    raw_data_path = os.path.join(root_path, 'raw_data')
    train_data_path = os.path.join(root_path, 'train')
else:
    root_path = os.path.join('data')
    raw_data_path = os.path.join(root_path, 'raw_data')
    train_data_path = os.path.join(root_path, 'train')

file_list = os.listdir(raw_data_path)

for i, file in enumerate(file_list):
    raw_data = np.load(os.path.join(raw_data_path, file))  # xyz, tag

    # generate airplane tags from 1~11
    raw_data[:, 3] = raw_data[:, 3] - 22
    raw_data[raw_data[:, 3] < 0] = 0

    # if data.shape[0] < 4500, multi itself until 4500
    while raw_data.shape[0] < opt.min_points:
        raw_data = np.concatenate((raw_data, raw_data), axis=0)

    # save data as xyzTag, shape = (n points, 4)
    np.save(os.path.join(train_data_path, file), raw_data)

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