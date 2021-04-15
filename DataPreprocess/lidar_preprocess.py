"""
preprocess the lidar data

in ObjectLabel.h, user defined tags
    enum class CityObjectLabel : uint8_t {
        None         =   0u,
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

write tags data inot labels folder
    turn tags to 0 for other object
    1~11 for airplane segments

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


def relu_airplane_label(x):
    return str(max(0, int(x) - 22))


file_name_list = os.listdir('D:\\mb95541\\aeroplane\\data\\lidarSem')

for i in range(0, len(file_name_list)):
    # read raw data
    with open('D:\\mb95541\\aeroplane\\data\\lidarSem\\' + file_name_list[i]) as f:
        lines = f.read().splitlines()[10:]
        # write xyz data
        with open('D:\\mb95541\\aeroplane\\data\\pointnet\\points\\' + file_name_list[i], 'w') as points:
            reg = r'^(\S*\s){3}'  # return 'x y z '
            for line in lines:
                points.write(re.match(reg, line).group() + '\n')
        # write label data
        with open('D:\\mb95541\\aeroplane\\data\\pointnet\\labels\\' + file_name_list[i] + '.seg', 'w') as labels:
            for line in lines:
                labels.write(relu_airplane_label(line.split(' ')[5]) + '\n')  # return 'tags'
