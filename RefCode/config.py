import glob
import os
import sys

import carla  # import from *.egg
import pygame
import random
import time
import numpy as np
import weakref
import math
import queue

try:
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_ESCAPE
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    sys.path.append(glob.glob('C:\carla\carla\PythonAPI\carla\dist\carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    print("add egg successfully")
except IndexError:
    pass


actor_list = list()


FPS = 20
IMG_WIDTH = 800
IMG_HEIGHT = 600
# surface = None
SHOW_CAM = True
