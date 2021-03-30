import glob
import os
import sys


# fix carla error bug
# try:
#     sys.path.append(glob.glob('PythonAPI')[0])
# except ImportError:
#     raise ImportError('cannot import carla PythonAPI')
#

try:
    sys.path.append(glob.glob('C:\\carla0.9.11\\PythonAPI\\carla\\dist\\carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    print(sys.path)
    print(sys.version_info.major, ".", sys.version_info.minor, "  add egg successfully")
except ImportError:
    raise ImportError('cannot import carla egg file')


try:
    import pygame
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_ESCAPE
except ImportError:
    raise ImportError('cannot import pygame package')


try:
    import carla
    import random
    import time
    import numpy as np
    import weakref
    import math
    import queue
    import argparse
except ImportError:
    raise ImportError('cannot import carla reference package')


try:
    # lidar
    import open3d as o3d
    from matplotlib import cm  # color map
except ImportError:
    raise ImportError('cannot import open3d or matplotlib package')


actor_list = list()

# pygame display
IMG_WIDTH = 800
IMG_HEIGHT = 600
surface = None
SHOW_CAM = True


# carla sync mode
class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', FPS)
        self._queues = []
        self._settings = None

    def __enter__(self):
        # some data about the simulation such as synchrony between client and server or rendering mode
        self._settings = self.world.get_settings()
        # ---- This is important carla.WorldSettings
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        # This method only has effect on synchronous mode, when both client and server move together.
        # The client tells the server when to step to the next frame and returns the id of the newly started frame.
        self.frame = self.world.tick()
        # get the data synchronous data
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            # client timestamp == world timestamp
            if data.frame == self.frame:
                return data


# for pygame display
def get_font():
    """
    show font
    :return:
    """
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 24)


def get_speed(vehicle):
    """
    Compute speed of a vehicle in Km/h.

        :param vehicle: the vehicle for which speed is calculated
        :return: speed as a float in Km/h
    """
    vel = vehicle.get_velocity()

    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)


def generate_vehicle_bp(world, blueprint_library):
    vehicle_bp = blueprint_library.filter('model3')[0]
    vehicle_bp.set_attribute('role_name', 'runner')
    white = '255.0, 255.0, 255.0'
    vehicle_bp.set_attribute('color', white)
    return vehicle_bp


def generate_rgb_bp(world, blueprint_library):
    rgb_camera_bp = blueprint_library.find('sensor.camera.rgb')
    rgb_camera_bp.set_attribute("image_size_x", "%f" % IMG_WIDTH)  # image width
    rgb_camera_bp.set_attribute("image_size_y", "%f" % IMG_HEIGHT)  # image height
    rgb_camera_bp.set_attribute("fov", "110")  # Horizontal field of view in degrees
    rgb_camera_bp.set_attribute("sensor_tick", "0.05")
    return rgb_camera_bp


def generate_rgb_sem_bp(world, blueprint_library):
    rgb_sem_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
    rgb_sem_bp.set_attribute("image_size_x", "%f" % IMG_WIDTH)  # image width
    rgb_sem_bp.set_attribute("image_size_y", "%f" % IMG_HEIGHT)  # image height
    rgb_sem_bp.set_attribute("fov", "110")  # Horizontal field of view in degrees
    rgb_sem_bp.set_attribute("sensor_tick", "0.05")
    return rgb_sem_bp

