import glob
import os
import sys

try:
    import pygame
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_ESCAPE
except ImportError:
    raise RuntimeError('cannot import pygame package')


try:
    import carla
    import random
    import time
    import numpy as np
    import weakref
    import math
    import queue
    import argparse
except IndexError:
    raise IndexError('cannot import carla reference package')


try:
    # lidar
    import open3d as o3d
    from matplotlib import cm  # color map
except IndexError:
    raise IndexError('cannot import lidar reference package')


try:
    sys.path.append(glob.glob('C:\carla0.9.11\PythonAPI\carla\dist\carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    print(sys.path)
    print(sys.version_info.major, ".", sys.version_info.minor, "  add egg successfully")
except IndexError:
    pass


# fix carla error bug
try:
    sys.path.append(glob.glob('PythonAPI')[0])
except IndexError:
    pass


actor_list = list()

# pygame display
FPS = 45
IMG_WIDTH = 800
IMG_HEIGHT = 600
# surface = None
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


def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))