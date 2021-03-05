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


class KeyboardControl(object):
    """Class that handles keyboard input."""

    def __init__(self, player):

        self._control = carla.VehicleControl()
        self._steer_cache = 0.0
        self.player = player

    def parse_events(self, clock):
        self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
        self.player.apply_control(self._control)

    def _parse_vehicle_keys(self, keys, milliseconds):
        if keys[K_w]:
            self._control.throttle = min(self._control.throttle + 0.1, 0.5)
        else:
            self._control.throttle = 0.0
            # fix the velocity
            # self._control.throttle = 0.40

        if keys[K_s]:
            self._control.brake = min(self._control.brake + 0.2, 1)
        else:
            self._control.brake = 0

        steer_increment = 5e-4 * milliseconds
        if keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 7)


def should_quit():
    """
    stop event
    :return:
    """
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


actor_list = list()


FPS = 20
IMG_WIDTH = 800
IMG_HEIGHT = 600
# surface = None
SHOW_CAM = True


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


surface = None


def process_img(image):
    """
    process the image
    """
    global surface
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1] # switch r,g,b
    array = array.swapaxes(0, 1)  # exchange the width and height
    surface = pygame.surfarray.make_surface(array)   # Copy an array to a new surface

    image_name = image.frame
    image.save_to_disk('D:\\mb95541\\aeroplane\\image\\%d' % image_name)


def get_speed(vehicle):
    """
    Compute speed of a vehicle in Km/h.

        :param vehicle: the vehicle for which speed is calculated
        :return: speed as a float in Km/h
    """
    vel = vehicle.get_velocity()

    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)


def carla_main():
    # --- pygame show --- #
    pygame.init()
    display = pygame.display.set_mode((IMG_WIDTH, IMG_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()
    try:
        # --- client --- #
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)  # connect with server

        # --- world --- #
        world = client.get_world()
        settings = world.get_settings()
        settings.synchronous_mode = False  # Disables synchronous mode
        world.apply_settings(settings)
        # map = world.get_map()

        # --- weather --- #
        # 1. change weather params: carla.WeatherParameters  https://carla.readthedocs.io/en/0.9.3/python_api_tutorial/
        # 2. set_weather:  https://carla.readthedocs.io/en/0.9.3/python_api/#carlaweatherparameters
        # world.set_weather(carla.WeatherParameters.ClearNoon)

        # --- vehicle --- #
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('model3')[0]
        # vehicle property  https://carla.readthedocs.io/en/0.9.3/python_api/#carlaactorblueprint
        vehicle_bp.set_attribute('role_name', 'runner')
        white = '255.0, 255.0, 255.0'
        vehicle_bp.set_attribute('color', white)

        # --- start point --- #
        # 1. spawn point
        spawn_point = carla.Transform(carla.Location(x=136, y=315, z=3),
                                      carla.Rotation(pitch=0.000000, yaw=270.000, roll=0.000000))
        print('spawn_point:', spawn_point)
        # generate the vehicle
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        print("vehicle is spawned!")
        # set the physics Determines whether an actor will be affected by physics or not
        vehicle.set_simulate_physics(True)
        actor_list.append(vehicle)
        # time.sleep(3.0)

        # --- rgb-camera sensor --- #
        # 1. blueprint
        rgb_camera_bp = blueprint_library.find('sensor.camera.rgb')
        # # 2. set the attribute of camera
        rgb_camera_bp.set_attribute("image_size_x", "%f" % IMG_WIDTH)  # image width
        rgb_camera_bp.set_attribute("image_size_y", "%f" % IMG_HEIGHT)  # image height
        rgb_camera_bp.set_attribute("fov", "110")  # Horizontal field of view in degrees
        rgb_camera_bp.set_attribute("sensor_tick", "0.05") # Simulation seconds between sensor captures (ticks).
        # # 3. add camera sensor to the vehicle, put the sensor in the car. rotation y x z
        spawn_point = carla.Transform(carla.Location(x=0.5, y=0.0, z=3),
                                      carla.Rotation(pitch=-15.0, yaw=0.0, roll=0.0))
        camera_rgb = world.spawn_actor(rgb_camera_bp, spawn_point,  attach_to=vehicle)

        camera_rgb.listen(lambda data: process_img(data))

        actor_list.append(camera_rgb)
        #
        # # --- controller --- #
        controller = KeyboardControl(vehicle)
        #
        # # --- Create a synchronous mode context ---#
        while True:
            if should_quit():
                return
            clock.tick(30)
            if (not surface):
                continue
        #     # Control vehicle by keyboard
            controller.parse_events(clock)
            vehicle_velocity = get_speed(vehicle)
            print(vehicle_velocity)
        #
            display.blit(font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)), (8, 10))
            # display.blit(font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),(8, 28))
            display.blit(font.render('% 5d mk/h (velocity)' % vehicle_velocity, True, (0, 0, 0)), (8, 46))
        #
            # time.sleep(10)
            pygame.display.flip()
            display.blit(surface, (0,0))

    finally:
        print('destroying actors')
        for actor in actor_list:
            actor.destroy()
        pygame.quit()
        print('done.')


if __name__ == '__main__':
    carla_main()
