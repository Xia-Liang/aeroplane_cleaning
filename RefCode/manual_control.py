"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    A/D          : steer left/right
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down

    L            : toggle next light type
    SHIFT + L    : toggle high beam
    Z/X          : toggle right/left blinker
    I            : toggle interior light

    TAB          : change sensor position
    ` or N       : next sensor
    [1-9]        : change to sensor [1-9]
    G            : toggle radar visualization
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle

    R            : toggle recording images to disk

    CTRL + R     : toggle recording of simulation (replacing any previous)
    CTRL + P     : start replaying last recorded simulation
    CTRL + +     : increments the start time of the replay by 1 second (+SHIFT = 10 seconds)
    CTRL + -     : decrements the start time of the replay by 1 second (+SHIFT = 10 seconds)

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""

try:
    from config import *
except ImportError:
    raise ImportError('cannot import config file')


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================

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
        map = world.get_map()

        # --- weather --- #
        # 1. change weather params: carla.WeatherParameters  https://carla.readthedocs.io/en/0.9.3/python_api_tutorial/
        # 2. set_weather:  https://carla.readthedocs.io/en/0.9.3/python_api/#carlaweatherparameters
        world.set_weather(carla.WeatherParameters.ClearNoon)

        # --- vehicle --- #
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('model3')[0]
        # vehicle property  https://carla.readthedocs.io/en/0.9.3/python_api/#carlaactorblueprint
        vehicle_bp.set_attribute('role_name', 'runner')
        white = '255.0, 255.0, 255.0'
        vehicle_bp.set_attribute('color', white)

        # --- start point --- #
        # 1. spawn point
        spawn_point = carla.Transform(carla.Location(x=300, y=315, z=3),

                                      carla.Rotation(pitch=0.000000, yaw=270.000, roll=0.000000))
        print('spawn_point:', spawn_point)
        # generate the vehicle
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        # set the physics Determines whether an actor will be affected by physics or not
        vehicle.set_simulate_physics(True)
        actor_list.append(vehicle)
        time.sleep(3.0)

        # --- rgb-camera sensor --- #
        # 1. blueprint
        rgb_camera_bp = blueprint_library.find('sensor.camera.rgb')
        # 2. set the attribute of camera
        rgb_camera_bp.set_attribute("image_size_x", "%f" % IMG_WIDTH)  # image width
        rgb_camera_bp.set_attribute("image_size_y", "%f" % IMG_HEIGHT)  # image height
        rgb_camera_bp.set_attribute("fov", "110")  # Horizontal field of view in degrees
        # 3. add camera sensor to the vehicle, put the sensor in the car. rotation y x z
        spawn_point = carla.Transform(carla.Location(x=0.5, y=0.0, z=3),
                                      carla.Rotation(pitch=-15.0, yaw=0.0, roll=0.0))
        camera_rgb = world.spawn_actor(rgb_camera_bp, spawn_point, attach_to=vehicle)
        actor_list.append(camera_rgb)

        # --- controller --- #
        controller = KeyboardControl(vehicle)

        # --- Create a synchronous mode context ---#
        synchronous_fps = 40
        with CarlaSyncMode(world, camera_rgb, fps=synchronous_fps) as sync_mode:
            while True:
                #  quit the while
                if should_quit():
                    return

                # start clock
                clock.tick_busy_loop(synchronous_fps)
                # clock.tick(synchronous_fps)

                # Advance the simulation and wait for the data.
                snapshot, image_rgb = sync_mode.tick(timeout=2.0)

                # Control vehicle by keyboard
                controller.parse_events(clock)

                # frame per second
                fps = round(1.0 / snapshot.timestamp.delta_seconds)

                # Draw the display
                draw_image(display, image_rgb)

                # vehicle velocity
                vehicle_velocity = get_speed(vehicle)

                display.blit(font.render('% 5d FPS (client)' % clock.get_fps(), True, (0, 0, 0)), (8, 10))
                display.blit(font.render('% 5d FPS (server)' % fps, True, (0, 0, 0)), (8, 28))
                display.blit(font.render('% 5d mk/h (velocity)' % vehicle_velocity, True, (0, 0, 0)), (8, 46))
                pygame.display.flip()

    finally:
        print('destroying actors')
        for actor in actor_list:
            actor.destroy()
        pygame.quit()
        print('done.')


if __name__ == '__main__':
    carla_main()
