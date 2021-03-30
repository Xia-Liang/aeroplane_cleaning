try:
    from config import *
    from manual_control import *
except ImportError:
    raise ImportError('cannot import config file')

# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


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
        spawn_point = carla.Transform(carla.Location(x=136, y=315, z=3),
                                      carla.Rotation(pitch=0.000000, yaw=270.000, roll=0.000000))
        print('spawn_point:', spawn_point)
        # generate the vehicle
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        # set the physics Determines whether an actor will be affected by physics or not
        vehicle.set_simulate_physics(True)
        actor_list.append(vehicle)
        # time.sleep(3.0)

        # --- rgb-camera sensor --- #
        # 1. blueprint
        rgb_camera_bp = blueprint_library.find('sensor.camera.rgb')
        # 2. set the attribute of camera
        rgb_camera_bp.set_attribute("image_size_x", "%f" % IMG_WIDTH)  # image width
        rgb_camera_bp.set_attribute("image_size_y", "%f" % IMG_HEIGHT)  # image height
        rgb_camera_bp.set_attribute("fov", "110")  # Horizontal field of view in degrees
        rgb_camera_bp.set_attribute("sensor_tick", "0.05") # Simulation seconds between sensor captures (ticks).
        # 3. add camera sensor to the vehicle, put the sensor in the car. rotation y x z
        spawn_point = carla.Transform(carla.Location(x=0.5, y=0.0, z=3),
                                      carla.Rotation(pitch=-15.0, yaw=0.0, roll=0.0))
        camera_rgb = world.spawn_actor(rgb_camera_bp, spawn_point,  attach_to=vehicle)

        actor_list.append(camera_rgb)

        # --- controller --- #
        controller = KeyboardControl(vehicle)

        # --- Create a synchronous mode context ---#
        synchronous_fps = 20
        with CarlaSyncMode(world, camera_rgb, fps=synchronous_fps) as sync_mode:
            while True:
                #  quit the while
                if should_quit():
                    return
                # start clock
                clock.tick()
                # clock.tick_busy_loop(synchronous_fps)

                # Advance the simulation and wait for the data.
                snapshot, image_rgb = sync_mode.tick(timeout=20)

                fps = round(1.0 / snapshot.timestamp.delta_seconds)
                # Control vehicle by keyboard
                # controller.parse_events(clock)
                controller.parse_events(clock)

                # Draw the display
                draw_image(display, image_rgb)

                # save_rgb(display, image_rgb)

                # vehicle velocity
                vehicle_velocity = get_speed(vehicle)

                display.blit(font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)), (8, 10))
                display.blit(font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),(8, 28))
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
