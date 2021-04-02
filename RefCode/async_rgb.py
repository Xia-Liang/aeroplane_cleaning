"""
async mode with rgb camera

save file in D:\mb95541\aeroplane\data\rgb
"""

try:
    from config import *
    from config_control import *
except ImportError:
    raise ImportError('cannot import config file')


def process_rgb(rgb):
    """
    process the image, update surface in pygame
    """
    global surface
    array = np.frombuffer(rgb.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (rgb.height, rgb.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]  # switch r,g,b
    array = array.swapaxes(0, 1)  # exchange the width and height
    surface = pygame.surfarray.make_surface(array)  # Copy an array to a new surface

    # rgb.save_to_disk('D:\\mb95541\\aeroplane\\data\\rgb\\%d' % rgb.frame)


def carla_main():
    # --- pygame show --- #
    pygame.init()
    display = pygame.display.set_mode((IMG_WIDTH, IMG_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()
    try:
        # --- client --- #
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)

        # --- world and blueprint_library --- #
        world = client.get_world()
        settings = world.get_settings()
        settings.synchronous_mode = False  # Disables synchronous mode
        world.apply_settings(settings)
        blueprint_library = world.get_blueprint_library()
        # --- weather --- #
        world.set_weather(carla.WeatherParameters.ClearNoon)

        # --- start point --- #
        spawn_point = carla.Transform(carla.Location(x=250, y=275, z=3),
                                      carla.Rotation(pitch=0.000000, yaw=270.000, roll=0.000000))
        print('spawn_point:', spawn_point)

        # --- vehicle --- #
        vehicle_bp = generate_vehicle_bp(world, blueprint_library)
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        vehicle.set_simulate_physics(True)
        actor_list.append(vehicle)

        # --- rgb-camera sensor --- #
        rgb_camera_bp = generate_rgb_bp(world, blueprint_library)
        rgb_spawn_point = carla.Transform(carla.Location(x=0.5, y=0.0, z=3),
                                      carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0))
        rgb_camera = world.spawn_actor(rgb_camera_bp, rgb_spawn_point, attach_to=vehicle)
        rgb_camera.listen(lambda data: process_rgb(data))
        actor_list.append(rgb_camera)

        # --- controller --- #
        controller = KeyboardControl(vehicle)

        # --- Create a asynchronous mode context ---#
        while True:
            if should_quit():
                return
            clock.tick(60)
            # don't delete ! will crash if surface is None
            if not surface:
                continue
            controller.parse_events(clock)
            vehicle_velocity = get_speed(vehicle)
            display.blit(font.render('% 5d mk/h (velocity)' % vehicle_velocity, True, (0, 0, 0)), (8, 10))
            pygame.display.flip()
            display.blit(surface, (0, 0))

    finally:
        print('destroying actors')
        for actor in actor_list:
            actor.destroy()
        pygame.quit()
        print('done.')


if __name__ == '__main__':
    carla_main()
