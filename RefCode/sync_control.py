"""
first edition for sync mode control

rgb and rgb_sem sensor

overlay display in pygame window

not saving
"""
try:
    from config import *
    from config_control import *
except ImportError:
    raise ImportError('cannot import config file')


def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    # https://www.pygame.org/docs/ref/surface.html?highlight=set_alpha#pygame.Surface.set_alpha
    # The alpha value is an integer from 0 to 255, 0 is fully transparent and 255 is fully opaque.
    if blend:
        image_surface.set_alpha(200)
    surface.blit(image_surface, (0, 0))


def main():
    pygame.init()
    display = pygame.display.set_mode((IMG_WIDTH, IMG_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()

    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()

        # --- start point --- #
        spawn_point = carla.Transform(carla.Location(x=260, y=315, z=3),
                                      carla.Rotation(pitch=0.000000, yaw=270.000, roll=0.000000))

        vehicle = world.spawn_actor(random.choice(blueprint_library.filter('vehicle.*')), spawn_point)
        vehicle.set_simulate_physics(True)
        actor_list.append(vehicle)

        rgb_camera = world.spawn_actor(
            generate_rgb_bp(world, blueprint_library),
            carla.Transform(carla.Location(x=0, y=0.0, z=2.8), carla.Rotation(pitch=0, yaw=0.0, roll=0.0)),
            attach_to=vehicle)
        actor_list.append(rgb_camera)

        rgb_sem = world.spawn_actor(
            generate_rgb_sem_bp(world, blueprint_library),
            carla.Transform(carla.Location(x=0, y=0.0, z=2.8), carla.Rotation(pitch=0, yaw=0.0, roll=0.0)),
            attach_to=vehicle)
        actor_list.append(rgb_sem)

        controller = KeyboardControl(vehicle)
        # Create a synchronous mode context.
        with CarlaSyncMode(world, rgb_camera, rgb_sem, fps=30) as sync_mode:
            while True:
                if should_quit():
                    return
                clock.tick()
                controller.parse_events(clock)
                # Advance the simulation and wait for the data.
                snapshot, image_rgb, image_sem = sync_mode.tick(timeout=2.0)
                image_sem.convert(carla.ColorConverter.CityScapesPalette)

                fps = round(1.0 / snapshot.timestamp.delta_seconds)
                vehicle_velocity = get_speed(vehicle)

                # Draw the display.
                draw_image(display, image_rgb)
                draw_image(display, image_sem, blend=True)
                display.blit(font.render('% 5d FPS (real)' % clock.get_fps(), True, (0, 0, 0)), (8, 10))
                display.blit(font.render('% 5d FPS (simulated)' % fps, True, (0, 0, 0)), (8, 28))
                display.blit(font.render('% 5d mk/h (velocity)' % vehicle_velocity, True, (0, 0, 0)), (8, 46))
                pygame.display.flip()

    finally:
        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()
        pygame.quit()
        print('done.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n Cancelled by user. Bye!')
