"""
sync

rgb and rgb semantic sensor

saving data

in ObjectLabel.h, user defined tags
    enum class CityObjectLabel : uint8_t {
        None         =   0u,
        Buildings    =   1u,
        Fences       =   2u,
        Other        =   3u,
        Pedestrians  =   4u,
        Poles        =   5u,
        RoadLines    =   6u,
        Roads        =   7u,
        Sidewalks    =   8u,
        Vegetation   =   9u,
        Vehicles     =  10u,
        Walls        =  11u,
        TrafficSigns =  12u,
        Sky          =  13u,
        Ground       =  14u,
        Bridge       =  15u,
        RailTrack    =  16u,
        GuardRail    =  17u,
        TrafficLight =  18u,
        Static       =  19u,
        Dynamic      =  20u,
        Water        =  21u,
        Terrain      =  22u,
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
        Any          =  0xFF
    };
"""


try:
    from config import *
    from config_control import *
except ImportError:
    raise ImportError('cannot import config file')


def draw_image(surface, image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    surface.blit(image_surface, (0, 0))

    # if image.frame % 10 == 0:
    #     image.save_to_disk('D:\\mb95541\\aeroplane\\data\\rgb\\%d' % image.frame)


def save_lidar(data):
    if data.frame % 10 == 0:
        data.save_to_disk('D:\\mb95541\\aeroplane\\data\\lidarSem\\%d' % data.frame)


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
        # spawn_point = carla.Transform(carla.Location(x=260, y=315, z=3),
        #                               carla.Rotation(pitch=0.000000, yaw=270.000, roll=0.000000))
        spawn_point = carla.Transform(carla.Location(x=250, y=280, z=3),
                                      carla.Rotation(pitch=0.000000, yaw=270.000, roll=0.000000))
        vehicle = world.spawn_actor(random.choice(blueprint_library.filter('vehicle.*')), spawn_point)
        vehicle.set_simulate_physics(True)
        actor_list.append(vehicle)

        rgb_camera = world.spawn_actor(
            generate_rgb_bp(world, blueprint_library),
            carla.Transform(carla.Location(x=-1, y=0.0, z=2.8), carla.Rotation(pitch=0, yaw=0.0, roll=0.0)),
            attach_to=vehicle)
        actor_list.append(rgb_camera)

        lidar_sem = world.spawn_actor(
            generate_lidar_sem_bp(world, blueprint_library),
            carla.Transform(carla.Location(x=-1, y=0.0, z=2.8), carla.Rotation(pitch=0, yaw=0.0, roll=0.0)),
            attach_to=vehicle)
        # lidar_sem.listen(lambda data: data.save_to_disk('D:\\mb95541\\aeroplane\\data\\lidarSem\\%d' % data.frame))
        actor_list.append(lidar_sem)

        controller = KeyboardControl(vehicle)
        # Create a synchronous mode context.
        with CarlaSyncMode(world, rgb_camera, lidar_sem, fps=30) as sync_mode:
            while True:
                if should_quit():
                    return
                clock.tick()
                controller.parse_events(clock)
                # Advance the simulation and wait for the data.
                snapshot, image_rgb, lidar_sem = sync_mode.tick(timeout=2.0)

                fps = round(1.0 / snapshot.timestamp.delta_seconds)
                vehicle_velocity = get_speed(vehicle)

                # Draw the display.
                draw_image(display, image_rgb)
                save_lidar(lidar_sem)
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


