"""
sync

rgb and lidar semantic sensor

saving data

Want data corresponding to each other, that is,
    Data from rgb and semLidar should have same name in different folder when collecting at same frame, and different from other frame's data
    Since in carla simulator, sensor listening will not have same timestamp
    Use (BIG_PRIME - current_frame) as name
    Each time running the simulator, BIG_PRIME will reduce by prevprime(n), which returns the prev prime smaller than n
    And save data every 19(also a prime) frame
    [p != q + k * 19 ] will always satisfied when p, q is different odd prime
    Done!

    Put the prime.txt into DataPreprocess folder


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


try:
    # Library functions for prime
    from sympy import prevprime
    # for each simulator, we give a different prime num
    # it's not right to use time.time(), since sensor will not listen at exactly same time
    # rgb and lidar file name = prime num - frame
    # prevprime(n): It returns the prev prime smaller than n.
except ImportError:
    raise ImportError('cannot import config file')


global PRIME


def generate_global_prime():
    global PRIME
    with open('D:\\mb95541\\aeroplane\\data\\prime.txt') as f:
        try:
            PRIME = int(f.read())
        except:
            PRIME = 1000000000000
        PRIME = prevprime(PRIME)
    with open('D:\\mb95541\\aeroplane\\data\\prime.txt', 'w') as f:
        f.write(str(PRIME))


def get_name(frame):
    global PRIME
    return PRIME - frame


def draw_image(surface, image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    surface.blit(image_surface, (0, 0))

    # if image.frame % 19 == 0:
    #     image.save_to_disk('D:\\mb95541\\aeroplane\\data\\rgb\\%d' % get_name(image.frame))


def sem_lidar(data):
    if data.frame % 300 == 0:
        array = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
        array = np.reshape(array, (-1, 6))
        col_index = [0, 1, 2, 5]  # xyz, tags
        array = array[:, col_index]
        print(array.shape, '  ',
              len(array[array[:, 2] > 0]), '  '
              )  # (npoints, 4)


def main():
    generate_global_prime()
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
        spawn_point = carla.Transform(carla.Location(x=250, y=280, z=3),
                                      carla.Rotation(pitch=0.000000, yaw=270.000, roll=0.000000))
        # spawn_point = carla.Transform(carla.Location(x=260, y=315, z=1.5),
        #                               carla.Rotation(pitch=0.000000, yaw=270.000, roll=0.000000))
        vehicle = world.spawn_actor(blueprint_library.filter('mercedesccc')[0], spawn_point)
        # print(blueprint_library.filter('vehicle.*.*'))
        vehicle.set_simulate_physics(True)
        actor_list.append(vehicle)

        rgb_camera = world.spawn_actor(
            generate_rgb_bp(world, blueprint_library),
            carla.Transform(carla.Location(x=1, y=0.0, z=1.8), carla.Rotation(pitch=0, yaw=0.0, roll=0.0)),
            attach_to=vehicle)
        actor_list.append(rgb_camera)

        lidar_sem = world.spawn_actor(
            generate_lidar_sem_bp(world, blueprint_library),
            carla.Transform(carla.Location(x=1, y=0.0, z=1.8), carla.Rotation(pitch=0, yaw=0.0, roll=0.0)),
            attach_to=vehicle)
        # lidar_sem.listen(lambda data: data.save_to_disk('D:\\mb95541\\aeroplane\\data\\lidarSem\\%d' % data.frame))
        actor_list.append(lidar_sem)

        controller = KeyboardControl(vehicle)
        # Create a synchronous mode context.
        with CarlaSyncMode(world, rgb_camera, lidar_sem, fps=FPS) as sync_mode:
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
                sem_lidar(lidar_sem)

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


