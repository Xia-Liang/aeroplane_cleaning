"""
generate data for training set, saving point of World coordinate system

Since sem-lidar and lidar(set in test_realtime.py) have z = 1.8
we will drop points near ground z < -1.7
and points very close to sensor abs(x) < 1, abs(y) < 1

(sem)lidar can not get whole info of plane
and pointnet performance bad
so we combine the data from six different place to generate a complete 3d scanning of plane

for each (sem)lidar location (x, y, z) and rotation (pitch, yaw, roll)
capture a point P_n (x_n, y_n, z_n) relative to (sem)lidar
so the P_n location in World coordinate system should be :

X = x + x_n * cos(yaw) + y_n * sin(yaw)
Y = y + y_n * sin(yaw) + y_n * cos(yaw)
Z = z + z_n

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

"""
func: generate_global_prime()
    Want data corresponding to each other, that is,
        Data from rgb and semLidar should have same name in different folder when collecting at same frame, and different from other frame's data
        Since in carla simulator, sensor listening will not have same timestamp
        Use (BIG_PRIME - current_frame) as name
        Each time running the simulator, BIG_PRIME will reduce by prevprime(n), which returns the prev prime smaller than n
        And save data every 19(also a prime) frame
        [p != q + k * 19 ] will always satisfied when p, q is different odd prime
        Done!

        Put the prime.txt into DataPreprocess folder
"""
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


def save_lidar(lidar):
    # just for test, in training model, 4500 points is suitable
    if lidar.frame % 3 == 0:
        data = np.copy(np.frombuffer(lidar.raw_data, dtype=np.dtype('f4')))
        data = np.reshape(data, (-1, 4))[:, :3]

        # drop ground
        drop_index = (data[:, 2] < -1.6) | (abs(data[:, 0]) < 1) | (abs(data[:, 1]) < 1)
        data = data[np.invert(drop_index)]

        # print('lidar: ', data.shape, end=' ')


def save_sem_lidar(lidar_sem, cases):
    if lidar_sem.frame % 13 == 0:
        # lidar_sem.save_to_disk('D:\\mb95541\\aeroplane\\data\\lidarSem\\%d_raw' % get_name(lidar_sem.frame))

        # in semLidar, xyz+cosAngle are float32, objIdx and objTag are int32;
        data_float = np.copy(np.frombuffer(lidar_sem.raw_data, dtype=np.dtype('f4')))
        data_float = np.reshape(data_float, (-1, 6))[:, :3]

        data_int = np.copy(np.frombuffer(lidar_sem.raw_data, dtype=np.dtype('i4')))
        data_int = np.reshape(data_int, (-1, 6))[:, 5:]

        # drop ground
        drop_index = (data_float[:, 2] < -1.6) | (abs(data_float[:, 0]) < 1) | (abs(data_float[:, 1]) < 1)
        data_float = data_float[np.invert(drop_index)]
        data_int = data_int[np.invert(drop_index)]

        # get world coordinate location
        global_x = lidar_sem.transform.location.x
        global_y = lidar_sem.transform.location.y
        global_z = lidar_sem.transform.location.z
        global_yaw = lidar_sem.transform.rotation.yaw / 180 * np.pi  # turn to pi

        global_location = np.concatenate(
            (data_float[:, 0:1] * np.cos(global_yaw) - data_float[:, 1:2] * np.sin(global_yaw),
             data_float[:, 0:1] * np.sin(global_yaw) + data_float[:, 1:2] * np.cos(global_yaw),
             data_float[:, 2:3]), axis=1)
        global_location = global_location + [global_x, global_y, global_z]

        data = np.concatenate((global_location, data_int), axis=1)  # xyz, tag

        print(data.shape, end=' ')
        print('location', global_x, global_y, global_z, global_yaw)

        if cases == 1:
            np.savetxt('D:\\mb95541\\aeroplane\\data\\new\\checkpoint1\\%d' % get_name(lidar_sem.frame), data)
        elif cases == 2:
            np.savetxt('D:\\mb95541\\aeroplane\\data\\new\\checkpoint2\\%d' % get_name(lidar_sem.frame), data)
        elif cases == 3:
            np.savetxt('D:\\mb95541\\aeroplane\\data\\new\\checkpoint3\\%d' % get_name(lidar_sem.frame), data)
        elif cases == 4:
            np.savetxt('D:\\mb95541\\aeroplane\\data\\new\\checkpoint4\\%d' % get_name(lidar_sem.frame), data)
        elif cases == 5:
            np.savetxt('D:\\mb95541\\aeroplane\\data\\new\\checkpoint5\\%d' % get_name(lidar_sem.frame), data)
        elif cases == 6:
            np.savetxt('D:\\mb95541\\aeroplane\\data\\new\\checkpoint6\\%d' % get_name(lidar_sem.frame), data)
        elif cases == 7:
            np.savetxt('D:\\mb95541\\aeroplane\\data\\new\\checkpoint7\\%d' % get_name(lidar_sem.frame), data)
        elif cases == 8:
            np.savetxt('D:\\mb95541\\aeroplane\\data\\new\\checkpoint8\\%d' % get_name(lidar_sem.frame), data)

        # np.save('D:\\mb95541\\aeroplane\\data\\new\\checkpoint1\\%d' % get_name(lidar_sem.frame), data)

        # print('semLidar: ', data.shape)
        #
        # for i in range(34, 40):
        #     print(len(data[data[:, 3] == i]), end=' ')
        # print(' ')


def main():
    generate_global_prime()
    pygame.init()
    display = pygame.display.set_mode((IMG_WIDTH, IMG_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()

    cases = 1
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()

        # # --- start point --- #
        if cases == 1:
            spawn_point = carla.Transform(carla.Location(x=280, y=320, z=1.8),
                                          carla.Rotation(pitch=0.000000, yaw=270.000, roll=0.000000))
        elif cases == 2 or cases == 7:
            spawn_point = carla.Transform(carla.Location(x=260, y=290, z=1.8),
                                          carla.Rotation(pitch=0.000000, yaw=270.000, roll=0.000000))
        elif cases == 3:
            spawn_point = carla.Transform(carla.Location(x=260, y=240, z=1.8),
                                          carla.Rotation(pitch=0.000000, yaw=270.000, roll=0.000000))
        elif cases == 4:
            spawn_point = carla.Transform(carla.Location(x=280, y=210, z=1.8),
                                          carla.Rotation(pitch=0.000000, yaw=270.000, roll=0.000000))
        elif cases == 5:
            spawn_point = carla.Transform(carla.Location(x=300, y=240, z=1.8),
                                          carla.Rotation(pitch=0.000000, yaw=270.000, roll=0.000000))
        elif cases == 6 or cases == 8:
            spawn_point = carla.Transform(carla.Location(x=300, y=290, z=1.8),
                                          carla.Rotation(pitch=0.000000, yaw=270.000, roll=0.000000))

        vehicle = world.spawn_actor(blueprint_library.filter('mercedesccc')[0], spawn_point)
        vehicle.set_simulate_physics(True)
        actor_list.append(vehicle)

        rgb_camera = world.spawn_actor(
            generate_rgb_bp(world, blueprint_library),
            carla.Transform(carla.Location(x=1, y=0.0, z=1.75), carla.Rotation(pitch=0, yaw=0.0, roll=0.0)),
            attach_to=vehicle)
        actor_list.append(rgb_camera)

        lidar = world.spawn_actor(
            generate_lidar_bp(world, blueprint_library),
            carla.Transform(carla.Location(x=1, y=0.0, z=1.8), carla.Rotation(pitch=0, yaw=0.0, roll=0.0)),
            attach_to=vehicle)
        actor_list.append(lidar)

        lidar_sem = world.spawn_actor(
            generate_lidar_sem_bp(world, blueprint_library),
            carla.Transform(carla.Location(x=1, y=0.0, z=1.8), carla.Rotation(pitch=0, yaw=0.0, roll=0.0)),
            attach_to=vehicle)
        actor_list.append(lidar_sem)

        controller = KeyboardControl(vehicle)
        # Create a synchronous mode context.
        with CarlaSyncMode(world, rgb_camera, lidar, lidar_sem, fps=30) as sync_mode:
            while True:
                if should_quit():
                    return
                clock.tick()
                controller.parse_events(clock)

                # Advance the simulation and wait for the data.
                snapshot, image_rgb, lidar, lidar_sem = sync_mode.tick(timeout=2.0)

                fps = round(1.0 / snapshot.timestamp.delta_seconds)
                vehicle_velocity = get_speed(vehicle)

                draw_image(display, image_rgb)
                save_lidar(lidar)
                save_sem_lidar(lidar_sem, cases)

                # Draw the display.
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
