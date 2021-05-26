"""
real time distance estimate after global scan

need to run test_global_scan.py first

env 0
AirplaneFrontCabin 1
AirplaneRearCabin 2
AirplaneTail 3
AirplaneWing 4
AirplaneEngine 5
AirplaneWheel 6

"""

try:
    from config import *
    from config_control import *
except ImportError:
    raise ImportError('cannot import config file')

try:
    # for each simulator, we give a different prime num
    # it's not right to use time.time(), since sensor will not listen at exactly same time
    # rgb and lidar file name = prime num - frame
    from sympy import prevprime
except ImportError:
    raise ImportError('cannot import config file')

try:
    from model.model import PointNetDenseCls
    import torch
except ImportError:
    raise ImportError('cannot import model')

global PRIME
global GLOBAL_AIRPLANE_LOCATION
global GLOBAL_AIRPLANE_TAG
global GLOBAL_AIRPLANE_INFO


def generate_global_prime():
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


def process_lidar(lidar, estimate_min_distance):
    global GLOBAL_AIRPLANE_INFO
    if lidar.frame % 13 == 0:
        x = lidar.transform.location.x
        y = lidar.transform.location.y
        z = lidar.transform.location.z

        for i in range(0, 7):
            index = np.where(GLOBAL_AIRPLANE_INFO[:, 3] == i)

            temp = GLOBAL_AIRPLANE_INFO[index][:, 0:3]
            if temp.shape[0] != 0:
                temp = temp - [x, y, z]
                distance_array = np.percentile(np.sqrt(np.sum(temp ** 2, axis=1)),
                                               (5, 7.5, 10, 12.5, 15))
                estimate_min_distance[i] = np.average(distance_array)
            else:
                estimate_min_distance[i] = 0

    return estimate_min_distance


def process_sem_lidar(lidar_sem, real_min_distance):
    if lidar_sem.frame % 13 == 0:
        # in semLidar, xyz+cosAngle are float32, objIdx and objTag are int32;
        data_float = np.copy(np.frombuffer(lidar_sem.raw_data, dtype=np.dtype('f4')))
        data_float = np.reshape(data_float, (-1, 6))[:, :3]
        data_int = np.copy(np.frombuffer(lidar_sem.raw_data, dtype=np.dtype('i4')))
        data_int = np.reshape(data_int, (-1, 6))[:, 5:]

        # drop ground
        drop_index = (data_float[:, 2] < -1.6) | (abs(data_float[:, 0]) < 1) | (abs(data_float[:, 1]) < 1)
        data_float = data_float[np.invert(drop_index)]
        data_int = data_int[np.invert(drop_index)]
        # concatenate
        data = np.concatenate((data_float, data_int), axis=1)
        # turn tags 0 for others, 1-6 for plane
        data[:, 3] = data[:, 3] - 33
        data[data[:, 3] < 0] = 0
        # calculate distance
        for i in range(0, 7):
            temp = data[np.where(data[:, 3] == i)][:, 0:3]
            if temp.shape[0] != 0:
                distance_array = np.percentile(np.sqrt(np.sum(temp ** 2, axis=1)),
                                               (5, 7.5, 10, 12.5, 15))
                real_min_distance[i] = np.average(distance_array)
            else:
                real_min_distance[i] = 0

    return real_min_distance


def get_global_airplane_location(path=os.path.join('D:', 'mb95541', 'aeroplane', 'data', 'new', 'testset_global_scan'),
                                 num_sample=10):
    global GLOBAL_AIRPLANE_LOCATION
    GLOBAL_AIRPLANE_LOCATION = [[0, 0, 0]]
    for j in range(num_sample):
        concat_data = [[0, 0, 0]]
        for i in range(1, 7):
            file_list = os.listdir(os.path.join(path, str(i)))
            file = random.choice(file_list)
            data = np.loadtxt(os.path.join(path, str(i), file))
            concat_data = np.concatenate((concat_data, data), axis=0)
        concat_data = concat_data[1:, :]
        # drop outlier (too far from main)

        index = np.random.choice(concat_data.shape[0], 4500)
        GLOBAL_AIRPLANE_LOCATION = np.concatenate((GLOBAL_AIRPLANE_LOCATION, concat_data[index, :]), axis=0)
    GLOBAL_AIRPLANE_LOCATION = GLOBAL_AIRPLANE_LOCATION[1:, :]


def get_global_points_tag(num_sample=10):
    global GLOBAL_AIRPLANE_LOCATION
    global GLOBAL_AIRPLANE_TAG

    classifier = PointNetDenseCls(k=7)
    classifier.load_state_dict(torch.load('model\\seg_model_141_best.pth'))
    classifier.eval()

    GLOBAL_AIRPLANE_TAG = [0]
    for i in range(num_sample):
        temp = np.copy(GLOBAL_AIRPLANE_LOCATION[i*4500:(i+1)*4500, :])

        temp = temp - np.expand_dims(np.mean(temp, axis=0), 0)
        max_dist = np.max(np.sqrt(np.sum(temp ** 2, axis=1)), 0)  # max distance after centering
        cuda_array = temp / max_dist  # scale
        # print(cuda_array.shape)
        cuda_array = cuda_array.reshape((1, 4500, 3))
        # gpu
        cuda_array = torch.from_numpy(cuda_array).float().transpose(2, 1)
        pred, _, _ = classifier(cuda_array)
        pred = pred.view(-1, 7)
        pred_choice = pred.data.max(1)[1].cpu().numpy()  # return indices of max val, shape = (length, )

        GLOBAL_AIRPLANE_TAG = np.concatenate((GLOBAL_AIRPLANE_TAG, pred_choice))
    GLOBAL_AIRPLANE_TAG = np.reshape(GLOBAL_AIRPLANE_TAG[1:], (-1, 1))

    # for i in range(0, 7):
    #     print(len(GLOBAL_AIRPLANE_TAG[GLOBAL_AIRPLANE_TAG == i]))
    # print(GLOBAL_AIRPLANE_LOCATION.shape)
    # print(GLOBAL_AIRPLANE_TAG.shape)


def get_global_points_info(keep_points=10000):
    global GLOBAL_AIRPLANE_LOCATION
    global GLOBAL_AIRPLANE_TAG
    global GLOBAL_AIRPLANE_INFO
    # GLOBAL_AIRPLANE_INFO = np.concatenate((GLOBAL_AIRPLANE_LOCATION, GLOBAL_AIRPLANE_TAG), axis=1)

    temp = np.concatenate((GLOBAL_AIRPLANE_LOCATION, GLOBAL_AIRPLANE_TAG), axis=1)

    # since tag 1-6 is FrontCabin RearCabin Tail Wing Engine Wheel, we random choose points to reduce calculation
    # keep all the wheel (too small)
    GLOBAL_AIRPLANE_INFO = temp[temp[:, 3] == 6]
    # downsampling
    index = np.random.choice(temp.shape[0], keep_points - GLOBAL_AIRPLANE_INFO.shape[0], replace=False)
    GLOBAL_AIRPLANE_INFO = np.concatenate((GLOBAL_AIRPLANE_INFO, temp[index]), axis=0)


def main():
    # ----------------get global points location, do pointnet classification -----------------------------------------
    global GLOBAL_AIRPLANE_LOCATION
    get_global_airplane_location()
    get_global_points_tag()
    get_global_points_info()

    estimate_min_distance = np.ones(7)
    real_min_distance = np.ones(7)

    # ----------------set up pygame------------------------------------------------------------------------------------
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

        # # --- start point --- #
        spawn_point = carla.Transform(carla.Location(x=280, y=320, z=1.8),
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

                estimate = process_lidar(lidar, estimate_min_distance)
                real = process_sem_lidar(lidar_sem, real_min_distance)

                # Draw the display.
                draw_image(display, image_rgb)

                display.blit(font.render('%2d FPS (real), %2d FPS (simulated), %2d km/h (velocity)'
                                         % (clock.get_fps(), fps, vehicle_velocity),
                                         True, (0, 0, 0)),
                             (8, 5))

                display.blit(font.render('Min distance (m) to each part',
                                         True, (0, 0, 0)),
                             (8, 40))

                display.blit(font.render('%-30s %-12s %-12s %-12s %-12s %-12s %-12s'
                                         % ('Part', 'FrontCabin', 'RearCabin', 'Tail', 'Wing', 'Engine', 'Wheel'),
                                         True, (0, 0, 0)),
                             (8, 75))

                display.blit(font.render('%-25s %-12.2f %-12.2f %-12.2f %-12.2f %-12.2f %-12.2f'
                                         % ('Estimate (global point)', estimate[1], estimate[2], estimate[3], estimate[4], estimate[5], estimate[6]),
                                         True, (0, 0, 0)),
                             (8, 100))

                display.blit(font.render('%-25s %-12.2f %-12.2f %-12.2f %-12.2f %-12.2f %-12.2f'
                                         % ('Real (unshaded point)', real[1], real[2], real[3], real[4], real[5], real[6]),
                                         True, (0, 0, 0)),
                             (8, 125))

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
