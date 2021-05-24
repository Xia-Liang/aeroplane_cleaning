"""
real time distance estimate after global scan

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
    # Library functions for prime
    from sympy import prevprime
    # for each simulator, we give a different prime num
    # it's not right to use time.time(), since sensor will not listen at exactly same time
    # rgb and lidar file name = prime num - frame
    # prevprime(n): It returns the prev prime smaller than n.
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
    if lidar.frame % 13 == 0:
        data = np.copy(np.frombuffer(lidar.raw_data, dtype=np.dtype('f4')))
        data = np.reshape(data, (-1, 4))[:, :3]

        # drop ground
        drop_index = (data[:, 2] < -1.6) | (abs(data[:, 0]) < 1) | (abs(data[:, 1]) < 1)
        data = data[np.invert(drop_index)]


def save_sem_lidar(lidar_sem):
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


def get_global_airplane_location(path=os.path.join('D:', 'mb95541', 'aeroplane', 'data', 'new', 'testset_global_scan'),
                                 num_sample=20):
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
        index = np.random.choice(concat_data.shape[0], 4500)
        GLOBAL_AIRPLANE_LOCATION = np.concatenate((GLOBAL_AIRPLANE_LOCATION, concat_data[index, :]), axis=0)
    GLOBAL_AIRPLANE_LOCATION = GLOBAL_AIRPLANE_LOCATION[1:, :]


def get_global_points_tag(num_sample=20):
    classifier = PointNetDenseCls(k=7)
    classifier.load_state_dict(torch.load('model\\seg_model_283_best.pth'))
    classifier.eval()

    global GLOBAL_AIRPLANE_LOCATION
    global GLOBAL_AIRPLANE_TAG
    GLOBAL_AIRPLANE_TAG = [0]
    for i in range(num_sample):
        temp = np.copy(GLOBAL_AIRPLANE_LOCATION[i*4500 : (i+1)*4500, :])

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
    GLOBAL_AIRPLANE_TAG = GLOBAL_AIRPLANE_TAG[1:]

    for i in range(0, 7):
        print(len(GLOBAL_AIRPLANE_TAG[GLOBAL_AIRPLANE_TAG == i]))


def main():
    # ----------------get global points location, do pointnet classification -----------------------------------------
    global GLOBAL_AIRPLANE_LOCATION
    get_global_airplane_location()
    get_global_points_tag()

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

                draw_image(display, image_rgb)
                save_lidar(lidar)
                save_sem_lidar(lidar_sem)

                # Draw the display.
                display.blit(font.render('% 5d FPS (real)' % clock.get_fps(), True, (0, 0, 0)), (8, 5))
                display.blit(font.render('% 5d FPS (simulated)' % fps, True, (0, 0, 0)), (8, 25))
                display.blit(font.render('% 5d km/h (velocity)' % vehicle_velocity, True, (0, 0, 0)), (8, 45))

                display.blit(font.render('    Min dist to each part', True, (0, 0, 0)), (8, 75))
                display.blit(font.render('%8s %8s %8s' % ('Part', 'Estimate', 'Real'), True, (0, 0, 0)), (8, 100))
                # display.blit(font.render('% 5d mk/h (velocity)' % vehicle_velocity, True, (0, 0, 0)), (8, 46))
                # display.blit(font.render('% 5d mk/h (velocity)' % vehicle_velocity, True, (0, 0, 0)), (8, 46))
                # display.blit(font.render('% 5d mk/h (velocity)' % vehicle_velocity, True, (0, 0, 0)), (8, 46))
                # display.blit(font.render('% 5d mk/h (velocity)' % vehicle_velocity, True, (0, 0, 0)), (8, 46))
                # display.blit(font.render('% 5d mk/h (velocity)' % vehicle_velocity, True, (0, 0, 0)), (8, 46))
                # display.blit(font.render('% 5d mk/h (velocity)' % vehicle_velocity, True, (0, 0, 0)), (8, 46))

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
