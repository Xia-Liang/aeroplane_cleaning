"""
print

        None         =   0u,
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

generate data for testset

use lidar, sem-lidar

print how many points in each tag (predict and real)
"""


try:
    from config import *
    from config_control import *
except ImportError:
    raise ImportError('cannot import config file')



'''
Library functions for prime
for each time simulator runs, we give a different prime num
it's not right to use time.time(), since sensor will not listen at exactly same time
rgb and lidar file name = prime num - frame
prevprime(n): It returns the prev prime smaller than n.
'''
try:
    from sympy import prevprime
except ImportError:
    raise ImportError('cannot import config file')


try:
    from model.model import PointNetDenseCls
    import torch
except ImportError:
    raise ImportError('cannot import model')


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


# def save_lidar(lidar):
#     if lidar.frame % 17 == 0:
#         lidar.save_to_disk('D:\\mb95541\\aeroplane\\data\\testset\\lidar\\%d' % lidar.frame)
#
#
# def save_sem_lidar(semLidar):
#     if semLidar.frame % 17 == 0:
#         semLidar.save_to_disk('D:\\mb95541\\aeroplane\\data\\testset\\semLidar\\%d' % semLidar.frame)


def savebin_lidar(lidar):
    # pass
    if lidar.frame % 17 == 0:
        data = np.copy(np.frombuffer(lidar.raw_data, dtype=np.dtype('f4')))
        data = np.reshape(data, (-1, 4))

        # drop ground
        drop_index = (data[:, 2] < -1.6) | (abs(data[:, 0]) < 1) | (abs(data[:, 1]) < 1)
        data = data[np.invert(drop_index)]

        # np.save('D:\\mb95541\\aeroplane\\data\\testset\\lidar\\%d' % lidar.frame, data)


def savebin_sem_lidar(lidar_sem):
    # pass
    if lidar_sem.frame % 17 == 0:
        data_float = np.copy(np.frombuffer(lidar_sem.raw_data, dtype=np.dtype('f4')))
        data_float = np.reshape(data_float, (-1, 6))[:, :4]

        data_int = np.copy(np.frombuffer(lidar_sem.raw_data, dtype=np.dtype('i4')))
        data_int = np.reshape(data_int, (-1, 6))[:, 4:]

        data = np.concatenate((data_float, data_int), axis=1)
        # data = np.hstack((data_float, data_int))
        # print(data.shape)
        data = data[:, [0, 1, 2, 5]]  # xyz, tag

        # drop ground
        drop_index = (data[:, 2] < -1.6) | (abs(data[:, 0]) < 1) | (abs(data[:, 1]) < 1)
        data = data[np.invert(drop_index)]

        # np.save('D:\\mb95541\\aeroplane\\data\\testset\\semLidar\\%d' % lidar_sem.frame, data)


'''
in ObjectLabel.h, user defined tags
    enum class CityObjectLabel : uint8_t {
        None         =   0u,
        Vehicles     =  10u,
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
        AirplaneFrontCabin = 34u,
        AirplaneRearCabin = 35u,
        AirplaneTail = 36u,
        AirplaneWing = 37u,
        AirplaneEngine = 38u,
        AirplaneWheel = 39u,
    };
will transform to 

    None                = 0
	AirplaneFrontCabin = 1
	AirplaneRearCabin = 2
	AirplaneTail = 3
	AirplaneWing = 4
	AirplaneEngine = 5
    Vehicle             = 0
'''


def airplane_label(x):
    x = int(x)
    if x == 10:
        return '0'
    elif x < 34:
        return '0'
    else:
        return str(x - 33)


def main():
    generate_global_prime()

    # --------------------------------------------------------------------------------------------------
    pygame.init()
    display = pygame.display.set_mode((IMG_WIDTH, IMG_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()

    # --------------------------------------------------------------------------------------------------

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # --- start point --- #
    # spawn_point = carla.Transform(carla.Location(x=260, y=315, z=3),
    #                               carla.Rotation(pitch=0.000000, yaw=270.000, roll=0.000000))
    spawn_point = carla.Transform(carla.Location(x=50, y=320, z=3),
                                  carla.Rotation(pitch=0.000000, yaw=270.000, roll=0.000000))
    vehicle = world.spawn_actor(blueprint_library.filter('mercedesccc')[0], spawn_point)
    vehicle.set_simulate_physics(True)
    actor_list.append(vehicle)

    controller = KeyboardControl(vehicle)
    # --------------------------------------------------------------------------------------------------

    rgb_camera = world.spawn_actor(
        generate_rgb_bp(world, blueprint_library),
        carla.Transform(carla.Location(x=1, y=0.0, z=1.8), carla.Rotation(pitch=0, yaw=0.0, roll=0.0)),
        attach_to=vehicle)
    actor_list.append(rgb_camera)

    lidar = world.spawn_actor(
        generate_lidar_bp(world, blueprint_library),
        carla.Transform(carla.Location(x=1, y=0.0, z=1.8), carla.Rotation(pitch=0, yaw=0.0, roll=0.0)),
        attach_to=vehicle)
    # lidar_sem.listen(lambda data: data.save_to_disk('D:\\mb95541\\aeroplane\\data\\lidarSem\\%d' % data.frame))
    actor_list.append(lidar)

    lidar_sem = world.spawn_actor(
        generate_lidar_sem_bp(world, blueprint_library),
        carla.Transform(carla.Location(x=1, y=0.0, z=1.8), carla.Rotation(pitch=0, yaw=0.0, roll=0.0)),
        attach_to=vehicle)
    actor_list.append(lidar_sem)

    # --------------------------------------------------------------------------------------------------

    classifier = PointNetDenseCls(k=13, feature_transform=False)
    classifier.load_state_dict(torch.load('model\\class13_points6000.pth'))
    classifier.eval()

    # --------------------------------------------------------------------------------------------------

    try:
        # Create a synchronous mode context.
        with CarlaSyncMode(world, rgb_camera, lidar, lidar_sem, fps=30) as sync_mode:
            while True:
                # --------------------------------------------------------------------------
                if should_quit():
                    return
                clock.tick()
                controller.parse_events(clock)
                # Advance the simulation and wait for the data.
                snapshot, image_rgb, lidar, lidar_sem = sync_mode.tick(timeout=2.0)
                fps = round(1.0 / snapshot.timestamp.delta_seconds)
                vehicle_velocity = get_speed(vehicle)

                # --------------------------------------------------------------------------
                # process lidar, update distance
                if lidar.frame > 200 & lidar.frame % 29 == 0:
                    lidar_array = np.copy(np.frombuffer(lidar.raw_data, dtype=np.dtype("f4")))
                    lidar_array = lidar_array.reshape((-1, 4))[:, 0:3]  # keep xyz, shape = (length, 3)

                    # drop ground
                    keep_index = (lidar_array[:, 2] > -1.6) & (abs(lidar_array[:, 0]) > 1) & (abs(lidar_array[:, 1]) > 1)
                    lidar_array = lidar_array[keep_index]

                    if lidar_array.shape[0] < 6000:
                        continue

                    # drop redundant point, choose only 6000 points
                    choice = np.random.choice(lidar_array.shape[0], 6000, replace=True)
                    lidar_array = lidar_array[choice, :]
                    dist = np.sqrt(np.sum(lidar_array ** 2, axis=1))  # real distance, shape = (length, )

                    lidar_array = lidar_array - np.expand_dims(np.mean(lidar_array, axis=0), 0)  # centering
                    max_dist = np.max(np.sqrt(np.sum(lidar_array ** 2, axis=1)), 0)  # max distance after centering
                    cuda_array = lidar_array / max_dist  # scale

                    # print(cuda_array.shape)
                    cuda_array = cuda_array.reshape((1, 6000, 3))

                    # gpu
                    cuda_array = torch.from_numpy(cuda_array).float().transpose(2, 1)
                    pred, _, _ = classifier(cuda_array)
                    pred = pred.view(-1, 13)
                    pred_choice = pred.data.max(1)[1].cpu().numpy()  # return indices of max val, shape = (length, )
                    # print(pred_choice.shape)

                    result = np.concatenate((dist.reshape(6000, 1), pred_choice.reshape(6000, 1)), axis=1)
                    # print(result.shape)  # shape = (length, 2)

                    for tag in range(1, 13):
                        print('%2.2f' % (len(dist[np.where(pred_choice == tag)]) / 45), end='% ')
                    print(' ')

                # --------------------------------------------------------------------------
                # process semLidar
                if lidar_sem.frame > 200 & lidar_sem.frame % 29 == 0:
                    ground_truth = np.frombuffer(lidar_sem.raw_data, dtype=np.dtype([
                        ('x', np.float32), ('y', np.float32), ('z', np.float32),
                        ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)]))
                    ground_truth = np.array([ground_truth['x'], -ground_truth['y'], ground_truth['z'], ground_truth['ObjTag']]).T  # shape: (npoints, 4)

                    # drop ground
                    drop_index = (ground_truth[:, 2] < -1.6) | (abs(ground_truth[:, 0]) < 1) | (abs(ground_truth[:, 1]) < 1)
                    ground_truth = ground_truth[np.invert(drop_index)]

                    for tag in range(23, 34):
                        print('%2.2f' % (ground_truth[np.where(ground_truth[:, 3] == tag)].shape[0] / ground_truth.shape[0] * 100), end='% ')
                    print(' ')
                    print('-------------------------------------')

                # --------------------------------------------------------------------------
                # Draw the display.
                draw_image(display, image_rgb)
                # save_lidar(lidar)
                # save_sem_lidar(lidar_sem)

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


