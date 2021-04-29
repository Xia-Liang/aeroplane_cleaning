"""
generate data for testset

use lidar, no sem-lidar

need to update

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


def save_lidar(data):
    # pass
    if data.frame % 41 == 0:
        data.save_to_disk('D:\\mb95541\\aeroplane\\data\\testset\\lidar\\%d' % get_name(data.frame))


def save_sem_lidar(data):
    # pass
    if data.frame % 41 == 0:
        data.save_to_disk('D:\\mb95541\\aeroplane\\data\\testset\\semLidar\\%d' % get_name(data.frame))


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
    };
will transform to 

    None                = 0
    CPcutCockpit        = 1
    CPcutDome           = 2
    CPcutEmpennage      = 3
    CPcutEngineLeft     = 4
    CPcutEngineRight    = 5
    CPcutGearFront      = 6
    CPcutGearLeft       = 7
    CPcutGearRight      = 8
    CPcutMainBody       = 9
    CPcutWingLeft       = 10
    CPcutWingRight      = 11
    Vehicle             = 12
'''


def airplane_label(x):
    x = int(x)
    if x == 10:
        return '12'
    elif x < 23:
        return '0'
    else:
        return str(x - 22)


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
    spawn_point = carla.Transform(carla.Location(x=260, y=315, z=3),
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
    classifier.load_state_dict(torch.load('model\\seg_model_244_best.pth'))
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
                if lidar.frame % 19 == 0:
                    lidar_array = np.copy(np.frombuffer(lidar.raw_data, dtype=np.dtype("f4")))
                    lidar_array = lidar_array.reshape((-1, 4))[:, 0:3]  # keep xyz, shape = (length, 3)
                    length = lidar_array.shape[0]
                    dist = np.sqrt(np.sum(lidar_array ** 2, axis=1))  # shape = (length, )

                    max_dist = np.max(dist, 0)
                    cuda_array = lidar_array / max_dist  # scale

                    if length < 6000:
                        zeros = np.zeros((6000 - length, 3), dtype='f4')
                        cuda_array = np.concatenate((lidar_array, zeros), axis=0)  # match size of model
                    else:
                        length = 6000
                        dist = dist[:6000]
                        cuda_array = cuda_array[:6000, :]

                    # print(cuda_array.shape)
                    cuda_array = cuda_array.reshape((1, 6000, 3))

                    # gpu
                    cuda_array = torch.from_numpy(cuda_array).float().transpose(2, 1)
                    pred, _, _ = classifier(cuda_array)
                    pred = pred.view(-1, 13)
                    pred_choice = pred.data.max(1)[1].cpu().numpy()[0:length]  # shape = (length, )
                    # print(pred_choice.shape)

                    result = np.concatenate((dist.reshape(length, 1), pred_choice.reshape(length, 1)), axis=1)
                    # print(result.shape)  # shape = (length, 2)

                    for tag in range(0, 13):
                        print(len(dist[np.where(pred_choice == tag)]), end=' ')
                    print(' ')

                # --------------------------------------------------------------------------
                # process semLidar

                # --------------------------------------------------------------------------
                # Draw the display.
                draw_image(display, image_rgb)
                save_lidar(lidar)
                save_sem_lidar(lidar_sem)

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


