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

try:
    from PIL import Image
except ImportError:
    raise RuntimeError('cannot import PIL, make sure "Pillow" package is installed')


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


def save_lidar_projection(rgb_camera, lidar, image_data, lidar_data, K, VID_RANGE, VIRIDIS):
    # Get the raw BGRA buffer and convert it to an array of RGB of
    # shape (image_data.height, image_data.width, 3).
    im_array = np.copy(np.frombuffer(image_data.raw_data, dtype=np.dtype("uint8")))
    im_array = np.reshape(im_array, (image_data.height, image_data.width, 4))
    im_array = im_array[:, :, :3][:, :, ::-1]

    # Get the lidar data and convert it to a numpy array.
    p_cloud_size = len(lidar_data)
    p_cloud = np.copy(np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4')))
    p_cloud = np.reshape(p_cloud, (p_cloud_size, 4))

    # Lidar intensity array of shape (p_cloud_size,) but, for now, let's focus on the 3D points.
    intensity = np.array(p_cloud[:, 3])

    # Point cloud in lidar sensor space array of shape (3, p_cloud_size).
    local_lidar_points = np.array(p_cloud[:, :3]).T

    # Add an extra 1.0 at the end of each 3d point so it becomes of
    # shape (4, p_cloud_size) and it can be multiplied by a (4, 4) matrix.
    local_lidar_points = np.r_[
        local_lidar_points, [np.ones(local_lidar_points.shape[1])]]

    # This (4, 4) matrix transforms the points from lidar space to world space.
    lidar_2_world = lidar.get_transform().get_matrix()

    # Transform the points from lidar space to world space.
    world_points = np.dot(lidar_2_world, local_lidar_points)

    # This (4, 4) matrix transforms the points from world to sensor coordinates.
    world_2_camera = np.array(rgb_camera.get_transform().get_inverse_matrix())

    # Transform the points from world space to camera space.
    sensor_points = np.dot(world_2_camera, world_points)

    # New we must change from UE4's coordinate system to an "standard"
    # camera coordinate system (the same used by OpenCV):
    # ^ z                       . z
    # |                        /
    # |              to:      +-------> x
    # | . x                   |
    # |/                      |
    # +-------> y             v y

    # This can be achieved by multiplying by the following matrix:
    # [[ 0,  1,  0 ],
    #  [ 0,  0, -1 ],
    #  [ 1,  0,  0 ]]
    # Or, in this case, is the same as swapping:
    # (x, y ,z) -> (y, -z, x)
    point_in_camera_coords = np.array([
        sensor_points[1],
        sensor_points[2] * -1,
        sensor_points[0]])

    # Finally we can use our K matrix to do the actual 3D -> 2D.
    points_2d = np.dot(K, point_in_camera_coords)

    # Remember to normalize the x, y values by the 3rd value.
    points_2d = np.array([
        points_2d[0, :] / points_2d[2, :],
        points_2d[1, :] / points_2d[2, :],
        points_2d[2, :]])

    # At this point, points_2d[0, :] contains all the x
    # points_2d[1, :] contains all the y values of our points.
    # In order to properly visualize everything on a screen, the points that are out of the screen
    # must be discarted, the same with points behind the camera projection plane.
    points_2d = points_2d.T
    intensity = intensity.T
    points_in_canvas_mask = \
        (points_2d[:, 0] > 0.0) & (points_2d[:, 0] < IMG_WIDTH) & \
        (points_2d[:, 1] > 0.0) & (points_2d[:, 1] < IMG_HEIGHT) & \
        (points_2d[:, 2] > 0.0)
    points_2d = points_2d[points_in_canvas_mask]
    intensity = intensity[points_in_canvas_mask]

    # Extract the screen coords (uv) as integers.
    u_coord = points_2d[:, 0].astype(np.int)
    v_coord = points_2d[:, 1].astype(np.int)

    # Since at the time of the creation of this script, the intensity function
    # is returning high values, these are adjusted to be nicely visualized.
    intensity = 4 * intensity - 3
    color_map = np.array([
        np.interp(intensity, VID_RANGE, VIRIDIS[:, 0]) * 255.0,
        np.interp(intensity, VID_RANGE, VIRIDIS[:, 1]) * 255.0,
        np.interp(intensity, VID_RANGE, VIRIDIS[:, 2]) * 255.0]).astype(np.int).T

    # # Draw the 2d points on the image as a single pixel using numpy.
    # im_array[v_coord, u_coord] = color_map

    # Draw the 2d points on the image as squares of extent args.dot_extent. (larger pixel)
    dot_extent = 1
    for i in range(len(points_2d)):
        # I'm not a NumPy expert and I don't know how to set bigger dots
        # without using this loop, so if anyone has a better solution,
        # make sure to update this script. Meanwhile, it's fast enough :)
        im_array[
        v_coord[i] - dot_extent: v_coord[i] + dot_extent,
        u_coord[i] - dot_extent: u_coord[i] + dot_extent] = color_map[i]

    if image_data.frame % 9 == 0:
        # Save the image using Pillow module.
        image = Image.fromarray(im_array)
        image.save("D:\\mb95541\\aeroplane\\data\\projection\\%d.png" % get_name(image_data.frame))


def main():
    generate_global_prime()
    pygame.init()
    display = pygame.display.set_mode((IMG_WIDTH, IMG_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()

    # lidar projection
    VIRIDIS = np.array(cm.get_cmap('plasma').colors)  # (256, 3)
    VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])  # Return evenly spaced numbers over a specified interval.

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
        # https://car.autohome.com.cn/config/series/5346.html, telsa parameter, delete car lidar point in preprocess
        # 4696 * 1850 * 1443
        # or delete vehicles tags = 10, nice!
        vehicle = world.spawn_actor(blueprint_library.filter('tesla')[1], spawn_point)
        vehicle.set_simulate_physics(True)
        actor_list.append(vehicle)

        rgb_camera = world.spawn_actor(
            generate_rgb_bp(world, blueprint_library),
            carla.Transform(carla.Location(x=1, y=0.0, z=1.8), carla.Rotation(pitch=0, yaw=0.0, roll=0.0)),
            attach_to=vehicle)
        actor_list.append(rgb_camera)

        lidar = world.spawn_actor(
            generate_lidar_bp(world, blueprint_library),
            carla.Transform(carla.Location(x=1, y=0.0, z=1.8), carla.Rotation(pitch=0, yaw=0.0, roll=0.0)),
            attach_to=vehicle)
        # lidar.listen(lambda data: data.save_to_disk('D:\\mb95541\\aeroplane\\data\\lidarSem\\%d' % data.frame))
        actor_list.append(lidar)

        controller = KeyboardControl(vehicle)

        # Build the K projection matrix:
        # K = [[Fx,  0, IMG_WIDTH/2],
        #      [ 0, Fy, IMG_HEIGHT/2],
        #      [ 0,  0,         1]]
        fov = 110.0  # same as config.py
        focal = IMG_WIDTH / (2.0 * np.tan(fov * np.pi / 360.0))
        # In this case Fx and Fy are the same since the pixel aspect ratio is 1
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = IMG_WIDTH / 2.0
        K[1, 2] = IMG_HEIGHT / 2.0

        with CarlaSyncMode(world, rgb_camera, lidar, fps=30) as sync_mode:
            while True:
                if should_quit():
                    return
                clock.tick()
                controller.parse_events(clock)
                # Advance the simulation and wait for the data.
                snapshot, image_data, lidar_data = sync_mode.tick(timeout=2.0)

                fps = round(1.0 / snapshot.timestamp.delta_seconds)
                vehicle_velocity = get_speed(vehicle)

                # Draw the display.
                draw_image(display, image_data)
                save_lidar_projection(rgb_camera, lidar, image_data, lidar_data, K, VID_RANGE, VIRIDIS)

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


