"""
give the open3d view of plane, both lidar and sem-lidar

change T/F in argparser

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
VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
LABEL_COLORS = np.array([
    (255, 255, 255), # None
    (70, 70, 70),    # Building
    (100, 40, 40),   # Fences
    (55, 90, 80),    # Other
    (220, 20, 60),   # Pedestrian
    (153, 153, 153), # Pole
    (157, 234, 50),  # RoadLines
    (128, 64, 128),  # Road
    (244, 35, 232),  # Sidewalk
    (107, 142, 35),  # Vegetation
    (0, 0, 142),     # Vehicle
    (102, 102, 156), # Wall
    (220, 220, 0),   # TrafficSign
    (70, 130, 180),  # Sky
    (81, 0, 81),     # Ground
    (150, 100, 100), # Bridge
    (230, 150, 140), # RailTrack
    (180, 165, 180), # GuardRail
    (250, 170, 30),  # TrafficLight
    (110, 190, 160), # Static
    (170, 120, 50),  # Dynamic
    (45, 60, 150),   # Water
    (145, 170, 100), # Terrain
    (102, 0, 204), # CPcutCockpit = 23, dark purple
    (153, 51, 255), # CPcutDome = 24, light purple
    (0, 255, 0),  # CPcutEmpennage = 25, green
    (255, 153, 51), # CPcutEngineLeft = 26, dark orange
    (204, 102, 0),  # CPcutEngineRight = 27, light orange
    (153, 255, 164), # CPcutGearFront = 28, light green
    (153, 255, 204), # CPcutGearLeft = 29,
    (153, 255, 244), # CPcutGearRight = 30,
    (255, 0, 0),  # CPcutMainBody = 31, red
    (204, 204, 0),# CPcutWingLeft = 32, dark yellow
    (255, 255, 51) # CPcutWingRight = 33, light yellow
]) / 255.0 # normalize each channel [0-1] since is what Open3D uses


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


def lidar_callback(point_cloud, point_list):
    """Prepares a point cloud with intensity
    colors ready to be consumed by Open3D"""
    data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
    data = np.reshape(data, (int(data.shape[0] / 4), 4))

    # drop ground
    drop_index = (data[:, 2] < -1.6) | (abs(data[:, 0]) < 1) | (abs(data[:, 1]) < 1)
    data = data[np.invert(drop_index)]
    print(data.shape)

    # Isolate the intensity and compute a color for it
    intensity = data[:, -1]
    intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
    int_color = np.c_[
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]

    # Isolate the 3D data
    points = data[:, :-1]

    # We're negating the y to correclty visualize a world that matches
    # what we see in Unreal since Open3D uses a right-handed coordinate system
    points[:, :1] = -points[:, :1]

    # # An example of converting points from sensor to vehicle space if we had
    # # a carla.Transform variable named "tran":
    # points = np.append(points, np.ones((points.shape[0], 1)), axis=1)
    # points = np.dot(tran.get_matrix(), points.T).T
    # points = points[:, :-1]
    point_list.points = o3d.utility.Vector3dVector(points)
    point_list.colors = o3d.utility.Vector3dVector(int_color)


def semantic_lidar_callback(point_cloud, point_list):
    """Prepares a point cloud with semantic segmentation
    colors ready to be consumed by Open3D"""
    data = np.frombuffer(point_cloud.raw_data, dtype=np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)]))

    # We're negating the y to correclty visualize a world that matches
    # what we see in Unreal since Open3D uses a right-handed coordinate system
    points = np.array([data['x'], -data['y'], data['z']]).T  # shape: (npoints, 3)

    # drop ground
    drop_index = (points[:, 2] < -2.8) | (abs(points[:, 0]) < 1) | (abs(points[:, 1]) < 1)
    points = points[np.invert(drop_index)]
    data = data[np.invert(drop_index)]

    # # An example of adding some noise to our data if needed:
    # points += np.random.uniform(-0.05, 0.05, size=points.shape)
    # Colorize the pointcloud based on the CityScapes color palette
    labels = np.array(data['ObjTag'])
    int_color = LABEL_COLORS[labels]

    # # In case you want to make the color intensity depending
    # # of the incident ray angle, you can use:
    # int_color *= np.array(data['CosAngle'])[:, None]
    point_list.points = o3d.utility.Vector3dVector(points)
    point_list.colors = o3d.utility.Vector3dVector(int_color)


def add_open3d_axis(vis):
    """Add a small 3D axis on Open3D Visualizer"""
    axis = o3d.geometry.LineSet()
    axis.points = o3d.utility.Vector3dVector(np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    axis.lines = o3d.utility.Vector2iVector(np.array([
        [0, 1],
        [0, 2],
        [0, 3]]))
    axis.colors = o3d.utility.Vector3dVector(np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    vis.add_geometry(axis)


def main(args):
    generate_global_prime()

    # -------------------------------------------------------------------------------------

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # --- start point --- #
    spawn_point = carla.Transform(carla.Location(x=260, y=315, z=3),
                                  carla.Rotation(pitch=0.000000, yaw=270.000, roll=0.000000))
    vehicle = world.spawn_actor(blueprint_library.filter('mercedesccc')[0], spawn_point)
    vehicle.set_simulate_physics(True)
    actor_list.append(vehicle)
    # -------------------------------------------------------------------------------------

    rgb_camera = world.spawn_actor(
        generate_rgb_bp(world, blueprint_library),
        carla.Transform(carla.Location(x=1, y=0.0, z=1.7), carla.Rotation(pitch=0, yaw=0.0, roll=0.0)),
        attach_to=vehicle)
    actor_list.append(rgb_camera)

    lidar = world.spawn_actor(
        generate_lidar_bp(world, blueprint_library),
        carla.Transform(carla.Location(x=1, y=0.0, z=1.75), carla.Rotation(pitch=0, yaw=0.0, roll=0.0)),
        attach_to=vehicle)
    actor_list.append(lidar)

    lidar_sem = world.spawn_actor(
        generate_lidar_sem_bp(world, blueprint_library),
        carla.Transform(carla.Location(x=1, y=0.0, z=3), carla.Rotation(pitch=0, yaw=0.0, roll=0.0)),
        attach_to=vehicle)
    actor_list.append(lidar_sem)

    # -------------------------------------------------------------------------------------

    global point_list
    point_list = o3d.geometry.PointCloud()
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name='Carla Lidar',
        width=1080,
        height=720,
        left=120,
        top=120)
    vis.get_render_option().background_color = [0.05, 0.05, 0.05]
    vis.get_render_option().point_size = 1
    vis.get_render_option().show_coordinate_frame = True

    try:
        # controller = KeyboardControl(vehicle)
        # Create a synchronous mode context.
        frame = 0
        with CarlaSyncMode(world, rgb_camera, lidar, lidar_sem, fps=FPS) as sync_mode:
            while True:
                # Advance the simulation and wait for the data.
                snapshot, image_rgb, lidar, lidar_sem = sync_mode.tick(timeout=2.0)
                time.sleep(1)
                if frame == 2:
                    vis.add_geometry(point_list)
                if args.semantic:
                    semantic_lidar_callback(lidar_sem, point_list)
                else:
                    lidar_callback(lidar, point_list)
                vis.update_geometry(point_list)

                vis.poll_events()
                vis.update_renderer()

                frame += 1
    finally:
        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()
        vis.destroy_window()
        print('done.')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='CARLA Automatic Control Client')
    argparser.add_argument(
        '--semantic',
        default=False,
        help='use the semantic lidar instead, which provides ground truth information')
    argparser.add_argument(
        '--show-axis',
        default=True,
        help='show the cartesian coordinates axis')

    args = argparser.parse_args()

    main(args)


