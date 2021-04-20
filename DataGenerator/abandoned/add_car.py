from config import *
import pygame
from pygame.locals import K_ESCAPE

actor_list = list()
surface = None

def should_quit():
    """
    stop event
    :return:
    """
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


def process_img(image):
    """
    process the image
    """
    global surface
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1] # switch r,g,b
    array = array.swapaxes(0, 1)  # exchange the width and height
    # print(array + "?")
    surface = pygame.surfarray.make_surface(array)   # Copy an array to a new surface
    # print(surface + "!")

    image_name = round(image.frame, 10)
    image.save_to_disk('D:\\mb95541\\aeroplane\\image\\rgb\\%d' % image_name)


def carla_main():
    pygame.init()
    display = pygame.display.set_mode((IMG_WIDTH, IMG_HEIGHT))
    try:
    # client
        client = carla.Client('localhost', 2000)
        client.set_timeout(3)

        # world
        world = client.get_world()
        # world = client.load_world('version2')
        # world = client.load_world('CarlaUE4')
        # # spectator
        # spectator = world.get_spectator()
        # transform = carla.Transform(carla.Location(x=0, y=0, z=20), carla.Rotation(pitch=-90.000000, yaw=0.000, roll=0.000000))
        # spectator.set_transform(transform)

        # vehicle
        blueprint_library = world.get_blueprint_library()

        # for i in blueprint_library.filter('vehicle'):
        #     print(i)
        vehicle_bp = blueprint_library.filter('bmw')[0]
        vehicle_bp.set_attribute('role_name', 'runner')
        white = '255.0, 255.0, 255.0'
        vehicle_bp.set_attribute('color', white)
        spawn_point = carla.Transform(carla.Location(x=136, y=315, z=1),
                                      carla.Rotation(pitch=0.000000, yaw=270.000, roll=0.000000))
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(vehicle)

        vehicle.set_simulate_physics(True)

        # --- rgb-camera sensor --- #
        # 1. blueprint
        rgb_camera_bp = blueprint_library.find('sensor.camera.rgb')
        # # 2. set the attribute of camera
        rgb_camera_bp.set_attribute("image_size_x", "%f" % IMG_WIDTH)  # image width
        rgb_camera_bp.set_attribute("image_size_y", "%f" % IMG_HEIGHT)  # image height
        rgb_camera_bp.set_attribute("fov", "110")  # Horizontal field of view in degrees
        # rgb_camera_bp.set_attribute("sensor_tick", "0.05")  # Simulation seconds between sensor captures (ticks).
        # # 3. add camera sensor to the vehicle, put the sensor in the car. rotation y x z
        spawn_point = carla.Transform(carla.Location(x=0.5, y=0.0, z=3),
                                      carla.Rotation(pitch=-15.0, yaw=0.0, roll=0.0))
        camera_rgb = world.spawn_actor(rgb_camera_bp, spawn_point, attach_to=vehicle)

        camera_rgb.listen(lambda data: process_img(data))

        actor_list.append(camera_rgb)

        clock = pygame.time.Clock()
        # vehicle.set_autopilot(enabled=True, tm_port=2000)
        while True:
            if should_quit():
                return
            clock.tick_busy_loop(30)
            if(not surface):
                continue
            pygame.display.flip()
            print(surface)

            display.blit(surface, (0, 0))

    finally:
        print('destroying actors')
        for actor in actor_list:
            actor.destroy()
        pygame.quit()
        print('done.')


if __name__ == '__main__':
    carla_main()
