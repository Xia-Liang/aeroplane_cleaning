try:
    from config import *
except ImportError:
    raise RuntimeError('cannot import config file')


def get_font():
    """
    show font
    :return:
    """
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 24)


def get_speed(vehicle):
    """
    Compute speed of a vehicle in Km/h.

        :param vehicle: the vehicle for which speed is calculated
        :return: speed as a float in Km/h
    """
    vel = vehicle.get_velocity()

    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)


# def process_img(image):
#     """
#     process the image
#     """
#     global surface
#     array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
#     array = np.reshape(array, (image.height, image.width, 4))
#     array = array[:, :, :3]
#     array = array[:, :, ::-1] # switch r,g,b
#     array = array.swapaxes(0, 1)  # exchange the width and height
#     surface = pygame.surfarray.make_surface(array)   # Copy an array to a new surface
#
#     # image_name = round(image.frame, 10)
#     # image.save_to_disk('./image/%d' % image_name)


def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def save_rgb(surface, image):
    # global depth_surface
    image.convert(carla.ColorConverter.Raw)
    if SHOW_CAM:
        image_name = round(image.frame, 10)
        image.save_to_disk('./image/%d' % image_name)
