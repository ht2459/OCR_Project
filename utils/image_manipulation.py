import math
import numpy as np
from PIL import Image

def rotate(image, angle):
    """
    :param image: an PIL image
    :param angle: degree to be rotated
    """
    image = image.convert('RGBA')
    rotated_image = image.rotate(angle, expand=1)
    padded_image = Image.new('RGBA', rotated_image.size, (255,)*4)
    output = Image.composite(rotated_image, padded_image, rotated_image)
    
    return output


def flip(image, mode='h'):
    """
    Flip self.image according to the mode specified
    :param mode: 'h' or 'v' or 'hv'. 'h' means horizontal and 'v' means vertical.
    :return flipped: flipped dataset
    """

    if mode == 'h':
        img_flipped = np.fliplr(image)

    if mode == 'v':
        img_flipped = np.flipud(image)
        
    if mode == 'hv':
        img_flipped = np.flip(image, (0, 1))

    return img_flipped


def add_noise(image, portion, amplitude):
    """
    :param image:
    :param portion: The portion of image to inject noise. 
    :param amplitude: An integer scaling factor of the noise.
    """
