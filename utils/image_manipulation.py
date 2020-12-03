from scipy import ndimage


def rotate(image, angle=0.0):
    """
    Rotate image by the angles (in degree) given.
    :param angle: Rotation angle in degrees.
    :return rotated: rotated dataset
    """
    rotated = ndimage.rotate(image, angle, reshape=True)

    return rotated


def flip(self, mode='h'):
    """
    Flip self.image according to the mode specified
    :param mode: 'h' or 'v' or 'hv'. 'h' means horizontal and 'v' means vertical.
    :return flipped: flipped dataset
    """

    x = self.x.copy()
    img = x.reshape(self.N, 28, 28)
    if mode == 'h':
        img_flipped = np.flip(img, axis=2)
        self.is_horizontal_flip = True
    if mode == 'v':
        img_flipped = np.flip(img, axis=2)
        self.is_vertical_flip = True

    return img_flipped.reshape(self.N, -1)