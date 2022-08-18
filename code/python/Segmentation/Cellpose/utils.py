import numpy as np

def remove_zero_pad(image):
    tmp = np.argwhere(image > 0.0)

    max_y = tmp[:, 0].max()
    min_y = tmp[:, 0].min()
    min_x = tmp[:, 1].min()
    max_x = tmp[:, 1].max()

    crop_image = image[min_y:max_y, min_x:max_x]
    return crop_image