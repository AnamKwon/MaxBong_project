import cv2
import numpy as np

def bbox_vflip(bbox, rows, cols):
    x_min, y_min, x_max, y_max = bbox[:4]
    return x_min, 1 - y_max, x_max, 1 - y_min

def bbox_hflip(bbox, rows, cols):
    x_min, y_min, x_max, y_max = bbox[:4]
    return 1 - x_max, y_min, 1 - x_min, y_max

# BoundingBox Flip
def bbox_flip(bbox, d, rows, cols):
    """Flip a bounding box either vertically, horizontally or both depending on the value of `d`.
    """
    if d == 0:
        bbox = bbox_vflip(bbox, rows, cols)
    elif d == 1:
        bbox = bbox_hflip(bbox, rows, cols)
    elif d == -1:
        bbox = bbox_hflip(bbox, rows, cols)
        bbox = bbox_vflip(bbox, rows, cols)
    else:
        raise ValueError("Invalid d value {}. Valid values are -1, 0 and 1".format(d))
    return bbox


def bbox_rotate(bbox, angle, rows, cols):
    """Rotates a bounding box by angle degrees.
    Args:
        bbox (tuple): A bounding box `(x_min, y_min, x_max, y_max)`.
        angle (int): Angle of rotation in degrees.
        rows (int): Image rows.
        cols (int): Image cols.
    Returns:
        A bounding box `(x_min, y_min, x_max, y_max)`.
    """
    x_min, y_min, x_max, y_max = bbox[:4]
    scale = cols / float(rows)
    x = np.array([x_min, x_max, x_max, x_min]) - 0.5
    y = np.array([y_min, y_min, y_max, y_max]) - 0.5
    angle = np.deg2rad(angle)
    x_t = (np.cos(angle) * x * scale + np.sin(angle) * y) / scale
    y_t = -np.sin(angle) * x * scale + np.cos(angle) * y
    x_t = x_t + 0.5
    y_t = y_t + 0.5

    x_min, x_max = min(x_t), max(x_t)
    y_min, y_max = min(y_t), max(y_t)

    return x_min, y_min, x_max, y_max

def vertical_flip(image):
    return cv2.flip(image, 0)


def horizontal_flip(image):
    return cv2.flip(image, 1)

def rotate(image, degree):
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), degree, 1)
    image = cv2.warpAffine(image, matrix, (w, h))
    return image

