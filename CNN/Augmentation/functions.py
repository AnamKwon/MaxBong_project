import cv2
import numpy as np
from functools import wraps

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
    x_min, y_min, x_max, y_max = bbox[:4]
    scale = cols / float(rows)
    x = np.array([x_min, x_max, x_max, x_min]) - 0.5
    y = np.array([y_min, y_min, y_max, y_max]) - 0.5
    angle = np.deg2rad(angle)
    x_t = (np.cos(angle) * x * scale + np.sin(angle) * y) / scale
    y_t = -np.sin(angle) * x * scale + np.cos(angle) * y
    x_t = x_t + 0.5
    y_t = y_t + 0.5
    print('y_t', y_t)

    x_min, x_max = min(x_t), max(x_t)
    y_min, y_max = min(y_t), max(y_t)
    print('yminmax', y_min,y_max)
    return x_min, y_min, x_max, y_max

def vertical_flip(image):
    return cv2.flip(image, 0)


def horizontal_flip(image):
    return cv2.flip(image, 1)

def rotate2(image, angle):
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    image = cv2.warpAffine(image, matrix, (w, h))
    return image

def preserve_channel_dim(func):
    """
    Preserve dummy channel dim.
    """

    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        shape = img.shape
        result = func(img, *args, **kwargs)
        if len(shape) == 3 and shape[-1] == 1 and len(result.shape) == 2:
            result = np.expand_dims(result, axis=-1)
        return result

    return wrapped_function


def get_num_channels(image):
    return image.shape[2] if len(image.shape) == 3 else 1


def _maybe_process_in_chunks(process_fn, **kwargs):
    """
    Wrap OpenCV function to enable processing images with more than 4 channels.
    Limitations:
        This wrapper requires image to be the first argument and rest must be sent via named arguments.
    Args:
        process_fn: Transform function (e.g cv2.resize).
        kwargs: Additional parameters.
    Returns:
        numpy.ndarray: Transformed image.
    """

    def __process_fn(img):
        num_channels = get_num_channels(img)
        if num_channels > 4:
            chunks = []
            for index in range(0, num_channels, 4):
                if num_channels - index == 2:
                    # Many OpenCV functions cannot work with 2-channel images
                    for i in range(2):
                        chunk = img[:, :, index + i : index + i + 1]
                        chunk = process_fn(chunk, **kwargs)
                        chunk = np.expand_dims(chunk, -1)
                        chunks.append(chunk)
                else:
                    chunk = img[:, :, index : index + 4]
                    chunk = process_fn(chunk, **kwargs)
                    chunks.append(chunk)
            img = np.dstack(chunks)
        else:
            img = process_fn(img, **kwargs)
        return img

    return __process_fn


@preserve_channel_dim
def rotate(img, angle, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, value=None):
    height, width = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)

    warp_fn = _maybe_process_in_chunks(
        cv2.warpAffine, M=matrix, dsize=(width, height), flags=interpolation, borderMode=border_mode, borderValue=value
    )
    return warp_fn(img)

def crop(img, x_min, y_min, x_max, y_max):
    height, width = img.shape[:2]
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(
            "We should have x_min < x_max and y_min < y_max. But we got"
            " (x_min = {x_min}, y_min = {y_min}, x_max = {x_max}, y_max = {y_max})".format(
                x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
            )
        )

    if x_min < 0 or x_max > width or y_min < 0 or y_max > height:
        raise ValueError(
            "Values for crop should be non negative and equal or smaller than image sizes"
            "(x_min = {x_min}, y_min = {y_min}, x_max = {x_max}, y_max = {y_max}, "
            "height = {height}, width = {width})".format(
                x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, height=height, width=width
            )
        )

    return img[y_min:y_max, x_min:x_max]



