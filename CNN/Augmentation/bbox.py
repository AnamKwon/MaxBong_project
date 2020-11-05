# 포맷 형식에 따라 BoundingBox 좌표값 변환
def convert(format, bbox):
    for box in bbox:
        if format == 'coco':
            x_min, y_min, w, h = box[:4]
            box[:4] = int(x_min), int(y_min), int(x_min + w), int(y_min + h)
        elif format == 'voc':
            return box
        elif format == 'yolo':
            x, y, w, h = box[:4]
            x_min, y_min, x_max, y_max = int(x - w / 2 + 1), int(y - h / 2 + 1), int(x_min + w), int(y_min + h)
            box[:4] = x_min, y_min, x_max, y_max


# BoundingBox Normalize
def normalize_bbox(bbox, rows, cols):
    """Normalize coordinates of a bounding box. Divide x-coordinates by image width and y-coordinates
    by image height.
    """
    x_min, y_min, x_max, y_max = bbox[:4]

    if rows <= 0:
        raise ValueError("Argument rows must be positive integer")
    if cols <= 0:
        raise ValueError("Argument cols must be positive integer")

    x_min, x_max = x_min / cols, x_max / cols
    y_min, y_max = y_min / rows, y_max / rows

    bbox[:4] = x_min, y_min, x_max, y_max
    return bbox


# Normalize a list of bounding boxes.
def normalize_bboxes(bboxes, rows, cols):
    """
    Args:
        bboxes List: Denormalized bounding boxes `[x_min, y_min, x_max, y_max]`.
        rows (int): Image height.
        cols (int): Image width.
    """
    return [normalize_bbox(bbox, rows, cols) for bbox in bboxes]


# Denormalize coordinates of a bounding box.
def denormalize_bbox(bbox, rows, cols):
    """Multiply x-coordinates by image width and y-coordinates
    by image height. This is an inverse operation for :func:`~albumentations.augmentations.bbox.normalize_bbox`.
    Args:
        bbox (tuple): Normalized bounding box `(x_min, y_min, x_max, y_max)`.
        rows (int): Image height.
        cols (int): Image width.
    Raises:
        ValueError: If rows or cols is less or equal zero
    """
    x_min, y_min, x_max, y_max= bbox[:4]

    if rows <= 0:
        raise ValueError("Argument rows must be positive integer")
    if cols <= 0:
        raise ValueError("Argument cols must be positive integer")

    x_min, x_max = x_min * cols, x_max * cols
    y_min, y_max = y_min * rows, y_max * rows

    bbox[:4] = x_min, y_min, x_max, y_max
    return bbox


# Denormalize a list of bounding boxes.
def denormalize_bboxes(bboxes, rows, cols):
    """Denormalize a list of bounding boxes.
    Args:
        bboxes (List[tuple]): Normalized bounding boxes `[(x_min, y_min, x_max, y_max)]`.
        rows (int): Image height.
        cols (int): Image width.
    """
    return [denormalize_bbox(bbox, rows, cols) for bbox in bboxes]



