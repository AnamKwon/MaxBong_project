import random

import cv2
from matplotlib import pyplot as plt
import bbox as B

BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, x_max, y_max = map(int, bbox)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)
    plt.show()


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


def bbox_vflip(bbox, rows, cols):  # skipcq: PYL-W0613
    x_min, y_min, x_max, y_max = bbox[:4]
    return x_min, 1 - y_max, x_max, 1 - y_min

def bbox_hflip(bbox, rows, cols):  # skipcq: PYL-W0613
    x_min, y_min, x_max, y_max = bbox[:4]
    return 1 - x_max, y_min, 1 - x_min, y_max

def vertical_flip(image):
    return cv2.flip(image, 0)


def horizontal_flip(image):
    return cv2.flip(image, 1)


image = cv2.imread('49269.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


rows, cols = image.shape[:2]
bboxes = [[6.48,315.47,269.47,317.4], [90.84,27.53,337.2,474.84]]
category_ids = [18, 19]

# We will use the mapping from category_id to the class name
# to visualize the class label for the bounding box on the image
category_id_to_name = {18: 'dog', 19: 'horse'}

B.convert('coco', bboxes)
B.normalize_bboxes(bboxes,rows,cols)
for bbox in bboxes:
    bbox[:4] = bbox_vflip(bbox, rows, cols)
B.denormalize_bboxes(bboxes, rows, cols)
image = vertical_flip(image)
visualize(image, bboxes, category_ids, category_id_to_name)
print(bboxes)


