import random
import cv2
from matplotlib import pyplot as plt
import bbox as B
import functions as F
import numpy as np
import copy

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


image = cv2.imread('./49269.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
rows, cols = image.shape[:2]
bboxes =  [[6.48,315.47,269.47,317.4], [90.84,27.53,337.2,474.84]]
bboxes2 = copy.deepcopy(bboxes)
category_ids = [18, 19]

# We will use the mapping from category_id to the class name
# to visualize the class label for the bounding box on the image
category_id_to_name = {18: 'dog', 19: 'horse'}



# 밝기
image_1 = F.brightness(image, 0.5, 3)
# 수직 접기
image_2 = F.vertical_flip(image)
# 일반 rotate 실행
image_3 = F.rotate2(image, 45)
# albumentation rotate 실행
image_4 = F.rotate(image, 45, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, value=None)
rows, cols = image.shape[:2]
B.convert('coco',bboxes)
B.convert('coco', bboxes2)
B.normalize_bboxes(bboxes, rows, cols)
B.normalize_bboxes(bboxes2, rows, cols)
for box in bboxes2:
    box[:4] = F.bbox_flip(box, 0, rows, cols)
# bbox 변환
for bbox in bboxes:
    bbox[:4] = F.bbox_rotate(bbox, 45, rows, cols)
    bbox[:4]= tuple(np.clip(bbox[:4], 0, 1.0))
B.denormalize_bboxes(bboxes, rows, cols)
B.denormalize_bboxes(bboxes2, rows, cols)
# image_3 = F.rotate_image(image, 45)
#화면출력
visualize(image_1, bboxes, category_ids, category_id_to_name)
visualize(image_2, bboxes2, category_ids, category_id_to_name)
visualize(image_3, bboxes, category_ids, category_id_to_name)
visualize(image_4, bboxes, category_ids, category_id_to_name)
print(bboxes)
