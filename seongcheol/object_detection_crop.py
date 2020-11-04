import cv2
import numpy as np
import matplotlib.pyplot as plt

read_img = 'timg.jpg'
path = "./l_img/"


def label_img_file(read_img,path):
    
    classes = ["person", "bicycle", "car", "motorcycle",
            "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
            "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
            "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
            "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
            "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
            "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
            "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
            "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
            "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" ]


    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))



    img = cv2.imread(read_img)
    # print(img.shape)
    height, width, channels = img.shape


    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    # print(blob.shape)




    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.1:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.4)


    font = cv2.FONT_HERSHEY_SIMPLEX
    count = 0

    img_idx_dict = dict()
    for i in range(len(boxes)):
        info_idx = []
        if i in indexes:
            x, y, w, h = boxes[i]
            if x < 0:
                x = 0

            label = str(classes[class_ids[i]])
            color = colors[i]

            cv2.rectangle(img, (x , y ), (x + w  , y + h  ), color, 1)
            cv2.putText(img, label, (x, y - 10), font, 1, color, 2)

            roi = img[y:y+h , x:x+w ]
            
            
            
            try:
                cv2.imwrite(path + str(count)+'_' + label + ".jpg", roi)
                info_idx.append(label)
                info_idx.append(x)
                info_idx.append(y)
                info_idx.append(w)
                info_idx.append(h)
                img_idx_dict[str(count)] = info_idx
                # print(roi.shape)
                count = count + 1
            except:
                continue
            


    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img_idx_dict
print(label_img_file(read_img,path))


