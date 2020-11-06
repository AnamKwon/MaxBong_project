import os
import json
import time
import glob
from cv2 import cv2
from xml.etree import ElementTree as ET
from convChar import convChar

class COCO:
    
    def __init__(self, img_dir_path, json_dir_path, input_type, input_name, input_x, input_y):
        
        # json_file 컨택을 위한 현재폴더 변경
        os.chdir(json_dir_path)
        json_file = glob.glob(f'*.json')[0]
        with open(json_file) as f :
            file = json.loads(f.read())
        file_list = []
        img_file_types = ['jpg','png','bmp']
        # glob(img.type)을 위한 현재폴더 변경
        os.chdir(img_dir_path)
        for type in img_file_types :
            file_list += glob.glob(f'*.{type}')
        
        # convChar을 위한 변수생성
        origin_file_list = file_list
        
        ## retype
        file_name_dict = None
        file_name_inv_dict = None
        file_type_dict = None
        if input_type != None:
            file_list, file_name_dict, file_name_inv_dict, file_type_dict = convChar().retype(origin_file_list, input_type)
        
        ## rename
        if input_name != None:
            file_list, file_name_dict, file_name_inv_dict, file_type_dict = convChar().rename(origin_file_list, input_name)
        
        self.files_dict = {}
        self.file_to_id = {}
        self.id_to_file = {}
        self.id_to_categories = {i['id']: i['name'] for i in file['categories']}
        self.categories_to_id = {i['name']: i['id']  for i in file['categories']}
        not_image = []
        
        # for문 이전에 사용할 변수선언
        origin_shape_dict = {}
        re_shape_dict = {}
        for num in file['images']:
            if num['file_name'] not in file_list :
                not_image.append(num['id'])
                continue
            
            ## retype, rename
            if file_name_dict != None:
                num['file_name'] = file_name_dict[num['file_name']]
            
            read_img = cv2.imread(num['file_name']).shape
            # origin_h = read_img[0]
            # origin_w = read_img[1]
            origin_shape_dict[num['id']] = list(read_img) 
            
            ## reshape
            if (input_x or input_y) != None:
                read_img = convChar().reshape(read_img, input_x, input_y)
                num['width'] = read_img[1]
                num['height'] = read_img[0]
                re_shape_dict[num['id']] = [read_img[1], read_img[0]]
            
            if (num['width'] != read_img[1]) or (num['height'] != read_img[0]):
                not_image.append(num['id'])
                continue
            self.files_dict[num['file_name']] = {'width': read_img[1],
                                            'height': read_img[0],
                                            'depth' : read_img[2],
                                            'object': []}
            self.id_to_file[num['id']] = num['file_name']
            self.file_to_id[num['file_name']] = num['id']
        
        for num in file['annotations']:
            if num['image_id'] in not_image :
                continue
            
            ## bbox ratio extract
            origin_size = origin_shape_dict[num['image_id']]
            center_x_ratio = (2*num['bbox'][0] + num['bbox'][2])/(2*origin_size[1])
            center_y_ratio = (2*num['bbox'][1] + num['bbox'][3])/(2*origin_size[0])
            box_width_ratio = num['bbox'][2]/origin_size[1]
            box_height_ratio = num['bbox'][3]/origin_size[0]
            
            ## bbox 
            if input_x or input_y != None:
                read_img[1], read_img[0] = re_shape_dict[num['image_id']]
                num['bbox'] = [
                    int(center_x_ratio*read_img[1]-(box_width_ratio*read_img[1])/2),
                    int(center_y_ratio*read_img[0]-(box_height_ratio*read_img[0])/2),
                    int(box_width_ratio*read_img[1]),
                    int(box_height_ratio*read_img[0])
                ]
            
            b = {self.id_to_categories[num['category_id']]: {'bbox': num['bbox'], 'segmentation': num['segmentation']}}
            self.files_dict[self.id_to_file[num['image_id']]]['object'].append(b)
        
        ## convChar -> img file save
        if (input_type or input_name or input_x or input_y) != None:
            pass
        
        print("file_list: ", file_list)
        print("label: ", self.files_dict)



class YOLO :
    
    def __init__(self, img_dir_path, txt_dir_path, label, input_type, input_name, input_x, input_y):
        file_list = []
        img_file_types = ['jpg','png','bmp','jpeg']
        # glob(img.type)을 위한 현재폴더 변경
        os.chdir(img_dir_path)
        for type in img_file_types :
            file_list += glob.glob(f'*.{type}')
        
        # convChar을 위한 변수생성
        origin_file_list = file_list
        
        ## retype
        file_name_dict = None
        file_name_inv_dict = None
        file_type_dict = None
        if input_type != None:
            file_list, file_name_dict, file_name_inv_dict, file_type_dict = convChar().retype(origin_file_list, input_type)
        
        ## rename
        if input_name != None:
            file_list, file_name_dict, file_name_inv_dict, file_type_dict = convChar().rename(origin_file_list, input_name)
        
        self.files_dict = {}
        self.file_to_id = {}
        file_to_id_num = 0
        self.id_to_file = {}
        # txt 파일 불러오기 위한 현재폴더 변경
        os.chdir(txt_dir_path)
        files = glob.glob(f'*.txt')
        # label 파일은 절대경로로 입력해 주어야 함
        with open(label) as f :
            f = f.readlines()
            self.categories_to_id = {categorie.strip() : idx for idx, categorie in enumerate(f)}
            self.id_to_categories = { idx : categorie.strip() for idx, categorie in enumerate(f)}
        
        # for문 이전에 사용할 변수선언
        relied_name = None
        for file in file_list:
            ## file_name_inv_dict
            if file_name_inv_dict != None:
                filename = file_name_inv_dict[file].rsplit('.', 1)[0]
                file = file_name_dict[file_name_inv_dict[file]]
            if file_name_inv_dict == None:
                filename = file.rsplit('.',1)[0]
                relied_name = filename
            
            filename = f"{filename}.txt"
            if filename not in files :
                continue
            # txt 파일 불러오기 위한 현재폴더 변경
            os.chdir(txt_dir_path)
            # objects
            with open(filename) as f :
                objects = f.readlines()
            
            ## retype, rename
            if (input_type != None) or (input_name != None):
                filename = relied_name
            
            self.file_to_id[filename] = file_to_id_num
            self.id_to_file[file_to_id_num] = filename
            file_to_id_num += 1
            # img 파일 읽어오기 위한 현재폴더 설정(openCV 경우 현재 폴더에서 이미지를 입력하지 않으면 안먹힘)
            os.chdir(img_dir_path)
            read_img = cv2.imread(file_name_inv_dict[file]).shape
            
            ## reshape
            if input_x or input_y != None:
                read_img = convChar().reshape(read_img, input_x, input_y)
            
            self.files_dict[file] = {'width': read_img[1],
                                            'height': read_img[0],
                                            'depth' : read_img[2],
                                            'object': []}
            for object in objects :
                object = object.strip().split()
                object_name = self.id_to_categories[int(object[0])]
                box = [
                    0, 
                    0,
                    round(float(object[3]) * read_img[1]),
                    round(float(object[4]) * read_img[0])
                    ]
                box[0] = round((float(object[1]) * read_img[1]) - (box[2]/2))
                box[1] = round((float(object[2]) * read_img[0]) - (box[3]/2))
                object_info = {object_name : {'bbox' : box, 'segmentation' : []}}
                self.files_dict[file]['object'].append(object_info)
        
        ## convChar -> img file save
        if input_type or input_name or input_x or input_y != None:
            pass
        
        print("file_list: ", file_list)
        print("label: ", self.files_dict)


class VOC :
    
    def __init__(self, img_dir_path, xml_dir_path, input_type, input_name, input_x, input_y):
        file_list = []
        img_file_types = ['jpg','png','bmp']
        # glob(img.type)을 위한 현재폴더 변경
        os.chdir(img_dir_path)
        for type in img_file_types :
            file_list += glob.glob(f'*.{type}')
        
        # convChar을 위한 변수생성
        origin_file_list = file_list
        print("~~~~~~~~~~~~~~~~", origin_file_list)
        
        ## retype
        file_name_dict = None
        file_name_inv_dict = None
        file_type_dict = None
        if input_type != None:
            file_list, file_name_dict, file_name_inv_dict, file_type_dict = convChar().retype(origin_file_list, input_type)
        print(file_list, file_name_dict, file_name_inv_dict, file_type_dict)
        ## rename
        if input_name != None:
            file_list, file_name_dict, file_name_inv_dict, file_type_dict = convChar().rename(origin_file_list, input_name)
        print(file_list, file_name_dict, file_name_inv_dict, file_type_dict)
        self.files_dict = {}
        self.file_to_id = {}
        file_to_id_num = 0
        self.id_to_file = {}
        self.id_to_categories = {}
        self.categories_to_id = {}
        categorie_num = 0
        # xml 파일 읽어오기 위한 현재폴더 설정
        os.chdir(xml_dir_path)
        files = glob.glob(f'*.xml')
        
        # for문 이전에 사용할 변수선언
        origin_w = None
        origin_h = None
        for file in files :
            # xml 파일 읽어오기 위한 현재폴더 설정
            os.chdir(xml_dir_path)
            raw_file = file.rsplit('.', 1)[0]
            tree = ET.parse(file)
            filename = tree.find('filename').text
            
            # if filename not in file_list :
            #     continue
            size = tree.find('size')
            
            # img 파일 읽어오기 위한 현재폴더 설정(openCV 경우 현재 폴더에서 이미지를 입력하지 않으면 안먹힘)
            os.chdir(img_dir_path)
            print("!!!!!!!!!!!!!!!!!!!!!!!", raw_file)
            print("!!!!!!!!!!!!!!!!!!!!!!!", file_type_dict[raw_file])
            read_img = cv2.imread(raw_file + file_type_dict[raw_file]).shape
            
            # xml 파일 읽어오기 위한 현재폴더 설정
            os.chdir(xml_dir_path)
            origin_w = int(size.find('width').text)
            origin_h = int(size.find('height').text)
            
            if int(size.find('width').text) != read_img[1] or int(size.find('height').text) != read_img[0] :
                continue
            
            ## retype, rename
            if file_name_dict != None:
                filename = file_name_dict[raw_file + file_type_dict[raw_file]]
            
            ## reshape
            if (input_x or input_y) != None:
                read_img = convChar().reshape(read_img, input_x, input_y)
            
            
            self.file_to_id[filename] = file_to_id_num
            self.id_to_file[file_to_id_num] = filename
            file_to_id_num += 1
            self.files_dict[filename] = {'width': read_img[1],
                                            'height': read_img[0],
                                            'depth' : read_img[2],
                                            'object': []}
            segmentation = [] if tree.find('segmented').text == '0' else [1]
            for object in tree.findall('object') :
                object_name = object.find('name').text
                if object_name not in self.id_to_categories.values() :
                    self.id_to_categories[categorie_num] = object_name
                    self.categories_to_id[object_name] = categorie_num
                    categorie_num += 1
                bndbox = object.find('bndbox')
                
                # x, y ratio
                x_min_ratio = int(bndbox.find('xmin').text)/origin_w
                y_min_ratio = int(bndbox.find('ymin').text)/origin_h 
                x_max_ratio = int(bndbox.find('xmax').text)/origin_w
                y_max_ratio = int(bndbox.find('ymax').text)/origin_h
                
                # Max and min of x, y 
                x_min = x_min_ratio * read_img[1] 
                y_min = y_min_ratio * read_img[0]
                x_max = x_max_ratio * read_img[1]
                y_max = y_max_ratio * read_img[0]
                
                box = [
                    int(x_min) - 1,
                    int(y_min) - 1,
                    int(x_max) - int(x_min),
                    int(y_max) - int(y_min)
                    ]
                object_info = {object_name : {'bbox' : box, 'segmentation' : segmentation}}
                self.files_dict[filename]['object'].append(object_info)
        
        # convChar -> img file save
        if input_type or input_name or input_x or input_y != None:
            pass
        
        print("file_list: ", file_list)
        print("label: ", self.files_dict)


def to_coco(dataset, path = 'voc_to_coco', user='max_bong'):

    try :
        os.mkdir(path)
    except: pass
    times = time.ctime(time.time())
    coco = \
        {"info" :
            {"year" : times[-4::],
            "version" : "1",
            "description" : "max_bong_project",
            "contributor": "",
            "date_created" : times},
        "licenses":
            [{"id": 1,
                "url": "",
                "name": user
                
            }],
        "id_to_categories" :
            [{"id":key,
                "name" : value,
                "supercategory": f"{user}_project"} for key, value in dataset.id_to_categories.items()],
        "imaes":
                [],
        "annotations" :
                []
        }
    images_num = 0
    annotations_num = 0
    for file, value in dataset.files_dict.items():
        image = {
            "id": images_num,
            "license": 1,
            "file_name": f"{file}",
            "height": int(f"{value['height']}"),
            "width": int(f"{value['width']}"),
            "date_captured": times
        }
        coco['images'].append(image)
        for object in value['object'] :
            category, bbox = list(object.items())[0]
            annotation = {
                "id": int(f"{annotations_num}"),
                "image_id": int(f"{images_num}"),
                "category_id": dataset.categories_to_id[category],
                "bbox": bbox['bbox'],
                "area": bbox['bbox'][-1]*bbox['bbox'][-2],
                "segmentation": f"{ 1 if bbox['segmentation'] != [] else [] }",
                "iscrowd": 0
            }
            coco['annotations'].append(annotation)
            annotations_num += 1
        images_num += 1
    with open(f"{path}/annotations.json","w") as f :
        json.dump(coco, f)


def to_yolo(dataset, path='voc_to_yolo') :

    try :
        os.mkdir(path)
    except :
        pass
    labels = [value for _, value in sorted(dataset.id_to_categories.items(), key=lambda t:t[0])]
    with open(f'{path}/labels.txt','w') as f :
        f.write('\n'.join(labels))
    for key, objects in dataset.files_dict.items():
        file = open(f'{path}/{key.rsplit(".",1)[0]}.txt','w')
        cols = []
        width = int(objects['width'])
        height = int(objects['height'])
        for object_child in objects['object']:
            name = list(object_child.keys())[0]
            object_child = object_child[name]
            col = [0,0,0,0,0]
            col[3] = object_child['bbox'][2] / width
            col[4] = object_child['bbox'][3] / height
            col[1] = (object_child['bbox'][0] + (object_child['bbox'][2] / 2)) / width
            col[2] = (object_child['bbox'][1] + (object_child['bbox'][3] / 2)) / height
            col[0] = labels.index(name)
            cols.append(' '.join(map(str,col)))
        file.write('\n'.join(cols))


def to_voc(dataset, path = 'coco_to_voc'):

    try :
        os.mkdir(path)
    except :
        pass
    for key, objects in dataset.files_dict.items():
        voc = \
            f'''<annotation>
        <folder></folder>
        <filename>{key}</filename>
        <source>
                <database>COCO To VOC Database</database>
                <annotation>PASCAL VOC2007</annotation>
        </source>
        <size>
                <width>{objects['width']}</width>
                <height>{objects['height']}</height>
                <depth>3</depth>
        </size>
        <segmented>0</segmented>'''
        for object_child in objects['object']:
            name = list(object_child.keys())[0]
            object_child = object_child[name]
            if object_child['segmentation'] != [] :
                voc.replace('<segmented>0</segmented>', '<segmented>1</segmented>')
            voc += f'''
        <object>
                <name>{name}</name>
                <pose>Unspecified</pose>
                <truncated>0</truncated>
                <difficult>0</difficult>
                <occluded>0</occluded>
                <bndbox>
                        <xmin>{round(object_child['bbox'][0] + 1)}</xmin>
                        <ymin>{round(object_child['bbox'][1] + 1)}</ymin>
                        <xmax>{round(object_child['bbox'][0] + object_child['bbox'][2] + 1)}</xmax>
                        <ymax>{round(object_child['bbox'][1] + object_child['bbox'][3] + 1)}</ymax>
                </bndbox>
        </object>'''
        voc += '''
</annotation>'''
        with open(f'{path}/{key.rsplit(".",1)[0]}.xml','w') as f :
            f.write(voc)


def convLabel(data, label_dir_path, ratio):
    os.chdir(label_dir_path)
    label_verify = glob.glob(f'*.*')[0]
    label_format = os.path.splitext(label_verify)[1].strip()
    if label_format == '.json':
        to_coco(data, 'convType_label_output_coco')
    elif label_format == '.xml':
        to_voc(data, 'convType_label_output_voc')
    elif label_format == '.txt':
        to_yolo(data, 'convType_label_output_yolo')

# a = COCO(r"C:\Users\tiale\Downloads\MaskWearing.v4-raw.coco\train", r"C:\Users\tiale\Downloads\MaskWearing.v4-raw.coco\train\jsonf", None, None, None, None)
# a
# to_voc(a,'coco_to_voc')
# to_yolo(a,'coco_to_yolo')
# b = VOC(r"C:\Users\tiale\Downloads\voc_sample\JPEG", r"C:\Users\tiale\Downloads\voc_sample\label", None, None, None, None)
# b 
# to_coco(b,'voc_to_coco')
# to_yolo(b,'voc_to_yolo')
# c = YOLO(r"C:\Users\tiale\Downloads\MaskWearing.v4-raw.darknet\train", r"C:\Users\tiale\Downloads\MaskWearing.v4-raw.darknet\train", r"C:\Users\tiale\Downloads\MaskWearing.v4-raw.darknet\train\_darknet.labels", None, "wearing_mask", 600, 600)
# c
# to_coco(c,'yolo_to_coco')
# to_voc(c,'yolo_to_voc')


