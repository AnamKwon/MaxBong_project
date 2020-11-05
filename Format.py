import os
import json
import time
import glob
import cv2
from xml.etree import ElementTree as ET

""" common dataset
dataset = {
            'filename':{
                        'width':<int>
                        'height':<int>
                        'depth':<int>
                        'object':[
                                    {
                                        'category':{
                                                    'bbox':[
                                                        xmin,
                                                        ymin,
                                                        width,
                                                        height
                                                    ],
                                                    'segmentation':[]
                                                    }
                                                    ...
                                    }
                                ]
                        }
                        ...
            } 
"""
class COCO:

    def __init__(self, img_dir_path, json_dir_path):
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
        self.files_dict = {}
        self.file_to_id = {}
        self.id_to_file = {}
        self.id_to_categories = {i['id']: i['name'] for i in file['categories']}
        self.categories_to_id = {i['name']: i['id']  for i in file['categories']}
        not_image = []
        for num in file['images']:
            if num['file_name'] not in file_list :
                not_image.append(num['id'])
                continue
            read_img = cv2.imread(num['file_name']).shape
            if num['width'] != read_img[1] or num['height'] != read_img[0] :
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
            b = {self.id_to_categories[num['category_id']]: {'bbox': num['bbox'], 'segmentation': num['segmentation']}}
            self.files_dict[self.id_to_file[num['image_id']]]['object'].append(b)



class YOLO :

    def __init__(self, img_dir_path, txt_dir_path, label='labels.txt'):
        file_list = []
        img_file_types = ['jpg','png','bmp','jpeg']
        # glob(img.type)을 위한 현재폴더 변경
        os.chdir(img_dir_path)
        for type in img_file_types :
            file_list += glob.glob(f'*.{type}')
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

        for file in file_list:
            filename = file.rsplit('.',1)[0]
            filename = filename+'.txt'
            if filename not in files :
                continue
            # txt 파일 불러오기 위한 현재폴더 변경
            os.chdir(txt_dir_path)
            with open(filename) as f :
                objects = f.readlines()
            self.file_to_id[filename] = file_to_id_num
            self.id_to_file[file_to_id_num] = filename
            file_to_id_num += 1
            # img 파일 읽어오기 위한 현재폴더 설정(openCV 경우 현재 폴더에서 이미지를 입력하지 않으면 안먹힘)
            os.chdir(img_dir_path)
            height, width, depth = cv2.imread(file).shape
            self.files_dict[file] = {'width': width,
                                            'height': height,
                                            'depth' : depth,
                                            'object': []}
            for object in objects :
                object = object.strip().split()
                object_name = self.id_to_categories[int(object[0])]
                box = [0,
                       0,
                       round(float(object[3]) * width),
                       round(float(object[4]) * height)
                ]
                box[0] = round((float(object[1]) * width) - (box[2]/2))
                box[1] = round((float(object[2]) * height) - (box[3]/2))
                object_info = {object_name : {'bbox' : box, 'segmentation' : []}}
                self.files_dict[file]['object'].append(object_info)


class VOC :

    def __init__(self, img_dir_path, xml_dir_path):
        file_list = []
        img_file_types = ['jpg','png','bmp']
        # glob(img.type)을 위한 현재폴더 변경
        os.chdir(img_dir_path)
        for type in img_file_types :
            file_list += glob.glob(f'*.{type}')
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
        for file in files :
            # xml 파일 읽어오기 위한 현재폴더 설정
            os.chdir(xml_dir_path)
            tree = ET.parse(file)
            filename = tree.find('filename').text
            if filename not in file_list :
                continue
            size = tree.find('size')
            # img 파일 읽어오기 위한 현재폴더 설정(openCV 경우 현재 폴더에서 이미지를 입력하지 않으면 안먹힘)
            os.chdir(img_dir_path)
            read_img = cv2.imread(filename).shape
            if int(size.find('width').text) != read_img[1] or int(size.find('height').text) != read_img[0] :
                continue
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
                box = [int(bndbox.find('xmin').text) - 1,
                       int(bndbox.find('ymin').text) - 1,
                       int(bndbox.find('xmax').text) - int(bndbox.find('xmin').text),
                       int(bndbox.find('ymax').text) - int(bndbox.find('ymin').text)]
                object_info = {object_name : {'bbox' : box, 'segmentation' : segmentation}}
                self.files_dict[filename]['object'].append(object_info)



def to_coco(dataset, path = 'voc_to_coco', user='max_bong'):

    try :
        os.mkdir(path)
    except: pass;
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
               }
              ],
         "id_to_categories" :
             [{"id":key,
               "name" : value,
               "supercategory": f"{user}_project"} for key, value in dataset.id_to_categories.items()],
         "images":
             [],
         "annotations" :
             []
         }
    images_num = 0
    annotations_num = 0
    for file, value in dataset.files_dict.items():
        image = {"id": images_num,
                 "license": 1,
                 "file_name": f"{file}",
                 "height": int(f"{value['height']}"),
                 "width": int(f"{value['width']}"),
                 "date_captured": times
                 }
        coco['images'].append(image)
        for object in value['object'] :
            category, bbox = list(object.items())[0]
            annotation = {"id": int(f"{annotations_num}"),
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


def to_yolo(dataset,path='voc_to_yolo') :

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


class convLabel():

    def __init__(self, dataset, label_dir_path, name_to_ratio_dict, name_to_rename_dict, name_to_retype_dict):

        self.data_dict = {}
        self.id_to_categories = dataset.id_to_categories
        self.categories_to_id = dataset.categories_to_id

        for file, file_dict in dataset.files_dict.items():
            filename = os.path.splitext(file)[0]
            filetype = os.path.splitext(file)[1]
            # rename verify
            if bool(name_to_rename_dict) != False:
                filename = os.path.splitext(name_to_rename_dict[file])[0]
            # retype verify
            if bool(name_to_retype_dict) != False:
                filetype = name_to_retype_dict[file]
            new_filename = filename + filetype

            new_file_dict = {}
            img_width = file_dict['width']
            img_height = file_dict['height']
            img_depth = file_dict['depth']
            file_object = file_dict['object']

            ratio = [1, 1]
            # rasize verify
            if bool(name_to_ratio_dict) != False:
                ratio = name_to_ratio_dict[file]

            object_list = []
            for obj in file_object:
                obj_dict = {}
                new_cate_dict = {}
                category = list(obj.items())[0][0]
                cate_dict = list(obj.items())[0][1]
                # x_center = (xmin + width/2) * dw
                bbox = cate_dict['bbox']
                x_center = (bbox[0] + bbox[2]/2) * ratio[0]
                y_center = (bbox[1] + bbox[3]/2) * ratio[1]
                bnd_width = bbox[2] * ratio[0]
                bnd_height = bbox[3] * ratio[1]
                segmentation = cate_dict['segmentation']

                re_xmin = x_center - bnd_width/2
                re_ymin = y_center - bnd_height/2

                new_cate_dict['bbox'] = [re_xmin,
                                        re_ymin,
                                        bnd_width,
                                        bnd_height
                                        ]
                new_cate_dict['segmentation'] = segmentation
                obj_dict[category] = new_cate_dict
                object_list.append(obj_dict)

            new_file_dict['width'] = img_width
            new_file_dict['height'] = img_height
            new_file_dict['depth'] = img_depth
            new_file_dict['object'] = object_list
            self.data_dict[new_filename] = new_file_dict

        self.files_dict = self.data_dict



# a = COCO('C:/Users/audrj/OneDrive/바탕 화면/me/final/convert/main_project/MaxBong_project/coco/val2014', 
        # 'C:/Users/audrj/OneDrive/바탕 화면/me/final/convert/main_project/MaxBong_project/coco/json')
# to_voc(a,'coco_to_voc')
# to_yolo(a,'coco_to_yolo')
# b = VOC('C:/Users/audrj/OneDrive/바탕 화면/me/final/convert/main_project/MaxBong_project/voc/JPEG', 
#         'C:/Users/audrj/OneDrive/바탕 화면/me/final/convert/main_project/MaxBong_project/voc/label')
# to_coco(b,'voc_to_coco')
# to_yolo(b,'voc_to_yolo')
# c = YOLO('','labels.txt')
# to_coco(c,'yolo_to_coco')
# to_voc(c,'yolo_to_voc')


