import os
import json
import xml

class COCO:
    def __init__(self, file):

        with open(file) as f :
            file = json.loads(f.read())
        self.files = {}
        self.file_to_id = {}
        self.categories = {i['id']: i['name'] for i in file['categories']}
        for num in file['images']:
            self.files[num['file_name']] = {'width': num['width'],
                                            'height': num['height'],
                                            'object': []}
            self.file_to_id[num['id']] = num['file_name']
        for num in file['annotations']:
            b = {self.categories[num['category_id']]: {'bbox': num['bbox'], 'segmentation': num['segmentation']}}
            self.files[self.file_to_id[num['image_id']]]['object'].append(b)

    def to_voc(self, path = 'coco_to_voc'):
        try :
            os.mkdir(path)
        except :
            pass
        for key, objects in self.files.items():
            voc = \
                f'''<annotation>
        <folder></folder>
        <filename>{key}</filename>
        <path>{key}</path>
        <source>
                <database>max_bong_project</database>
        </source>
        <size>
                <width>{objects['width']}</width>
                <height>{objects['height']}</height>
                <depth>3</depth>
        </size>
        <segmented>0</segmented>'''
            for object_ in objects['object']:
                name = list(object_.keys())[0]
                object_ = object_[name]
                if object_['segmentation'] != [] :
                    voc.replace('<segmented>0</segmented>','<segmented>1</segmented>')
                voc += f'''
        <object>
                <name>{name}</name>
                <pose>Unspecified</pose>
                <truncated>0</truncated>
                <difficult>0</difficult>
                <occluded>0</occluded>
                <bndbox>
                        <xmin>{round(object_['bbox'][0]+1)}</xmin>
                        <xmax>{round(object_['bbox'][0] + object_['bbox'][2]+1)}</xmax>
                        <ymin>{round(object_['bbox'][1]+1)}</ymin>
                        <ymax>{round(object_['bbox'][1] + object_['bbox'][3]+1)}</ymax>
                </bndbox>
        </object>'''
            voc += '''
</annotation>'''
            with open(f'{path}/{key.rsplit(".",1)[0]}.xml','w') as f :
                f.write(voc)



    def to_yolo(self,path='coco_to_yolo') :
        try :
            os.mkdir(path)
        except :
            pass
        labes = list(self.categories.values())[1::]
        with open(f'{path}/_darknet.labels','w') as f :
            f.write('\n'.join(labes))
        for key, objects in self.files.items():
            file = open(f'{path}/{key.rsplit(".",1)[0]}.txt','w')
            cols = []
            width = objects['width']
            height = objects['height']
            for object_ in objects['object']:
                name = list(object_.keys())[0]
                object_ = object_[name]
                col = [0,0,0,0,0]
                col[3] = object_['bbox'][2]/width
                col[4] = object_['bbox'][3]/height
                col[1] = (object_['bbox'][0]+(object_['bbox'][2]/2))/width
                col[2] = (object_['bbox'][1]+(object_['bbox'][3]/2))/height
                col[0] = labes.index(name)
                cols.append(' '.join(map(str,col)))
            file.write('\n'.join(cols))
