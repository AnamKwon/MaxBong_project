
class COCO:
    def __init__(self, a):
        self.files = {}
        self.file_to_id = {}
        self.categories = {i['id']: i['name'] for i in a['categories']}
        for num in a['images']:
            self.files[num['file_name']] = {'width': num['width'],
                                            'height': num['height'],
                                            'object': []}
            self.file_to_id[num['id']] = num['file_name']
        for num in a['annotations']:
            b = {self.categories[num['category_id']]: {'bbox': num['bbox'], 'segmentation': num['segmentation']}}
            self.files[self.file_to_id[num['image_id']]]['object'].append(b)

    def to_voc(self):
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
    </size>'''

            for object_ in objects['object']:
                name = list(object_.keys())[0]
                object_ = object_[name]
                voc += f'''
    <segmented>{'0' if object_['segmentation'] == [] else object_['segmentation']}</segmented>
    <object>
        <name>{name}</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <occluded>0</occluded>
        <bndbox>
            <xmin>{str(int(object_['bbox'][0]))}</xmin>
            <xmax>{str(int(object_['bbox'][0] + object_['bbox'][2]))}</xmax>
            <ymin>{str(int(object_['bbox'][1]))}</ymin>
            <ymax>{str(int(object_['bbox'][1] + object_['bbox'][3]))}</ymax>
        </bndbox>
    </object>'''
            voc += '''
</annotation>'''
            print(voc)
