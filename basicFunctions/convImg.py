import os
import sys
import cv2
import glob
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import Format

def convCharacter(img_dir_path, label_dir_path, label_name_file, return_dataset_type, retype_index, rename_input, resize_width=1, resize_height=1):
    os.chdir(label_dir_path)
    label_verify = glob.glob(f'*.*')[0]
    label_format = os.path.splitext(label_verify)[1].strip()
    
    if label_format == '.json':
        dataset = Format.COCO(img_dir_path, label_dir_path)
    elif label_format == '.xml':
        dataset = Format.VOC(img_dir_path, label_dir_path)
    elif label_format == '.txt':
        dataset = Format.YOLO(img_dir_path, label_dir_path, label_name_file)
    # print(dataset.files_dict)
    os.chdir(img_dir_path)
    return_type_list = ['None', '.jpg', '.png', '.bmp']
    type_list = ['jpg', 'png', 'bmp']
    file_list = []
    for type in type_list:
        file_list += glob.glob(f'*.{type}')

    name_to_newname_dict = {}
    name_to_newtype_dict = {}
    name_to_ratio_dict = {}
    output_dir_path = 'convImg_output'
    os.mkdir(output_dir_path)

    
    for idx, file in enumerate(file_list):
        # print(file + " process start")
        img = cv2.imread(file)
        height, width, depth = img.shape
        resize_ratio = []
        dw = 1
        dh = 1
        filename = os.path.splitext(file)[0]
        filetype = os.path.splitext(file)[1]

        # reshape
        # resize_width, resize_height default = 1
        if (resize_width != 1) and (resize_height == 1):
            dw = resize_width / width
            img = cv2.resize(img, (resize_width, height))
        elif (resize_width == 1) and (resize_height != 1):
            dh = resize_height / height
            img = cv2.resize(img, (width, resize_height))
        elif (resize_width != 1) and (resize_height != 1):
            dw = resize_width / width
            dh = resize_height / height
            img = cv2.resize(img, (resize_width, resize_height))
        # ratio data dict
        resize_ratio.append(dw)
        resize_ratio.append(dh)
        name_to_ratio_dict[file] = resize_ratio

        # retype
        if retype_index != 0:
            filetype = return_type_list[retype_index]
            name_to_newtype_dict[file] = filetype
        
        # rename
        newfilename = filename + filetype
        if bool(rename_input) != False:
            newfilename = rename_input + '_' + str(idx + 1) + filetype
            name_to_newname_dict[file] = newfilename
        
        cv2.imwrite(os.path.join(output_dir_path, newfilename), img)
        # print(file + "process succese")

    conv_dataset = Format.convLabel(dataset, label_dir_path, name_to_ratio_dict, name_to_newname_dict, name_to_newtype_dict)
    if bool(return_dataset_type) == False:
        return_dataset_type = label_format
    if return_dataset_type == '.json':
        Format.to_coco(conv_dataset, 'convType_label_output_coco')
    elif return_dataset_type == '.xml':
        Format.to_voc(conv_dataset, 'convType_label_output_voc')
    elif return_dataset_type == '.txt':
        Format.to_yolo(conv_dataset, 'convType_label_output_yolo')

if __name__ == "__main__":
    # convCharacter('C:/Users/audrj/OneDrive/바탕 화면/me/final/convert/main_project/MaxBong_project/coco/val2014',
    #                 'C:/Users/audrj/OneDrive/바탕 화면/me/final/convert/main_project/MaxBong_project/coco/json',
    #                 '',
    #                 '.xml',
    #                 0,
    #                 'test',
    #                 225,
    #                 225)
    
    convCharacter('C:/Users/audrj/OneDrive/바탕 화면/me/final/convert/main_project/MaxBong_project/voc/JPEG',
                    'C:/Users/audrj/OneDrive/바탕 화면/me/final/convert/main_project/MaxBong_project/voc/label',
                    '',
                    '.txt',
                    0,
                    'test',
                    225,
                    225)