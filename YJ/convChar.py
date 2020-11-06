import os

class convChar():
    
    def retype(self, file_list, input_type):
        file_name_dict = {}
        file_name_inv_dict = {}
        file_type_dict = {}
        for i, img_file_name in enumerate(file_list):
            retyped_img_file_name = f"{img_file_name.rsplit('.', 1)[0]}.{input_type}"
            file_list[i] = retyped_img_file_name
            raw_file_type = f".{img_file_name.rsplit('.', 1)[1]}"
            raw_img_file_name = f"{img_file_name.rsplit('.', 1)[0]}"
            file_name_dict[img_file_name] = retyped_img_file_name
            file_name_inv_dict[retyped_img_file_name] = img_file_name
            file_type_dict[raw_img_file_name] = raw_file_type
        return file_list, file_name_dict, file_name_inv_dict, file_type_dict
    
    def rename(self, file_list, input_name):
        file_name_dict = {}
        file_name_inv_dict = {}
        file_type_dict = {}
        for i, img_file_name in enumerate(file_list):
            renamed_img_file_name = f"{input_name}_{i}.{img_file_name.rsplit('.', 1)[1]}"
            file_list[i] = renamed_img_file_name
            raw_file_type = f".{img_file_name.rsplit('.', 1)[1]}"
            raw_img_file_name = f"{img_file_name.rsplit('.', 1)[0]}"
            file_name_dict[img_file_name] = renamed_img_file_name
            file_name_inv_dict[renamed_img_file_name] = img_file_name
            file_type_dict[raw_img_file_name] = raw_file_type 
        return file_list, file_name_dict, file_name_inv_dict, file_type_dict
    
    def reshape(self, read_img, input_x, input_y):
        read_img = list(read_img)
        if input_y != None:
            read_img[0] = int(input_y)
        if input_x != None:
            read_img[1] = int(input_x)
        return read_img
