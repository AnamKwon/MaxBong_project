# 파이썬 버전확인
import sys
print(sys.version)

#supported data Type in PIL: [BMP, DIB, EPS, GIF, ICNS, ICO, IM, JPEG, MSP, PCX, PNG, PPM, SGI, SPIDER, TGA, TIFF, Webp]
#reference URL: https://appia.tistory.com/353
# file_list = ['D:/dacon/SMILES/data/data\\train_0.png', 'D:/dacon/SMILES/data/data\\train_1.png', 'D:/dacon/SMILES/data/data\\train_10.png', 'D:/dacon/SMILES/data/data\\train_100.png', 'D:/dacon/SMILES/data/data\\train_1000.png', 'D:/dacon/SMILES/data/data\\train_10000.png', 'D:/dacon/SMILES/data/data\\train_100000.png', 'D:/dacon/SMILES/data/data\\train_100001.png', 'D:/dacon/SMILES/data/data\\train_100002.png', 'D:/dacon/SMILES/data/data\\train_100003.png', 'D:/dacon/SMILES/data/data\\train_100004.png', 'D:/dacon/SMILES/data/data\\train_100005.png', 'D:/dacon/SMILES/data/data\\train_100006.png', 'D:/dacon/SMILES/data/data\\train_100007.png', 'D:/dacon/SMILES/data/data\\train_100008.png', 'D:/dacon/SMILES/data/data\\train_100009.png', 'D:/dacon/SMILES/data/data\\train_10001.png', 'D:/dacon/SMILES/data/data\\train_100010.png', 'D:/dacon/SMILES/data/data\\train_100011.png', 'D:/dacon/SMILES/data/data\\train_100012.png', 'D:/dacon/SMILES/data/data\\train_100013.png', 'D:/dacon/SMILES/data/data\\train_100014.png', 'D:/dacon/SMILES/data/data\\train_100015.png', 'D:/dacon/SMILES/data/data\\train_100016.png', 'D:/dacon/SMILES/data/data\\train_100017.png', 'D:/dacon/SMILES/data/data\\train_100018.png', 'D:/dacon/SMILES/data/data\\train_100019.png', 'D:/dacon/SMILES/data/data\\train_10002.png', 'D:/dacon/SMILES/data/data\\train_100020.png', 'D:/dacon/SMILES/data/data\\train_100021.png', 'D:/dacon/SMILES/data/data\\train_100022.png', 'D:/dacon/SMILES/data/data\\train_100023.png', 'D:/dacon/SMILES/data/data\\train_100024.png', 'D:/dacon/SMILES/data/data\\train_100025.png', 'D:/dacon/SMILES/data/data\\train_100026.png', 'D:/dacon/SMILES/data/data\\train_100027.png', 'D:/dacon/SMILES/data/data\\train_100028.png', 'D:/dacon/SMILES/data/data\\train_100029.png', 'D:/dacon/SMILES/data/data\\train_10003.png', 'D:/dacon/SMILES/data/data\\train_100030.png', 'D:/dacon/SMILES/data/data\\train_100031.png', 'D:/dacon/SMILES/data/data\\train_100032.png', 'D:/dacon/SMILES/data/data\\train_100033.png', 'D:/dacon/SMILES/data/data\\train_100034.png', 'D:/dacon/SMILES/data/data\\train_100035.png', 'D:/dacon/SMILES/data/data\\train_100036.png', 'D:/dacon/SMILES/data/data\\train_100037.png', 'D:/dacon/SMILES/data/data\\train_100038.png', 'D:/dacon/SMILES/data/data\\train_100039.png', 'D:/dacon/SMILES/data/data\\train_10004.png', 'D:/dacon/SMILES/data/data\\train_100040.png', 'D:/dacon/SMILES/data/data\\train_100041.png', 'D:/dacon/SMILES/data/data\\train_100042.png', 'D:/dacon/SMILES/data/data\\train_100043.png', 'D:/dacon/SMILES/data/data\\train_100044.png', 'D:/dacon/SMILES/data/data\\train_100045.png', 'D:/dacon/SMILES/data/data\\train_100046.png', 'D:/dacon/SMILES/data/data\\train_100047.png', 'D:/dacon/SMILES/data/data\\train_100048.png', 'D:/dacon/SMILES/data/data\\train_100049.png', 'D:/dacon/SMILES/data/data\\train_10005.png', 'D:/dacon/SMILES/data/data\\train_100050.png', 'D:/dacon/SMILES/data/data\\train_100051.png', 'D:/dacon/SMILES/data/data\\train_100052.png', 'D:/dacon/SMILES/data/data\\train_100053.png', 'D:/dacon/SMILES/data/data\\train_100054.png', 'D:/dacon/SMILES/data/data\\train_100055.png', 'D:/dacon/SMILES/data/data\\train_100056.png', 'D:/dacon/SMILES/data/data\\train_100057.png', 'D:/dacon/SMILES/data/data\\train_100058.png', 'D:/dacon/SMILES/data/data\\train_100059.png', 'D:/dacon/SMILES/data/data\\train_10006.png', 'D:/dacon/SMILES/data/data\\train_100060.png', 'D:/dacon/SMILES/data/data\\train_100061.png', 'D:/dacon/SMILES/data/data\\train_100062.png', 'D:/dacon/SMILES/data/data\\train_100063.png', 'D:/dacon/SMILES/data/data\\train_100064.png', 'D:/dacon/SMILES/data/data\\train_100065.png', 'D:/dacon/SMILES/data/data\\train_100066.png', 'D:/dacon/SMILES/data/data\\train_100067.png', 'D:/dacon/SMILES/data/data\\train_100068.png', 'D:/dacon/SMILES/data/data\\train_100069.png', 'D:/dacon/SMILES/data/data\\train_10007.png', 'D:/dacon/SMILES/data/data\\train_100070.png', 'D:/dacon/SMILES/data/data\\train_100071.png', 'D:/dacon/SMILES/data/data\\train_100072.png', 'D:/dacon/SMILES/data/data\\train_100073.png', 'D:/dacon/SMILES/data/data\\train_100074.png', 'D:/dacon/SMILES/data/data\\train_100075.png', 'D:/dacon/SMILES/data/data\\train_100076.png', 'D:/dacon/SMILES/data/data\\train_100077.png', 'D:/dacon/SMILES/data/data\\train_100078.png', 'D:/dacon/SMILES/data/data\\train_100079.png', 'D:/dacon/SMILES/data/data\\train_10008.png', 'D:/dacon/SMILES/data/data\\train_100080.png', 'D:/dacon/SMILES/data/data\\train_100081.png']
# input_type = 'BMP'
# new_name = 'aaa'
# x = '1000'.strip()
# y = '   '.strip()

#pyQT적용시, newfile_path 불필요?!
def convType(file_list, input_type, new_name, x, y):
    
    print("\n")
    print("--start point--")
    
    import os
    from PIL import Image

    file_path = file_list[0].rsplit('\\',1)[0]
    try :
        os.mkdir(file_path+r'/conv')
    except:
        pass
    # 데이터타입 변경
    print("<converting type progress>")
    cnt = 0
    for file_name in file_list:
        img = Image.open(file_name)
        if new_name == '':
            newfile_name = file_name.rsplit('\\',1)[1].split(".")[0]
        else :
            newfile_name = f'{new_name}_{cnt}'
            cnt += 1
        width, height = img.size
        if x != '' or y != '' :
            print(x,y)
            if x == '' :
                y = int(y)
                width = int(width*(y/height))
                height=y
            elif y == '' :
                x = int(x)
                height = int(height*(x/width))
                width=x
            else :
                width, height = int(x), int(y)
        img = img.resize((width, height))
        print(file_path + r"/conv/" + f"{newfile_name}.{input_type}")
        img.save(file_path + r"/conv/" + f"{newfile_name}.{input_type}", input_type)
        print(f">>> now progressing: converting {file_name} to {input_type} format")
        
        
        cnt = cnt + 1
    
    print("--end point--")
    print("\n")
#convType(file_list,input_type,new_name,x,y)
#convType(r"D:\dacon\SMILES\data\data", r"D:\dacon\SMILES\data\data\conv", "png")