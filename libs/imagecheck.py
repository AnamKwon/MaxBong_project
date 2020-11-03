import os
import mahotas as mh
import numpy as np
from glob import glob
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
# %matplotlib inline


path = "./imgimg/*.jpg"

def feature_extraction(images):

    # images = glob(path)
    features = []
    # # labels.txt = []
    for i in images:
    #    labels.txt.append(i[14:-len('.jpg')])
        i = mh.imread(i)
        i = mh.colors.rgb2grey(i, dtype = np.uint8)
        features.append(mh.features.haralick(i).ravel())
    features = np.array(features)
    sc = StandardScaler()
    features = sc.fit_transform(features)
    dists = distance.squareform(distance.pdist(features))

    return images, dists
# print(len(feature_extraction(path)[0]))

def select_all(n, dists, images, C):
    dists_idx_sorted = dists[n].argsort()                                      
    similar_list = []
    for i in dists_idx_sorted:
        if n == i :
            continue
        if dists[n][i] < C:
            similar_list.append(images[i])
    return similar_list

def check_img(images, path, C=3.5) :
    os.chdir(path)
    img_dict = {}
    features = feature_extraction(images)
    for i in range(len(features[0])) :
        img_dict[str(features[0][i])] = select_all(i, features[1],features[0],C)
    return img_dict

#print(check_img(['000001_jpg.rf.cab894412fc394ddc10ffab614099c5b.jpg', '000003_jpg.rf.7882b87dd3241d56b35e293c2c8287ed.jpg', '000004_jpg.rf.b11b49bb7d0b720c6bd2c968fe0e3ae1.jpg', '000006_jpg.rf.343f1e2ef86a875e2feadb4765c5049c.jpg'],'D:/company_pro/pythonProject/dataset'))
#
# img_dict = dict()
# for i in range(len(feature_extraction(path)[0])):
#     img_dict[str(feature_extraction(path)[0][i])] = select_all(i,feature_extraction(path)[1],feature_extraction(path)[0])
# #---------------------------------------------------------------------------------------------
# #---------------------------------------------------------------------------------------------
# print(img_dict)
