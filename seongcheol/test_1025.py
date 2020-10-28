import os
import mahotas as mh
import numpy as np
import cv2
from matplotlib import pyplot as plt
from glob import glob
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
# %matplotlib inline


path = "./imgimg/*.jpg"

def feature_extraction(path):

    images = glob(path)
    features = []
    # labels = []
    for i in images:
    #    labels.append(i[14:-len('.jpg')])
        i = mh.imread(i)
        i = mh.colors.rgb2grey(i, dtype = np.uint8)
        features.append(mh.features.haralick(i).ravel())
    features = np.array(features)
    sc = StandardScaler()
    features = sc.fit_transform(features)
    dists = distance.squareform(distance.pdist(features))

    return images, dists
# print(len(feature_extraction(path)[0]))

def select_all(n, dists, images):
    dists_idx_sorted = dists[n].argsort()                                      
    similar_list = []
    for i in dists_idx_sorted:       
        if dists[n][i] < 3.5:
            similar_list.append(images[i])
    return similar_list



img_dict = dict()
for i in range(len(feature_extraction(path)[0])):
    img_dict[str(feature_extraction(path)[0][i])] = select_all(i,feature_extraction(path)[1],feature_extraction(path)[0])
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
print(img_dict)



def size_normal(path):

    g_path = glob.glob(path)
    cv_img = []

    for i in g_path:
        n = cv2.imread(i)
        cv_img.append(n)

    # print(cv_img[0].shape[0])


    m_width = 0            
    m_height = 0
    
    for j in range(len(cv_img)):
        if m_width < cv_img[j].shape[0]:
            m_width = cv_img[j].shape[0]
        elif m_height < cv_img[j].shape[1]:
            m_height = cv_img[j].shape[1]
    
    return cv_img, m_width, m_height



def r_size(img_list, w, h):

    re_size = [] 
    for i in img_list:
        re_size.append(cv2.resize(i, dsize = (w,h), interpolation=cv2.INTER_AREA ))
    return re_size


#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------

# os.getcwd()  path

def imgwrite(slist,f_name):
    os.mkdir(str(f_name))
    count = 0
    for i in slist:
        
        imgfile = i
        img = cv2.imread(imgfile, cv2.IMREAD_COLOR)
        cv2.imwrite(str(f_name)+'/' + str(count)+'.jpg',img)
        count = count + 1

# imgwrite(select_all(5,feature_extraction(path)[1],feature_extraction(path)[0]),'test_sim_img')




