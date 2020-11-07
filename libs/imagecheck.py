import os
import mahotas as mh
import numpy as np
from glob import glob
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler

def feature_extraction(images):

    features = []
    for i in images:
        i = mh.imread(i)
        i = mh.colors.rgb2grey(i, dtype = np.uint8)
        features.append(mh.features.haralick(i).ravel())
    features = np.array(features)
    sc = StandardScaler()
    features = sc.fit_transform(features)
    dists = distance.squareform(distance.pdist(features))

    return images, dists

def select_all(n, dists, images, C):
    dists_idx_sorted = dists[n].argsort()                                      
    similar_list = []
    for i in dists_idx_sorted:
        if n == i :
            continue
        if dists[n][i] < C:
            similar_list.append(images[i].rsplit('/',1)[1])
    return similar_list

def check_img(images, path, C=3.5) :
    os.chdir(path)
    img_dict = {}
    features = feature_extraction(images)
    for i in range(len(features[0])) :
        img_dict[str(features[0][i]).rsplit('/',1)[1]] = select_all(i, features[1],features[0],C)
    return img_dict
