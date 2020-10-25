import mahotas as mh
import numpy as np
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


def select_all(n, dists, images):
    dists_idx_sorted = dists[n].argsort()                                      
    similar_list = []
    for i in dists_idx_sorted:       
        if dists[n][i] < 3.5:
            similar_list.append(images[i])
    return similar_list




