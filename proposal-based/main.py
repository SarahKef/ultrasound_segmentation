import numpy as np
from sklearn.linear_model import LogisticRegression as LR
import cv2
import sw

DATAPATH = "../../USNS/RawData/train/"
img1 = cv2.imread(DATAPATH+'1_1.tif',cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(DATAPATH+'1_2.tif',cv2.IMREAD_GRAYSCALE)
img1_mask = cv2.imread(DATAPATH+'1_1_mask.tif',cv2.IMREAD_GRAYSCALE)
img2_mask = cv2.imread(DATAPATH+'1_2_mask.tif',cv2.IMREAD_GRAYSCALE)

img1 = cv2.resize(img1, (330,330))
img2 = cv2.resize(img2, (330,330))
img1_mask = cv2.resize(img1_mask, (330,330))
img2_mask = cv2.resize(img2_mask, (330,330))

img1 = img1.astype(np.float32)/255.0
img2 = img2.astype(np.float32)/255.0
img1_mask = img1_mask.astype(np.float32)/255.0
img2_mask = img2_mask.astype(np.float32)/255.0

img1_windows = sw.sliding_window(img1, (30,30),(10,10))
img2_windows = sw.sliding_window(img2, (30,30),(10,10))
img1_mask_windows = sw.sliding_window(img1_mask, (30,30),(10,10))
img2_mask_windows = sw.sliding_window(img2_mask, (30,30),(10,10))

img1_windows = img1_windows.reshape(img1_windows.shape[0], img1_windows.shape[1]*img1_windows.shape[2])
img2_windows = img1_windows.reshape(img2_windows.shape[0], img2_windows.shape[1]*img2_windows.shape[2])
img1_mask_windows_labels = np.array([w.sum()/900.0 > 0.5 for w in img1_mask_windows]).reshape(img1_mask_windows.shape[0])
img2_mask_windows_labels = np.array([w.sum()/900.0 > 0.5 for w in img2_mask_windows]).reshape(img2_mask_windows.shape[0])

classifier = LR().fit(img1_windows, img1_mask_windows_labels)

print classifier.score(img2_windows, img2_mask_windows_labels)

img2_windows_labels = classifier.predict_proba(img2_windows)
img2_windows_labels = img2_windows_labels.reshape(31,31,2)

def get_interval_begins(x, w, s, N):
    p = []
    i =0
    while s*i + w <= N:
        if s*i + w < x:
            i = i + 1
        elif s*i > x:
            break
        else:
            p.append(i)
            i = i+ 1
    return p

img = [list(w) for w in list(np.zeros(shape=(330,330)))]

for j in range(330):
    for i in range(330):
        ilist = get_interval_begins(i, 30, 10, 330)
        jlist = get_interval_begins(j, 30, 10, 330)
        points = []
        for jp in jlist:
            for ip in ilist:
                points.append([ip,jp])
        if points != []:
            img[i][j] = points

mask = [list(w) for w in list(np.zeros(shape=(330,330)))]

for j in range(330):
    for i in range(330):
        windows_list = img[i][j]
        mask[i][j] = sum([img2_windows_labels[r][s][1] for r,s in windows_list ])/len(windows_list)

mask = np.array(mask)
# mask = (mask.squeeze()*255).astype(np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(50,50))
# mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
# cv2.imwrite("poo.png",mask)

import numpy as np
import os
import cv2
import sys
_data = np.load("../../USNS/ResizedData/train/compressed/trainData.npz")
Y_true = _data["Y"][:,None,:,:]

del _data
print "Loaded all Data"
from sklearn.decomposition import TruncatedSVD
# pca2 = TruncatedSVD(n_components=20).fit(Y_true.reshape(-1, 128*128))
# print "PCA Fit Done"
from sklearn.externals import joblib
pca2 = joblib.load('pca.pkl')


Y128 = cv2.resize(mask.squeeze(), (128, 128))
Y128 = Y128*255
Y_r = pca2.transform(Y128.reshape(-1))
corr_mask = pca2.inverse_transform(Y_r).reshape(128, 128)
corr_mask = cv2.resize(mask, (330, 330)) > 0.5
corr_mask = corr_mask.astype(np.uint8)
corr_mask = cv2.morphologyEx(corr_mask, cv2.MORPH_CLOSE, kernel)
cv2.imwrite("poo_corr.png",corr_mask)