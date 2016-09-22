import os
import numpy as np
import cv2
import glob
import skimage.util
import scipy.spatial.distance as spdist
import pylab as pl
DATAPATH = "../USNS/RawData/test"
DATASAVEPATH = "../USNS/ResizedData/test"

NEWIMGSIZE = (128,128)
def loadPreProcess(imgName):
    img = cv2.imread(DATAPATH+"/"+imgName, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,NEWIMGSIZE)
    cv2.imwrite(DATASAVEPATH+"/"+imgName, img)
    img = img.astype(np.float32)/255.0
    np.ascontiguousarray(img)    
    return img
imgNames = [fname for fname in os.listdir(DATAPATH)]
imgs = zip(*[loadPreProcess(fname) for fname in imgNames])
X_test = np.array(imgs)
np.savez(DATASAVEPATH+"/compressed/testData.npz",X_test=X_test)
