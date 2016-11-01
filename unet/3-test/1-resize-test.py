import os
import numpy as np
import cv2
import glob
import skimage.util
import scipy.spatial.distance as spdist
import pylab as pl
import sys
DATAPATH = "../../USNS/RawData/test"
DATASAVEPATH = "../../USNS/ResizedData/test"

NEWIMGSIZE = (128,128)
def loadPreProcess(imgName):
    print '\r %s' % (imgName),
    sys.stdout.flush()
    img = cv2.imread(DATAPATH+"/"+str(imgName)+".tif", cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,NEWIMGSIZE)
    
    imgname = str(imgName) + ".png"
    cv2.imwrite(DATASAVEPATH+"/"+imgname, img)
    
    img = img.astype(np.float32)/255.0
    np.ascontiguousarray(img)
    return img

imgNames = [int(fname.split(".")[0]) for fname in os.listdir(DATAPATH)]
imgNames.sort()
imgs = [loadPreProcess(fname) for fname in imgNames]
X = np.array(imgs)
print X.shape
np.savez(DATASAVEPATH+"/compressed/testData.npz",X_test=X)