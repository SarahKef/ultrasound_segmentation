import os
import numpy as np
import cv2
import glob
import sys
import skimage.util
import dice
import scipy.spatial.distance as spdist
import pylab as pl
DATAPATH = "../USNS/RawData/train"
DATASAVEPATH = "../USNS/ResizedData/train"

NEWIMGSIZE = (128,128)
def loadPreProcess(imgName):
    print '\r %s' % (imgName),
    sys.stdout.flush()
    pnumber, imnumber = imgName.split(".")[0].split('_')
    maskName = imgName.split(".")[0]+"_mask.tif"
    
    img = cv2.imread(DATAPATH+"/"+imgName, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(DATAPATH+"/"+maskName, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,NEWIMGSIZE)
    mask = cv2.resize(mask,NEWIMGSIZE)
    
    imgname = imgName.split(".")[0] + ".png"
    mask_fname = imgName.split(".")[0] + "_mask.png"
    cv2.imwrite(DATASAVEPATH+"/"+imgname, img)
    cv2.imwrite(DATASAVEPATH+"/"+mask_fname, mask)
    

    img = img.astype(np.float32)/255.0
    mask = (mask > 128).astype(np.float32)
    np.ascontiguousarray(img)
    
    return img,mask,int(pnumber),imgName
imgNames = [fname for fname in os.listdir(DATAPATH) if "mask" not in fname]
imgs, masks, pnumbers, imgNames = zip(*[loadPreProcess(fname) for fname in imgNames])
X = np.array(imgs)
Y = np.array(masks)
pids = np.array(pnumbers)
imgNames = np.array(imgNames)
np.savez(DATASAVEPATH+"/compressed/trainData.npz",X=X,Y=Y,pids=pids,imgNames=imgNames)
i=0
dope = {}
for pid in set(pids):
    dope[pid]={"X":[],"Y":[],"imgNames":[]}

for i in range(len(X)):
    dope[pids[i]]["X"].append(X[i])
    dope[pids[i]]["Y"].append(Y[i])    
    dope[pids[i]]["imgNames"].append(imgNames[i])

for pid in set(pids):
    np.savez(DATASAVEPATH+"/compressed/trainData"+str(pid)+".npz",X=dope[pid]["X"],Y=dope[pid]["Y"],imgNames=dope[pid]["imgNames"])    
