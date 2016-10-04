import os
import numpy as np
import cv2
import glob

import skimage.util
import dice
import scipy.spatial.distance as spdist
import pylab as pl
DATAPATH = "../USNS/CleanedData/train"
DATASAVEPATH = "../USNS/CleanedData/train/compressed"

def loadPreProcess(imgName):
    pnumber, imnumber = imgName.split(".")[0].split('_')
    maskName = imgName.split(".")[0]+"_mask.png"
    
    img = cv2.imread(DATAPATH+"/"+imgName, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(DATAPATH+"/"+maskName, cv2.IMREAD_GRAYSCALE)

    img = img.astype(np.float32)/255.0
    mask = (mask > 128).astype(np.float32)
    np.ascontiguousarray(img)
    
    return img,mask,int(pnumber),imgName

imgNames = [fname for fname in os.listdir(DATAPATH) if "mask" not in fname and "compressed" not in fname]
imgs, masks, pnumbers, imgNames = zip(*[loadPreProcess(fname) for fname in imgNames])
X = np.array(imgs)
Y = np.array(masks)
pids = np.array(pnumbers)
imgNames = np.array(imgNames)
np.savez(DATASAVEPATH+"/trainData.npz",X=X,Y=Y,pids=pids,imgNames=imgNames)
i=0
dope = {}
for pid in set(pids):
    dope[pid]={"X":[],"Y":[],"imgNames":[]}

for i in range(len(X)):
    dope[pids[i]]["X"].append(X[i])
    dope[pids[i]]["Y"].append(Y[i])    
    dope[pids[i]]["imgNames"].append(imgNames[i])

for pid in set(pids):
    np.savez(DATASAVEPATH+"/trainData"+str(pid)+".npz",X=dope[pid]["X"],Y=dope[pid]["Y"],imgNames=dope[pid]["imgNames"])    
