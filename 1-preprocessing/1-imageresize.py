import os
import numpy as np
import cv2
import glob
DATAPATH = "../USNS/RawData/train"
DATASAVEPATH = "../USNS/ResizedData/train"
NEWIMGSIZE = (128,128)
def loadPreProcess(imgName):
    pnumber, imnumber = imgName.split(".")[0].split('_')
    maskName = imgName.split(".")[0]+"_mask.tif"
    img = cv2.imread(os.path.join(DATAPATH, imgName), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(os.path.join(DATAPATH, maskName), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,NEWIMGSIZE)
    mask = cv2.resize(img,NEWIMGSIZE)
    img = img.astype(np.float32)/255.0
    mask = (mask > 128).astype(np.float32)
    np.ascontiguousarray(img)
    print pnumber
    return img,mask,int(pnumber)
imgNames = [os.path.basename(fname) for fname in glob.glob(DATAPATH + "/*.tif") if "mask" not in fname]
imgs, masks, pnumbers = zip(*[loadPreProcess(fname) for fname in imgNames])
X = np.array(imgs)
Y = np.array(masks)
pids = np.array(pnumbers)
np.savez(DATASAVEPATH+"/trainData.npz",X=X,Y=Y,pids=pids)
