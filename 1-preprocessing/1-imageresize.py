import os
import numpy as np
import cv2
import glob
DATAPATH = "../USNS/RawData/train"
DATASAVEPATH = "../USNS/ResizedData/train"
def loadPreProcess(imgName):
    print imgName
    maskName = imgName.split(".")[0]+"_mask.tif"
    img = cv2.imread(os.path.join(DATAPATH, imgName), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(os.path.join(DATAPATH, maskName), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(128,128))
    mask = cv2.resize(img,(128,128))
    img = img.astype(np.float32)/255.0
    mask = (mask > 128).astype(np.float32)
    np.ascontiguousarray(img)
    return img,mask
imgNames = [os.path.basename(fname) for fname in glob.glob(DATAPATH + "/*.tif") if "mask" not in fname]
imgs, masks = zip(*[loadPreProcess(fname) for fname in imgNames])
X = np.array(imgs)
Y = np.array(masks)
np.savez(DATASAVEPATH+"/trainData.npz",X=X,Y=Y)
