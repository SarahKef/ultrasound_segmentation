import numpy as np
import os
import cv2

DATAPATH = "../USNS/RawData/train"
PREDPATH = "../USNS/Predicted/Raw"
CORRPATH = "../USNS/Predicted/Corrected"
rawMaskNames = [fname for fname in os.listdir(DATAPATH) if "mask" in fname]
predictedMaskNames = [fname for fname in os.listdir(PREDPATH)]

def loadRawMask(fname):
    mask = cv2.imread(DATAPATH + "/" + fname, cv2.IMREAD_GRAYSCALE)
    mask = (mask > 128).astype(np.float32)
    return mask

def loadPredictedMask(fname):
    mask = cv2.imread(PREDPATH + "/" + fname, cv2.IMREAD_GRAYSCALE)
    mask = (mask > 128).astype(np.float32)
    
    return mask

rawMasks = np.array([loadRawMask(fname) for fname in rawMaskNames])[:,None,:,:]
print "Loaded Raw Masks",rawMasks.shape

from sklearn.decomposition import TruncatedSVD
pca = TruncatedSVD(n_components=20).fit(rawMasks.reshape(-1, 580*420))
print "Computed PCA Model"
predictedMasks = np.array([loadPredictedMask(fname) for fname in predictedMaskNames])
print "Loaded Predicted Masks"
correctPredictedMasks = np.array(pca.transform(predictedMasks))
print "Corrected the Predicted Masks"
correctPredictedMasks = np.array([cmask.reshape(580,420) for cmask in correctPredictedMasks])
for i in range(correctPredictedMasks.shape[0]):
    cv2.imwrite(CORRPATH + "/" + i + ".png")

np.savez(CORRPATH + "/compressed/corrected.npz",final=correctPredictedMasks)