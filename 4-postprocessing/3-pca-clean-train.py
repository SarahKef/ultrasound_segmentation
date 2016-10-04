import numpy as np
import os
import cv2
import sys
_data = np.load("../USNS/ResizedData/train/compressed/trainData.npz")
Y_true = _data["Y"][:,None,:,:]

_data = np.load("../USNS/Predicted/128-train/compressed/128_train_pred.npz")
Y_pred = _data["predicted"]

del _data
print "Loaded all Data"
from sklearn.decomposition import TruncatedSVD
pca2 = TruncatedSVD(n_components=20).fit(Y_true.reshape(-1, 128*128))
print "PCA Fit Done"

def correct_mask(Y_pred):
    Y128 = cv2.resize(Y_pred.squeeze(), (128, 128))
    Y128 = (Y128 > 0).astype(np.float32)
    Y_r = pca2.transform(Y128.reshape(-1))
    mask = pca2.inverse_transform(Y_r).reshape(128, 128)
    mask = cv2.resize(mask, (580, 420)) > 0.5
    return mask

with_masks = np.sum(np.sum(Y_pred.squeeze(), axis=2), axis=1) > 10000
with_masks = with_masks.astype(np.float32)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(50,50))
for i in range(Y_pred.shape[0]):
    if with_masks[i]>0:
        imre = correct_mask(Y_pred[i])
        imre = (imre.squeeze() * 255).astype(np.uint8)
        imre = cv2.morphologyEx(imre, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite("../USNS/Predicted/Corrected-train/"+str(i)+".png", imre)
    else:
        imre = np.zeros(shape=(420,580))
        imre = (imre.squeeze() * 255).astype(np.uint8)
        imre = cv2.morphologyEx(imre, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite("../USNS/Predicted/Corrected-train/"+str(i)+".png",imre)
    print '\r%d %d / %d' % (i, with_masks[i], Y_pred.shape[0]),
    sys.stdout.flush()