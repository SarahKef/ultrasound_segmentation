import os
os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS'] = 'floatX=float32,device=gpu,lib.cnmem=0.8,dnn.conv.algo_bwd_filter=deterministic,dnn.conv.algo_bwd_data=deterministic,blas.ldflags=-LC:/toolkits/openblas-0.2.14-int32/bin -lopenblas'
import theano
import keras
from unet import *
from keras.layers import Input, Dense, Dropout, Activation, Flatten, merge, RepeatVector, Permute, Reshape
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, MaxoutDense
from keras.layers import LeakyReLU, BatchNormalization
from keras.layers import Layer, InputSpec
import keras.initializations
from keras.models import Model
from keras.optimizers import SGD, RMSprop
from keras.utils.layer_utils import print_summary
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import numpy as np
import sys

trainDATAPATH = "../../USNS/ResizedData/train/compressed"
data = np.load(trainDATAPATH+"/trainData.npz")
X_train = data["X"]
del data


print X_train.shape
# load json and create model
json_file = open('unet_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json,{'Maxout2D':Maxout2D})
# load weights into new model
model.load_weights("unet_model_weights.h5")
# model.compile(optimizer='adam', loss='binary_crossentropy',loss_weights=[1., 0.01])
print("Loaded model from disk")

Y_pred, Y_pred_88, Y_pred_44, Y_pred_binary = model.predict(X_train[:][:,None,:,:])
# Y_pred_binary = Y_pred_binary.squeeze()
# Y_pred_proba = Y_pred

# def Y_to_binary(Y_pred, Y_pred_binary, map_thresh, bin_thresh):
#     Y_pred = Y_pred > map_thresh
#     for i in range(Y_pred.shape[0]):
#         if np.sum(Y_pred[i]) < 150 or Y_pred_binary[i] < bin_thresh:
#             Y_pred[i,:] = 0
#         pass
#     return Y_pred, Y_pred_proba, Y_pred_binary

# map_threshs = np.linspace(0, 1, num=11)
# bin_threshs = np.linspace(0, 1, num=11)

# masks_per = np.zeros((len(map_threshs), len(bin_threshs)))
# for i in range(len(map_threshs)):
#     for j in range(len(bin_threshs)):
#         m_t = map_threshs[i]
#         b_t = bin_threshs[j]
#         _pred, _pred_proba, _pred_binary = Y_to_binary(Y_pred, Y_pred_binary, m_t, b_t)
#         # Turn labels into boolean present/missing
#         _pred_binary = np.sum(np.sum(np.sum(_pred, axis=2), axis=1), axis=1) > 0
#         _pred_binary = _pred_binary.astype(np.float32)
#         per_mask = (100 * np.count_nonzero(_pred_binary) / float(len(_pred_binary)))

#         masks_per[i, j] = per_mask

map_thresh = 0.5
bin_thresh = 0.7

import cv2
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(50,50))

OUTDIR='../../USNS/Predicted/128-train'

for i in range(Y_pred.shape[0]):
    imre = (Y_pred[i].squeeze() * 255).astype(np.uint8)
    # imre = cv2.morphologyEx(imre, cv2.MORPH_CLOSE, kernel)
    basename = str(i)
    cv2.imwrite(os.path.join(OUTDIR, basename + '.png'), imre)
    Y_pred[i] = imre
    print '\r%d / %d' % (i, Y_pred.shape[0]),
    sys.stdout.flush()
Y_pred = np.array(Y_pred)
np.savez("../../USNS/Predicted/128-train/compressed/128_train_pred.npz",predicted=Y_pred)
