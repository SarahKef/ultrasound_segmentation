import os
os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS'] = 'floatX=float32,device=gpu,lib.cnmem=0.8,dnn.conv.algo_bwd_filter=deterministic,dnn.conv.algo_bwd_data=deterministic,blas.ldflags=-LC:/toolkits/openblas-0.2.14-int32/bin -lopenblas'
import theano
import keras
from unet import *
from dice import *
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
import pylab as pl

AUGMENTEDDATAPATH = "../USNS/AugmentedData/compressed"
RESIZEDDATAPATH="../USNS/ResizedData/train/compressed"
# fullData = np.load(RESIZEDDATAPATH+"/trainData.npz")
# X_full = fullData['X']
# Y_full = fullData['Y']
# del fullData
# print "X_full",X_full.shape

# X_full = X_full[:][:,None,:,:]

# print "X_full",X_full.shape

data = np.load(AUGMENTEDDATAPATH+"/Data.npz")
X_train=data['X_train']
Y_train=data['Y_train']
Y_train_binary=data['Y_train_binary']
X_test=data['X_test']
Y_test=data['Y_test']
Y_test_binary=data['Y_test_binary']
del data

print "X_train",X_train.shape
print "X_test",X_test.shape

import cv2
def _resize(Y, outshape):
    return cv2.resize(Y.squeeze(), outshape, interpolation=cv2.INTER_AREA)[None, :, :]

Y_train_88 = np.array([_resize(img, (8, 8)) for img in Y_train])
Y_test_88 = np.array([_resize(img, (8, 8)) for img in Y_test])
Y_train_44 = np.array([_resize(img, (4, 4)) for img in Y_train])
Y_test_44 = np.array([_resize(img, (4, 4)) for img in Y_test])



# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json,{'Maxout2D':Maxout2D})
# load weights into new model
model.load_weights("model_weights.h5")
# model.compile(optimizer='adam', loss='binary_crossentropy',loss_weights=[1., 0.01])
print("Loaded model from disk")


def compute_scores_map(X, Y):
    Y_pred, Y_pred_88, Y_pred_44, Y_pred_binary = model.predict(X)
    
    map_threshs = np.linspace(0, 1, num=21)
    bin_threshs = np.linspace(0, 1, num=21)
    
    scores = np.zeros((len(map_threshs), len(bin_threshs)))
    for i in range(len(map_threshs)):
        for j in range(len(bin_threshs)):
            _Y_pred = Y_pred > map_threshs[i]
            
            for k in range(_Y_pred.shape[0]):
                # Clear prediction where we detect to little pixels
                if np.sum(_Y_pred[k]) < 150 or Y_pred_binary[k] < bin_threshs[j]:
                    _Y_pred[k,:] = 0
            
            scores[i, j] = averageDiceCoefficient(_Y_pred, Y)
    
    # pl.imshow(scores, interpolation='nearest')
    # pl.yticks(np.arange(len(map_threshs)) + 0.5, ['%.2f' % v for v in map_threshs])
    # pl.xticks(np.arange(len(bin_threshs)) + 0.5, ['%.2f' % v for v in bin_threshs], rotation=90)
    # pl.ylabel('map thresh')
    # pl.xlabel('bin thresh')
    # pl.colorbar()
    
    best_i, best_j = np.unravel_index(np.argmax(scores.ravel()), scores.shape)
    best_map_thresh = map_threshs[best_i]
    best_bin_thresh = bin_threshs[best_j]
    return best_map_thresh, best_bin_thresh

map_thresh, bin_thresh = compute_scores_map(X_test, Y_test)
print 'Best map_thresh=%f, bin_thresh=%f' % (map_thresh, bin_thresh)


def predict(X, map_thresh, bin_thresh):
    Y_pred, Y_pred_88, Y_pred_44, Y_pred_binary = model.predict(X)
    Y_pred_binary = Y_pred_binary.squeeze()
    Y_pred_proba = Y_pred
    Y_pred = Y_pred > map_thresh
    for i in range(Y_pred.shape[0]):
        if np.sum(Y_pred[i]) < 150 or Y_pred_binary[i] < bin_thresh:
            Y_pred[i,:] = 0
        pass
    return Y_pred, Y_pred_proba, Y_pred_binary

Y_pred, Y_pred_proba, Y_pred_binary = predict(X_test, map_thresh, bin_thresh)
# Y_full_pred, Y_full_pred_proba, Y_full_pred_binary = predict(X_full, map_thresh, bin_thresh)

print "test set score : ", diceCoefficient(Y_pred, Y_test)
print "avg test set score : ", averageDiceCoefficient(Y_pred, Y_test)
# print "full set score : ", diceCoefficient(Y_full_pred, Y_full)
# print "avg full set score : ", averageDiceCoefficient(Y_full_pred, Y_full)