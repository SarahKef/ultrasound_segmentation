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

X_test = np.load("../USNS/AlexNet/test/Data.npz")["X_train"]


json_file = open('alexnet_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("alexnet_model_weights.h5")
# model.compile(optimizer='adam', loss='binary_crossentropy',loss_weights=[1., 0.01])
print("Loaded model from disk")

Y_pred = model.predict(X_test)

import cv2
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(50,50))

OUTDIR='../USNS/AlexNet/Predicted/128'

for i in range(Y_pred.shape[0]):
    imre = (Y_pred[i].squeeze() * 255).astype(np.uint8)
    # imre = cv2.morphologyEx(imre, cv2.MORPH_CLOSE, kernel)
    basename = str(i)
    cv2.imwrite(os.path.join(OUTDIR, basename + '.png'), imre)
    Y_pred[i] = imre
    print '\r%d / %d' % (i, Y_pred.shape[0]),
    sys.stdout.flush()
Y_pred = np.array(Y_pred)
np.savez("../USNS/AlexNet/Predicted/128/128_pred.npz",predicted=Y_pred)
