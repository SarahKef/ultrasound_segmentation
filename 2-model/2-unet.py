import os
os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,device=gpu0,floatX=float32'
import theano
import keras
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
import unet
AUGMENTEDDATAPATH = "../USNS/AugmentedData/compressed"
data = np.load(AUGMENTEDDATAPATH+"/Data.npz")
X_train=data['X_train']
Y_train=data['Y_train']
Y_train_binary=data['Y_train_binary']
X_test=data['X_test']
Y_test=data['Y_test']
Y_test_binary=data['Y_test_binary']
del data
import cv2
def _resize(Y, outshape):
    return cv2.resize(Y.squeeze(), outshape, interpolation=cv2.INTER_AREA)[None, :, :]
Y_train_88 = np.array([_resize(img, (8, 8)) for img in Y_train])
Y_test_88 = np.array([_resize(img, (8, 8)) for img in Y_test])
Y_train_44 = np.array([_resize(img, (4, 4)) for img in Y_train])
Y_test_44 = np.array([_resize(img, (4, 4)) for img in Y_test])

input_shape = X_train[0].shape
print input_shape


model = unet.generate_model(input_shape)


json_string = model.to_json()
open('architecture.json', 'w').write(json_string)

model_checkpoint = ModelCheckpoint('weights.hdf5', monitor='val_loss', save_best_only=True, verbose=True)

history = model.fit(
    X_train,
    {
        'outmap': Y_train,
        'outmap4' : Y_train_88,
        'outmap5' : Y_train_44,
        'outbin': Y_train_binary
    },
    batch_size=64,
    validation_data=(X_test, {
            'outmap': Y_test,
            'outmap4' : Y_test_88,
            'outmap5' : Y_test_44,
            'outbin': Y_test_binary
    }),
    nb_epoch=40,# 40 + 10 + 10 + 5 + 5
    verbose=1,
    shuffle=True,
    callbacks=[model_checkpoint]
)

