import os
os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS'] = 'floatX=float32,device=gpu,lib.cnmem=0.8,dnn.conv.algo_bwd_filter=deterministic,dnn.conv.algo_bwd_data=deterministic,blas.ldflags=-LC:/toolkits/openblas-0.2.14-int32/bin -lopenblas'
import theano
import keras
from keras.layers import Input, Dense, Dropout, Activation, Flatten, merge, RepeatVector, Permute, Reshape
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, MaxoutDense
from keras.layers import LeakyReLU, BatchNormalization
from keras.layers import Layer, InputSpec, Reshape
import keras.initializations
from keras.models import Model
from keras.optimizers import SGD, RMSprop
from keras.utils.layer_utils import print_summary
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import numpy as np

from sklearn.cross_validation import train_test_split

def generate_model():
    inp = Input(shape=(1000,))
    x = inp
    x = Dense(32*32)(x)
    # x = Dense(64*64)(x)
    x = Dense(32*32)(x)
    x = Reshape((1,32,32))(x)
    
    x = Convolution2D(32, 3, 3, border_mode='same')(x)
    x = Convolution2D(32, 3, 3, border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)

    x = Convolution2D(16, 3, 3, border_mode='same')(x)
    x = Convolution2D(16, 3, 3, border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)

    x = Convolution2D(8, 3, 3, border_mode='same')(x)
    x = Convolution2D(8, 3, 3, border_mode='same')(x)
    
    out = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same', name='outmap')(x)
    model = Model(input = inp, output=out)
    model.compile(optimizer='adam', loss='binary_crossentropy')

    print_summary(model.layers)
    return model

model = generate_model()

_d = np.load("../USNS/AlexNet/train/Data.npz")
DATAPATH= "../USNS/ResizedData/train/compressed"
Y = np.load(DATAPATH+"/trainData.npz")["Y"]
nsamples = _d["X_train"].shape[0]
train_indices, test_indices = train_test_split(np.arange(nsamples), test_size=0.1)
X_train = _d["X_train"][train_indices]
Y_train = Y[train_indices][:,None,:,:]
X_test = _d["X_train"][test_indices]
Y_test = Y[test_indices][:,None,:,:]



model.fit(
    X_train,
    Y_train,
    validation_data=(
        X_test,
        Y_test
        ),
    nb_epoch = 5,
    verbose=1,
    shuffle=True
)



model_json = model.to_json()
with open("alexnet_model.json","w") as json_file:
    json_file.write(model_json)
model.save_weights("alexnet_model_weights.h5")
print("Saved the Model to Disk")
