import os
os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS'] = 'floatX=float32,device=gpu,lib.cnmem=0.8,dnn.conv.algo_bwd_filter=deterministic,dnn.conv.algo_bwd_data=deterministic,blas.ldflags=-LC:/toolkits/openblas-0.2.14-int32/bin -lopenblas'
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

class Maxout2D(Layer):
    def __init__(self, output_dim, cardinality, init='glorot_uniform', **kwargs):
        super(Maxout2D, self).__init__(**kwargs)
        # the k of the maxout paper
        self.cardinality = cardinality
        # the m of the maxout paper
        self.output_dim = output_dim
        self.init = keras.initializations.get(init)
    
    def build(self, input_shape):
        self.input_dim = input_shape[1]
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_shape[1], input_shape[2], input_shape[3]))]
        self.W = self.init((self.input_dim, self.output_dim, self.cardinality),
                           name='{}_W'.format(self.name))
        self.b = K.zeros((self.output_dim, self.cardinality))
        self.trainable_weights = [self.W, self.b]
              
    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        # flatten the spatial dimensions
        flat_x = K.reshape(x, (-1, input_shape[1], input_shape[2] * input_shape[3]))
        output = K.dot(
            K.permute_dimensions(flat_x, (0, 2, 1)),
            K.permute_dimensions(self.W, (1, 0, 2))
        )
        output += K.reshape(self.b, (1, 1, self.output_dim, self.cardinality))
        output = K.max(output, axis=3)
        output = output.transpose(0, 2, 1)
        output = K.reshape(output, (-1, self.output_dim, input_shape[2], input_shape[3]))
        return output
    
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim, input_shape[2], input_shape[3])
    
    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'cardinality': self.cardinality
        }
        base_config = super(Maxout2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def generate_model(input_shape):
    input_img = x = Input(shape=input_shape, name='input_img')
    x = Dropout(0.25)(x)
    x = Convolution2D(32, 3, 3, border_mode='same')(input_img)
    x = Convolution2D(32, 3, 3, border_mode='same')(x)
    x = Maxout2D(16, 2)(x)
    pool1 = x = MaxPooling2D((2, 2), name='pool1')(x)

    x = Dropout(0.25)(x)
    x = Convolution2D(32, 3, 3, border_mode='same')(x)
    x = Convolution2D(32, 3, 3, border_mode='same')(x)
    x = Maxout2D(16, 2)(x)
    pool2 = x = MaxPooling2D((2, 2), name='pool2')(x)

    x = Dropout(0.25)(x)
    x = Convolution2D(32, 3, 3, border_mode='same')(x)
    x = Convolution2D(32, 3, 3, border_mode='same')(x)
    x = Maxout2D(16, 2)(x)
    pool3 = x = MaxPooling2D((2, 2), name='pool3')(x)

    # -- binary presence part
    x = Dropout(0.25)(x)
    x = Convolution2D(32, 3, 3, border_mode='same')(x)
    x = Convolution2D(32, 3, 3, border_mode='same')(x)
    x = Maxout2D(16, 2)(x)
    pool4 = x = MaxPooling2D((2, 2), name='pool4')(x)

    x = Dropout(0.25)(x)
    x = Convolution2D(32, 3, 3, border_mode='same')(x)
    x = Convolution2D(32, 3, 3, border_mode='same')(x)
    x = Maxout2D(16, 2)(x)
    pool5 = x = MaxPooling2D((2, 2), name='pool5')(x)

    # Since some images have not mask, the hope is that the innermost units capture this
    x = Flatten()(pool5)
    x = Dense(32)(x)
    x = LeakyReLU()(x)
    x = Dense(16)(x)
    x = LeakyReLU()(x)
    outbin = Dense(1, activation='sigmoid', name='outbin')(x)

    x = Maxout2D(8, 2)(pool5)
    outmap5 = Convolution2D(1, 1, 1, border_mode='same', activation='sigmoid', name='outmap5')(x)
    x = UpSampling2D((2, 2))(x)


    x = merge([x, pool4], mode='concat', concat_axis=1)
    x = Convolution2D(16, 3, 3, border_mode='same')(x)
    x = Maxout2D(8, 2)(x)
    outmap4 = Convolution2D(1, 1, 1, border_mode='same', activation='sigmoid', name='outmap4')(x)

    x = UpSampling2D((2, 2))(x)

    x = merge([x, pool3], mode='concat', concat_axis=1)
    x = Convolution2D(16, 3, 3, border_mode='same')(x)
    x = Maxout2D(8, 2)(x)
    x = UpSampling2D((2, 2))(x)

    x = merge([x, pool2], mode='concat', concat_axis=1)
    x = Convolution2D(16, 3, 3, border_mode='same')(x)
    x = Maxout2D(8, 2)(x)
    x = UpSampling2D((2, 2))(x)

    x = merge([x, pool1], mode='concat', concat_axis=1)
    x = Convolution2D(16, 3, 3, border_mode='same')(x)
    x = Maxout2D(8, 2)(x)
    x = UpSampling2D((2, 2))(x)

    x = Dropout(0.25)(x)
    x = Convolution2D(8, 3, 3, border_mode='same')(x)
    x = Convolution2D(8, 3, 3, border_mode='same')(x)
    outmap = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same', name='outmap')(x)

    model = Model(
        input=input_img,
        output=[outmap, outmap4, outmap5, outbin]
    )

    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #model.compile(optimizer=sgd, loss='binary_crossentropy')
    #rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
    #model.compile(optimizer=rmsprop, loss='binary_crossentropy')
    metrics={'outbin': 'accuracy'}
    model.compile(optimizer='adam', loss='binary_crossentropy',
                loss_weights=[1., 0.2, 0.05, 0.01], metrics=metrics)

    print_summary(model.layers)
    return model