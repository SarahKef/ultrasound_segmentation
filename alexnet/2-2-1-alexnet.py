import os
os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS'] = 'floatX=float32,device=gpu,lib.cnmem=0.8,dnn.conv.algo_bwd_filter=deterministic,dnn.conv.algo_bwd_data=deterministic,blas.ldflags=-LC:/toolkits/openblas-0.2.14-int32/bin -lopenblas'

from keras.optimizers import SGD
from convnetskeras.convnets import preprocess_image_batch, convnet
import numpy as np
DATAPATH = "../USNS/RawData/train/"
imgNames = [DATAPATH+fname for fname in os.listdir(DATAPATH) if "mask" not in fname]

x_train = preprocess_image_batch(imgNames,img_size=(256,256), crop_size=(227,227), color_mode="rgb")
print "Pre-processing done"
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model = convnet('alexnet',weights_path="../USNS/AlexNet/weights/alexnet_weights.h5", heatmap=False)
model.compile(optimizer=sgd, loss='mse')
print "Model Compiled"
train = model.predict(x_train)
print "Predictions made"
np.savez("../USNS/AlexNet/train/Data.npz",X_train=train)