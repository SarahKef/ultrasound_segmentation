import sys
import os
import numpy as np
import glob
import cv2
import pylab as pl
import shutil
import matplotlib.cm as cm
from sklearn.cross_validation import train_test_split
import skimage.transform as sktransf
        
CLEANEDDATAPATH = "../../USNS/CleanedData/train/compressed"
AUGMENTEDDATAPATH = "../../USNS/AugmentedData/compressed"

def rand_float(rng):
    return rng[0] + np.random.random() * (rng[1] - rng[0])

def generate_image(X_img, Y_img, angle_range=None, zoom_range=None, shear_range=None, horiz_shift_range=None,
                   vert_shift_range=None):
    """
    Randomly shift an image by resizing and then random croping
    the resized image
    """
    if angle_range is None:
        angle_range = (-0.01, 0.01)
    if zoom_range is None:
        zoom_range = (0.99, 1.01)
    if shear_range is None:
        shear_range = (-5, 5) # shear angle in degrees
    if horiz_shift_range is None:
        horiz_shift_range = (-5, 5)
    if vert_shift_range is None:
        vert_shift_range = (-5, 5)
    inshape = X_img.shape
    
    X2 = X_img.copy()
    Y2 = Y_img.copy()
    
    # -- random rotation
    angle = np.deg2rad(rand_float(angle_range))
    zoom = rand_float(zoom_range)
    shear = np.deg2rad(rand_float(shear_range))
    horiz_shift = rand_float(horiz_shift_range)
    vert_shift = rand_float(vert_shift_range)
    
    tform = sktransf.AffineTransform(
        scale=(zoom, zoom),
        rotation=angle,
        shear=shear,
        translation=(horiz_shift, vert_shift)
    )
    X2 = sktransf.warp(X2, tform, order=5, mode='reflect')
    Y2 = sktransf.warp(Y2, tform, order=1)

    # gaussian noise
    #noise_scale = rand_float((0, 0.1))
    #X2 += np.random.normal(scale=noise_scale, size=X2.shape)
    Y2 = Y2 > 0.5
    return X2[None,:,:].astype(np.float32), Y2[None,:,:].astype(np.float32)



# Contains X,Y,pids,imgNames
data = np.load(CLEANEDDATAPATH+"/trainData.npz")

X = data["X"]
Y = data["Y"]
pids = data["pids"]
imgNames = data["imgNames"]

Y_binary = np.sum(np.sum(Y, axis=2), axis=1) > 0
Y_binary = Y_binary.astype(np.float32)
nsamples = X.shape[0]

train_indices, test_indices = train_test_split(np.arange(nsamples), test_size=0.1)
X_train = X[train_indices][:,None,:,:]
Y_train_binary = Y_binary[train_indices][:,None]
Y_train = Y[train_indices][:,None,:,:]
X_test = X[test_indices][:,None,:,:]
Y_test_binary = Y_binary[test_indices][:,None]
Y_test = Y[test_indices][:,None,:,:]


X_train_augmented = []
Y_train_augmented = []
for i in range(X_train.shape[0]):
    if i % 10 == 0:
        print '\r%d / %d' % (i, X_train.shape[0]),
        sys.stdout.flush()
    X_train_augmented.append(X_train[i])
    Y_train_augmented.append(Y_train[i])
    for j in range(2):
        X2, Y2 = generate_image(X_train[i].squeeze(), Y_train[i].squeeze())
        X_train_augmented.append(X2)
        Y_train_augmented.append(Y2)
    
X_train = np.array(X_train_augmented, dtype=np.float32)
Y_train = np.array(Y_train_augmented, dtype=np.float32)

Y_train_binary = np.sum(np.sum(Y_train.squeeze(), axis=2), axis=1) > 0
Y_train_binary = Y_train_binary.astype(np.float32)
Y_train_binary = Y_train_binary[:,None]

np.savez(AUGMENTEDDATAPATH+"/Data.npz", \
    X_train=X_train, \
    Y_train=Y_train, \
    Y_train_binary=Y_train_binary, \
    X_test=X_test, \
    Y_test=Y_test, \
    Y_test_binary=Y_test_binary \
    )
