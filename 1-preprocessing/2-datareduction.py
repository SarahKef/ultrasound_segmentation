import os
import numpy as np
import cv2
import glob
import skimage.util
import dice
import scipy.spatial.distance as spdist
import pylab as pl
DATAPATH= "../USNS/ResizedData/train"
OUTDIR = "../USNS/CleanData/train"
def loadPatient(pid):
    data = np.load(DATAPATH+"/trainData"+str(pid)+".npz")
    print "Data Loaded for ",pid
    return data["X"],data["Y"],data["imgNames"]

def computeImgHist(img):
    # Divide the image in blocks and compute per-block histogram
    blocks = skimage.util.view_as_blocks(img, block_shape=(16,16))
    imgHists = [np.histogram(block, bins=np.linspace(0, 1, 10))[0] for block in blocks]
    return np.concatenate(imgHists)

def filterImagesForPatient(pid):
    imgs, masks, names = loadPatient(pid)
    hists = np.array(map(computeImgHist, imgs))
    D = spdist.squareform(spdist.pdist(hists, metric='cosine'))
    
    # Used 0.005 to train at 0.67
    closePairs = D + np.eye(D.shape[0]) < 0.008
    
    close_ij = np.transpose(np.nonzero(closePairs))
    incoherent_ij = [(i, j) for i, j in close_ij if dice.diceCoefficient(masks[i], masks[j]) < 0.2]
    incoherent_ij = np.array(incoherent_ij)
    
    #i, j = incoherent_ij[np.random.randint(incoherent_ij.shape[0])]
    
    valids = np.ones(len(imgs), dtype=np.bool)
    for i, j in incoherent_ij:
        if np.sum(masks[i]) == 0:
            valids[i] = False
        if np.sum(masks[j]) == 0:
            valids[i] = False

    for i in np.flatnonzero(valids):
        imgname = names[i].split(".")[0] + ".png"
        mask_fname = names[i].split(".")[0] + "_mask.png"
        img = skimage.img_as_ubyte(imgs[i])
        cv2.imwrite(OUTDIR+"/"+imgname, img)
        mask = skimage.img_as_ubyte(masks[i])
        cv2.imwrite(OUTDIR+"/"+mask_fname, mask)
        os.listdir(OUTDIR)
    
    print 'Discarded ', np.count_nonzero(valids), " images for patient %d" % pid

for pid in range(1,47):
    print "Working on ",pid
    filterImagesForPatient(pid)
