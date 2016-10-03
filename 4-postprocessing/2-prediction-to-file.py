
import cv2
import numpy as np
import sys
import os
def runlen_encode(img):
    c_img = img.reshape(img.shape[0] * img.shape[1], order='F')
    runs = []
    npixels = len(c_img)
    
    run_start = 1
    run_length = 0
    for i in range(npixels):
        c = c_img[i]
        if c == 0:
            if run_length != 0:
                # for kaggle, pixels are numbered from 1, hence the + 1
                runs.append((run_start + 1, run_length))
                run_length = 0
        else:
            if run_length == 0:
                run_start = i
            run_length += 1
    
    if run_length != 0:
        # for kaggle, pixels are numbered from 1, hence the + 1
        runs.append((run_start + 1, run_length))
    return runs

fnames = [fname for fname in os.listdir("../USNS/Predicted/Corrected") if "compressed" not in fname]

with open("../USNS/Predicted/submission.csv","w") as f:
    f.write("img,pixels\n")
    for i in range(len(fnames)):
        # print i
        img = cv2.imread("../USNS/Predicted/Corrected/"+str(i)+".png", cv2.IMREAD_GRAYSCALE)
        runs = runlen_encode(img)
        runtext = ' '.join(['%d %d' % rr for rr in runs])
        f.write('%d,%s\n' % (i + 1, runtext))
        print '\r%d / %d' % (i,  len(fnames)),
        sys.stdout.flush()