
import cv2
import numpy as np
import sys
import os
import rle

fnames = [fname for fname in os.listdir("../USNS/Predicted/Corrected") if "compressed" not in fname]

with open("../USNS/Predicted/submission-0.2.csv","w") as f:
    f.write("img,pixels\n")
    for i in range(len(fnames)):
        # print i
        print '\r%d / %d' % (i + 1,  len(fnames))
        
        img = cv2.imread("../USNS/Predicted/Corrected/"+str(i)+".png", cv2.IMREAD_GRAYSCALE)
        print img.shape
        runs = rle.runlen_encode(img)
        runtext = ' '.join(['%d %d' % rr for rr in runs])
        f.write('%d,%s\n' % (i + 1, runtext))
        