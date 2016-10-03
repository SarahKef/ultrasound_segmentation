
import cv2
import numpy as np
import sys
import os
import rle

fnames = [fname for fname in os.listdir("../USNS/Predicted/Corrected") if "compressed" not in fname]

with open("../USNS/Predicted/submission-0.2-p.csv","w") as f:
    f.write("img,pixels\n")
    for i in range(len(fnames)):
        # print i
        img = cv2.imread("../USNS/Predicted/Corrected/"+str(i)+".png", cv2.IMREAD_GRAYSCALE)
        runs = rle.runlen_encode(img)
        runtext = ' '.join(['%d %d' % rr for rr in runs])
        f.write('%d,%s\n' % (i + 1, runtext))
        print '\r%d / %d' % (i + 1,  len(fnames)),
        sys.stdout.flush()