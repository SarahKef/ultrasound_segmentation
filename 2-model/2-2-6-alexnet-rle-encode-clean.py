
import cv2
import numpy as np
import sys
import os
import rle
import threading
fnames = [fname for fname in os.listdir("../USNS/AlexNet/Predicted/Corrected") if "compressed" not in fname]

NumOfThreads = 1
class break_and_run(threading.Thread):
    def __init__(self,a):
        threading.Thread.__init__(self)
        self.a = a
        self.b = a + (len(fnames)/NumOfThreads)
    def run(self):
        a = self.a
        b = self.b
        with open("../USNS/AlexNet/Predicted/submission-"+str(a/(len(fnames)/NumOfThreads))+".csv","w") as f:
            f.write("img,pixels\n")
            for i in range(a,b):
                # print i
                print '\r %d / %d' % ( i + 1,  len(fnames))
                sys.stdout.flush()
                try:
                    img = cv2.imread("../USNS/AlexNet/Predicted/Corrected/"+str(i)+".png", cv2.IMREAD_GRAYSCALE)
                    runs = rle.runlen_encode(img)
                    runtext = ' '.join(['%d %d' % rr for rr in runs])
                    f.write('%d,%s\n' % (i + 1, runtext))
                except:
                    pass

threads = []    
for i in range(NumOfThreads):
    thread = break_and_run(i*(len(fnames)/NumOfThreads))
    thread.start()
    threads.append(thread)
    # try:
    #     thread.start_new_thread(break_and_write,(i*55,i*55 + 55) )
    # except:
    #     print "Thread didn\'t start"
for thread in threads:
    thread.join()