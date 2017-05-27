import sys
import os

import cv2
import numpy as np

img  = cv2.imread(str(sys.argv[1]))
gray = cv2.imread(str(sys.argv[1]),0)

erc1 = cv2.text.loadClassifierNM1(pathname+'/trained_classifierNM1.xml')
er1 = cv2.text.createERFilterNM1(erc1)

erc2 = cv2.text.loadClassifierNM2(pathname+'/trained_classifierNM2.xml')
er2 = cv2.text.createERFilterNM2(erc2)

regions = cv2.text.detectRegions(gray,er1,er2)

#Visualization
rects = [cv2.boundingRect(p.reshape(-1, 1, 2)) for p in regions]
for rect in rects:
  cv2.rectangle(img, rect[0:2], (rect[0]+rect[2],rect[1]+rect[3]), (0, 0, 0), 2)
for rect in rects:
  cv2.rectangle(img, rect[0:2], (rect[0]+rect[2],rect[1]+rect[3]), (255, 255, 255), 1)
cv2.imshow("Text detection result", img)
cv2.waitKey(0)
