#Autorzy
#Patryk Chmielecki 145190 I7.1
#Bartek Demut 145324 I7.1

import cv2
from matplotlib import pyplot as plt
import numpy as np
from skimage import io

name = 'kostki\hard\kostki_11.png'
nameW = 'kostki\hard\kostki_11W.png'

orgImage = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
image = cv2.GaussianBlur(orgImage,(9,9),2)
thresh, binary = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)

kontury, hier = cv2.findContours(binary, cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
points = [[0]*2 for i in range(len(kontury))]
tempAR = [1 for i in range(len(kontury))]
for i, c in enumerate(kontury):
    rect = cv2.minAreaRect(c)
    if rect[1][1]!=0:
        aspect = rect[1][0]/rect[1][1]
    elif rect[1][0]!=0:
        aspect = rect[1][1]/rect[1][0]
    else:
        aspect = 0
    box = cv2.boxPoints(rect)
    points[i][0] = (min([box[i][0] for i in range(4)]) + max([box[i][0] for i in range(4)]))/2
    points[i][1] = max([box[i][1] for i in range(4)]) + 10
    if aspect < 0.7 or aspect > 1.3:
        tempAR[i]=0
        continue
    box = np.int0(box)
    cv2.drawContours(orgImage,[box],0,(120,120,120),2)

temp = [0] * len(kontury)
for i, h in enumerate(hier[0]):
    if h[3]!=-1:
        if tempAR[i] == 1:
            temp[h[3]] = temp[h[3]] + 1

for i, t in enumerate(temp):
    if t>=1 and t<=6:        
        cv2.putText(orgImage, str(t), (int(points[i][0]),int(points[i][1])), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (120, 120, 120, 255), 1)

#plt.imshow(orgImage, cmap='gray')
#plt.show()
io.imsave(nameW, orgImage)