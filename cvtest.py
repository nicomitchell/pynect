#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 17:21:08 2017

@author: max
"""

import cv2
 
print("OpenCV version:")
print(cv2.__version__)

img = cv2.imread("flowerpic.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("Over the Clouds", img)
cv2.imshow("Over the Clouds - gray", gray)

cv2.waitKey(0)
cv2.destroyAllWindows()