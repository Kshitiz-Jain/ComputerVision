import numpy as np
import cv2
from matplotlib import pyplot as plt

imgL = cv2.imread('left.jpg',0)
imgR = cv2.imread('right.jpg',0)

stereo = cv2.StereoBM_create(numDisparities=48, blockSize=19)
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()

#resource for code : https://docs.opencv.org/3.1.0/dd/d53/tutorial_py_depthmap.html
