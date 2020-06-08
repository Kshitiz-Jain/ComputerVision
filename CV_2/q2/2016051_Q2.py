#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import cv2
#Reference from OpenCV documentation
# Referred link : https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
x=cv2.TERM_CRITERIA_EPS 
y=x+cv2.TERM_CRITERIA_MAX_ITER
criteria = (y, 30, 0.001)
objp = np.zeros((12*12,3))
objp=objp.astype(np.float32)

objpoints = []
mean_error = 0
imgpoints = []
image=np.zeros((12,12))
objp[:,:2] = np.mgrid[0:12,0:12].T.reshape(-1,2)

for i in range(15):
    gray = cv2.imread('Left'+str(i+1)+'.bmp',0)
    ret, corners = cv2.findChessboardCorners(gray, (12,12),None)
    if ret != False:
        i=i+1
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        i=i-1
        objpoints.append(objp)
        imgpoints.append(corners2)
        img = cv2.drawChessboardCorners(gray, (12,12), corners2,ret)
        cv2.imwrite('img'+str(i+1)+'.jpg',gray)


# In[18]:


parameters = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
for i in parameters[3]:
    result=cv2.Rodrigues(i)
    print(result[0])


# In[19]:


l=len(objpoints)
for i in range(l):
    p = cv2.projectPoints(objpoints[i], parameters[3][i], parameters[4][i], parameters[1], parameters[2])
    error = cv2.norm(imgpoints[i],p[0], cv2.NORM_L2)/len(p[0])
    print (error)
    mean_error =mean_error+ error

print ("Mean error: ", mean_error/len(objpoints))


# In[ ]:




