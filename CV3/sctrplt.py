import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


img=cv2.imread("./Q1-images/2or4objects.jpg")
colors=np.reshape(img,(np.shape(img)[0]*np.shape(img)[1],3))
imx=img[:,:,0]
x=np.reshape(imx,(np.shape(img)[0]*np.shape(img)[1]))
imy=img[:,:,1]
y=np.reshape(imy,(np.shape(img)[0]*np.shape(img)[1]))
imz=img[:,:,2]
z=np.reshape(imz,(np.shape(img)[0]*np.shape(img)[1]))


# C = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x, y, z, c = colors)
plt.show()