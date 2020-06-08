import cv2
import glob
import numpy as np


ims=glob.glob('./Q3-faces/*.jpg')
faces=[]
for i in ims:
	# print(i)
	im=cv2.imread(i)
	# im=cv2.resize(im,(0,0),fx=0.5,fy=0.)
	faces.append(im)


def skinhue(face,i):
	face=cv2.cvtColor(face,cv2.COLOR_BGR2HSV)
	threslow=np.array([0,40,80],dtype="uint8")
	threshigh=np.array([25,250,250],dtype="uint8")
	r,c,d=np.shape(face)
	img=np.zeros((r,c))
	for i in range(r):
		for j in range(c):
			if(np.all(np.less_equal(face[i][j],threshigh)) and np.all(np.greater_equal(face[i][j],threslow))):
				img[i][j]=250
	cv2.imwrite("hue"+str(i)+".png",img.astype(np.uint8))

for i in range(len(faces)):
	skinhue(faces[i],i+1)