import cv2
import numpy as np
import math

def padding(image,padd):
    psize=math.floor(padd/2)
    row,col = image.shape
    nimg=np.zeros((row+2*psize,col+2*psize))
    print (nimg)
    print(image)
    for i in range(row):
        for j in range(col):
            nimg[i+psize][j+psize]=image[i][j]
    return nimg

def difference(img,x,y,ker):
	row,col=np.shape(ker)
	total=0
	padd=math.floor(row/2)
	for i in range(row):
		for j in range(col):
			diff=ker[i][j]-img[x-padd+i][y-padd+j]
			total=total+math.pow(diff,2)
	return total


def bestmatch(img,row,ker):
	minimum=99999999
	opti=0
	padd=math.floor(np.shape(ker)[0]/2)
	for i in range(padd,np.shape(img)[1]-2*padd):
		diff=difference(img,row,i,ker)
		# print(diff)
		if(diff<minimum):
			minimum=diff
			# print("hello")
			opti=i
	return opti

def findker(img,x,y,ker):
	kernel=np.zeros((ker,ker))
	padd=math.floor(ker/2)
	for i in range(ker):
		for j in range(ker):
			kernel[i][j]=img[x-padd+i][y-padd+j]
	return kernel

def depth(img1,img2,ker):
	row,col=np.shape(img1)
	depth=np.zeros(np.shape(img1))
	img2=padding(img2,ker)
	img=padding(img1,ker)
	padd=math.floor(ker/2)
	for i in range(row):
		for j in range(col):
			kernel=findker(img,i+padd,j+padd,ker)
			best=bestmatch(img2,i+padd,kernel)
			# print(best)
			# exit(0)
			dis=abs(j-best)+1
			# print(dis)
			depth[i][j]=int(dis*255/190)
		# exit(0)
		print(i) 
	cv2.imwrite("depth.jpg",depth)


img1=cv2.imread("left.jpg",0)
# img1=cv2.resize(img1,(0,0),fx=0.5,fy=0.5)
img2=cv2.imread("right.jpg",0)
# img2=cv2.resize(img2,(0,0),fx=0.5,fy=0.5)
depth(img1,img2,7)


