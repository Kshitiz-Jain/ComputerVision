import cv2
import numpy as np
import math
import copy 

import matplotlib.pyplot as plt

#source : https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html


def matches(img1,img2,thresh):
	sift = cv2.xfeatures2d.SIFT_create()
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)
	# BFMatcher with default params
	print("SIFT CALCULATED")
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des2,des1,k=2)
	list_kp1 = []
	list_kp2 = []
	# For each match...
	for mat in matches:
		if mat[0].distance < thresh*mat[1].distance:
			# Get the matching keypoints for each of the images
			img1_idx = mat[0].queryIdx
			img2_idx = mat[0].trainIdx
			# print(img1_idx,img2_idx)
			# x - columns
			# y - rows
			# Get the coordinates
			(x1,y1) = kp2[img1_idx].pt
			(x2,y2) = kp1[img2_idx].pt

			# Append to each list
			list_kp2.append((y1, x1))
			list_kp1.append((y2, x2))

	good = []
	for m,n in matches:
		if m.distance < thresh*n.distance:
			# print(m.distance,n.distance)
			good.append([m])
	# cv.drawMatchesKnn expects list of lists as matches.
	img3 = cv2.drawMatchesKnn(img2,kp2,img1,kp1,good,None,singlePointColor=[0,0,250],matchColor=[250,250,250])
	plt.imshow(img3),plt.show()
	return (list_kp1,list_kp2)

def transformation(list_kp1,list_kp2,img1,img2):
	m1=np.array(list_kp1).astype(int)
	m2=np.array(list_kp2).astype(int)
	nimg1=copy.deepcopy(img1)
	nimg2=copy.deepcopy(img2)
	for i in range(len(m1)):
		nimg1[m1[i][0]][m1[i][1]]=0
	for i in range(len(m2)):
		nimg2[m2[i][0]][m2[i][1]]=0
	matrix1=np.zeros((np.shape(m1)[0]*2,1))
	for i in range(np.shape(m1)[0]):
		matrix1[2*i]=m1[i][0]
		matrix1[2*i+1]=m1[i][1]
	matrix=np.zeros((2*np.shape(m2)[0],6))
	for i in range(np.shape(m1)[0]):
		matrix[2*i][0]=m2[i][0]
		matrix[2*i][1]=m2[i][1]
		matrix[2*i][2]=1
		matrix[2*i+1][3]=m2[i][0]
		matrix[2*i+1][4]=m2[i][1]
		matrix[2*i+1][5]=1
	matrix2=np.matmul(np.transpose(matrix),matrix)
	matrix3=np.linalg.pinv(matrix2)
	matrix4=np.matmul(matrix3,np.transpose(matrix))
	matrix5=np.matmul(matrix4,matrix1)
	fmat=np.zeros((3,3))
	fmat[0][0]=matrix5[0]
	fmat[0][1]=matrix5[1]
	fmat[0][2]=matrix5[2]
	fmat[1][0]=matrix5[3]
	fmat[1][1]=matrix5[4]
	fmat[1][2]=matrix5[5]
	fmat[2][2]=1
	# h, w = img2.shape
 #    border = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
	# fmat=cv2.perspectiveTransform(border,fmat)
	return fmat


def boundary(img1,img2,oimg1,oimg2,thresh):
	listkps=matches(img1,img2,thresh)
	fmat=transformation(listkps[0],listkps[1],img1,img2)
	print(fmat)
	row,col,dim=np.shape(img2)
	for i in range(row):
		tup1=np.array([i,0,1])
		tup2=np.array([i,col-1,1])
		ftup1=np.matmul(fmat,tup1)
		ftup2=np.matmul(fmat,tup2)
		oimg1[int(ftup1[0])][int(ftup1[1])]=[0,0,255]
		oimg1[int(ftup2[0])][int(ftup2[1])]=[0,0,255]

	for i in range(col):
		tup1=np.array([0,i,1])
		tup2=np.array([row-1,i,1])
		ftup1=np.matmul(fmat,tup1)
		ftup2=np.matmul(fmat,tup2)
		oimg1[int(ftup1[0])][int(ftup1[1])]=[0,0,255]
		oimg1[int(ftup2[0])][int(ftup2[1])]=[0,0,255]

	cv2.imwrite("box.png",oimg1)



img1 = cv2.imread('collage.jpg',cv2.IMREAD_GRAYSCALE)          # queryImage
img2 = cv2.imread('test1.jpeg',cv2.IMREAD_GRAYSCALE) # trainImage
img3 = cv2.imread('test2.jpeg',cv2.IMREAD_GRAYSCALE)
oimg1=cv2.imread('collage.jpg')
oimg2=cv2.imread('test1.jpeg')
oimg3=cv2.imread('test2.jpeg')

# fimg=panaroma(img1,img2,oimg1,oimg2)
# bwimg=cv2.cvtColor(fimg.astype(np.uint8),cv2.COLOR_BGR2GRAY)
# matches(img1,img2)
# matches(img1,img3)

boundary(oimg1,oimg2,oimg1,oimg2,0.5)
# bwimg=cv2.cvtColor(fimg.astype(np.uint8),cv2.COLOR_BGR2GRAY)
boundary(oimg1,oimg3,oimg1,oimg3,0.37)


# good = []
# for m,n in matches:
# 	if m.distance < 0.3*n.distance:
# 		# print(m.distance,n.distance)
# 		good.append([m])
# # cv.drawMatchesKnn expects list of lists as matches.
# img3 = cv2.drawMatchesKnn(img2,kp2,img1,kp1,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# plt.imshow(img3),plt.show()
