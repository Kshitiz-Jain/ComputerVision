import cv2
import numpy as np
import math
import copy 

import matplotlib.pyplot as plt




def matches(img1,img2):
	sift = cv2.xfeatures2d.SIFT_create()
	# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)
	# BFMatcher with default params
	print("SIFT CALCULATED")
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des2,des1,k=2)

	# matches=sorted(matches, key= lambda x:x.distance)
	list_kp1 = []
	list_kp2 = []
	# For each match...
	for mat in matches:
		if mat[0].distance < 0.3*mat[1].distance:
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

	# good = []
	# for m,n in matches:
	#   if m.distance < 0.3*n.distance:
	#       # print(m.distance,n.distance)
	#       good.append([m])
	# # cv.drawMatchesKnn expects list of lists as matches.
	# img3 = cv2.drawMatchesKnn(img2,kp2,img1,kp1,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
	# plt.imshow(img3),plt.show()
	return (list_kp1,list_kp2)

def transformation(list_kp1,list_kp2,img1,img2):
	m1=np.array(list_kp1).astype(np.float64)
	m2=np.array(list_kp2).astype(np.float64)

	matrix1=np.zeros((np.shape(m1)[0]*2,1))
	for i in range(np.shape(m1)[0]):
		matrix1[2*i]=m1[i][0]
		matrix1[2*i+1]=m1[i][1]

	matrix=np.zeros((2*np.shape(m2)[0],6))
	for i in range(np.shape(m1)[0]):
		# print("hello")
		matrix[2*i][0]=m2[i][0]
		matrix[2*i][1]=m2[i][1]
		matrix[2*i][2]=1
		matrix[2*i+1][3]=m2[i][0]
		matrix[2*i+1][4]=m2[i][1]
		matrix[2*i+1][5]=1

	# print(matrix)
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
	return fmat




def interpolate(img,ker):
	nimg=np.zeros(np.shape(img))
	row,col,d=np.shape(img)
	for i in range(row):
		for j in range(col):
			x=i-ker
			y=j-ker
			temp=np.array([0,0,0])
			r=[]
			g=[]
			b=[]
			count=0
			for u in range(ker):
				for v in range(ker):
					if(img[x+u][y+v][0]!=0 or img[x+u][y+v][1]!=0 or img[x+u][y+v][2]!=0):
						temp=temp+img[x+u][y+v]
						r.append(img[x+u][y+v][0])
						g.append(img[x+u][y+v][1])
						b.append(img[x+u][y+v][2])
						count=count+1
			if(count==0 or count==(ker*ker-1)):
				nimg[i][j]=temp
			else:
				nimg[i][j]=temp/count
	return nimg

def ifempty(array):
	for i in range(len(array)):
		if(array[i][0]!=0 or array[i][1]!=0 or array[i][2]!=0):
			return False
	return True

def cutblack(img):
	row,col,d=np.shape(img)
	rmax=row-1
	rmin=0
	cmax=col-1
	cmin=0
	for i in range(row):
		if(ifempty(img[i])):
			rmin=i
		else:
			break
	for i in range(rmin+1,row):
		if(ifempty(img[i])==False):
			rmax=i
		else:
			break
	print(np.shape(img))
	print(rmin,rmax,cmin,cmax)
	fimg=np.zeros((rmax-rmin+1,col,3))
	r,c,d=np.shape(fimg)
	for i in range(r):
		for j in range(c):
			fimg[i][j]=img[rmin+i][j]
	return fimg

	



def panaroma(img1,img2,oimg1,oimg2):
	print(np.shape(img1))
	print(np.shape(img2))
	listkps=matches(img1,img2)
	fmat=transformation(listkps[0],listkps[1],img1,img2)
	#tranformation formed
	row1,col1=np.shape(img1)
	row2,col2=np.shape(img2)
	finalimg=np.zeros((row1+(2*row2),col1+(2*col2),3))
	for i in range(row1):
		for j in range(col1):
			finalimg[i+row2][j+col2]=oimg1[i][j]

	print(np.shape(finalimg))
	for i in range(row2):
		for j in range(col2):
			tup=np.array([i,j,1])
			ftup=np.matmul(fmat,tup)
			x=int(ftup[0])
			y=int(ftup[1])
			b=max(oimg2[i][j][0],finalimg[x+row2][y+col2][0])
			g=max(oimg2[i][j][1],finalimg[x+row2][y+col2][1])
			r=max(oimg2[i][j][2],finalimg[x+row2][y+col2][2])
			finalimg[x+row2][y+col2]=[b,g,r]

	# finalimg=interpolate(finalimg)
	finalimg=cutblack(finalimg)
	new_img = np.transpose(finalimg[:,:,0])
	new_img1 = np.transpose(finalimg[:,:,1])
	new_img2 = np.transpose(finalimg[:,:,2])
	img_new = np.zeros((finalimg.shape[1], finalimg.shape[0], 3))
	# img_new = np.stack((new_img, new_img1, new_img2))
	img_new[:,:,0] = new_img
	img_new[:,:,1] = new_img1
	img_new[:,:,2] = new_img2
	print (img_new.shape, "shape")
	finalimg=cutblack(img_new)
	# finalimg=np.transpose(finalimg)

	new_img = np.transpose(finalimg[:,:,0])
	new_img1 = np.transpose(finalimg[:,:,1])
	new_img2 = np.transpose(finalimg[:,:,2])
	img_new = np.zeros((finalimg.shape[1], finalimg.shape[0], 3))
	# img_new = np.stack((new_img, new_img1, new_img2))
	img_new[:,:,0] = new_img
	img_new[:,:,1] = new_img1
	img_new[:,:,2] = new_img2
	print (img_new.shape, "shape")
	finalimg = img_new
	cv2.imwrite("finalimg1.jpg",finalimg)
	return finalimg









img1 = cv2.imread('1a.jpg',cv2.IMREAD_GRAYSCALE)          # queryImage
img2 = cv2.imread('1b.jpg',cv2.IMREAD_GRAYSCALE) # trainImage
img3 = cv2.imread('1c.jpg',cv2.IMREAD_GRAYSCALE)
oimg1=cv2.imread('1a.jpg')
oimg2=cv2.imread('1b.jpg')
oimg3=cv2.imread('1c.jpg')

# fimg=panaroma(img1,img2,oimg1,oimg2)
# bwimg=cv2.cvtColor(fimg.astype(np.uint8),cv2.COLOR_BGR2GRAY)
fimg=panaroma(img2,img3,oimg2,oimg3)
bwimg=cv2.cvtColor(fimg.astype(np.uint8),cv2.COLOR_BGR2GRAY)
panaroma(bwimg,img1,fimg,oimg1)


