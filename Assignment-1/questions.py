import cv2
import numpy as np
import math
import random
import pywt
import filters


#Queston1

##3x3 average filter
#img=cv2.imread('./image_1.jpg',0)
#img=filters.padding(img,3)
#img=filters.filter(img,3,0)
#img=filters.removepadd(img,3)
#img=img.astype(np.uint8)
#cv2.imshow("Average filter",img)

##5x5 average filter
#img2=cv2.imread('./image_1.jpg',0)
#img2=filters.padding(img2,5)
#img2=filters.filter(img2,5,0)
#img2=filters.removepadd(img2,5)
#img2=img2.astype(np.uint8)
#cv2.imshow("Average filter",img2)

##11x11 average filter
#img=cv2.imread('./image_1.jpg',0)
#img=filters.padding(img,11)
#img=filters.filter(img,11,0)
#img=filters.removepadd(img,11)
#img=img.astype(np.uint8)
#cv2.imshow("Average filter",img)

##15x15 average filter
#img=cv2.imread('./image_1.jpg',0)
#img=filters.padding(img,15)
#img=filters.filter(img,15,0)
#img=filters.removepadd(img,15)
#img=img.astype(np.uint8)
#cv2.imshow("Average filter",img)

#Question2

#10% noise
#img=cv2.imread('./image_2.png',0)
#img1=filters.addnoise(img,10)
#img=img.astype(np.uint8)
#cv2.imshow("Average filter",img)

##20% noise
#img=cv2.imread('./image_2.png',0)
#img=filters.addnoise(img,20)
#img=img.astype(np.uint8)
#cv2.imshow("Average filter",img)

##3x3 median filter
#img=cv2.imread('./image_2.png',0)
#img=filters.padding(img,3)
#img=filters.filter(img,3,1)
#img=filters.removepadd(img,3)
#img=img.astype(np.uint8)
#cv2.imshow("Average filter",img)

##5x5 median filter
#img2=cv2.imread('./image_2.png',0)
#img2=filters.padding(img2,5)
#img2=filters.filter(img2,5,1)
#img2=filters.removepadd(img2,5)
#img2=img2.astype(np.uint8)
#cv2.imshow("Average filter",img2)

##11x11 median filter
#img=cv2.imread('./image_2.png',0)
#img=filters.padding(img,11)
#img=filters.filter(img,11,1)
#img=filters.removepadd(img,11)
#img=img.astype(np.uint8)
#cv2.imshow("Average filter",img)

#Question3

##3x3 Gaussian filter
#img=cv2.imread('./image_3.png',0)
#img=filters.padding(img,3)
#img=filters.filter(img,3,2)
#img=filters.removepadd(img,3)
#img=img.astype(np.uint8)
#cv2.imshow("Average filter",img)
#
##5x5 Gaussian filter
#img2=cv2.imread('./image_3.png',0)
#img2=filters.padding(img2,5)
#img2=filters.filter(img2,5,2)
#img2=filters.removepadd(img2,5)
#print(img2[98])
#img2=img2.astype(np.uint8)
#
#cv2.imshow("Average filter",img2)

##11x11 Gaussian filter
#img=cv2.imread('./image_3.png',0)
#img=filters.padding(img,11)
#img=filters.filter(img,11,2)
#img=filters.removepadd(img,11)
#img=img.astype(np.uint8)
#cv2.imshow("Average filter",img)

##15x15 Gaussian filter
#img=cv2.imread('./image_3.png',0)
#img=filters.padding(img,15)
#img=filters.filter(img,15,2)
#img=filters.removepadd(img,15)
#img=img.astype(np.uint8)
#cv2.imshow("Average filter",img)

#Question4

##Laplacian level 1
#img=cv2.imread('./image_3.png',0)
#img2=filters.padding(img,5)
#img2=filters.filter(img,5,2)
#img2=filters.downsample(img2,5)
#print(img2)
#img2=filters.upsample(img2,5)
#img=filters.sub(img,img2)
#img2=filters.removepadd(img,5)
#print(img2)
#img2=img2.astype(np.uint8)
#print(img2)
#cv2.imshow("Average filter",img2)

##Laplacian level 2
#img=cv2.imread('./image_3.png',0)
###img=np.array([[11,12,5,2],[0,15,6,10],[11,12,5,2],[0,15,6,10]])
#img2=filters.padding(img,5)
#img2=filters.filter(img,5,2)
#img=filters.downsample(img2,5)
#img2=filters.filter(img,5,1)
#img2=filters.downsample(img2,5)
###img=addnoise(img,20)
##img2=haar(img2,3)
#print(img2)
##img2=removepadd(img2,3)
###img2=filter(img,3,1)
#img2=filters.upsample(img2,5)
#img=filters.sub(img,img2)
#img2=filters.removepadd(img,5)
#print(img2)
#img2=img2.astype(np.uint8)
#print(img2)
#cv2.imshow("Average filter",img2)

##Laplacian level 3
#img=cv2.imread('./image_3.png',0)
###img=np.array([[11,12,5,2],[0,15,6,10],[11,12,5,2],[0,15,6,10]])
#img2=filters.padding(img,5)
#print (img2.shape)
#img2=filters.filter(img2,5,1)
#img=filters.downsample(img2,5)
#print ("After downsample",img.shape)
#img2=filters.filter(img,5,1)
#img=filters.downsample(img2,5)
#print ("After 2 downsample",img.shape)
#img2=filters.filter(img,5,1)
#img2=filters.downsample(img2,5)
#print ("After 2 downsample",img2.shape)
###img=addnoise(img,20)
##img2=haar(img2,3)
#print(img2)
##img2=removepadd(img2,3)
###img2=filter(img,3,1)
#img2=filters.upsample(img2,5)
#print ("After upsample",img2.shape)
#img=filters.sub(img,img2)
#img2=filters.removepadd(img,5)
#print(img2)
#img2=img2.astype(np.uint8)
#print(img2)
#cv2.imshow("Average filter",img2)
#cv2.waitKey(0)

#Question5

#Average
#img1=cv2.imread('./image_1.jpg',0)
#img=filters.padding(img1,15)
#img=filters.filter(img,15,0)
#img=filters.removepadd(img,15)
#blur=cv2.blur(img1,(15,15))
#blur=filters.sub(img,blur)
#blur=blur*5
#cv2.imshow("Average filter",blur)


##Median
img1=cv2.imread('./image_2.png',0)
img1=filters.addnoise(img1,10)
img=filters.padding(img1,15)
img=filters.filter(img,15,1)
img=filters.removepadd(img,15)
print(img)
img=img.astype(np.uint8)
blur=cv2.medianBlur(img1,15)
blur=filters.sub(img,blur)
blur=blur*5
cv2.imshow("Median filter",blur)


###Gaussian
#img1=cv2.imread('./image_3.png',0)
#img=filters.padding(img1,15)
#img=filters.filter(img,15,2)
#img=filters.removepadd(img,15)
#print(img)
#img=img.astype(np.uint8)
#blur=cv2.GaussianBlur(img1,(15,15),5)
#blur=filters.sub(img,blur)
#cv2.imshow("Gaussian filter",blur)

##Question 6
#img=cv2.imread("./image_3.png",0)
#imgn=filters.addnoise(img,10)
##imgn=imgn.astype(np.uint8)
#imgl=cv2.Laplacian(img,cv2.CV_64F)
##imgl=imgl.astype(np.uint8)
#img=imgn+imgl
#c=pywt.dwt2(img,'haar')
#ll,(lh,hl,hh)=c
#z=np.zeros(hh.shape)
#c=ll,(z,z,z)
#img=pywt.idwt2(c,'haar')
#img=img.astype(np.uint8)
#cv2.imshow("Smoothening",img)




###Question 7
#wtrmrk=cv2.imread("./Watermark.jpg",0)
#img=cv2.imread("./image_3.png",0)
#c=pywt.dwt2(img,'haar')
#ll,(lh,hl,hh)=c
#row, col= ll.shape
#print (ll.shape)
#print (wtrmrk.shape)
#for i in range(row):
#    for j in range(col):
#        ll[i][j]=ll[i][j]+wtrmrk[i][j]
#c=ll, (lh,hl,hh)
#img=pywt.idwt2(c,'haar')
##img=img.astype(np.uint8)
#cv2.imwrite('WK.jpg',img)



cv2.waitKey(0)


