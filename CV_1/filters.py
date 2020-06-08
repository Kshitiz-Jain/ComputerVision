import cv2
import numpy as np
import math
import random

def padding(image,padd):
    psize=math.floor(padd/2)
    row,col = image.shape
    nimg=np.zeros((row+2*psize,col+2*psize))
    print (nimg)
    print(image)
    for i in range(row):
        for j in range(col):
            nimg[i+psize][j+psize]=image[i][j]
#    For Symmetric padding
#    for j in range(psize):
#        for i in range(row+2*psize):
#            nimg[i][psize-j-1]=nimg[i][psize-j]
#            nimg[i][col+psize-1+j]=nimg[i][col+psize-2+j]
#    for j in range(psize):
#        for i in range(col+2*psize):
#            nimg[psize-j-1][i]=nimg[psize-j][i]
#            nimg[row+psize-1+j][i]=nimg[row+psize-2+j][i]
    return nimg

def gaussian(st,x,y):
    a=1
    if(st!=0):
        exp=math.exp(-(math.pow(x,2)+math.pow(y,2))/(2*math.pow(st,2)))
        a=(1/((2*math.pi)*st*st)) * exp
    return a


def gaussfil(ker):
    mat=np.zeros((ker,ker))
    a=-math.floor(ker/2)
    b=-math.floor(ker/2)
    for i in range(ker):
        for j in range(ker):
            mat[i][j]=gaussian(15,a+i,b+j)
    mat=mat/np.sum(mat)
#    print(mat)
    return mat

def lapfil(ker):
    mat=np.zeros((ker,ker))
    a=-math.floor(ker/2)
    b=-math.floor(ker/2)
    for i in range(ker):
        for j in range(ker):
            mat[i][j]=gaussian(5,a+i,b+j)
    mat[int(math.floor(ker/2))][int(math.floor(ker/2))]=-mat[int(math.floor(ker/2))][int(math.floor(ker/2))]
    mat=np.divide(mat,np.sum(mat))
    return mat


def avgfil(ker):
    a=np.ones((ker,ker))
    a=a.astype(np.uint8)
    return a

def applyfil(img,matfil,ker,r,c):
    a=r-math.floor(ker/2)
    b=c-math.floor(ker/2)
    sum=0.0
    for x in range(ker):
        for y in range(ker):
            sum = sum + (img[a+x][b+y]*matfil[x][y])
    sum=sum/(ker*ker)
    return sum

def medianfil(img,ker,r,c):
    a=r-math.floor(ker/2)
    b=c-math.floor(ker/2)
    array=[]
    for x in range(ker):
        for y in range(ker):
            array.append(img[a+x][b+y])
    array.sort()
    return array[math.floor(len(array)/2)]

def filter(img,ker,case):
    row,col=img.shape
    pad=math.floor(ker/2)
    newimg = np.zeros((row,col))
    for i in range(row-2*pad):
        for j in range(col-2*pad):
            if(case==0):
                matfil=avgfil(ker)
                newimg[pad+i][pad+j]=applyfil(img,matfil,ker,pad+i,pad+j)
            elif(case==2):
#                print ("median")
                matfil=gaussfil(ker)
#                print(matfil)
                newimg[pad+i][pad+j]=applyfil(img,matfil,ker,pad+i,pad+j)*ker*ker
            elif(case==1):
                newimg[pad+i][pad+j]=medianfil(img,ker,pad+i,pad+j)
#    print (newimg)
    return newimg

def addnoise(img,quan):
    num=int(img.size * quan / 100)
    row,col=img.shape
    for i in range(num):
        a=int(random.random()*row)
        b=int(random.random()*col)
        c=random.random()
        if(c<0.5):
            img[a][b]=255
        else:
            img[a][b]=0
    return img


def avg4(img,r,c):
    ans=0
    for i in range(2):
        for j in range(2):
            ans=ans+img[r+i][c+j]
    ans=ans/4.0
    return ans

def upsample(img,ker):
    row,col=img.shape
    pad=math.floor(ker/2)
    r1=2*row-2*pad
    c1=2*col-2*pad
    newimg = np.zeros((2*row-2*pad,2*col-2*pad))
    for i in range(row-2*pad):
        for j in range(col-2*pad):
            newimg[pad+2*i][pad+2*j]=img[pad+i][pad+j]

    for i in range(row-2*pad):
        for j in range(col-2*pad-1):
            newimg[pad+2*i][pad+1+2*j]=(newimg[pad+2*i][pad+2+2*j]+newimg[pad+2*i][pad+2*j])/2

    for i in range(row-2*pad-1):
        for j in range(c1-2*pad):
            newimg[pad+1+2*i][pad+j]=(newimg[pad+2+2*i][pad+j]+newimg[pad+2*i][pad+j])/2
    for i in range(2*row-2*pad):
        newimg[i][2*col-3*pad-1]=newimg[i][2*col-3*pad-2]
    for i in range(2*row-2*pad):
        newimg[2*row-3*pad-1][i]=newimg[2*row-3*pad-2][i]

    return newimg

def haarrow(matrix,ker):
    row,col=matrix.shape
    newimg=np.zeros((row,col))
    pad=math.floor(ker/2)
    for i in range(row-2*pad):
        temp=[]
        temp.append(0)
        temp.append(1)
        temp[0]=[]
        temp[1]=[]
        for j in range(int((col-2*pad)/2)):
            temp[0].append((matrix[pad+i][pad+2*j]+matrix[pad+i][pad+2*j+1])/2)
            temp[1].append((matrix[pad+i][pad+2*j]-matrix[pad+i][pad+2*j+1])/2)
        for j in range(int((col-2*pad)/2)):
            newimg[pad+i][pad+j]=temp[0][j]
        for j in range(int((col-2*pad)/2)):
            newimg[pad+i][pad+j+int((col-2*pad)/2)]=temp[1][j]
    return newimg

def haarcol(matrix,ker):
    row,col=matrix.shape
    newimg=np.zeros((row,col))
    pad=math.floor(ker/2)
    for i in range(col-2*pad):
        temp=[]
        temp.append(0)
        temp.append(1)
        temp[0]=[]
        temp[1]=[]
        for j in range(int((row-2*pad)/2)):
            temp[0].append((matrix[pad+2*j][pad+i]+matrix[pad+2*j+1][pad+i])/2)
            temp[1].append((matrix[pad+2*j][pad+i]-matrix[pad+2*j+1][pad+i])/2)
        for j in range(int((row-2*pad)/2)):
            newimg[pad+j][pad+i]=temp[0][j]
        for j in range(int((row-2*pad)/2)):
            newimg[pad+j+int((row-2*pad)/2)][pad+i]=temp[1][j]
    return newimg

def haar(img,ker):
    img=haarrow(img,ker)
    img=haarcol(img,ker)
    return img
                                                                                      
                                                                                      

def downsample(img,ker):
    row,col=img.shape
    print ("Downsample input",img.shape)
    pad=math.floor(ker/2)
#    print (int(row/2)+pad,int(col/2)+pad)
    newimg = np.zeros( (int(row/2)+pad,int(col/2)+pad))
    print ("Downsample fxn",newimg.shape)
    for i in range(int(row/2)-pad):
        for j in range(int(col/2)-pad):
            newimg[i+pad][j+pad]=avg4(img,i*2+pad,j*2+pad)
    return newimg

def removepadd(image, ker):
    row,col=image.shape
    pad=int(math.floor(ker/2))
    newimg=np.zeros((row-2*pad,col-2*pad))
    for i in range(row-2*pad):
        for j in range(col-2*pad):
            newimg[i][j]=image[i+pad][j+pad]
    return newimg
                                                                                      
#def sub(mat1,mat2):
#    row,col=mat1.shape
#    mat=np.zeros((row,col))
#    for i in range(row):
#        for j in range(col):
#            a=mat1[i][j]-mat2[i][j]
#            if(a<0):
#                mat[i][j]=mat2[i][j]-mat1[i][j]
#            else:
#                mat[i][j]=mat1[i][j]-mat2[i][j]
#    return mat

def sub(img1, img2):
    a1=np.int16(img1)-img2
    a2=np.uint8(absolute(a1))
    return a2

def absolute(mat):
    row,col=mat.shape
    for i in range(row):
        for j in range(col):
            if(mat[i][j]<0):
                mat[i][j]=-mat[i][j]
    return mat



#img=cv2.imread('./image_3.png',0)
###img=np.array([[11,12,5,2],[0,15,6,10],[11,12,5,2],[0,15,6,10]])
#img2=padding(img,5)
#print (img2.shape)
#img2=filter(img2,5,1)
#img2=downsample(img2,5)
##print ("After downsample",img.shape)
#img2=filter(img2,5,1)
#img2=downsample(img2,5)
#print ("After 2 downsample",img.shape)
#img2=filter(img,5,1)
#img2=downsample(img2,5)
#print ("After 2 downsample",img2.shape)
###img=addnoise(img,20)
##img2=haar(img2,3)
#print(img2)
##img2=removepadd(img2,3)
###img2=filter(img,3,1)
#img2=upsample(img2,5)
#print ("After upsample",img2.shape)
#img=img-img2
#img2=removepadd(img2,5)
##print(img2)
#img2=img2.astype(np.uint8)
##print(img2)
#cv2.imshow("Average filter",img2)
#cv2.waitKey(0)
##print (type(img))
#print ('RGB shape: ', img.shape)
#print ('img.dtype: ', img.dtype)
#print ('img.size: ', img.size)
