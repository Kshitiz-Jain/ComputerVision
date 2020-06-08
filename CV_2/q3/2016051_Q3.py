import numpy as np
import cv2
import math
from copy import deepcopy

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

# def fx(img):
#     row,col = img.shape
#     nimg=np.zeros((row,col))
#     for i in range(row):
#         for j in range(col-1):
#             nimg[i][j]=abs(img[i][j+1]-img[i][j])
#     return nimg

# def fy(img):
#     row,col = img.shape
#     nimg=np.zeros((row,col))
#     for i in range(col):
#         for j in range(row-1):
#             nimg[j][i]=abs(img[j+1][i]-img[j][i])
#     return nimg

def applyfil(img,matfil,ker,r,c):
    a=r-math.floor(ker/2)
    b=c-math.floor(ker/2)
    sum=0.0
    for x in range(ker):
        for y in range(ker):
            sum = sum + (img[a+x][b+y]*matfil[x][y])
    sum=(sum)/(ker*ker)
    return sum


def gaussian(st,x,y):
    a=1
    if(st!=0):
        exp=math.exp(-(math.pow(x,2)+math.pow(y,2))/(2*math.pow(st,2)))
        a=(1/(2*math.pi*st*st)) * exp
    return a

def gaussfil(ker):
    mat=np.zeros((ker,ker))
    a=-math.floor(ker/2)
    b=-math.floor(ker/2)
    for i in range(ker):
        for j in range(ker):
            mat[i][j]=gaussian(1.5,a+i,b+j)
    mat=mat/np.sum(mat)
#    print(mat)
    return mat

def sobelfilx(ker):
    mat=np.zeros((ker,ker))
    mat[0][0]=-1
    mat[0][1]=0
    mat[0][2]=1
    
    mat[1][0]=-2
    mat[1][1]=0
    mat[1][2]=2
    
    mat[2][0]=-1
    mat[2][1]=0
    mat[2][2]=1

    return mat/8

def sobelfily(ker):
    mat=np.zeros((ker,ker))
    mat[0][0]=-1
    mat[0][1]=-2
    mat[0][2]=-1
    
    mat[1][0]=0
    mat[1][1]=0
    mat[1][2]=0
    
    mat[2][0]=1
    mat[2][1]=2
    mat[2][2]=1

    return mat/8

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
            elif(case==3):
                matfil=sobelfilx(3)
                newimg[pad+i][pad+j]=applyfil(img,matfil,ker,pad+i,pad+j)*ker*ker
            elif(case==4):
                matfil=sobelfily(3)
                newimg[pad+i][pad+j]=applyfil(img,matfil,ker,pad+i,pad+j)*ker*ker
#    print (newimg)
    return newimg
def avg4(img,r,c):
    ans=0
    for i in range(2):
        for j in range(2):
            ans=ans+img[r+i][c+j]
    ans=ans/4.0
    return ans

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

ker=7
img=cv2.imread('./yosemite1.jpg',0)
#img=np.rot90(img,3)
#img=np.flip(img,axis=1)
# img=downsample(img,0)
# cv2.imwrite('Q3Graychess.png',img)
img=padding(img,ker)
# cv2.imwrite('orig.png',img)
img1=filter(img,3,3)
#img1=img1.astype(np.uint8)
cv2.imwrite('Q3Xgradflow.png',img1)
# cv2.imwrite('fx.png',img1)
img2=filter(img,3,4)
img2=img2.astype(np.uint8)
cv2.imwrite('Q3Ygradflow.png',img2)
# cv2.imwrite('fy.png',img2)
fil=gaussfil(ker)


def sum(array,i,j,key,fil):
    sum=0
    i=i-math.floor(key/2)
    j=j-math.floor(key/2)
    for l in range(key):
        for m in range(key):
            sum=sum+(math.pow(array[i+l][j+m],2)*fil[l][m])
    return sum

def dot(arr1,arr2,i,j,key,fil):
    sum=0
    i=i-math.floor(key/2)
    j=j-math.floor(key/2)
    for l in range(key):
        for m in range(key):
            sum=sum+(arr1[i+l][j+m]*arr2[i+l][j+m]*fil[l][m])
    return sum

def covmat(img,fx,fy,ker,fil,thres):
    row,col=img.shape
    pad=math.floor(ker/2)
    nimg=deepcopy(img)
    print(row)
    tempp=0
    for i in range(pad,row-pad):
        for j in range(pad,col-pad):
            mat=np.zeros((2,2))
            mat[0][0]=sum(fx,i,j,ker,fil)
            mat[1][1]=sum(fy,i,j,ker,fil)
            mat[1][0]=dot(fx,fy,i,j,ker,fil)
            mat[0][1]=mat[1][0]
            # print(mat) 
            eign,vec = np.linalg.eig(mat)
            trace=eign[0]+eign[1]
            det=eign[0]*eign[1]
            val=det-0.06*math.pow(trace,2)
            if( val>thres):
                nimg[i][j]=128
                tempp=tempp+1

            # print (eign)
        print(i)
    print(tempp)
    temp=0
    cv2.imwrite('Q3down'+str(thres)+'finyose.jpg',nimg)

#covmat(img,img1,img2,ker,fil,100)
#covmat(img,img1,img2,ker,fil,1000)
covmat(img,img1,img2,ker,fil,10000)








