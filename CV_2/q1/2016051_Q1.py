import numpy as np
import cv2
import math



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

img=cv2.imread('./coins.jpeg',0)
img=downsample(img,0)
img=downsample(img,0)
blur=cv2.GaussianBlur(img,(5,5),1)
blur=blur.astype(np.uint8)
edges=cv2.Canny(blur,100,170)
cv2.imwrite('canny1.jpg',edges)
row,col=edges.shape
for i in range(row):
	a=[]
	for j in range(col):
		a.append(edges[i][j])
	print(a)


rad=min(row,col)
hough=np.zeros((rad,row,col))
dic={}

for r in range(10,rad):
	for x in range(row):
		for y in range(col):
			if(edges[x][y]>250):
				for t in range(360):
					a=x+(r*math.cos(math.pi*t/180))
					b=y+(r*math.sin(math.pi*t/180))
					key=
					if(a>=0 and a<row-1 and b>=0 and b<col-1):
						hough[r][int(a)][int(b)]=hough[r][int(a)][int(b)]+1
					else:
						continue
			else:
				continue
	print("hough",r)
dic=sorted(dic.items(),key=lamda x:x[1], reverse=True)

# rad,row,col=hough.shape
# for r in range(10,rad):
# 	for i in range(row):
# 		for j in range(col):
# 			print(edges[r][i][j])

# for r in range(10,rad):
# 	for x in range(row):
# 		for y in range(col):
# 			if(hough[r][x][y]>40):
# 				for t in range(360):
# 					a=x+(r*math.cos(math.pi*t/180))
# 					b=y+(r*math.sin(math.pi*t/180))
# 					key=str(int(a))+" "+str(int(b))+" "+str(int(r))
# 					if(key not in dic):
# 						dic[key]=1
# 					else:
# 						dic[key]=dic[key]+1
# 			else:
# 				continue
# 	print("circles",r)


cv2.imwrite('circles.jpg',edges)











