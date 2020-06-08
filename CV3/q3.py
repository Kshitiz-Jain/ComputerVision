import cv2
import glob
import math
import numpy as np

def padding(image,padd):
    psize=math.floor(padd/2)
    row,col = image.shape
    nimg=np.zeros((row+2*psize,col+2*psize))
    for i in range(row):
        for j in range(col):
            nimg[i+psize][j+psize]=image[i][j]
    return nimg


ims=glob.glob('./Q3-faces/*.jpg')
faces=[]
seeds=[]
for i in ims:
	# print(i)
	im=cv2.imread(i)
	im=cv2.resize(im,(0,0),fx=0.5,fy=0.5)
	im0=padding(im[:,:,0],3)
	im1=padding(im[:,:,1],3)
	im2=padding(im[:,:,2],3)
	img=[im0,im1,im2]
	faces.append(img)
	row,col = np.shape(im0)
	seeds.append([int(row/2),int(col/2)])

finfaces=[]
for i in range(len(faces)):
	row,col=np.shape(faces[i][0])
	finfaces.append(np.zeros((row,col)))

visited=[]
for i in range(len(faces)):
	row,col=np.shape(faces[i][0])
	visited.append(np.zeros((row,col)))


def eucl(r1,g1,b1,r2,g2,b2):
	r=abs(r1-r2)
	g=abs(g1-g2)
	b=abs(b1-b2)
	dist=math.pow(r,2) + math.pow(g,2) + math.pow(b,2)
	print(math.sqrt(dist))
	return math.sqrt(dist)

def manhatn(r1,g1,b1,r2,g2,b2):
	r=abs(r1-r2)
	g=abs(g1-g2)
	b=abs(b1-b2)
	dist = r+g+b
	return dist

def findnear(img,seed,thresh,i):
	print(seed)
	d,rmax,cmax=np.shape(img)
	row=seed[0]
	col=seed[1]	#convert to int from unit8
	if(row>0 and col>0 and row<rmax-2 and col<cmax-2):
		if(visited[i][row][col]==-1):
			print("repeated")
			return
		visited[i][row][col]=-1
		if(eucl(img[0][row][col],img[1][row][col],img[2][row][col],img[0][row+1][col],img[1][row+1][col],img[2][row+1][col]) <= thresh):
			findnear(img,[row+1,col],thresh,i)
			finfaces[i][row+1][col]=255

		if(eucl(img[0][row][col],img[1][row][col],img[2][row][col],img[0][row-1][col],img[1][row-1][col],img[2][row-1][col]) <= thresh):
			findnear(img,[row-1,col],thresh,i)
			finfaces[i][row-1][col]=255

		if(eucl(img[0][row][col],img[1][row][col],img[2][row][col],img[0][row][col+1],img[1][row][col+1],img[2][row][col+1]) <= thresh):
			findnear(img,[row,col+1],thresh,i)
			finfaces[i][row][col+1]=255

		if(eucl(img[0][row][col],img[1][row][col],img[2][row][col],img[0][row][col-1],img[1][row][col-1],img[2][row][col-1]) <= thresh):
			findnear(img,[row,col-1],thresh,i)
			finfaces[i][row][col-1]=255
	else:
		return

	return

def findfaces(fcs,sds,thresh):
	finf=[]
	for i in range(len(fcs)):
		print(np.shape(fcs[i]))
		findnear(fcs[i],sds[i],thresh,i)
		cv2.imwrite("fc"+str(i+1)+"_"+str(thresh)+".png",finfaces[i])

print(len(faces))
findfaces(faces,seeds,3)











