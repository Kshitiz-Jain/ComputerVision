import cv2
import math
import glob
import numpy as np

ims=glob.glob('./Q3-faces/*.jpg')
faces=[]
seeds=[]
for i in ims:
	# print(i)
	im=cv2.imread(i)
	im=cv2.resize(im,(0,0),fx=0.5,fy=0.5)
	faces.append(im)
	row,col,d = np.shape(im)
	seeds.append([int(row/2-3),int(col/2-3)])


finface=None
visited=None
row,col,d=np.shape(faces[0])
finface=np.zeros((row,col))
visited=np.zeros((row,col))

def eucl(i1,i2):
	r1=i1[0]
	g1=i1[1]
	b1=i1[2]
	r2=i2[0]
	b2=i2[1]
	g2=i2[2]
	r=abs(r1-r2)
	g=abs(g1-g2)
	b=abs(b1-b2)
	dist=math.pow(r,2) + math.pow(g,2) + math.pow(b,2)
	print("dist",math.sqrt(dist))
	return math.sqrt(dist)

def manh(i1,i2):
	r1=i1[0]
	g1=i1[1]
	b1=i1[2]
	r2=i2[0]
	b2=i2[1]
	g2=i2[2]
	r=abs(r1-r2)
	g=abs(g1-g2)
	b=abs(b1-b2)
	# dist=math.pow(r,2) + math.pow(g,2) + math.pow(b,2)
	dist=r+g+b
	print(dist)
	return dist

def p(img):
	row,col=np.shape(img)
	# print(np.shape(img))
	for x in range(row):
		for y in range(col):
			print(img[x][y],end=" ")
			pass
		print(" ")
		pass

def findnear(img,seed,thresh):
	rmax,cmax,d=np.shape(img)
	row=seed[0]
	col=seed[1]	
	# p(finface)
	if(row>0 and col>0 and row<rmax-2 and col<cmax-2):
		if(visited[row][col]==-1):
			# print("repeated")
			return
		visited[row][col]=-1
		if(manh(img[row][col],img[row+1][col]) <= thresh):
			finface[row][col-1]=255
			findnear(img,[row+1,col],thresh)

		if(manh(img[row][col],img[row-1][col]) <= thresh):
			finface[row][col-1]=255
			findnear(img,[row-1,col],thresh)

		if(manh(img[row][col],img[row][col+1]) <= thresh):
			finface[row][col-1]=255
			findnear(img,[row,col+1],thresh)

		if(manh(img[row][col],img[row][col-1]) <= thresh):
			finface[row][col-1]=255
			findnear(img,[row,col-1],thresh)	
	else:
		return

	return


def queue(img,seed,thresh):
	queue=[]
	queue.append(seed)
	r,c,d=np.shape(img)
	ff=np.zeros((r,c))
	visi=np.zeros((r,c))
	avg=np.zeros(np.shape(img[0][0]))
	tot=0
	while(len(queue)!=0):
		print(len(queue))
		s=queue.pop(0)
		row=s[0]
		col=s[1]
		ff[row][col]=250
		avg=avg+img[row][col]
		tot=tot+1
		if(visi[row][col]!=-1 and row>0 and col>0 and row<r-2 and col<c-2):
			x = manh(img[row][col],img[row+1][col])
			# print(thresh)
			if(x <= thresh and manh(avg/tot,img[row+1][col]) < thresh):
				print("down")
				queue.append([row+1,col])

			if(manh(img[row][col],img[row-1][col]) <= thresh and manh(avg/tot,img[row-1][col]) < thresh):
				queue.append([row-1,col])
				print("up")

			if(manh(img[row][col],img[row][col+1]) <= thresh and manh(avg/tot,img[row][col+1]) < thresh):
				queue.append([row,col+1])
				print("right")

			if(manh(img[row][col],img[row][col-1]) <= thresh and manh(avg/tot,img[row][col-1]) < thresh):
				queue.append([row,col-1])
				print("left")
		visi[row][col]=-1
	print("done")
	cv2.imwrite("./Q3-faces/img.png",ff.astype(np.uint8))
	# exit(0)


def findface(thresh):
	for i in range(1):
		queue(faces[i].astype(np.int32),seeds[i],thresh)
		# row,col,d=np.shape(faces[i+2])
		# finface=np.zeros((row,col))
		# visited=np.zeros((row,col))
		# findnear(faces[i].astype(int),seeds[i],thresh)
		# print(finface)
		# cv2.imwrite("img"+str(i)+".png",finface)

findface(160)

