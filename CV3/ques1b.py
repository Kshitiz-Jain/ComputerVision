import cv2
import numpy as np
import math
import copy
import random
import glob

def minfrmcen(centers,dpt):
	ind=0
	maxi=9999
	for i in range(len(centers)):
		# print(np.shape(dpt),np.shape(centers[i]))
		dist=np.linalg.norm(centers[i]-dpt,axis=0)
		if(dist<maxi):
			ind=i
			maxi=dist
	return ind

def clustering(data,num):
	centers=[]
	for i in range(num):
		r=random.randint(0,len(data)-1)
		centers.append(copy.deepcopy(data[r]))
	convergence=True
	labels=np.zeros((len(data)))
	while(convergence):
		newcen=copy.deepcopy(centers)
		freq=np.zeros((num))
		print(np.shape(freq))
		for i in range(len(data)):
			# print(i)
			labl=minfrmcen(centers,data[i])
			newcen[labl]=newcen[labl]+data[i]
			freq[labl]=freq[labl]+1
			labels[i]=labl
		# newcen=(newcen-centers)
		for i in range(num):
			newcen[i]=newcen[i]-centers[i]
			newcen[i]=newcen[i]/freq[i]
		d=0
		for i in range(num):
			d=d+np.abs(centers[i]-newcen[i])
		print(d)
		if (np.all(np.less_equal(d,np.array([5,5,5,5,5,])))):
			convergence=False
		# print(centers)
		centers=copy.deepcopy(newcen)
		# print(centers)
	return centers,labels.astype(np.int32)

ims=glob.glob('./Q1-images/*.jpg')
faces=[]
shapes=[]
for i in ims:
	print(i)
	im=cv2.imread(i)
	# im=cv2.resize(im,(0,0),fx=0.5,fy=0.5)
	shapes.append(np.shape(im))
	r,c,d=np.shape(im)
	temp=[]
	for i in range(r):
		for j in range(c):
			array=np.array([i,j,im[i][j][0],im[i][j][1],im[i][j][2]])
			temp.append(array)
			pass
	# im=np.reshape(im,(np.shape(im)[0]*np.shape(im)[1],3))
	im=cv2.resize(im,(0,0),fx=0.5,fy=0.5)
	faces.append(np.asarray(temp))

for j in range(1,2):
	# print(j)
	print(np.shape(faces[j]))
	centers,lbls=clustering(faces[j].astype(np.int32),5)
	# colors=[np.array([250,250,250]),np.array([0,250,0]),np.array([0,0,250]),np.array([250,0,0]),np.array([0,0,0])]
	image=np.zeros((shapes[j]))
	for i in range(len(faces[j])):
		image[faces[j][i][0]][faces[j][i][1]]=[centers[lbls[i]][2],centers[lbls[i]][3],centers[lbls[i]][4]]
	

	# f1=np.reshape(faces[j],shapes[j])
	cv2.imwrite("cluster5d"+str(j+1)+"("+str(5)+")"+".png",image)








