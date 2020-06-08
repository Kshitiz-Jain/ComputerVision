import cv2
import numpy as np
import math
import copy
import random
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
		if (np.all(np.less_equal(d,np.array([5,5,5])))):
			convergence=False
		print(centers)
		centers=copy.deepcopy(newcen)
		print(centers)
	return centers,labels.astype(np.int32)

ims=glob.glob('./Q1-images/*.jpg')
faces=[]
shapes=[]
for i in ims:
	print(i)
	im=cv2.imread(i)
	# im=cv2.resize(im,(0,0),fx=0.5,fy=0.5)
	shapes.append(np.shape(im))
	im=np.reshape(im,(np.shape(im)[0]*np.shape(im)[1],3))
	# im=cv2.resize(im,(0,0),fx=0.5,fy=0.)
	faces.append(im)

for j in range(3,4):
	print(j)
	print(np.shape(faces[j]))
	centers,lbls=clustering(faces[j].astype(np.int32),4)
	# colors=[np.array([250,250,250]),np.array([0,250,0]),np.array([0,0,250]),np.array([250,0,0]),np.array([0,0,0])]
	

	img=faces[j]
	imx=img[:,0]
	x=np.reshape(imx,(np.shape(img)[0]))
	imy=img[:,1]
	y=np.reshape(imy,(np.shape(img)[0]))
	imz=img[:,2]
	z=np.reshape(imz,(np.shape(img)[0]))

	fig = plt.figure()
	ax = Axes3D(fig)
	ax.scatter(x, y, z, c = lbls)
	plt.show()
	
	for i in range(len(faces[j])):
		faces[j][i]=centers[lbls[i]]


	# C = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
	

	f1=np.reshape(faces[j],shapes[j])
	cv2.imwrite("cluster3D"+str(j+1)+"("+str(6)+")"+".png",f1)













