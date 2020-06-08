import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster
import glob

imgs=[]
shapes=[]
ims=glob.glob('./Q2-images/*.jpg')
for i in ims:
	img=cv2.imread(i)
	shapes.append(np.shape(img))
	img=np.reshape(img,(np.shape(img)[0]*np.shape(img)[1],3))
	imgs.append(img)


x=0.05
while(x<0.5):
	for y in range(200,600,50):	
		for i in range(len(imgs)):
			shp=shapes[i]
			print("Hello1")
			image=imgs[i]
			bndwdth=sklearn.cluster.estimate_bandwidth(image,quantile=x,n_samples=y)
			ms = sklearn.cluster.MeanShift(bndwdth,bin_seeding=True)
			ms.fit(image)
			lbls=ms.labels_
			pic=np.reshape(lbls,(shp[0],shp[1]))
			# cv2.imwrite("Image_"+str(i+1)+"(quan,samp)_("+str(x)+" "+str(y)+").png",pic.astype(int))
			plt.imshow(pic)
			plt.savefig("Image_"+str(i+1)+"(quan,samp)_("+str(x)+" "+str(y)+").png")
			plt.clf()
			print("Hello")
	x=x+0.05


