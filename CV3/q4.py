#!/usr/bin/env python
# coding: utf-8

# In[46]:


import cv2
import numpy
import numpy as np
import copy
from sklearn.cluster import KMeans
import math
import random
from sklearn.naive_bayes import GaussianNB
from skimage.feature import local_binary_pattern

def confmat(clf, data, label,total):
    matrix=np.zeros((total,total))
    pros=[]
    for i in range(len(data)):
        temp=[]
        temp.append(data[i])
        cl=clf.predict(temp)
        pros.append(clf.predict_proba(temp))
        matrix[label[i]-1][cl-1]=matrix[label[i]-1][cl-1]+1
    cor=0
    for i in range(total):
        cor=cor+matrix[i][i]
    print("Accuracy :",np.sum(matrix.diagonal())/np.sum(matrix))
    return matrix,pros

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def data2():
    traind=[]
    trainl=[]
    testd=[]
    testl=[]
    for i in range(5):
        d=unpickle('./cifar-10-batches-py/data_batch_'+str(i+1))
        data=d[b'data']
        lbl=d[b'labels']
        for i in range(len(data)):
            img=cv2.cvtColor(np.reshape(data[i],(32,32,3)),cv2.COLOR_BGR2GRAY)
            # img=np.ravel(img)
            traind.append(img)
            trainl.append(lbl[i])
    d2=unpickle('./cifar-10-batches-py/test_batch')
    data2=d2[b'data']
    lbl2=d2[b'labels']
    for i in range(len(data2)):
        img=cv2.cvtColor(np.reshape(data2[i],(32,32,3)),cv2.COLOR_BGR2GRAY)
        # img=np.ravel(img)
        testd.append(img)
        testl.append(lbl2[i])
    return traind,trainl,testd,testl,(32,32)

def distribute(data,label):
    datacp=copy.deepcopy(data)
    labelcp=copy.deepcopy(label)
    a=int(len(labelcp)*0.5)
    testdata=[]
    testlabel=[]
    length=len(labelcp)
    for i in range(a):
        ind=int(random.random()*(len(labelcp)-1))
        testdata.append(datacp.pop(ind))
        testlabel.append(labelcp.pop(ind))
    print(len(datacp),len(labelcp),len(testdata),len(testlabel))
    return datacp,labelcp,testdata,testlabel

a,b,c,d,shape=data2()

a1,b1,a2,b2=distribute(a,b)

def hogdes():
	calc_HOG =	cv2.HOGDescriptor((8,8),(4,4),(2,2),(2,2),9,1,4.0,0,0.20000000000001,0,8)
	return calc_HOG

def divide(img,size):
	row,col=np.shape(img)
	patches=[]
	for i in range(0,row,size):
		for j in range(0,col,size):
			part=img[i:i+size,j:j+size]
			patches.append(copy.deepcopy(part))
	return patches

# def make_lbphist(train_x, size):
#   histogram = []
#   for i in range(len(train_x)):
#     for r in range(0,train_x[i].shape[0] - size, size):
#       for c in range(0,train_x[i].shape[1] - size, size):
#           patch = train_x[i][r : r+size, c : c+size]
#           lbp = local_binary_pattern(patch, 8*3, 3, 'uniform')
#         #   print (lbp.shape)
#           (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 8*3 + 3), range=(0, 8*3 + 2))

#             # normalize the histogram
#           hist = hist.astype("float")
#           eps=1e-7
#           hist /= (hist.sum() + eps)
#           histogram.append(hist)
#   return histogram

def computehogs(data,size):
    hogf=[]
    hog=hogdes()
    # hog=cv2.HOGDescriptor()
    for i in range(len(data)):
        patches=divide(data[i],size)
        for j in range(len(patches)):
            hf=hog.compute(patches[j])
            hogf.append(hf.reshape((np.shape(hf)[0],)))
#         print(i)
    print(len(data))
    print(len(hogf))
    return hogf

def computelbps(data,size):
    lbpf=[]
    e=1e-7
    k=0
    for i in range(len(data)):
        patches=divide(data[i],size)
        for j in range(len(patches)):
#             hf=hog.compute(patches[j])
            lbp = local_binary_pattern(patches[j], 8*3, 3, 'uniform')
            lbp.ravel()
            k=k+1
            (h, _) = np.histogram(lbp, bins=np.arange(0, 8*3 + 3), range=(0, 8*3 + 2))
            h=h.astype("float")
            h /= (h.sum() + e)
            lbpf.append(h)
#             hogf.append(hf.reshape((np.shape(hf)[0],)))
#         print(i)
    print(len(data))
    print(len(lbpf))
    return lbpf

# words=computehogs(a1,8)
# print("HOGS computed")
words=computelbps(a1,8)
print("LBPS computed")
kmeans = KMeans(n_clusters=10, random_state=0).fit(np.array(words))
print("Clustering done")
bow=kmeans.cluster_centers_
print("Bags of words created")


# In[47]:


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

def cnvrttrn(a,bow,size):
    data=[]
# 	row=np.shape(a[0])
    num=int(math.pow(32/size,2))
    for i in range(0,len(a),num):
        temp=np.zeros((np.shape(bow)[0]))
        for j in range(i,i+num):
            ind=minfrmcen(bow,a[j])
            temp[ind]=temp[ind]+1
        data.append(temp)
#         print(i)
    return data


# In[48]:


# newa2=computehogs(a2,8)
newa2=computelbps(a2,8)
trainx=cnvrttrn(newa2,bow,8)


# In[50]:


# newc=computehogs(c,8)
newc=computelbps(c,8)
testx=cnvrttrn(newc,bow,8)


# In[52]:


clf = GaussianNB()
clf.fit(trainx,b2)
clf.score(testx,d)
# confmat(clf,testx,d,10)


# In[ ]:






# In[ ]:




