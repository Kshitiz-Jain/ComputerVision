import numpy as np
import math
import pickle
from skimage.feature import hog
import matplotlib.pyplot as plt
import cv2
from skimage.feature import local_binary_pattern


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


a,b,c,d,shp=data2()

# btr=cv2.imread("./Q3-faces/face4.jpg",0)
# fd, hog_image = hog(btr, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualise=True)
# plt.imshow(hog_image)
# plt.savefig("Facehog.png")
# plt.clf()

# lbp = local_binary_pattern(btr, 8*3, 3, 'uniform')
# plt.imshow(lbp)
# plt.savefig("Facelbp.png")
# plt.clf()

# plt.savefig("btr.png")
# plt.clf()

pxlpc=(16,16)
clpb=(1,1)

btr=a[10]
fd, hog_image = hog(btr, orientations=8, pixels_per_cell=pxlpc,cells_per_block=clpb, visualize=True)
plt.imshow(hog_image)
plt.savefig("CIFAR1hog.png")
plt.clf()

lbp = local_binary_pattern(btr, 8*3, 3, 'uniform')
plt.imshow(lbp)
plt.savefig("CIFAR1lbp.png")
plt.clf()

plt.imshow(btr)
plt.savefig("CIFAR1.png")
plt.clf()

btr=a[100]
fd, hog_image = hog(btr, orientations=8, pixels_per_cell=pxlpc,cells_per_block=clpb, visualize=True)
plt.imshow(hog_image)
plt.savefig("CIFAR2hog.png")
plt.clf()

lbp = local_binary_pattern(btr, 8*3, 3, 'uniform')
plt.imshow(lbp)
plt.savefig("CIFAR2lbp.png")
plt.clf()

plt.imshow(btr)
plt.savefig("CIFAR2.png")
plt.clf()

btr=a[1000]
fd, hog_image = hog(btr, orientations=8, pixels_per_cell=pxlpc,cells_per_block=clpb, visualize=True)
plt.imshow(hog_image)
plt.savefig("CIFAR3hog.png")
plt.clf()

lbp = local_binary_pattern(btr, 8*3, 3, 'uniform')
plt.imshow(lbp)
plt.savefig("CIFAR3lbp.png")
plt.clf()

plt.imshow(btr)
plt.savefig("CIFAR3.png")
plt.clf()


