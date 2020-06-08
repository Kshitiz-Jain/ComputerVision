#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import os
import cv2
import copy
import torch
import torchvision.transforms as transforms
from torchvision import models

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
        d=unpickle('./data/cifar-10-batches-py/data_batch_'+str(i+1))
        data=d[b'data']
        lbl=d[b'labels']
        for i in range(len(data)):
            img=np.reshape(data[i],(32,32,3))
            traind.append(img)
            trainl.append(lbl[i])
#             img=cv2.cvtColor(np.reshape(data[i],(32,32,3)),cv2.COLOR_BGR2GRAY)
#             img=np.ravel(img)
#             traind.append(img)
#             trainl.append(lbl[i])
    d2=unpickle('./data/cifar-10-batches-py/test_batch')
    data2=d2[b'data']
    lbl2=d2[b'labels']
    for i in range(len(data2)):
        img=np.reshape(data2[i],(32,32,3))
        testd.append(img)
        testl.append(lbl2[i])
        
#     for i in range(len(data2)):
        
#         img=cv2.cvtColor(np.reshape(data2[i],(32,32,3)),cv2.COLOR_BGR2GRAY)
#         img=np.ravel(img)
#         testd.append(img)
#         testl.append(lbl2[i])
    a=np.asarray(traind)
    b=np.asarray(trainl)
    c=np.asarray(testd)
    d=np.asarray(testl)
    return a,b,c,d

trnx,trny,tstx,tsty=data2()
print(np.shape(trnx))
print(np.shape(trny))
print(np.shape(tstx))
print(np.shape(tsty))
#[size,mean,standdev]
params=[(224,224),[0.485, 0.465, 0.406],[0.229, 0.224, 0.225]]
t=[transforms.ToPILImage(), transforms.Resize(size=params[0]), transforms.ToTensor(),transforms.Normalize(mean =params[1], std=params[2])]
AlexNet=models.alexnet(pretrained=True)
trans =transforms.Compose(t)


# In[6]:


train_x=[]
tot=0
for i in range(200):
    batch=[]
    print(i)
    for j in range(250):
        image=trans(trnx[j+tot])
        batch.append(image)
    tot=tot+250
    tensors=torch.stack(batch)
    final=AlexNet(tensors)
    extract= final.detach().numpy()
    train_x.extend(extract)


# In[20]:


test_x=[]
tot=0
for i in range(40):
    batch=[]
    for j in range(250):
        image=trans(tstx[j+tot])
        batch.append(image)
    tot=tot+250
    tensors=torch.stack(batch)
    final=AlexNet(tensors)
    extract= final.detach().numpy()
    test_x.extend(extract)


# In[21]:


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(train_x, trny)
# y_pred=clf.predict(test_x)


# In[26]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
y_pred=clf.predict(test_x)
print(np.shape(tsty))
print(np.shape(test_x))
cm = confusion_matrix(tsty, y_pred)
print(np.sum(np.diagonal(cm))/np.sum(cm))
plt.matshow(cm)
plt.colorbar()
plt.title("Confusion Matrix")
plt.savefig("Confusion")
# plt.clf()


# In[29]:


from sklearn.neural_network import MLPClassifier

clf1=MLPClassifier(hidden_layer_sizes=(500, 500, ), max_iter=500)
clf1.fit(train_x,trny)


# In[30]:


y_pred=clf1.predict(test_x)
cm = confusion_matrix(tsty, y_pred)
print(np.sum(np.diagonal(cm))/np.sum(cm))
plt.clf()
plt.matshow(cm)
plt.colorbar()
plt.title("Confusion Matrix")
plt.savefig("ConfusionNN")


# In[31]:


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
clf.fit(train_x, trny)


# In[32]:


y_pred=clf.predict(test_x)
cm = confusion_matrix(tsty, y_pred)
print(np.sum(np.diagonal(cm))/np.sum(cm))
plt.clf()
plt.matshow(cm)
plt.colorbar()
plt.title("Confusion Matrix")
plt.savefig("ConfusionLR")


# In[ ]:




