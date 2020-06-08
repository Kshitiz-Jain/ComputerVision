#!/usr/bin/env python
# coding: utf-8

# In[52]:

#https://blog.algorithmia.com/convolutional-neural-nets-in-pytorch/
#https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
#https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py

import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np

class ConvNetwork(nn.Module):
    
    def __init__(self):
        super(ConvNetwork,self).__init__()
        #[in_channels,out_channels,kernel,stride,padding]
        layer1=[3,16,3,1,0]
        layer2=[16,32,3,1,0]
        layer3=[32,64,3,1,0]
        self.conv1=torch.nn.Conv2d(layer1[0],layer1[1],kernel_size=layer1[2],stride=layer1[3],padding=layer1[4])
        self.conv2=torch.nn.Conv2d(layer2[0],layer2[1],kernel_size=layer2[2],stride=layer2[3],padding=layer2[4])
        self.conv3=torch.nn.Conv2d(layer3[0],layer3[1],kernel_size=layer3[2],stride=layer3[3],padding=layer3[4])
        self.fc1=nn.Linear(64*26*26,64)
        self.fc2=torch.nn.Linear(64,10)
        
        
    def forward(self,inp):
        out1=F.relu(self.conv1(inp))
        out2=F.relu(self.conv2(out1))
        out3=F.relu(self.conv3(out2))
        out=out3.view(-1,out3.shape[1]*out3.shape[2]*out3.shape[3])
        outfc1=self.fc1(out)
        outfc2=self.fc2(outfc1)
        outf=F.softmax(outfc2,_stacklevel=4)
        return outf

model = ConvNetwork()


# In[63]:


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
    keys={1:0,3:1,5:2,9:3}
    for i in range(5):
        d=unpickle('./data/cifar-10-batches-py/data_batch_'+str(i+1))
        data=d[b'data']
        lbl=d[b'labels']
        for i in range(len(data)):
            if(lbl[i] in keys):
                img=np.reshape(data[i],(32,32,3))
                nimg=[]
                i1=img[:,:,0]
                i2=img[:,:,1]
                i3=img[:,:,2]
                nimg.append(i1)
                nimg.append(i2)
                nimg.append(i3)
                traind.append(nimg)
                trainl.append(keys[lbl[i]])
    d2=unpickle('./data/cifar-10-batches-py/test_batch')
    data2=d2[b'data']
    lbl2=d2[b'labels']
    for i in range(len(data2)):
        if(lbl2[i] in keys):
            img=np.reshape(data2[i],(32,32,3))
            nimg=[]
            i1=img[:,:,0]
            i2=img[:,:,1]
            i3=img[:,:,2]
            nimg.append(i1)
            nimg.append(i2)
            nimg.append(i3)
            testd.append(nimg)
            testl.append(keys[lbl2[i]])
        
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

train_x,train_y,test_x,test_y=data2()
print(np.shape(train_x),np.shape(train_y),np.shape(test_x),np.shape(test_y))


# In[54]:


import random
c = list(zip(train_x, train_y))
random.shuffle(c)
train_x, train_y = zip(*c)
train_x=np.asarray(train_x)
train_y=np.asarray(train_y)
print(np.shape(train_x),np.shape(train_y))


# In[ ]:





# In[55]:


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


# In[56]:


epochs=10
batches=1000
bsize=20
for i in range(epochs):
    curr=0
    for j in range(batches):
        trnx=[]
        trny=[]
        for k in range(bsize):
            trnx.append(train_x[k+curr])
            trny.append(train_y[k+curr])
        curr=curr+bsize
        trnx=np.array(trnx).astype(np.float32)
        trny=np.array(trny).astype(np.long)
        trnx=torch.from_numpy(trnx)
        trny=torch.from_numpy(trny)

        outputs = model(trnx)
        optimizer.zero_grad()
        loss = criterion(outputs, trny)
        loss.backward()
        optimizer.step()
        if(j%100==0):
            print(j,i,loss.item())


# In[68]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
model.eval()
with torch.no_grad():
    accurate=0
    total=0
    y_pred=[]
    curr=0
    for i in range(200):
        tstx=[]
        tsty=[]
        for j in range(bsize):
            tstx.append(test_x[j+curr])
            tsty.append(test_y[j+curr])
        curr=curr+bsize
        tstx=np.array(tstx).astype(np.float32)
        tsty=np.array(tsty).astype(np.long)
        
        print(np.shape(tstx))

        tstx=torch.from_numpy(tstx)
        tsty=torch.from_numpy(tsty)
        
        output=model(tstx)
        _, predicted=torch.max(output.data,1)
        y_pred.extend(predicted.numpy().tolist())
        total=total+tsty.size(0)
        accurate=accurate+(predicted==tsty).sum().item()
    print(np.shape(y_pred))
#     y_pred=clf.predict(test_x)
    cm = confusion_matrix(test_y, y_pred)
    print(np.sum(np.diagonal(cm))/np.sum(cm))
    plt.clf()
    plt.matshow(cm)
    plt.colorbar()
    plt.title("Confusion Matrix")
    plt.savefig("ConfusionLR")
    print("Accuracy :",accurate/total)


train_x=torch.from_numpy(train_x)
train_y=torch.from_numpy(train_y)
test_x=torch.from_numpy(test_x)
test_y=torch.from_numpy(test_y)
print(train_x.size(),train_y.size(),test_x.size(),test_y.size())


# In[ ]:





# In[3]:





# In[18]:


import tensorflow as tf

#[batchsize,learning_rate,epochs,classes]



# Train the model
total_step = len(trainset)
for epoch in range(param[2]):
    for i, data in enumerate(trainset):
        images, labels = data
        imgs1=images[labels==1]
        imgs3=images[labels==3]
        imgs5=images[labels==5]
        imgs9=images[labels==9]
        imgs1=torch.cat([imgs1, imgs3], 0)
        imgs1=torch.cat([imgs1, imgs5], 0)
        imgs1=torch.cat([imgs1, imgs9], 0)
#         tf.concat([imgs1, imgs5], 0)
#         tf.concat([imgs1, imgs9], 0)
        
#         lbl1=labels[labels==1]
#         lbl3=labels[labels==3]
#         lbl5=labels[labels==5]
#         lbl9=labels[labels==9]
#         lbl1=torch.cat([lbl1, lbl3], 0)
#         lbl1=torch.cat([lbl1, lbl5], 0)
#         lbl1=torch.cat([lbl1, lbl9], 0)

        lbl1=np.zeros(len(labels[labels==1]))
        lbl3=np.zeros(len(labels[labels==3]))+1
        lbl5=np.zeros(len(labels[labels==5]))+2
        lbl9=np.zeros(len(labels[labels==9]))+3
        lbl1=np.concatenate((lbl1, lbl3), axis=0)
        lbl1=np.concatenate((lbl1, lbl5), axis=0)
        lbl1=np.concatenate((lbl1, lbl9), axis=0)
        
        lbl1 = torch.tensor(lbl1.astype(np.long))
        
#         tf.concat([lbl1, lbl3], 0)
#         tf.concat([lbl1, lbl5], 0)
#         tf.concat([lbl1, lbl9], 0)
    
        outputs = model(imgs1)
#         print(np.shape(outputs),np.shape(lbl1))

        
        optimizer.zero_grad()
        loss = criterion(outputs, lbl1)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, param[2], i+1, total_step, loss.item()))


# In[ ]:





# In[19]:


model.eval()
with torch.no_grad():
    accurate=0
    total=0
    for images, labels in testset:
        imgs1=images[labels==1]
        imgs3=images[labels==3]
        imgs5=images[labels==5]
        imgs9=images[labels==9]
        imgs1=torch.cat([imgs1, imgs3], 0)
        imgs1=torch.cat([imgs1, imgs5], 0)
        imgs1=torch.cat([imgs1, imgs9], 0)
#         tf.concat([imgs1, imgs3], 0)
#         tf.concat([imgs1, imgs5], 0)
#         tf.concat([imgs1, imgs9], 0)
        
#         lbl1=labels[labels==1]
#         lbl3=labels[labels==3]
#         lbl5=labels[labels==5]
#         lbl9=labels[labels==9]
#         lbl1=torch.cat([lbl1, lbl3], 0)
#         lbl1=torch.cat([lbl1, lbl5], 0)
#         lbl1=torch.cat([lbl1, lbl9], 0)
        
        
        lbl1=np.zeros(len(labels[labels==1]))
        lbl3=np.zeros(len(labels[labels==3]))+1
        lbl5=np.zeros(len(labels[labels==5]))+2
        lbl9=np.zeros(len(labels[labels==9]))+3
        lbl1=np.concatenate((lbl1, lbl3), axis=0)
        lbl1=np.concatenate((lbl1, lbl5), axis=0)
        lbl1=np.concatenate((lbl1, lbl9), axis=0)
        
        lbl1 = torch.tensor(lbl1.astype(np.long))
        
        
#         tf.concat([lbl1, lbl3], 0)
#         tf.concat([lbl1, lbl5], 0)
#         tf.concat([lbl1, lbl9], 0)
        output=model(imgs1)
        _, predicted=torch.max(output.data,1)
        p=predicted.numpy()
        total=total+lbl1.size(0)
        accurate=accurate+(predicted==lbl1).sum().item()
#         print(accurate)
    print("Accuracy :",accurate/total)

torch.save(model.state_dict(),"Model.ckpt")


# In[ ]:




