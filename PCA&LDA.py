#!/usr/bin/env python
# coding: utf-8

# # Q4

# ### Importing neccesary libraries

# In[1]:


import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
# Reading an animated GIF file using Python Image Processing Library - Pillow
from PIL import Image
from tensorflow.math import confusion_matrix as cm


# ###  [Principal Component Analysis](#To-PCA-algo)

# In[2]:


def pca(k,x):
    #number of data samples
    N=x.shape[0]
    
    #centering the data
    x_mean=np.mean(x,axis=0)
    X=(x-x_mean)
    
    #calculating eigenvalues and eigenvectors
    eigenVal, eigenVec = np.linalg.eigh(X@X.T/N)
    
    #sort the eigenvalues in descending order
    sorted_index = np.argsort(eigenVal)[::-1]
    sorted_eigenVal = eigenVal[sorted_index]
    
    #similarly sort the eigenvectors 
    sorted_eigenVec = eigenVec[:,sorted_index]
    
    #take top k values
    eigenVal_subset = sorted_eigenVal[:k]
    eigenVec_subset = sorted_eigenVec[:,:k]
    
    u=X.T@eigenVec_subset/np.sqrt(N*eigenVal_subset)
    
    y=X@u
    
    return y,u


# ### [Linear Discriminant Analysis](#To-LDA-algo)

# In[3]:


def lda(x,train_target):
    
    #number of features
    k=x.shape[1]
    
    #take mean of each feature
    x_global_mean=np.mean(x,axis=0)
    
    #total unique class
    unq=np.unique(train_target)
    c=len(np.unique(train_target))
    
    #initializing the withinClass and betweenClass covariance matrices
    Sw=np.zeros((k,k))
    Sb=np.zeros((k,k))
    
    #run through each class
    for i in unq:
        #take data belongs to i class
        x_class = x[train_target == i]
        x_class_mean = np.mean(x_class,axis = 0)
        
        #number of samples belongs to i class
        n = len(train_target[train_target == i]) 
                
        temp = (x_class_mean - x_global_mean).reshape(k,1)
        Sb += n * temp @ temp.T
        
        temp= (x_class - x_class_mean)
        Sw += temp.T @ temp
    
    #calculating eigenValues and eigenVectors of (Sw^-1) Sb
    eigenVal, eigenVec = np.linalg.eig(np.linalg.inv(Sw)@Sb)
    eigenVec = eigenVec.T
    
    #Taking eigenVector corresponding to Max eigenValue
    w = eigenVec[np.argmax(eigenVal)].reshape(k,1).real
    ym=x@w
    
    #Fisher Ratio
    f_ratio=(w.T@Sb@w)/(w.T@Sw@w)
    
    return ym,w,f_ratio.reshape(1,)


# In[4]:


#load all gif in train folder
train_images = [Image.open(file) for file in glob.glob("./emotion_classification/train/*.gif")]

#Dimension of image
height,width=np.array(train_images[0]).shape
size=height*width

#number of data samples
N1=len(train_images)

#Target values corresponding to data samples (happy = 1 & sad = 0)
train_target=np.array([1,1,0,0,1,0,0,1,0,1,0,1,0,1,0,0,1,0,1,0])
unq=np.unique(train_target)


# In[5]:


#flattening the data sample images from 101x101 to 1x10201  and storing into x_train
x_train=np.zeros((N1,size))
for i in range(len(train_images)):
    x_train[i,:]=np.array(train_images[i]).reshape(1,size)


# ###### [To PCA algo](#Principal-Component-Analysis) 
# ###### [To LDA algo](#Linear-Discriminant-Analysis)

# In[6]:


#Plotting the one dimension features for each image from K=1 to 19
plt.figure(figsize = (25, 25))
plt.subplot(5,4,19)
ax=[0]*N1
fisher_ratio=[]
for i in range(1,N1):
    ax[i-1]=plt.subplot(5,4,i)
    y,u=pca(i,x_train)
    ym,w,f_ratio=lda(y,train_target)
    fisher_ratio.append(f_ratio)
    for j in unq:
        x_lda=list(ym[train_target==j])
        n=len(train_target[train_target==j])
        ax[i-1].scatter(x_lda,[0]*n,label='happy' if j==1 else 'sad')
        
    ax[i-1].legend()
    ax[i-1].axvline(x = 0, color = 'r', label = 'axvline - full height')
    ax[i-1].set_title(f'K={i}', size=16)
    ax[i-1].grid()
plt.show()


# ### Fisher Ratio for different values of K

# In[7]:


data={'values of K':range(1,20),'Fisher ratio':fisher_ratio}
df=pd.DataFrame(data)
df


# In[8]:


plt.plot(range(1,20),fisher_ratio)


# ### Taking K=12 as Optimum separability

# In[9]:


y,u=pca(12,x_train)
ym,w,f_ratio=lda(y,train_target)

plt.figure(figsize = (8, 5))
for j in unq:
    x_lda=list(ym[train_target==j])
    n=len(train_target[train_target==j])
    plt.scatter(x_lda,[0]*n,label='happy' if j==1 else 'sad')
plt.grid(which='both')
plt.legend()
plt.axvline(x = 0, color = 'r', label = 'axvline - full height')
plt.show()
thresold=np.mean(ym)
thresold


# ### Reading Test files

# In[10]:


test_images = [Image.open(file) for file in glob.glob("./emotion_classification/test/*.gif")]
N=len(test_images)
test_target=np.array([0,1,1,1,0,1,1,0,1,0])
x_test=np.zeros((N,size))
for i in range(len(test_images)):
    x_test[i,:]=np.array(test_images[i]).reshape(1,size)
    
X_test=x_test-np.mean(x_train,axis=0)


# ### Checking Accuracy on test data

# In[11]:


y_test=X_test@u
predict=[]
ym_test=y_test@w

sad_mean=np.mean(sum(ym_test[test_target==0])).real

for i in range(N):
    temp=0 if (sad_mean>0 and ym_test[i]>0)                 else 1 if (sad_mean>0 and ym_test[i]<0)             else 0 if (sad_mean<0 and ym_test[i]<0) else 1
    predict.append(temp)
false_predict=sum( [1 for i in range(N) if test_target[i]!=predict[i]])
print('False prediction =',false_predict)
accuracy=(N1-false_predict)*100/N1
print(f'Test accuracy = {accuracy}%')


# ### Creating Confusion Matrix

# In[13]:


confusion_matrix=cm(labels=test_target,predictions=predict)
plt.figure(figsize=(18,6))
plt.subplot(1,2,2)
plt.subplot(1,2,1)
sn.set(font_scale=1.5)
sn.heatmap(confusion_matrix,annot=True,fmt='d')
plt.xticks([0.5,1.5],['Sad','Happy'])
plt.yticks([0.5,1.5],['Sad','Happy'])
plt.xlabel('Predictions', fontsize=20)
plt.ylabel('Truth',fontsize=20)

plt.subplot(1,2,2)
for j in unq:
    x_lda=list(ym_test[test_target==j])
    n=len(test_target[test_target==j])
    plt.scatter(x_lda,[0]*n,s=200,marker="*",label='test happy' if j==1 else 'test sad')

for j in unq:
    x_lda=list(ym[train_target==j])
    n=len(train_target[train_target==j])
    plt.scatter(x_lda,[0]*n,s=100,marker=".",label='train happy' if j==1 else 'train sad')
plt.axvline(x = 0, color = 'r', label = 'Thresold')
plt.title('Sad', size=16, loc='Left' if sad_mean<0 else'Right')
plt.title('Happy', size=16, loc='Right' if sad_mean<0 else'Left')
plt.legend()
plt.grid()
plt.show()

