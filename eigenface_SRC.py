#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import sys
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cvx
from os import listdir
import re
import glob
import numpy as np
from skimage.measure import block_reduce
#from PIL import Image
import random
import pandas as pd


# In[2]:


# Parameters of the images
DIM_1 = 192
DIM_2 = 168


# In[3]:


# Dense Noise
#Epsilon => we will set epsilon just before calling SOCP
CLASSES = 12
SAMPLE_EACH_CLASS = 68
#DOWNSAMPLE_COEFFICIENT = 2

total_imgs = CLASSES*SAMPLE_EACH_CLASS


# In[4]:


###if problem change float to float64
X = np.empty((DIM_1*DIM_2, total_imgs), dtype='float')
delta = np.zeros(shape=(CLASSES, total_imgs)) ## for SRC only
label = np.zeros(total_imgs)


# In[5]:


currPath = '/home/swapnil/Desktop/BTP sem VII/CroppedYale/'
#os.chdir(currPath)


# In[6]:


X.shape


# In[7]:


delta.shape


# ## Read Images

# In[8]:


def read_images(path):
    os.chdir(path)
    directories = sorted(glob.glob('yale*'))
    count=0
    for i in range(CLASSES):
        os.chdir(path+directories[i])
        images = glob.glob('*pgm')
        for image in images:
            img = cv2.imread(image, 0)
            if(img.shape[0]*img.shape[1] == X.shape[0]):
                ### if want to normalize while reading execute below statement
                #img = (img-np.mean(img))/np.std(img)
                X[:,count] = img.flatten()
                label[count]=i+1 ## label is i+1
                count += 1
                delta[i,count]=1
            
    return X, label


# In[231]:


X, y = read_images(currPath)
X[:,5]


# In[11]:


#plt.imshow(X[:,5].reshape((DIM_1,DIM_2)), cmap='gray')


# ## Train Test Split

# In[12]:


def train_test_data(L, label, delta, frac):
    if 0 in label:
        label = label[:np.where(label==0)[0][0]]
    classes = np.unique(label)
    df = pd.concat([pd.Series(range(len(label)), name='img_no'),pd.Series(label, name='label')],axis=1)
    train_df = df[df.label==1].sample(frac=frac, random_state=42)
    test_df = df[df.label==1].drop(df[df.label==1].sample(frac=frac, random_state=42).index)
    for i in range(len(classes)-1):
        train_df = pd.concat([train_df,df[df.label==i+2].sample(frac=frac, random_state=42)])
        test_df = pd.concat([test_df,df[df.label==i+2].drop(df[df.label==i+2].sample(frac=frac, random_state=42).index)])
    X_train = L[:,train_df.img_no]
    delta = delta[:,train_df.img_no] ###changing delta for only training
    X_test = L[:,test_df.img_no]
    y_train = train_df.label.values
    y_test = test_df.label.values
    return X_train,y_train,X_test,y_test,delta
    


# In[13]:


fraction=0.5 ##3to prove accuracy amke it 0.75 or 0.8
X_train,y_train,X_test,y_test,delta = train_test_data(X, y, delta, fraction)


# In[14]:


X_train.shape


# In[15]:


delta.shape


# In[16]:


mean_img = np.mean(X_train, axis=1)
mean_img


# In[17]:


#sd = np.tile(mean_img, (tmp.shape[1],1))


# In[18]:


##### mean image
plt.imshow(np.mean(X_train, axis=1).reshape((DIM_1,DIM_2)), cmap='gray')
plt.axis('off')
plt.savefig('mean_img', bbox_inches='tight')


# In[20]:


X_train.shape


# In[ ]:





# ## Eigenspace

# In[19]:


def eigenfaces(L,pca_components = 90):
    
    ###mean 
    mean_img = np.mean(L, axis=1) #### calculating mean image
    mean_matrix = np.tile(mean_img, (L.shape[1],1)) #### copying mean image to matrix of same size of X
    L = L - mean_matrix.T  ### subtracting mean from X
    l = L.shape[1] ### l = no. of images
    C = np.matrix(L.transpose()) * np.matrix(L) ### here c= X.T * X
    C /= l-1
    evalues, evectors = np.linalg.eig(C)  ### eigenvectors are column vectors
    ### [::-1] just reverses whatever array you have
    idx = evalues.argsort()[::-1]   ### reversing index max->min
    evalues = evalues[idx]
    evalues = evalues[:pca_components]
    evectors = evectors[:,idx]
    evectors = evectors[:,:pca_components]
    ##you still need to normalize new eigenvectors CHECK1
    evectors = L * evectors ###opencv trick for computation
    evectors = evectors.real ### evectors are complex with 0j
    ## just by normalizing accuracy goes from 36 to 86
    norms = np.linalg.norm(evectors, axis=0)
    evectors = evectors / norms
    #plt.imshow(tmp[:,pca_components/2].reshape((DIM_1, DIM_2)), cmap='gray') ##3 to plot eigenface
    ## now project given images into this subspace
    #features = evectors.T * X
    ## for single image do evectors.T * img(as column)
    return evectors


# In[20]:


get_ipython().run_cell_magic('time', '', '### here we have a column of size max_components which is feature to that image\nk=200 ## specify how many eigenvctors \nevectors = eigenfaces(X_train,k)')


# In[21]:


evectors.shape


# In[22]:


X_train.shape


# In[39]:


os.chdir('/home/swapnil/Desktop/BTP sem VII')


# In[35]:


plt.plot([1,2,3], [1,2,3])
plt.show()
plt.savefig('tmp.png',bbox_inches='tight')
fig1 = plt.gcf()
fig1.savefig('tessstttyyy.png', dpi=100)


# In[168]:


###### reconstruction for image in occluded
test_img = err
test_img = test_img.reshape(-1,1) 
print(evectors.shape, test_img.shape)
reconstructed_img = (evectors*test_img).ravel()
#plt.imshow(X_test[:,img_no].reshape((DIM_1,DIM_2)), cmap='gray')
#plt.axis('off')
plt.imshow(reconstructed_img.reshape((DIM_1,DIM_2)), cmap='gray')


# In[114]:


#%matplotlib inline
###reconstruction for image in eigenspace
img_no = 200
test_img = X_test[:,img_no] 
test_img = evectors.T * test_img.reshape(-1,1) 
reconstructed_img = mean_img + (evectors*test_img).ravel()
plt.imshow(X_test[:,img_no].reshape((DIM_1,DIM_2)), cmap='gray')
plt.axis('off')
plt.imshow(reconstructed_img.reshape((DIM_1,DIM_2)), cmap='gray')
#plt.savefig('eigen.png',bbox_inches='tight', pad_inches=0)


# ## RandomFaces

# In[68]:


new_dim = 500
np.random.seed(42)
R = np.random.standard_normal(size=(new_dim,DIM_1*DIM_2))


# In[69]:


R.shape


# In[70]:


R_norm = np.linalg.norm(R, axis=1).shape


# In[71]:


R = R/R_norm 


# In[72]:


plt.imshow(R[200].reshape((DIM_1, DIM_2)), cmap='gray')
#plt.imshow(X_test[:,img_no].reshape((DIM_1,DIM_2)), cmap='gray')


# In[73]:


features = np.matmul(R,X_train)
test_features = np.matmul(R,X_test)


# In[81]:


random_img = np.matmul(test_features[:,img_no],R)
plt.axis('off')
plt.imshow(random_img.reshape((DIM_1,DIM_2)), cmap='gray')
plt.savefig('random.png', bbox_inches='tight')


# In[229]:


svc.fit(X_train.T, y_train)


# In[63]:


predictions = svc.predict(X_test.T)


# In[64]:


(y_test==predictions).sum()*100/y_test.shape[0]


# ## Downsampling

# In[62]:


DOWNSAMPLE_COEFFICIENT = 12
features = []
for img in X_train.T:
    img = img.reshape((DIM_1, DIM_2))
    img = block_reduce(np.array(img), block_size=(DOWNSAMPLE_COEFFICIENT, DOWNSAMPLE_COEFFICIENT), func=np.mean)
    features.append(img.flatten())
features = np.array(features).T


# In[63]:


test_features = []
for img in X_test.T:
    img = img.reshape((DIM_1, DIM_2))
    img = block_reduce(np.array(img), block_size=(DOWNSAMPLE_COEFFICIENT, DOWNSAMPLE_COEFFICIENT), func=np.mean)
    test_features.append(img.flatten())
test_features = np.array(test_features).T


# In[64]:


print(features.shape, test_features.shape) ### every image is still column


# In[66]:


plt.imshow(test_features[:,200].reshape(img.shape), cmap='gray')
plt.savefig('downsample.png', bbox_inches='tight')


# In[91]:


svc.fit(X_train.T, y_train)


# In[92]:


predictions = svc.predict(X_test.T)


# In[93]:


(y_test==predictions).sum()*100/y_test.shape[0]


# ## Nearest Neighbour

# In[187]:


## pass features(K components of test_img)
def nearest_neighbour(L, test_img):
    ###columns of X are images
    copy_matrix = np.tile(test_img, (L.shape[1])) ###copying image into matrix
    residual = L - copy_matrix ### calculate residual 
    ### take absolute difference 
    min_index = np.argmin(np.abs(residual).sum(axis=0)) ### image number with minimum residual
    return min_index ## returning image number
    
    
    


# In[188]:


#features.shape


# In[189]:


acc=0.0 ### calculating accuracy
features = evectors.T * X_train ### calculate training X in eigenspace
for i in range (X_test.shape[1]):
    test_img = X_test[:,i] 
    test_img = evectors.T * test_img.reshape(-1,1) ###project test image into eigenspace
    #print(i,nearest_neighbour(features, test_img))
    if (y_test[i]==y_train[nearest_neighbour(features, test_img)]) :
        acc +=1
    


# In[26]:


X_test.shape[1]


# In[190]:


acc*100/X_test.shape[1]


# In[28]:


### use eigen with src from below cells 
###plot k vs accuracy


# In[29]:


k_values = [20,50,70,80,100,120,130,150,180,200,250]
accuracies = []
for k in k_values:
    evectors = eigenfaces(X_train,k)
    acc=0.0 ### calculating accuracy
    features = evectors.T * X_train ### calculate training X in eigenspace
    for i in range (X_test.shape[1]):
        test_img = X_test[:,i] 
        test_img = evectors.T * test_img.reshape(-1,1) ###project test image into eigenspace
        if (y_test[i]==y_train[nearest_neighbour(features, test_img)]) :
            acc +=1
    accuracies.append(acc*100/X_test.shape[1])
    print(acc*100/X_test.shape[1])


# In[30]:


plt.plot(k_values, accuracies)
plt.xlabel('no. of pca aparameters')
plt.ylabel('accuracy')


# ## Nearest Subspace

# In[134]:


## pass features(K components of test_img)
subspace_size = 3
##IMP OBS keep subspace size smaller , more accuracy
def nearest_subspace(L, test_img):
    #subspace_size = 12
    ###columns of X are images
    copy_matrix = np.tile(test_img, (L.shape[1])) ###copying image into matrix
    residual = L - copy_matrix ### calculate residual 
    ### take absolute difference 
    diff = np.abs(residual).sum(axis=0)
    #print(type(diff))
    ##matrix to numpy array problem
    diff = np.array(diff[0].ravel()).ravel()
    #print(type(diff))
    
    return diff ## returning image number
    
    
    


# In[135]:


##IMP you can try weighted nearest subspace


# In[136]:


k_values = [20,50,70,80,100,120,130,150,180,200,250]
accuracies = []
for k in k_values:
    evectors = eigenfaces(X_train,k)
    acc=0.0 ### calculating accuracy
    features = evectors.T * X_train ### calculate training X in eigenspace
    for i in range (X_test.shape[1]):
        test_img = X_test[:,i] 
        test_img = evectors.T * test_img.reshape(-1,1) ###project test image into eigenspace
        diff = nearest_subspace(features, test_img)
        
        if (y_test[i]==np.argmax(np.bincount(y_train[np.argpartition(diff,subspace_size)[:subspace_size]].astype(int)))) :
            acc +=1
    accuracies.append(acc*100/X_test.shape[1])
    print(acc*100/X_test.shape[1])


# In[137]:


plt.plot(k_values, accuracies)
plt.xlabel('no. of pca aparameters')
plt.ylabel('accuracy')


# ## SRC

# In[165]:


def src(A,y, Epsilon, robust=False):
    
    if (robust == False):
        size = A.shape[1]
        x = cvx.Variable(size)
        obj = cvx.norm(x,1)
        obj = cvx.Minimize(obj)

        constraints = [cvx.norm(A*x - y,2) <= Epsilon]


    else:
        x_size = A.shape[1]
        err_size = A.shape[0]

        # Define the variables, constraints and object of the optimization problem
        x = cvx.Variable(x_size)
        err = cvx.Variable(err_size)
        obj = cvx.Minimize(cvx.norm(x,1) + cvx.norm(err,1))
        constraints = [ A*x - err == y]
    
    prob = cvx.Problem(obj, constraints)
    prob.solve()
    #print("Status:",prob.status)    
    X_hat = x.value
    X_hat.resize((X_hat.shape[0],1))
    residual = X_hat*delta.T
    testCopy = np.tile(y,(CLASSES,1))
    testCopy = np.transpose(testCopy)
    #print(testCopy.shape, np.dot(A, residual).shape)
    M = (testCopy - np.dot(A, residual)) ##IMP **2 removed
    mistake=np.array(M.ravel()).ravel()**2
    mistake=mistake.reshape(M.shape)
    #print(type(mistake))
    #print(mistake.shape)
    return  np.argmin(np.sum(mistake, axis=0))+1 ###returning class number


# In[69]:


X_train[:,(y_train==7)][:,0].shape


# In[35]:


k_values = [20,50,70,80,100,120,130,150,180,200,250]
k_values = [100]
## OBS as k increases src accuracy increses
accuracies = []
for k in k_values:
    evectors = eigenfaces(X_train,k)
    acc=0.0 ### calculating accuracy
    features = evectors.T * X_train ### calculate training X in eigenspace
    for i in range (X_test.shape[1]):
        test_img = X_test[:,i] 
        test_img = evectors.T * test_img.reshape(-1,1) ###project test image into eigenspace
        test_img = np.array([number.item(0) for number in test_img])
        Epsilon=1000
        if (y_test[i]==src(features,test_img, Epsilon)) :
            acc +=1
    accuracies.append(acc*100/X_test.shape[1])
    print(acc*100/X_test.shape[1])


# ## SVM

# In[58]:


from sklearn import svm


# In[59]:


C = 1.0 # SVM regularization parameter
##pass X_train.T beacuse svm takes rows as sample
svc = svm.SVC(kernel='linear', C=C, decision_function_shape='ovr')


# In[43]:


svc.fit(X_train.T, y_train)


# In[46]:


predictions = svc.predict(X_test.T)


# In[51]:


(y_test==predictions).sum()*100/y_test.shape[0]


# In[55]:


y_train.shape


# In[64]:


features = evectors.T * X_train
test_features = evectors.T * X_test
svc = svm.SVC(kernel='linear', C=1, decision_function_shape='ovr')
svc.fit(features.T, y_train)
predictions = svc.predict(test_features.T)
print((y_test==predictions).sum()*100/y_test.shape[0])


# In[138]:


k_values = [20,50,70,80,100,120,130,150,180,200,250]
#k_values = [100]
## OBS as k increases src accuracy increses
accuracies = []
for k in k_values:
    evectors = eigenfaces(X_train,k)
    acc=0.0 ### calculating accuracy
    features = evectors.T * X_train ### calculate training X in eigenspace
    features = evectors.T * X_train
    test_features = evectors.T * X_test
    svc = svm.SVC(kernel='linear', C=1, decision_function_shape='ovr')
    svc.fit(features.T, y_train)
    predictions = svc.predict(test_features.T)
    print((y_test==predictions).sum()*100/y_test.shape[0])
    accuracies.append((y_test==predictions).sum()*100/y_test.shape[0])


# In[139]:


plt.plot(k_values, accuracies)
plt.xlabel('no. of pca aparameters')
plt.ylabel('accuracy')


# ## Plotting multiple plots algoithmwise

# ### nearest neighbour

# In[167]:


### feature extraction  eigen
k_values = [100,200,300,400,500]
accuracies1 = []
for k in k_values:
    evectors = eigenfaces(X_train,k)
    acc=0.0 ### calculating accuracy
    features = evectors.T * X_train ### calculate training X in eigenspace
    for i in range (X_test.shape[1]):
        test_img = X_test[:,i] 
        test_img = evectors.T * test_img.reshape(-1,1) ###project test image into eigenspace
        if (y_test[i]==y_train[nearest_neighbour(features, test_img)]) :
            acc +=1
    accuracies1.append(acc*100/X_test.shape[1])
    print(acc*100/X_test.shape[1])


# In[207]:


###downsampling
coeffs = [8,10,12,16,20]
accuracies2 = []
dimension_size = []
for down_coef in coeffs:
    features = []
    for img in X_train.T:
        img = img.reshape((DIM_1, DIM_2))
        img = block_reduce(np.array(img), block_size=(down_coef, down_coef), func=np.mean)
        features.append(img.flatten())
    dimension_size.append(img.shape[0]*img.shape[1])
    features = np.array(features).T
    test_features = []
    for img in X_test.T:
        img = img.reshape((DIM_1, DIM_2))
        img = block_reduce(np.array(img), block_size=(down_coef, down_coef), func=np.mean)
        test_features.append(img.flatten())
    test_features = np.array(test_features).T
    acc=0
    for i in range (test_features.shape[1]):
        test_img = test_features[:,i] 
        test_img = test_img.reshape(-1,1)
        #print(features.shape, test_img.shape)
        if (y_test[i]==y_train[nearest_neighbour(features, test_img)]) :
            acc +=1
    accuracies2.append(acc*100/X_test.shape[1])
    print(acc*100/X_test.shape[1])    

#### reversing the dimension sizes
dimension_size = dimension_size[::-1]
accuracies2 = accuracies2[::-1]


# In[199]:


### random 
k_values = [100,200,300,400,500]
accuracies3 = []
for new_dim in k_values:
    np.random.seed(42)
    R = np.random.standard_normal(size=(new_dim,DIM_1*DIM_2))
    R_norm = np.linalg.norm(R, axis=1).shape
    R = R/R_norm 
    features = np.matmul(R,X_train)
    test_features = np.matmul(R,X_test)
    acc=0
    for i in range (test_features.shape[1]):
        test_img = test_features[:,i] 
        test_img = test_img.reshape(-1,1)
        #print(features.shape, test_img.shape)
        if (y_test[i]==y_train[nearest_neighbour(features, test_img)]) :
            acc +=1
    accuracies3.append(acc*100/X_test.shape[1])
    print(acc*100/X_test.shape[1])       


# In[228]:


###plotting
plt.plot(k_values, accuracies1, color='orange', label='Eigen+NN')
plt.plot(dimension_size, accuracies2, color='green', label='Downsampling+NN')
plt.plot(k_values, accuracies3, color='blue', label='Random+NN')
plt.ylim(40,100)
plt.legend()
plt.xlabel('Feature Dimension')
plt.ylabel('Recognition Rate Percentage ')
plt.title('Nearest Neighbour')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.savefig('Nearest Neighbour.png', bbox_inches='tight')
plt.show()


# ## Occlusion

# In[131]:


plt.imshow(X_test[:,110].reshape((DIM_1,DIM_2)), cmap='gray')


# In[145]:


#%matplotlib inline
###reconstruction for image in eigenspace
img_no = 123
data= X_train

test_img = data[:,img_no] 
test_img = evectors.T * test_img.reshape(-1,1) 
reconstructed_img = mean_img + (evectors*test_img).ravel()
plt.axis('off')
plt.imshow(data[:,img_no].reshape((DIM_1,DIM_2)), cmap='gray')
plt.savefig('2st_orig.png',bbox_inches='tight')
plt.imshow(reconstructed_img.reshape((DIM_1,DIM_2)), cmap='gray')
plt.savefig('2st_feature_orig.png',bbox_inches='tight')


# In[157]:


#### USUAL for single image
acc=0.0 ### calculating accuracy
features = evectors.T * X_train ### calculate training X in eigenspace
Epsilon=1000
test_img = X_test[:,110] 
test_img = evectors.T * test_img.reshape((test_img.shape[0],1)) ###project test image into eigenspace
test_img = np.array([number.item(0) for number in test_img])
# ,robust=True sent this for robust
res, xhat , output = src(features,test_img, Epsilon)


# In[158]:


res.shape


# In[161]:


##residual plot
x_axis_values = range(res.shape[1]+1)[1:]
residual_values = res.mean(axis=0)
plt.bar(x_axis_values,residual_values, color='black')
plt.savefig('residual_plot.png',bbox_inches='tight')


# In[151]:


res.mean(axis=0)


# In[140]:


###xhat plot
x_axis_values = range(xhat.shape[0]+1)[1:]
sparse_values = xhat.ravel()
plt.bar(x_axis_values,sparse_values)
plt.savefig('barplot_xvalues.png',bbox_inches='tight')


# In[78]:


sdf = [10.00,15312,20,35]
np.argsort(sdf)


# In[135]:


np.argmax(xhat[:127])


# In[143]:


plt.imshow(X_train[:,xhat.argmax()].reshape((DIM_1, DIM_2)), cmap='gray')
print(xhat.argmax())


# In[136]:


plt.imshow(X_train[:,123].reshape((DIM_1, DIM_2)), cmap='gray')
print(123)


# In[139]:


os.chdir('/home/swapnil/Desktop/BTP sem VII')


# In[175]:


#### OCCLUDED
os.chdir('/home/swapnil/Desktop/BTP sem VII')
test_img = cv2.imread('glass.jpg',0)
print(test_img.shape)
test_img = cv2.resize(test_img, (DIM_2, DIM_1))
plt.imshow(test_img,cmap='gray')

print(test_img.shape)
test_img = test_img.flatten()
print(test_img.shape)

#os.curdir


# In[176]:


#### OCCLUDED
acc=0.0 ### calculating accuracy
features =  X_train ### calculate training X in eigenspace
Epsilon=1000
#test_img = X_test[:,100] 
test_img =  test_img.reshape((test_img.shape[0],1)) ###project test image into eigenspace
test_img = np.array([number.item(0) for number in test_img])
# ,robust=True sent this for robust
xhat , err, output = src(features,test_img, Epsilon,robust=True)


# In[184]:


axx = np.matmul(X_train,xhat)
axx.shape
axx_img = axx.reshape(DIM_1,DIM_2)
axx_img.shape
plt.imshow(axx_img, cmap='gray')
plt.savefig('ax.png', bbox_inches='tight')


# In[185]:


plt.imshow(err.reshape((DIM_1,DIM_2)), cmap='gray')
plt.savefig('err.png', bbox_inches='tight')


# In[186]:


err_image = err.reshape((DIM_1,DIM_2))
plt.imshow(axx_img-err_image, cmap='gray')
plt.savefig('ax-err=y.png', bbox_inches='tight')


# In[25]:


#plt.bar(np.array(range(err.shape[0])), err)


# In[ ]:





# In[ ]:




