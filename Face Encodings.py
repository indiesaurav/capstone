import os
import cv2
import sys
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

path = os.getcwd()+'\\archive'

# %%
images = []

# %%
m = 92                                                                      # number of columns of the image
n = 112                                                                     # number of rows of the image
mn = m * n 
## her l is totl no. of images
l = 400
L = np.empty(shape=(mn, l), dtype='float64')
total_subjects = 40

# %%
curr_img = 0
for j in range(1, total_subjects+1):
    path = os.path.join(path, 's'+str(j)+os.sep)
    for i in range(1,11):
        path_to_img = os.path.join(path, str(i)+'.pgm')
        #print(path_to_img)
        img = cv2.imread(path_to_img, 0)
        images.append(img)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_arr = np.array(img, dtype='float64').flatten()
        L[:, curr_img] = img_arr[:]
        curr_img += 1
        #plt.imshow(img)
        #plt.title('my picture')
        #plt.show()
        

# %%
# every image is a column here 
L.shape

# %%
def show_images(images, cols = 1):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
#     assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
#     if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, image in enumerate(images):
        a = fig.add_subplot(cols, int(np.ceil(n_images/int(cols))), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
#         a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()

# %%
len(images)

# %%
images[5].shape

# %%
len(images[:])

# %%
show_images(images[:20])

# %%
# use opposite of flatten to show reverse 

# %%
L.shape

# %%
L1 = []
for i in range(20):
    L1.append(L[:,i])

# %%
#plt.imshow(L[:,125].reshape(112, 92))

# %%
L[:,1].shape

# %%
#plt.imshow(L[:,0].reshape(112,92))

# %%
mean_img_col = np.sum(L, axis=1) / l

# %%
mean_img_col.shape

# %%
plt.imshow(mean_img_col.reshape(112,92))

# %%
for j in xrange(0, l):                                             # subtract from all training images
    L[:, j] -= mean_img_col[:]

# %%
L.shape

# %%
L1 = []
for i in range(20):
    L1.append(L[:,i].reshape((112, 92)))

# %%
show_images(L1)

# %%
L1[1].shape

# %%
show_images(L1)

# %%
C = np.matrix(L.transpose()) * np.matrix(L)

# %%
C /= l

# %%
C.shape

# %%
#C

# %%
#np.linalg.eig(C)

# %%
evalues, evectors = np.linalg.eig(C) 

# %%
evectors.shape

# %%
sort_indices = evalues.argsort()[::-1]

# %%
evalues = evalues[sort_indices]
evectors = evectors[sort_indices]

# %%
evalues_sum = sum(evalues[:])

# %%
evalues_count = 0 
evalues_energy = 0.0

# %%
## comparing energy here 
energy = 0.95

# %%
for evalue in evalues:
            evalues_count += 1
            evalues_energy += evalue / evalues_sum

            if evalues_energy >= energy:
                break

# %%
evalues_count

# %%
evalues = evalues[0:evalues_count]
evectors = evectors[0:evalues_count]

# %%
evectors = evectors.transpose()

# %%
evectors.shape

# %%
evectors = L * evectors

# %%
evectors.shape

# %%
#### showing eigenfaces for 6 training examples out of 10
eigen_faces = []
for i in range (20):
    eigen_faces.append(evectors[:,i].reshape(112,92))

# %%
show_images(eigen_faces)

# %%
norms = np.linalg.norm(evectors, axis=0) 

# %%
evectors = evectors / norms

# %%
W = evectors.transpose() * L

# %%
## weights are having size of same as top k eigens that too  for every image
W.shape


