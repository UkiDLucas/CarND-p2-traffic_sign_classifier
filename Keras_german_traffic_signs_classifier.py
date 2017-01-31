
# coding: utf-8

# In[1]:

from urllib.request import urlretrieve
from os.path import isfile
from tqdm import tqdm
# conda install -c conda-forge tqdm


# In[4]:

# This is a wrapper class to show the progrees of the download.
class DownloadProgress(tqdm):
    last_block = 0

    # This hook function that will be called: 
    # - once on establishment of the network connection and 
    # - once after each block read thereafter. 
    # The hook will receive three arguments: 
    # - a count of blocks transferred so far, 
    # - a block size in bytes, and 
    # - the total size of the file.
    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


# In[8]:

# urllib.request.urlretrieve(url[, filename[, reporthook[, data]]])

# I usually keep my DATA one folder below the git repo
data_dir = "../DATA/german_traffic_signs/"

file_name = 'train.p'
if not isfile(data_dir + file_name):
    with DownloadProgress(unit='B', unit_scale=True, miniters=1, desc='Train Dataset') as pbar:
        urlretrieve(
            'https://s3.amazonaws.com/udacity-sdc/datasets/german_traffic_sign_benchmark/train.p',
            data_dir + file_name,
            pbar.hook)

        
file_name = 'test.p'
if not isfile(data_dir + file_name):
    with DownloadProgress(unit='B', unit_scale=True, miniters=1, desc='Test Dataset') as pbar:
        urlretrieve(
            'https://s3.amazonaws.com/udacity-sdc/datasets/german_traffic_sign_benchmark/test.p',
            data_dir + file_name,
            pbar.hook)
        
        
# Train Dataset: 120MB [01:20, 1.50MB/s]                           
# Test Dataset: 38.8MB [00:28, 1.37MB/s]  

print('Training and Test data downloaded at ' + data_dir)


# In[9]:

import pickle
import numpy as np
import math

# Switch Keras to TensorFlow (from Theano)
import tensorflow as tf
tf.python.control_flow_ops = tf

print('Modules loaded.')


# In[11]:

with open(data_dir + 'train.p', 'rb') as f:
    data = pickle.load(f)

# TODO: Load the feature data to the variable X_train
X_train = data['features']
# TODO: Load the label data to the variable y_train
y_train = data['labels']

# Assert that the data sets loaded correctly
assert np.array_equal(X_train, data['features']), 'X_train not set to data[\'features\'].'
assert np.array_equal(y_train, data['labels']), 'y_train not set to data[\'labels\'].'
print('Training data sets are loaded correctly')


# In[13]:

from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train) 

assert X_train.shape == data['features'].shape, 'X_train has changed shape. The shape shouldn\'t change when shuffling.'
assert y_train.shape == data['labels'].shape, 'y_train has changed shape. The shape shouldn\'t change when shuffling.'
assert not np.array_equal(X_train, data['features']), 'X_train not shuffled.'
assert not np.array_equal(y_train, data['labels']), 'y_train not shuffled.'
print('Tests passed.')


# In[15]:

def normalize_color_intensity(image_data):
    # desired range
    a = -0.5
    b = 0.5
    # expected values
    grayscale_min = 0
    grayscale_max = 255
    return a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )


# In[16]:

X_normalized = normalize_color_intensity(X_train)


# In[ ]:



