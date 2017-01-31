
# coding: utf-8

# In[1]:

from urllib.request import urlretrieve
from os.path import isfile
from tqdm import tqdm
# conda install -c conda-forge tqdm


# In[2]:

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


# In[3]:

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


# In[4]:

import pickle
import numpy as np
import math

# Switch Keras to TensorFlow (from Theano)
import tensorflow as tf
tf.python.control_flow_ops = tf

print('Modules loaded.')


# In[5]:

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


# In[6]:

from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train) 

assert X_train.shape == data['features'].shape, 'X_train has changed shape. The shape shouldn\'t change when shuffling.'
assert y_train.shape == data['labels'].shape, 'y_train has changed shape. The shape shouldn\'t change when shuffling.'
assert not np.array_equal(X_train, data['features']), 'X_train not shuffled.'
assert not np.array_equal(y_train, data['labels']), 'y_train not shuffled.'
print('Tests passed.')


# In[7]:

def normalize_color_intensity(image_data):
    # desired range
    a = -0.5
    b = 0.5
    # expected values
    grayscale_min = 0
    grayscale_max = 255
    return a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )


# In[8]:

X_normalized = normalize_color_intensity(X_train)


# In[9]:

assert math.isclose(np.min(X_normalized), -0.5, abs_tol=1e-5) and math.isclose(np.max(X_normalized), 0.5, abs_tol=1e-5), 'The range of the training data is: {} to {}.  It must be -0.5 to 0.5'.format(np.min(X_normalized), np.max(X_normalized))
print('Tests passed.')


# In[10]:

from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
y_one_hot = label_binarizer.fit_transform(y_train)


# In[11]:

import collections

assert y_one_hot.shape == (39209, 43), 'y_one_hot is not the correct shape.  It\'s {}, it should be (39209, 43)'.format(y_one_hot.shape)
assert next((False for y in y_one_hot if collections.Counter(y) != {0: 42, 1: 1}), True), 'y_one_hot not one-hot encoded.'
print('Tests passed.')


# In[12]:

from keras.models import Sequential



# In[13]:

# Create the Sequential model
# The keras.models.Sequential class is a wrapper for the neural network model. 
# Just like many of the class models in scikit-learn, it provides common functions like 
# - fit()
# - evaluate()
# - compile() 

model = Sequential()


# In[15]:

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten


# In[ ]:

# Create the Sequential model
model = Sequential()

# 1st Layer - Add a flatten layer
model.add(Flatten(input_shape=(32, 32, 3)))

# 2nd Layer - Add a fully connected layer
model.add(Dense(100))

# 3rd Layer - Add a ReLU activation layer
model.add(Activation('relu'))

# 4th Layer - Add a fully connected layer
model.add(Dense(60))

# 5th Layer - Add a ReLU activation layer
model.add(Activation('relu'))


# Keras will automatically infer the shape of all layers after the first layer. This means you only have to set the input dimensions for the first layer.
# The first layer from above, model.add(Flatten(input_shape=(32, 32, 3))), sets the input dimension to (32, 32, 3) and output dimension to (3072=32*32*3). The second layer takes in the output of the first layer and sets the output dimenions to (100). This chain of passing output to the next layer continues until the last layer, which is the output of the model.

# # Build a Multi-Layer Feedforward Network
# Build a multi-layer feedforward neural network to classify the traffic sign images.
# Set the first layer to a Flatten layer with the input_shape set to (32, 32, 3)
# Set the second layer to Dense layer width to 128 output.
# Use a ReLU activation function after the second layer.
# Set the output layer width to 43, since there are 43 classes in the dataset.
# Use a softmax activation function after the output layer.
# To get started, review the Keras documentation about models and layers.
# The Keras example of a Multi-Layer Perceptron network is similar to what you need to do here. Use that as a guide, but keep in mind that there are a number of differences.

# In[16]:

from keras.models import Sequential
model = Sequential()
# TODO: Build a Multi-layer feedforward neural network with Keras here.
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten

model.add(Flatten(input_shape=(32, 32, 3)))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(43))
model.add(Activation('softmax'))


# In[17]:

# STOP: Do not change the tests below. Your implementation should pass these tests.
from keras.layers.core import Dense, Activation, Flatten
from keras.activations import relu, softmax

def check_layers(layers, true_layers):
    assert len(true_layers) != 0, 'No layers found'
    for layer_i in range(len(layers)):
        assert isinstance(true_layers[layer_i], layers[layer_i]), 'Layer {} is not a {} layer'.format(layer_i+1, layers[layer_i].__name__)
    assert len(true_layers) == len(layers), '{} layers found, should be {} layers'.format(len(true_layers), len(layers))

check_layers([Flatten, Dense, Activation, Dense, Activation], model.layers)

assert model.layers[0].input_shape == (None, 32, 32, 3), 'First layer input shape is wrong, it should be (32, 32, 3)'
assert model.layers[1].output_shape == (None, 128), 'Second layer output is wrong, it should be (128)'
assert model.layers[2].activation == relu, 'Third layer not a relu activation layer'
assert model.layers[3].output_shape == (None, 43), 'Fourth layer output is wrong, it should be (43)'
assert model.layers[4].activation == softmax, 'Fifth layer not a softmax activation layer'
print('Tests passed.')


# ## Training a Sequential Model
# You built a multi-layer neural network in Keras, now let's look at training a neural network.
# ```python
# from keras.models import Sequential
# from keras.layers.core import Dense, Activation
# 
# model = Sequential()
# ...
# 
# # Configures the learning process and metrics
# model.compile('sgd', 'mean_squared_error', ['accuracy'])
# 
# # Train the model
# # History is a record of training loss and metrics
# history = model.fit(x_train_data, Y_train_data, batch_size=128, nb_epoch=2, validation_split=0.2)
# 
# # Calculate test score
# test_score = model.evaluate(x_test_data, Y_test_data)
# ```
# The code above configures, trains, and tests the model.  The line `model.compile('sgd', 'mean_squared_error', ['accuracy'])` configures the model's optimizer to `'sgd'`(stochastic gradient descent), the loss to `'mean_squared_error'`, and the metric to `'accuracy'`.  
# 
# You can find more optimizers [here](https://keras.io/optimizers/), loss functions [here](https://keras.io/objectives/#available-objectives), and more metrics [here](https://keras.io/metrics/#available-metrics).
# 
# To train the model, use the `fit()` function as shown in `model.fit(x_train_data, Y_train_data, batch_size=128, nb_epoch=2, validation_split=0.2)`.  The `validation_split` parameter will split a percentage of the training dataset to be used to validate the model.  The model can be further tested with the test dataset using the `evaluation()` function as shown in the last line.

# ## Train the Network
# 
# 1. Compile the network using adam optimizer and categorical_crossentropy loss function.
# 2. Train the network for ten epochs and validate with 20% of the training data.

# In[ ]:



