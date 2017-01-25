
# coding: utf-8

# # Traffic Sign Classification
# 
# by Uki D. Lucas
# 
# ** Self-Driving Car Engineer Nanodegree - Deep Learning **

# # Overview
# 
# In this notebook I design and implement a deep learning model that learns to recognize traffic signs. I train and test my model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
# 
# Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). 
# 
# **NOTE:** 
# 
# The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) is a solid starting point.

# ---
# # Set-up
# 
# ## Hyper-parameters
# 
# This section stay on top for easy access to parameters.

# In[1]:

# Hyperparameters
# Arguments used for tf.truncated_normal, 
# which randomly defines variables for the weights and biases for each layer
mu = 0 # 0 seems OK (EPOCH 10 -> 0.821)
sigma = 0.2 # 0.2 is better than 0.1 (EPOCH 10 -> 0.821)

EPOCHS = 100 # When I have time I run about 100, 70 is plenty enough to achive good model.
BATCH_SIZE = 256 #  memory limited
# on MacBook Pro 2.3GHz i7, 16GB 1600MHz DDR3 RAM: 
# 128 (slowest), 256 (faster), 512 (slower)
DROPOUT = 0.75 
# 0.80 (EPOCH 10 -> 0.75)
# 0.85 (EPOCH 10 -> 0.82)
validation_split = 0.15 # we will use x% of TRAIN data for validation (0.15 -> EPOCH 10 -> 0.821)
best_model = "./models/model_0.884893309667"
goal_rate = 0.8


# In[2]:

def print_hyper_parameters():
    print("- mu = {},".format(mu))
    print("- sigma = {},".format(sigma), "0.2 achieves 82% in 10 epochs." )
    print("- EPOCHS = {},".format(EPOCHS), "I am achieving peak at about 70") 
    print("- BATCH SIZE = {},".format(BATCH_SIZE), "best results achived with 256 on my computer") 
    print("- DROPOUT = {},".format(DROPOUT), "used for keep_prob") 
    print("- validation_split = {},".format(validation_split), "part of training set left for validation")
    print("- best_model = {},".format(best_model), "best saved model I have") 
    print("- goal_rate = {},".format(goal_rate), "rate after which I will quit, highest learning rate I recorded was 0.887") 
    
print_hyper_parameters()


# **Table of Contents**
# 
# <div id="toc"></div>

# In[3]:

import time
start = time.time()
import tensorflow as tf
end = time.time()
print('import tensorflow as tf took {} seconds'.format(round(end - start,1)))


# In[4]:

from tqdm import tqdm
# tqdm shows a smart progress meter
# usage: tqdm(iterable)


# ## Load The Data

# In[5]:

# Load pickled German street signs dataset from:
# http://bit.ly/german_street_signs_dataset
# If file location is not correct you get
# FileNotFoundError: [Errno 2] No such file or directory
training_file = "/Users/ukilucas/dev/DATA/traffic-signs-data/train.p" # 120.7MB
testing_file = "/Users/ukilucas/dev/DATA/traffic-signs-data/test.p" # 38.9 MB

import pickle

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']


# In[6]:

# Make sure the number of images in TRAIN set matches the number of labels
assert(len(X_train) == len(y_train))

# Make sure the number of images in TEST set matches the number of labels
assert(len(X_test) == len(y_test))

# name 'X_validation' is not defined
# assert(len(X_validation) == len(y_validation)) 


# In[7]:

# print example of one image to see the dimentions of data
print()
print("Image Shape: {}".format(X_train[0].shape))
print()
# Image Shape: (32, 32, 3) - good for LeNet, no need for padding with zero


# In[8]:

# print size of each set
training_set_size = len(X_train)
print("Training Set:   {} samples".format(training_set_size))
# Training Set:   39209 samples

testing_set_size = len(X_test)
print("Test Set:       {} samples".format(testing_set_size))
# Test Set:       12630 samples

#print("Validation Set: {} samples".format(len(X_validation)))


# ---
# 
# # Dataset Exploration
# 
# The pickled data is a dictionary with 4 key/value pairs:
# 
# - `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
# - `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
# - `'sizes'` is a list containing tuples, (width, height) representing the the original width and height the image.
# - `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**
# 
# Complete the basic data summary below.

# ## Dataset Summary
# 
# MEETS SPECIFICATIONS:
# Student performs basic data summary.

# In[9]:

import numpy as np


# In[10]:

### Replace each question mark with the appropriate value.
 
# TODO: Number of training examples
# Training Set:   39209 samples
n_train = X_train.shape[0]

# TODO: Number of testing examples.
# Test Set:       12630 samples
n_test = X_test.shape[0]

# TODO: What's the shape of an traffic sign image?
# Image Shape: (32, 32, 3)
image_shape = X_test.shape[1:3]

input_image_size = 32
number_of_channels = 3 # trying to keep color

# TODO: How many unique classes/labels there are in the dataset.
# see signnames.csv 43 elements (0 to 42)
number_train_labels = np.unique(y_train).shape[0]

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes (training labels) =", number_train_labels)

np.unique(y_train)

# Output:
# Number of training examples = 39209
# Number of testing examples = 12630
# Image data shape = (32, 32)
# Number of classes = 43


# ## Exploratory Visualization
# 
# MEETS SPECIFICATIONS: 
# Student performs an exploratory visualization on the dataset.
# 
# Overview:
# 
# Visualize the German Traffic Signs Dataset using the pickled file(s). 
# This is open ended, suggestions include: plotting traffic sign images, 
# plotting the count of each sign, etc.
# 
# The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.
# 
# **NOTE:** It's recommended you start with something simple first. If you wish to do more, 
# come back to it after you've completed the rest of the sections.

# In[11]:

def human_readable_sign_names(sign_number):
    return {
        0: "Speed limit (20km/h)",
        1: "Speed limit (30km/h)",
        2: "Speed limit (50km/h)",
        3: "Speed limit (60km/h)",
        4: "Speed limit (70km/h)",
        5: "Speed limit (80km/h)",
        6: "End of speed limit (80km/h)",
        7: "Speed limit (100km/h)",
        8: "Speed limit (120km/h)",
        9: "No passing",
        10: "No passing for vehicles over 3.5 metric tons",
        11: "Right-of-way at the next intersection",
        12: "Priority road",
        13: "Yield",
        14: "Stop",
        15: "No vehicles",
        16: "Vehicles over 3.5 metric tons prohibited",
        17: "No entry",
        18: "General caution",
        19: "Dangerous curve to the left",
        20: "Dangerous curve to the right",
        21: "Double curve",
        22: "Bumpy road",
        23: "Slippery road",
        24: "Road narrows on the right",
        25: "Road work",
        26: "Traffic signals",
        27: "Pedestrians",
        28: "Children crossing",
        29: "Bicycles crossing",
        30: "Beware of ice/snow",
        31: "Wild animals crossing",
        32: "End of all speed and passing limits",
        33: "Turn right ahead",
        34: "Turn left ahead",
        35: "Ahead only",
        36: "Go straight or right",
        37: "Go straight or left",
        38: "Keep right",
        39: "Keep left",
        40: "Roundabout mandatory",
        41: "End of no passing",
        42: "End of no passing by vehicles over 3.5 metric tons"
    }.get(sign_number, "Error: sign not found") # default if x not found

# TEST function
print( human_readable_sign_names(0))
print( human_readable_sign_names(28))
print( human_readable_sign_names(42))
print( human_readable_sign_names(43))


# ## Display sample images with corresponding labels
# 
# Please note the terrible quality of the images.
# 
# In this case, the **computer might be able to detect the right sign better than human eye**.

# In[12]:

import random
# import numpy as np # already imported
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

figure, axes = plt.subplots(10, 5, figsize=(32,32))
for rows in range(10):
    for columns in range(5):
        index = random.randint(0, len(X_train))
        image = X_train[index]#.squeeze()

        axes[rows,columns].imshow(image)
        axes[rows,columns].set_title(human_readable_sign_names(y_train[index]))
plt.show()


# ### Training Data Labels Distribution

# In[13]:

# train is the pickle file
histOut = plt.hist(train['labels'],number_train_labels, facecolor='g', alpha=0.60)


# ### Testing Data Labels Distribution

# In[14]:

histOut = plt.hist(test['labels'],number_train_labels, facecolor='g', alpha=0.60)


# ---
# 
# # Design and Test a Model Architecture
# 
# ## Preprocessing
# 
# 
# <font color='PURPLE'>
# Students provides sufficient details of the preprocessing techniques used. Additionally, the student discusses why the techniques were chosen.
# </font>
# 
# 
# 
# #### Question 1 (preprocessing)
# 
# _Describe how you preprocessed the data. Why did you choose that technique?_
# 
# Answer:
# - Shuffle given data each time to avoid image order ralated problems
# - There is no need to resize the images as they are already 32x32
# - I am keeping 3 color chanels as they should be beneficial in categorization
# - image color depth scaling to values between -0.5 and +0.5
# - Create validation set as 20% of the training set 
# 
# 
# ** Further Suggestions **
# 
# To inspire further improvements, here are some preprocessing techniques that have proven to work on this dataset:
# 
# - Performing translations, scalings and rotations of the training set considerably improves generalization
# - Values for translation, rotation and scaling can be drawn from a uniform distribution in a specified range, i.e. ±T % of the image size for translation, 1±S/100 for scaling and ±R◦ for rotation.
# - Convertion to grayscale can also help.
# - A normalization method that yields very impressive results is the Contrast-limited Adaptive Histogram Equalization (CLAHE)
# - Normalization can help in case of High contrast variation among the images
# - More preprocessing steps could be included
# - These resources (1 & 2) might provide some more intuition on the subject
# 
# 
# http://people.idsia.ch/~juergen/nn2012traffic.pdf
# 
# http://people.idsia.ch/~juergen/ijcnn2011.pdf

# ### Scale Image Color Depth from 0..255 to -0.5 to +0.5

# In[15]:

def scale_image_color_depth(value):
    """ 
    normalizes image color depth values 0..255
    to values between -0.5 and +0.5
    """
    # image color depth has values 0 to 255
    max_value = 255.0
    # take the half value = 127.5
    return ((value - max_value/2) / max_value)

# TEST:
print("normalized", scale_image_color_depth(0)) # min value
print("normalized", scale_image_color_depth(128)) # half value
print("normalized", scale_image_color_depth(255)) # max value
    


# In[16]:

def scale_image_color_depth_for_all(image_set):
    results = np.copy(image_set) # create placeholder
    for i in tqdm(range(image_set.shape[0])):
        results[i] = scale_image_color_depth(image_set[i].astype(float))
    return results


# ### Histogram Equalization (even out brighness)
# 
# The effect should be that the images with little contrast should be very readable now.

# In[17]:

# http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
import cv2
import numpy as np
from matplotlib import pyplot as plt

def histogram_equalization(image):
    hist,bins = np.histogram(image.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()
    plt.plot(cdf_normalized, color = 'b')
    plt.hist(image.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.show()
    return image

def histogram_clahe(image):
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(image)
    #cv2.imwrite('clahe_2.jpg',cl1)
    return cl1



image = cv2.imread('images/sample_set.jpg',0)
plt.imshow(image)
plt.show()
# histogram_equalization(image)
new_image = histogram_clahe(image)
plt.imshow(new_image)
plt.show()


# ## Preprocess image sets (scale, histogram)

# In[18]:

# TRAIN FEATURES
# Apply histogram before changing color depth
# train_features = histogram_clahe(X_train.astype(float))
# Scale training set
train_features = scale_image_color_depth_for_all(X_train.astype(float))


# TEST FEATURES
# Apply histogram before changing color depth
# test_features = histogram_clahe(X_test.astype(float))
# Scale testing set
test_features = scale_image_color_depth_for_all(X_test.astype(float))

# TODO the application of histogram needs more love.


# In[ ]:




# ### Create validation set as 20% of the training set
# 
# Use scaled values of the training set.

# #### Question 2 (describe training, validation and testing)
# 
# <br/>
# <font color='Purple'>
# Student describes how the model was trained and evaluated. If the student generated additional data they discuss their process and reasoning. Additionally, the student discusses the difference between the new dataset with additional data, and the original dataset.
# </font>
# 
# _Describe how you set up the training, validation and testing data for your model. **Optional**: If you generated additional data, how did you generate the data? Why did you generate the data? What are the differences in the new dataset (with generated data) from the original dataset?_
# 
# **Answer:**
# - sklearn.cross_validation  train_test_split
# - create 20% validation set from the training features set
# - assert that the number of features is same as corresponding labels
# - display the count of each

# In[19]:

from sklearn.cross_validation import train_test_split
# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train, X_validation, y_train, y_validation = train_test_split(train_features, 
                                                                y_train, 
                                                                test_size = validation_split, 
                                                                random_state=42)


# In[20]:

# Make sure the number of images in TRAIN set matches the number of labels
assert(len(X_train) == len(y_train))
print("len(X_train)", len(X_train))

# Make sure the number of images in TEST set matches the number of labels
assert(len(X_test) == len(y_test))
print("len(X_test)", len(X_test))

# name 'X_validation' is not defined
assert(len(X_validation) == len(y_validation))
print("len(X_validation)", len(X_validation))


# In[21]:

from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)


# ### Features and Labels
# 
# - `x` is a placeholder for a batch of input images.
# - `y` is a placeholder for a batch of output labels.

# In[22]:

# x variable (Tensor) stores input batches
# None - later accepts batch of any size
# image dimentions 32x32x1
x = tf.placeholder(tf.float32, (None, 
                                input_image_size, 
                                input_image_size, 
                                number_of_channels)) # (None, 32, 32, 3)

# y variable (Tensor) stores labels
y = tf.placeholder(tf.int32, (None)) # if using "None,number_of_channels" -> (None, 3)


# encode our labels
one_hot_y = tf.one_hot(y, number_train_labels) # 43

# See definition of the DROPOUT below
keep_prob = tf.placeholder(tf.float32)


# ---
# ## Model Architecture
# 
# <font color='Purple'>
# Student provides sufficient details of the characteristics and qualities of the architecture, such as the type of model used, the number of layers, the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged.
# </font>
# 
# ### Convolutional Deep Neural Network
# 
# Implement the neural network architecture based on LeNet-5.
# 
# 
# **Input **
# 
# The LeNet architecture accepts a 32x32xC number_color_channels. 
# 
# - MNIST images are grayscale, C is 1.
# - German street sign images have **3 color channels**, C is 3.
# 
# **Suggestions & Comments **
# 
# - Using StratifiedShuffleSplit over the traditional train-test-split, enables the use of the entire training data while keeping data for cross-validation: the distribution of the labels in both would be similar. Around 1% of the data can be used for the cross-validation set.
# - Also here's a nice discussion on how to choose the batch_size of Stochastic Gradient Decent
# - And a nice Discussion on Adam Optimizer
# 
# http://stats.stackexchange.com/questions/140811/how-large-should-the-batch-size-be-for-stochastic-gradient-descent
# 
# http://sebastianruder.com/optimizing-gradient-descent/index.html#adam

# #### Question 3: ( architecture of CDNN)
# 
# <font color='Purple'>
# Student thoroughly discusses the approach taken for deriving and designing a model architecture fit for solving the problem given.
# </font>
# 
# _What does your final architecture look like? (Type of model, layers, sizes, connectivity, etc.)  For reference on how to build a deep neural network using TensorFlow, see [Deep Neural Network in TensorFlow
# ](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/b516a270-8600-4f93-a0a3-20dfeabe5da6/concepts/83a3a2a2-a9bd-4b7b-95b0-eb924ab14432) from the classroom._
# 
# **Answer:**
# 
# CONVOLUTIONAL LAYER L1
# - Calculated output size 28.0
# - Convolution Output Tensor("add:0", shape=(?, 28, 28, 6), dtype=float32)
# - ReLU Output Tensor("Relu:0", shape=(?, 28, 28, 6), dtype=float32)
# - Pooling Layer Output Tensor("MaxPool:0", shape=(?, 14, 14, 6), dtype=float32)
# 
# CONVOLUTIONAL LAYER L2
# - Calculated output size 28.0
# - Convolution Output Tensor("add_1:0", shape=(?, 10, 10, 16), dtype=float32)
# - ReLU Output Tensor("Relu_1:0", shape=(?, 10, 10, 16), dtype=float32)
# - Pooling Layer Output Tensor("MaxPool_1:0", shape=(?, 5, 5, 16), dtype=float32)
# 
# FLATTENING LAYER L3
# - Flattened Output Tensor("Flatten/Reshape:0", shape=(?, 400), dtype=float32)
# - CONVOLUTIONAL FULLY CONNECTED LAYER L4
# - Convolution Output Tensor("add_2:0", shape=(?, 120), dtype=float32)
# - ReLU Output Tensor("Relu_2:0", shape=(?, 120), dtype=float32)
# 
# CONVOLUTIONAL FULLY CONNECTED LAYER L5
# - Convolution Output Tensor("add_3:0", shape=(?, 84), dtype=float32)
# - ReLU Output Tensor("Relu_3:0", shape=(?, 84), dtype=float32)
# 
# CONVOLUTIONAL FULLY CONNECTED LAYER L6
# - Convolution Output Tensor("add_4:0", shape=(?, 43), dtype=float32)
# - ReLU Output Tensor("Relu_4:0", shape=(?, 43), dtype=float32)
# 
# 
# ** Suggestions & Comments **
# 
# Here are few questions to think about while designing the architecture:
# 
# - How would I choose the optimizer? What is its Pros & Cons and how would I evaluate it?
# - How would I decide the number and type of layers?
# - How would I tune the hyperparameter? How many values should I test and how to decide the values?
# - How would I preprocess my data? Why do I need to apply a certain technique?
# - How would I train the model?
# - How would I evaluate the model? What is the metric? How do I set the benchmark?
# Also, here's an interesting Inception's example.
# 
# https://github.com/tflearn/tflearn/blob/master/examples/images/googlenet.py

# In[23]:

def convolution_output_size(input_size=32, filter_size=5, stride_veritcal=1):
    output_size = (input_size - filter_size + 1)/stride_veritcal
    print("Calculated output size", output_size)


# In[24]:

def pooling_layer(input_tensor):
    # POOLING (SUBSAMPLING) LAYER L2
    # Input = 28x28x6. 
    # Output = 14x14x6.
    # value: A 4-D Tensor with shape [batch, height, width, channels] and type tf.float32.
    # ksize: A list of ints that has length >= 4. The size of the window for each dimension of the input tensor.
    # strides: A list of ints that has length >= 4. The stride of the sliding window for each dimension of the input tensor.
    # padding: A string, either 'VALID' or 'SAME'. 
    # name: Optional name for the operation.
    tensor = tf.nn.max_pool(value = input_tensor, 
                           ksize=[1, 2, 2, 1], 
                           strides=[1, 2, 2, 1], 
                           padding='VALID')
    print("Pooling Layer Output", tensor)
    #L2 output Tensor("L2:0", shape=(?, 14, 14, 6), dtype=float32)
    return tensor


# In[25]:

def convolution_layer(input_tensor, filter_size=5, input_depth=3, output_depth=6):
    # L1 filter (5,5,3,6)
    # L2 filter (5,5,6,16) 
    
    filter_tensor = tf.Variable(tf.truncated_normal(
            shape=(filter_size, 
                   filter_size, 
                   input_depth, 
                   output_depth), 
            mean = mu, 
            stddev = sigma))
    
    bias = tf.Variable(tf.zeros(output_depth))
    tensor = tf.nn.conv2d(input = input_tensor, 
                         filter = filter_tensor, 
                         strides = [1, 1, 1, 1], 
                         padding='VALID'
                        ) + bias
    
    convolution_output_size(input_size=32, filter_size=filter_size, stride_veritcal=1)
    # calculated output size 28.0
    
    print("Convolution Output", tensor)
    # L1 output Tensor("add:0", shape=(?, 28, 28, 6), dtype=float32)
    
    # ReLU Activation function.
    tensor = tf.nn.relu(features = tensor)
    print("ReLU Activation funtion Output", tensor)
    # ReLU output Tensor("Relu:0", shape=(?, 28, 28, 6), dtype=float32)
    
    tensor = pooling_layer(input_tensor = tensor)
    return tensor


# In[26]:

def convolution_fully_connected(input_tensor, input_size=400, output_size=120):
    # Fully Connected. Input = 400. Output = 120.
    filter_tensor = tf.Variable(tf.truncated_normal(
            shape=(input_size, output_size), 
            mean = mu, stddev = sigma))
    
    bias = tf.Variable(tf.zeros(output_size))
    
    tensor   = tf.matmul(input_tensor, filter_tensor) + bias
    print("Convolution Output", tensor)
    
    # ReLu Activation.
    tensor    = tf.nn.relu(tensor)
    print("ReLU Activation funtion Output", tensor)
    return tensor


# In[27]:

from tensorflow.contrib.layers import flatten


    
def convolutional_neural_network(tensor): 

    print("CONVOLUTIONAL LAYER L1")
    tensor = convolution_layer(
        input_tensor = tensor, filter_size=5, input_depth=3, output_depth=6)
    
    print("CONVOLUTIONAL LAYER L2")
    tensor = convolution_layer(
        input_tensor = tensor, filter_size=5, input_depth=6, output_depth=16)

    # Input Tensor("MaxPool_1:0", shape=(?, 5, 5, 16), dtype=float32)
    print("FLATTENING LAYER L3")
    # Flattens Input 5x5x16 = 400
    tensor   = flatten(tensor)
    print("Flattened Output", tensor)
    
    # Tensor("Flatten/Reshape:0", shape=(?, 400), dtype=float32)
    
    print("CONVOLUTIONAL FULLY CONNECTED LAYER L4")
    tensor = convolution_fully_connected(input_tensor=tensor, input_size=400, output_size=120)
    # Convolution output tensor Tensor("add_2:0", shape=(?, 120), dtype=float32)
    # ReLU output tensor Tensor("Relu_2:0", shape=(?, 120), dtype=float32)
    
    print("CONVOLUTIONAL FULLY CONNECTED LAYER L5")
    tensor = convolution_fully_connected(input_tensor=tensor, input_size=120, output_size=84)
    # Convolution output tensor Tensor("add_3:0", shape=(?, 84), dtype=float32)
    #ReLU output tensor Tensor("Relu_3:0", shape=(?, 84), dtype=float32)
    
    print("CONVOLUTIONAL FULLY CONNECTED LAYER L6")
    tensor = convolution_fully_connected(input_tensor=tensor, input_size=84, output_size=43)
    # Convolution output tensor Tensor("add_4:0", shape=(?, 43), dtype=float32)
    # ReLU output tensor Tensor("Relu_4:0", shape=(?, 43), dtype=float32)
    
    return tensor # logits


# ## Dataset and Training
# 
# <font color='Purple'>
# Student describes how the model was trained and evaluated. If the student generated additional data they discuss their process and reasoning. Additionally, the student discusses the difference between the new dataset with additional data, and the original dataset.
# </font>
# 
# ** Answer: **
# - Run the training data through the training pipeline to train the model.
# - Before each epoch, shuffle the training set.
# - After each epoch, measure the loss and accuracy of the validation set.
# - Save the model after training.

# Create a training pipeline that uses the model to classify sign data.

# In[28]:

logits = convolutional_neural_network(x)


# ### Model Evaluation
# Evaluate how well the loss and accuracy of the model for a given dataset.

# #### Softmax Cross Entropy
# 
# Cross Entropy is the measure of how different are 
# the logits (output classes) from ground truth training labels

# In[29]:

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
print(cross_entropy)

# average entropy from all the training images
mean_loss_tensor = tf.reduce_mean(cross_entropy)
print("mean_loss_tensor",mean_loss_tensor)



# #### Question 4 (training, optimizer, batch size, epochs, params)
# 
# <font color='Purple'>
# _How did you train your model? (Type of optimizer, batch size, epochs, hyperparameters, etc.)_
# </font>
# 
# **Answer:**
# 
# #### Adam optimizer
# 
# This is a minimize the loss function (similar to Stockastic Gradient Decent SGD does)
# 
# AdamOptimizer is more sofisticated, so it is a good default.
# 
# It uses moving averages of the parameters (momentum); 
# Bengio discusses the reasons for why this is beneficial in Section 3.1.1 of this paper. 
# 
# Simply put, this enables Adam to use a larger effective step size (rate), 
# and the algorithm will converge to this step size without fine tuning

# #### Dropout
# 
# To reduce overfitting, we will apply dropout before the readout layer. We create a placeholder for the probability that a neuron's output is kept during dropout. This allows us to turn dropout on during training, and turn it off during testing. TensorFlow's tf.nn.dropout op automatically handles scaling neuron outputs in addition to masking them, so dropout just works without any additional scaling.1

# In[30]:

# learning rate (how quickly to update the networks weights)
rate = 0.001

adam_optimizer = tf.train.AdamOptimizer(learning_rate = rate)
print("adam_optimizer", adam_optimizer)

# uses backpropagation 
adam_optimizer_minimize = adam_optimizer.minimize(mean_loss_tensor)
print("adam_optimizer_minimize", adam_optimizer_minimize)


# In[31]:

# is prediction correct
are_preditions_correct = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
print("are_preditions_correct", are_preditions_correct)

# calc model's overall accuracy by avegaring individual prediction acuracies 
predition_mean = tf.reduce_mean(tf.cast(are_preditions_correct, tf.float32))
print("predition_mean", predition_mean)


# In[32]:

def evaluate(X_data, y_data):
    num_examples = len(X_data) 
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in tqdm(range(0, num_examples, BATCH_SIZE)):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(predition_mean, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


# In[33]:

import time
start = time.time()

saver = tf.train.Saver()
vector_accurancies = []


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    print("Training...")
    print_hyper_parameters()
    print()
    for i in range(EPOCHS):
        # shuffle: make sure it is not biased by the order of images
        X_train, y_train = shuffle(X_train, y_train)
        
        # break training model into batches,
        # train the model on each batch
        for offset in range(0, training_set_size, BATCH_SIZE):
            end = offset + BATCH_SIZE
            # print("running batch from", offset, " step ", end, " up to ", training_set_size)
            batch_x = X_train[offset:end]
            batch_y = y_train[offset:end]
             
            sess.run(adam_optimizer_minimize, 
                     feed_dict={x: batch_x, y: batch_y, keep_prob: DROPOUT})
            
        # at the end of each epoch, evaluate against validation set
        validation_accuracy = evaluate(X_validation, y_validation)
        vector_accurancies.extend([validation_accuracy])
        print("EPOCH {} ...".format(i+1),
              "Validation Accuracy = {:.3f}".format(validation_accuracy))
        # EPOCH 1 ... Validation Accuracy = 0.300 - very low
        # EPOCH 2 ... Validation Accuracy = 0.500 - growing
        # training for 2 epochs and 256 batch size took 41.1 seconds
        print()
        if validation_accuracy > goal_rate: # I want to beat my best prediction to date and stop there
            print ("stopping after achieving",validation_accuracy )
            break
    
    end = time.time()
    print('Training for {} epochs and {} batch size took {} seconds'.format(
            EPOCHS, BATCH_SIZE, round(end - start,1)))
        
    # upon training complete, save it so we do not have to train again
    # assure to save the model with achieved accuracy,
    # this way later I can select the best run
    saver.save(sess, './models/model_' + str(validation_accuracy))
    print("Model saved")
    


# ## Learning accuracy graph

# In[34]:

import matplotlib.pyplot as plt
plt.plot(vector_accurancies)
plt.xlabel('EPOCHS')
plt.ylabel('PERCENTAGE')
plt.show()


# In[35]:



# Example Run 100 epochs
vector_accurancies  = [ 0.210, 0.357, 0.419, 0.583, 0.625, 0.633, 0.639, 
                       0.724, 0.694, 0.705, 0.734, 0.676, 0.747, 0.770, 
                       0.749, 0.728, 0.803, 0.807, 0.786, 0.760, 0.784, 
                       0.833, 0.826, 0.826, 0.838, 0.835, 0.789, 0.764, 
                       0.825, 0.841, 0.847, 0.849, 0.853, 0.851, 0.842, 
                       0.807, 0.832, 0.760, 0.821, 0.847, 0.849, 0.858, 
                       0.848, 0.854, 0.857, 0.863, 0.860, 0.861, 0.862, 
                       0.862, 0.849, 0.714, 0.852, 0.859, 0.841, 0.856, 
                       0.862, 0.861, 0.864, 0.855, 0.837, 0.847, 0.763, 
                       0.870, 0.875, 0.873, 0.866, 0.880, 0.877, 0.883, 
                       0.882, 0.886, 0.885, 0.886, 0.884, 0.886, 0.885, 
                       0.886, 0.887, 0.886, 0.886, 0.886, 0.886, 0.887, 
                       0.886, 0.887, 0.887, 0.887, 0.365, 0.827, 0.867, 
                       0.865, 0.875, 0.882, 0.885, 0.885, 0.885, 0.885, 
                       0.886, 0.885]

import numpy as np
print ("max achieved", np.amax(vector_accurancies))
import matplotlib.pyplot as plt
plt.plot(vector_accurancies)
plt.xlabel('EPOCHS')
plt.ylabel('PERCENTAGE')
plt.show()


# #### Question 5 (describe final solution)
# 
# <font color='Purple'>
# _What approach did you take in coming up with a solution to this problem? It may have been a process of trial and error, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think this is suitable for the current problem._
# </font>
# 
# **Answer:**
# 
# - I have started with LeNet architecture as is was recommended in class
# - Made sure it runs on MINST dataset
# - Made sure to change it to German signs (32x32x3)
# - Stepped thru each piece of functionality, refactored names, functions  to understand it better
# - Looked up most of the functions in TensorFlow official documentation
# - Started to play with Hyper-parameters to get better results
# - I am not good enough yet to start switching the architecture
# 
# 
# 
# Training...
# - mu 0
# - sigma 0.3
# - EPOCHS 100 more the better, but achieving > 98% is a proof
# - BATCH SIZE 512 best results with 256 on my computer
# - DROPOUT 0.8 used for keep_prob
# 
# 100%|██████████| 16/16 [00:03<00:00,  4.91it/s]
# EPOCH 1 ... Validation Accuracy = 0.181
# 
# 
# 100%|██████████| 16/16 [00:01<00:00,  8.14it/s]
# EPOCH 97 ... Validation Accuracy = 0.803
# 
# 100%|██████████| 16/16 [00:02<00:00,  7.70it/s]
# EPOCH 98 ... Validation Accuracy = 0.802
# 
# 100%|██████████| 16/16 [00:01<00:00,  8.09it/s]
# EPOCH 99 ... Validation Accuracy = 0.803
# 
# 100%|██████████| 16/16 [00:02<00:00,  7.89it/s]
# EPOCH 100 ... Validation Accuracy = 0.803
# 
# Training for 100 epochs and 512 batch size took 3649.5 seconds
# Model saved

# ---
# # Evaluate saved model agaist the TEST set

# In[36]:

# I will be updating the model name to the HIGHEST accurancy achieved

import time # takes no time if imported previously
start = time.time()

import tensorflow as tf # takes no time if imported previously
saver = tf.train.Saver()

end = time.time()
print('import tensorflow as tf took {} seconds'.format(round(end - start,1)))



with tf.Session() as sess:
    print ('Re-loading saved model' + best_model)
    saver.restore(sess, best_model)

    test_accuracy = evaluate(X_test, y_test)
    print("Evaluating the TEST set agaist restored model trained to 88.4%, result = {:.1f}% accurancy".format(test_accuracy*100))


# # Test a Model on New Images
# 
# #### Question 6 (choose and describe 5 new images)
# 
# 
# <font color='Purple'>
# 
# Student chooses five candidate images of traffic signs taken and visualizes them in the report. Discussion is made as to any particular qualities of the images or traffic signs in the images that may be of interest, such as whether they would be difficult for the model to classify.
# 
# 
# Take several pictures of traffic signs that you find on the web or around you (at least five), and run them through your classifier on your computer to produce example results. The classifier might not recognize some local signs but it could prove interesting nonetheless.
# 
# You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.
# 
# 
# _Choose five candidate images of traffic signs and provide them in the report. Are there any particular qualities of the image(s) that might make classification difficult? It could be helpful to plot the images in the notebook._
# 
# </font>
# 
# 
# **Answer**
# 
# - I found 7 signs that indicate 23: "Slippery road",
# - I think they are a VERY IMTERESTING test as each is a little different
# - 1) triangle/red/white on blue background
# - 2) triangle/red/white on green background, sightly different car and crossing over tire marks
# - 3) triangle/red/**yellow** on blue/yellow background, very different tire marks
# - 4) **rectangle/black**/yellow on white background - **I DO NOT expect it to be recognized**
# - 5) triangle/red/white on green background, sightly different car and crossing over tire marks, This is similar to 2, but not the same design
# - 6) triangle/red/white on **winter-blue background with snow cap**
# - 7) triangle/red/white on **winter-blue background with 70% of sign covered in snow**
# - I resized each to square
# - I created a notebook with code to resize whole directory of images, see same folder
# - I resized each to 32x32 - the QUALITY IS TERRIBLE
# - Image slip_008_32x32.png is from the TRAINING set to see if it will be guessed
# 
# I do not understand why we use such a bad training set of 32x32 images, 
# it makes sense for characted recognition, but not for signs with important text inside.

# ### Show NEW images (original and resized 32x32)

# In[37]:

directory = "images/verification"
prepended_by = "slip_"


import os
listing = os.listdir(directory)
print (len(listing))
listing[5]


# In[38]:

from skimage import io
import numpy as np
from matplotlib import pyplot as plt
 

# count and display valid images
counter = 0
for i in range(len(listing)):
    if ".jpg" not in listing[i]: 
        print("ignoring", listing[i])
        continue
    if prepended_by not in listing[i]: 
        print("ignoring", listing[i])
        continue
        
    image = io.imread(directory + "/" + listing[i])
    plt.figure(figsize=(2,2))
    plt.imshow(image)
    plt.show()
    
    if "32x32" in listing[i]: 
        counter = counter + 1
        


# In[39]:

print("counter", counter)      
image_matrix = np.uint8(np.zeros((counter, 32, 32, 3))) 
print("image_matrix", image_matrix.shape)

index = -1
for i in range(len(listing)):
    if ".jpg" not in listing[i]: 
        print("ignoring", listing[i])
        continue
    if prepended_by not in listing[i]: 
        print("ignoring", listing[i])
        continue

    if "32x32" in listing[i]: 
        image = io.imread(directory + "/" + listing[i])
        index = index + 1
        image_matrix[index] = image
        print("adding", listing[i], "@ index", index)
image_matrix.shape


# #### Question 7 (performance of new images)
# 
# <font color='Purple'>
# Student documents the performance of the model when tested on the captured images and compares it to the results of testing on the dataset.
# 
# <p/>
# **Required:**
# <p/>
# - Compare them to the test results of the dataset.
# <br/>
# - The aim here is to document the performance of your model on the new images and compare it to the test accuracy on the dataset.
# <br/>
# - Your explanation can look something like: the accuracy on the captured images is X% while it was Y% on the testing set thus It seems the model is overfitting
# </font>
# 
# 
# _Is your model able to perform equally well on captured pictures when compared to testing on the dataset? The simplest way to do this check the accuracy of the predictions. For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate._
# 
# _**NOTE:** You could check the accuracy manually by using `signnames.csv` (same directory). This file has a mapping from the class id (0-42) to the corresponding sign name. So, you could take the class id the model outputs, lookup the name in `signnames.csv` and see if it matches the sign from the image._
# 
# 
# **Answer:**
# 
# The accuracy on the captured images is **0% while it was 68.2%** on the testing set thus it seems the model is was overfit.
# 
# - I created image_matrix with 7 of my NEW #23: "Slippery road"
# - I preprocess them the same way as training set, I got new_images
# - I created lables (all same): new_labels = [23, 23, 23, 23, 23, 23, 23]
# - I run evaluate function (test set returned 0.682)
# - My set of NEW images returned 0% accurancy (disappointing)
# - I cannot explain this, **considering that I had 1 image from training set**

# In[40]:

items = len(image_matrix)
print("Number of new images:", items)


# 

# In[41]:

new_images = scale_image_color_depth_for_all(
    image_matrix.reshape((-1, 32, 32, 3)).astype(float))

new_labels =  [23] * items
print(new_labels)
human_readable_sign_names(23)


# In[42]:

# I will be updating the model name to the HIGHEST accurancy achieved

with tf.Session() as sess:
    print ('Re-loading saved model' + best_model)
    saver.restore(sess,   best_model)

    test_accuracy = evaluate(new_images, new_labels)
    print("Evaluating the TEST set agaist restored model trained to 88.4%, result = {:.1f}% accurancy".format(test_accuracy*100))


# In[ ]:




# #### Question 8 (visualize softmax on new images)
# 
# <font color='Purple'>
# The softmax probabilities of the predictions on the captured images are visualized. The student discusses how certain or uncertain the model is of its predictions.
# </font>
# 
# ** Answer: **
# - Predition is totally misleading
# - For some there is 100%, for other 20%
# - It is hard to draw any conclusion
# 
# 
# 
# ** Suggestions & Comments **
# 
# 
# - Usually I'd recommend for each of the five signs, a histogram of softmax probabilities for each class (or for the 5-10 classes with highest probabilities).
# - Here is an example code from a student for visualizing the softmax probabilities:
# 
# **(I plan to review and implemnt this in the future)**
# 
# <img src="./visualize_softmax.png" />

# In[43]:

softmax_tensor = tf.nn.softmax(logits)

def classify_images(X_data):
    session = tf.get_default_session()
    predicted_tensor = session.run(softmax_tensor, feed_dict={x: X_data, keep_prob: 0.8})
    return predicted_tensor
    
with tf.Session() as sess:
    print ('Re-loading saved model' + best_model)
    saver.restore(sess,   best_model)
    
    predictions = classify_images(new_images)
    top_k_tensor = sess.run(tf.nn.top_k(predictions, 5, sorted=True))
    label_indexes = np.argmax(top_k_tensor, 1)

values = label_indexes[1,1:]  

for index in tqdm(range(len(values))):
    print(human_readable_sign_names(values[index]), values[index])


# In[44]:

### Visualize the softmax probabilities
top = 5

for i in range(top):
    predictions = top_k_tensor[0][i]
    plt.title('Top {} Softmax probabilities for option {}'.format(top, str(i)))
    plt.figure(i)
    plt.xlabel('label #')
    plt.ylabel('prediction')
    plt.bar(range(top), predictions, 0.10, color='b')
    plt.xticks(np.arange(top) + 0.10, tuple(predictions))

plt.show()


# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

# In[ ]:



// This updates Table of Contents section on top, run on the bottom of the notebook
%%javascript
$.getScript('https://kmahelona.github.io/ipython_notebook_goodies/ipython_notebook_toc.js')
# In[ ]:



