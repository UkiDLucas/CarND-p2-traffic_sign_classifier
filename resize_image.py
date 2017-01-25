
# coding: utf-8

# In[1]:

directory = "images/verification"
prepended_by = "slip_"


# In[2]:

from PIL import Image

def resize_32x32(image_name):
    image_name = remove_extension(image_name)
    img = Image.open(directory + '/' + image_name + ".jpg")
    new_img = img.resize((32,32))
    new_img.save(directory + "/" + image_name + "_32x32.jpg", "JPEG", optimize=True)
    

def remove_extension(text):
    if text.endswith('.jpg'):
        text = text[:-4]
    return text

#TEST:
print (remove_extension("stop.jpg"))

resize_32x32("stop.jpg")


# In[3]:

import os
listing = os.listdir(directory)
print (len(listing))
listing


# In[4]:

from skimage import io
import numpy as np
from matplotlib import pyplot as plt
 
image_matrix = np.uint8(np.zeros((len(listing), 32, 32, 3))) # I know I have 30 images

counter = -1
for i in range(len(listing)):
    if ".jpg" not in listing[i]: 
        print("ignoring", listing[i])
        continue
    if prepended_by not in listing[i]: 
        print("ignoring", listing[i])
        continue
    counter = counter + 1
    image = io.imread(directory + "/" + listing[i])
    plt.imshow(image)
    plt.show()
    resize_32x32(listing[i])
    #image_matrix[i-1] = image_array
    print(counter, listing[i])
    
    
image_matrix.shape
#plt.imshow(image_matrix[counter])
#plt.show()


# In[ ]:



