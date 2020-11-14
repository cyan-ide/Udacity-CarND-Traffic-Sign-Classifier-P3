#!/usr/bin/env python
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# ## Deep Learning
# 
# ## Project: Build a Traffic Sign Recognition Classifier


#all imports
import tensorflow as tf
import numpy as np
import pandas as pd

import time
import random
import matplotlib.pyplot as plt

from tensorflow.contrib.layers import flatten

#notebook paramters
EPOCHS = 20
#BATCH_SIZE = 256
BATCH_SIZE = 128
#BATCH_SIZE = 64

#image color channels
COLOR_CHANNELS = 3 #3


# ---
# ## Step 0: Load The Data

# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = "./data/train.p"
validation_file= "./data/valid.p"
testing_file = "./data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

# ---
# 
# ## Step 1: Dataset Summary & Exploration
# 
# The pickled data is a dictionary with 4 key/value pairs:
# 
# - `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
# - `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
# - `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
# - `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**
# 
# Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the [pandas shape method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shape.html) might be useful for calculating some of the summary results. 

# ### Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas

### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

# ----
# 
# ## Step 2: Design and Test a Model Architecture
# 
# Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
# 
# The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play! 
# 
# With the LeNet-5 solution from the lecture, you should expect a validation set accuracy of about 0.89. To meet specifications, the validation set accuracy will need to be at least 0.93. It is possible to get an even higher accuracy, but 0.93 is the minimum for a successful project submission. 
# 
# There are various aspects to consider when thinking about this problem:
# 
# - Neural network architecture (is the network over or underfitting?)
# - Play around preprocessing techniques (normalization, rgb to grayscale, etc)
# - Number of examples per label (some have more than others).
# - Generate fake data.
# 
# Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.

# ### Pre-process the Data Set (normalization, grayscale, etc.)

# Minimally, the image data should be normalized so that the data has mean zero and equal variance. For image data, `(pixel - 128)/ 128` is a quick way to approximately normalize the data and can be used in this project. 
# 
# Other pre-processing steps are optional. You can try different techniques to see if it improves performance. 
# 
# Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.

# In[9]:


### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.

from sklearn.utils import shuffle

#turn to grayscale
# X_train_p = np.sum(X_train/np.float32(3), axis=3, keepdims=True)
# X_valid_p = np.sum(X_valid/np.float32(3), axis=3, keepdims=True)
# X_test_p = np.sum(X_test/np.float32(3), axis=3, keepdims=True)
#regular color images
X_train_p = X_train
X_valid_p = X_valid
X_test_p = X_test

#normalize (ie. change from rgb unit8 0-255 to float32 0-1 range)
X_train_p = X_train_p/np.float32(255)
y_train_p = y_train
X_test_p = X_test/np.float32(255)
X_valid_p = X_valid_p/np.float32(255)
#tst = X_train/float(255)
#print(tst[0][0][0][0])

#rotate by random angles
# Tensorflow random angle rotation
input_size = X_train_p.shape[1]
side_size = int(np.sqrt(input_size))

images = tf.placeholder(tf.float32, (None, 32, 32, COLOR_CHANNELS)) #tf.placeholder(tf.float32, [None, input_size])
#images = tf.reshape(dataset, (-1, side_size, side_size, 1))
random_angles = tf.random.uniform(shape = (tf.shape(images)[0], ), minval = -np.pi / 8, maxval = np.pi / 8)

rotated_images = tf.contrib.image.transform(
    images,
    tf.contrib.image.angles_to_projective_transforms(
        random_angles, tf.cast(tf.shape(images)[1], tf.float32), tf.cast(tf
            .shape(images)[2], tf.float32)
    ))

sess = tf.Session()
result = sess.run(rotated_images, feed_dict = {
    images: X_train_p,
})
#print(len(rotated_images))

#shuffle
#X_train_p, y_train_p = shuffle(X_train_p, y_train)


# In[ ]:


# # test rotation - Print 10 random samples
# fig, axes = plt.subplots(2, 10, figsize = (15, 4.5))
# choice = np.random.choice(range(len(X_train_p)), 10)
# for k in range(10):
#     axes[0][k].set_axis_off()
#     axes[0][k].imshow(X_train_p[choice[k, ]]) #, interpolation = 'nearest', \
#     #cmap = 'gray')
#     axes[1][k].set_axis_off()
#     axes[1][k].imshow(result[choice[k, ]]) #, interpolation = 'nearest', cmap = 'gray')


# # In[ ]:


# #append rotated images to the train set
# X_train_p = np.append(X_train_p,result,axis=0)
# y_train_p = np.append(y_train_p,y_train_p)


# In[10]:


from skimage import io
from skimage import transform as trans

#iamge shear
images = tf.placeholder(tf.float32, (None, 32, 32, COLOR_CHANNELS)) #tf.placeholder(tf.float32, [None, input_size])

afine_tf = trans.AffineTransform(shear=0.2)
transformed = tf.contrib.image.matrices_to_flat_transforms(tf.linalg.inv(afine_tf.params))
transformed = tf.cast(transformed, tf.float32)
sheared_images = tf.contrib.image.transform(images, transformed)  # Image here is a tensor 


# rotated_images = tf.contrib.image.transform(
#     images,
#     tf.contrib.image.angles_to_projective_transforms(
#         random_angles, tf.cast(tf.shape(images)[1], tf.float32), tf.cast(tf
#             .shape(images)[2], tf.float32)
#     ))

sess = tf.Session()
sheared_images_result = sess.run(sheared_images, feed_dict = {
    images: X_train_p,
})


# #test shear
# fig, axes = plt.subplots(2, 10, figsize = (15, 4.5))
# choice = np.random.choice(range(len(X_train_p)), 10)
# for k in range(10):
#     axes[0][k].set_axis_off()
#     axes[0][k].imshow(X_train_p[choice[k, ]]) #, interpolation = 'nearest', \
#     #cmap = 'gray')
#     axes[1][k].set_axis_off()
#     axes[1][k].imshow(sheared_images_result[choice[k, ]]) #, interpolation = 'nearest', cmap = 'gray')



# #append rotated images to the train set
# X_train_p = np.append(X_train_p,sheared_images_result,axis=0)
# y_train_p = np.append(y_train_p,y_train_p)



#flip images

# flip up-down using np.flipud
# up_down = np.flipud(image)
# left_right = np.fliplr(image)

# plt.figure(figsize=(1,1))
# plt.imshow(image)
# plt.imshow(up_down)
# plt.imshow(left_right)
def flip_updown_img():
    fliped_updown = []
    for i in range(len(X_train_p)):
        fliped_updown.append(np.flipud(X_train_p[i]))

    len(fliped_updown)

    #test shear
    fig, axes = plt.subplots(2, 10, figsize = (15, 4.5))
    choice = np.random.choice(range(len(X_train_p)), 10)
    for k in range(10):
        axes[0][k].set_axis_off()
        axes[0][k].imshow(X_train_p[choice[k, ]]) #, interpolation = 'nearest', \
        #cmap = 'gray')
        axes[1][k].set_axis_off()
        axes[1][k].imshow(fliped_updown[choice[k, ]]) #, interpolation = 'nearest', cmap = 'gray')


#append rotated images to the train set
# X_train_p = np.append(X_train_p,fliped_updown,axis=0)
# y_train_p = np.append(y_train_p,y_train_p)


# In[ ]:


#flip images (left-right)

def flip_leftright_img():
    fliped_leftright = []
    for i in range(len(X_train_p)):
        fliped_leftright.append(np.fliplr(X_train_p[i]))

    len(fliped_leftright)

    #test shear
    fig, axes = plt.subplots(2, 10, figsize = (15, 4.5))
    choice = np.random.choice(range(len(X_train_p)), 10)
    for k in range(10):
        axes[0][k].set_axis_off()
        axes[0][k].imshow(X_train_p[choice[k, ]]) #, interpolation = 'nearest', \
        #cmap = 'gray')
        axes[1][k].set_axis_off()
        axes[1][k].imshow(fliped_leftright[choice[k, ]]) #, interpolation = 'nearest', cmap = 'gray')


#append rotated images to the train set
# X_train_p = np.append(X_train_p,fliped_leftright,axis=0)
# y_train_p = np.append(y_train_p,y_train_p)


# In[11]:


from skimage.transform import rescale
import random


def random_zoom(img):
    img_section = random.randrange(0, 4, 1)
    #rescale
    result = rescale(img, (2,2,1))
    if img_section == 0:
        result = result[0:32,0:32,:]
    elif img_section == 1:
        result = result[32:64,0:32,:]
    elif img_section == 2:
        result = result[0:32,32:64,:]
    else:
        result = result[32:64,32:64,:]
    return result

def random_zoom_small(img):
    img_section = random.randrange(0, 4, 1)
    #rescale
    result = rescale(img, (1.5,1.5,1))
    return result[8:40,8:40,:]

zoomed = []

# 2x zoom into random corner
# for i in range(len(X_train_p)):
#     zoomed.append(random_zoom(X_train_p[i]))

#1.5 zoom into the center
for i in range(len(X_train_p)):
    zoomed.append(random_zoom_small(X_train_p[i]))

# len(zoomed)

# #test zoom
# fig, axes = plt.subplots(2, 10, figsize = (15, 4.5))
# choice = np.random.choice(range(len(X_train_p)), 10)
# for k in range(10):
#     axes[0][k].set_axis_off()
#     axes[0][k].imshow(X_train_p[choice[k, ]]) #, interpolation = 'nearest', \
#     #cmap = 'gray')
#     axes[1][k].set_axis_off()
#     axes[1][k].imshow(zoomed[choice[k, ]]) #, interpolation = 'nearest', cmap = 'gray')


#append rotated images to the train set
# X_train_p = np.append(X_train_p,zoomed,axis=0)
# y_train_p = np.append(y_train_p,y_train_p)


# In[12]:


#zoomed and shear
# X_train_p2 = np.append(X_train_p,zoomed,axis=0)
# y_train_p2 = np.append(y_train_p,y_train_p)

# X_train_p = np.append(X_train_p2,sheared_images_result,axis=0)
# y_train_p = np.append(y_train_p2,y_train_p)

#zoomed, rotated and shear
X_train_p2 = np.append(X_train_p,zoomed,axis=0)
y_train_p2 = np.append(y_train_p,y_train_p)

X_train_p2 = np.append(X_train_p2,result,axis=0)
y_train_p2 = np.append(y_train_p2,y_train_p)

X_train_p = np.append(X_train_p2,sheared_images_result,axis=0)
y_train_p = np.append(y_train_p2,y_train_p)


#final check data 
# print(len(X_train_p))

# fig, axes = plt.subplots(2, 10, figsize = (15, 4.5))
# choice = np.random.choice(range(34799), 10)
# print(choice)
# for k in range(10):
#     axes[0][k].set_axis_off()
#     axes[0][k].imshow(X_train_p[choice[k,]]) #, interpolation = 'nearest', \
#     #cmap = 'gray')
#     axes[1][k].set_axis_off()
#     axes[1][k].imshow(X_train_p[choice[k,]+34799]) #, interpolation = 'nearest', cmap = 'gray')
    


# ### Model Architecture

# In[24]:


### Define your architecture here.
### Feel free to use as many code cells as needed.
def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    ## TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    # Weight and bias
    k_output = 6
    weight = tf.Variable(tf.truncated_normal( [5, 5, COLOR_CHANNELS, k_output], mean = mu, stddev = sigma ))
    bias = tf.Variable(tf.zeros(k_output))
    network = tf.nn.conv2d(x, weight, strides=[1, 1, 1, 1], padding='VALID') # Apply Convolution
    network = tf.nn.bias_add(network, bias) # Add bias
    # TODO: Activation.
    network = tf.nn.relu(network)
    #network = tf.nn.dropout(network, 0.5)
    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    network = tf.nn.max_pool(network, ksize= [1,2,2,1], strides= [1,2,2,1], padding= 'VALID')

    ## TODO: Layer 2: Convolutional. Output = 10x10x16.
    k_output = 16
    weight = tf.Variable(tf.truncated_normal( [5, 5, 6, k_output], mean = mu, stddev = sigma ))
    bias = tf.Variable(tf.zeros(k_output))
    network = tf.nn.conv2d(network, weight, strides=[1, 1, 1, 1], padding='VALID') # Apply Convolution
    network = tf.nn.bias_add(network, bias) # Add bias
    # TODO: Activation.
    network = tf.nn.relu(network)
    #network = tf.nn.dropout(network, 0.5)
    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    network = tf.nn.max_pool(network, ksize= [1,2,2,1], strides= [1,2,2,1], padding= 'VALID')
    # TODO: Flatten. Input = 5x5x16. Output = 400.
    #network   = flatten(network)
    
    ### TEST
    ## TODO: Layer 2.5: Convolutional. Output = 10x10x16.
    k_output = 100
    weight = tf.Variable(tf.truncated_normal( [2, 2, 16, k_output], mean = mu, stddev = sigma ))
    bias = tf.Variable(tf.zeros(k_output))
    network = tf.nn.conv2d(network, weight, strides=[1, 1, 1, 1], padding='VALID') # Apply Convolution
    network = tf.nn.bias_add(network, bias) # Add bias
    # TODO: Activation.
    network = tf.nn.relu(network)
    #network = tf.nn.dropout(network, 0.5)
    # TODO: Pooling. Input = 4x4x100. Output = 2x2x100.
    network = tf.nn.max_pool(network, ksize= [1,2,2,1], strides= [1,2,2,1], padding= 'VALID')
    # TODO: Flatten. Input = 2x2x100. Output = 400.
    network   = flatten(network)
    ### END TEST
    
    ## TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    k_output = 120
    weight = tf.Variable(tf.truncated_normal(shape=(400, k_output), mean = mu, stddev = sigma))
    bias = tf.Variable(tf.zeros(k_output))
    network   = tf.matmul(network, weight) + bias
    # TODO: Activation.
    network = tf.nn.relu(network)
    network = tf.nn.dropout(network, 0.5)

    ## TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    k_output = 84
    weight = tf.Variable(tf.truncated_normal(shape=(120, k_output), mean = mu, stddev = sigma))
    bias = tf.Variable(tf.zeros(k_output))
    network   = tf.matmul(network, weight) + bias
    # TODO: Activation.
    network = tf.nn.relu(network)
    #network = tf.nn.dropout(network, 0.5)

    ## TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    k_output = 43
    weight = tf.Variable(tf.truncated_normal(shape=(84, k_output), mean = mu, stddev = sigma))
    bias = tf.Variable(tf.zeros(k_output))
    network   = tf.matmul(network, weight) + bias
    return network


# In[ ]:


#test
# print(type(X_train_p[0][0][0][0]))
# logits = LeNet(tf.constant(X_train_p[0], shape = (1, 32, 32, COLOR_CHANNELS)))
# print(logits)


# ### Train, Validate and Test the Model

# A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
# sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.

# In[27]:


### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.

#EPOCHS = 10
#BATCH_SIZE = 128

x = tf.placeholder(tf.float32, (None, 32, 32, COLOR_CHANNELS))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

#train pipeline
rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

#evalation
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    total_loss = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy, loss = sess.run([accuracy_operation, loss_operation], feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
        total_loss += (loss * len(batch_x))
    return total_accuracy / num_examples, total_loss / num_examples


# -----------------------------------------------------------------------

valid_acc = []
train_acc = []

valid_loss = []
train_loss = []

#run training 

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train_p)
    
    print("Training...")
    print()
    total_time = 0
    for i in range(EPOCHS):
        start = time.time()
        X_train_p, y_train_p = shuffle(X_train_p, y_train_p)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train_p[offset:end], y_train_p[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
        
        end = time.time()
        epoch_time = end - start
        total_time += epoch_time
        
        train_accuracy, train_loss_val = evaluate(X_train_p, y_train_p)
        validation_accuracy, validation_loss = evaluate(X_valid_p, y_valid)
        #save for later analysis
        train_acc.append(train_accuracy)
        train_loss.append(train_loss_val)
        valid_acc.append(validation_accuracy)
        valid_loss.append(validation_loss)
#         print("EPOCH {} ...".format(i+1))
#         print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print("EPOCH {} ...".format(i+1), " - Validation Accuracy = {:.3f}".format(validation_accuracy)," time: {:.4f}s".format(epoch_time))
        print()
    
    print("Total training time: : {:.4f}".format(total_time))
    saver.save(sess, './lenet_signs')
    print("Model saved")


# In[26]:


# plot accuracy and loss

#%matplotlib notebook
# get_ipython().run_line_magic('matplotlib', 'inline')
# import matplotlib.pyplot as plt

# epochs = range(1, len(train_acc) + 1)

# line_train, = plt.plot(epochs, train_loss, 'ro', label ='Training loss')
# line_valid, = plt.plot(epochs, valid_loss, 'b', label ='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend(handles=[line_train, line_valid])


# plt.show()

# line_train, = plt.plot(epochs, train_acc, 'ro', label ='Training accuracy')
# line_valid, = plt.plot(epochs, valid_acc, 'b', label ='Validation accuracy')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend(handles=[line_train, line_valid])

# plt.show()


# ---
# 
# ## Step 3: Test a Model on New Images
# 
# To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.
# 
# You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

# ### Load and Output the Images

# In[ ]:


### Load the images and plot them here.
### Feel free to use as many code cells as needed.


# ### Predict the Sign Type for Each Image

# In[ ]:


### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.


# ### Analyze Performance

# In[ ]:


### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.


# ### Output Top 5 Softmax Probabilities For Each Image Found on the Web

# For each of the new images, print out the model's softmax probabilities to show the **certainty** of the model's predictions (limit the output to the top 5 probabilities for each image). [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. 
# 
# The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.
# 
# `tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.
# 
# Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. `tf.nn.top_k` is used to choose the three classes with the highest probability:
# 
# ```
# # (5, 6) array
# a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
#          0.12789202],
#        [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
#          0.15899337],
#        [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
#          0.23892179],
#        [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
#          0.16505091],
#        [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
#          0.09155967]])
# ```
# 
# Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:
# 
# ```
# TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
#        [ 0.28086119,  0.27569815,  0.18063401],
#        [ 0.26076848,  0.23892179,  0.23664738],
#        [ 0.29198961,  0.26234032,  0.16505091],
#        [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
#        [0, 1, 4],
#        [0, 5, 1],
#        [1, 3, 5],
#        [1, 4, 3]], dtype=int32))
# ```
# 
# Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.

# In[ ]:


### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.


# ### Project Writeup
# 
# Once you have completed the code implementation, document your results in a project writeup using this [template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) as a guide. The writeup can be in a markdown or pdf file. 

# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

# ---
# 
# ## Step 4 (Optional): Visualize the Neural Network's State with Test Images
# 
#  This Section is not required to complete but acts as an additional excersise for understaning the output of a neural network's weights. While neural networks can be a great learning device they are often referred to as a black box. We can understand what the weights of a neural network look like better by plotting their feature maps. After successfully training your neural network you can see what it's feature maps look like by plotting the output of the network's weight layers in response to a test stimuli image. From these plotted feature maps, it's possible to see what characteristics of an image the network finds interesting. For a sign, maybe the inner network feature maps react with high activation to the sign's boundary outline or to the contrast in the sign's painted symbol.
# 
#  Provided for you below is the function code that allows you to get the visualization output of any tensorflow weight layer you want. The inputs to the function should be a stimuli image, one used during training or a new one you provided, and then the tensorflow variable name that represents the layer's state during the training process, for instance if you wanted to see what the [LeNet lab's](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) feature maps looked like for it's second convolutional layer you could enter conv2 as the tf_activation variable.
# 
# For an example of what feature map outputs look like, check out NVIDIA's results in their paper [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) in the section Visualization of internal CNN State. NVIDIA was able to show that their network's inner weights had high activations to road boundary lines by comparing feature maps from an image with a clear path to one without. Try experimenting with a similar test to show that your trained network's weights are looking for interesting features, whether it's looking at differences in feature maps from images with or without a sign, or even what feature maps look like in a trained network vs a completely untrained one on the same sign image.
# 
# <figure>
#  <img src="visualize_cnn.png" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your output should look something like this (above)</p> 
#  </figcaption>
# </figure>
#  <p></p> 
# 

# In[ ]:


### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")

