# **Traffic Sign Recognition** 

## Udacity Project 3 Writeup

---

**Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

---

In comparison to course material the key code modificatoons / customizations are in:
* Image Augmentation (added rotation/fliping/zooming/shear)
* Hyper-parameters (epoch, batch size)
* Network architecture (added one more conv later and dropout)
* Experiments with various combinations of all above

**Summary:** The solution was trained/validated/tested on the provided dataset and additional test on 6 images downloaded form the web ( located in  'new_images' directory). The final accuracy achieved on validation is: **0.97**. The key milestones in achieving this (moving from the starting 0.85 accuracy of LeNet-5) were: increasing epoch count, adding image augmentation (the ones that worked for me where: rotate with small angle, zoom and shear). Moving to grayscale made small change but helped to speed up training. Finally dropout helped to mitigate overfiting that started to be quite visible after I incresed epoch count.



[//]: # (Image References)

[image_1]: ./results_charts/1_dataset_distribution.png "Dataset class distribution"


[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"


---

**Key files:**
* Traffic_Sign_Classifier.ipynb - final code for the project
* README.md - (this file) - writeup on project coding, challenges and possible improvements

Additional files:
* original_README.md - original project readme file with the description of problem to solve
* new_images/* - additional testing images downloaded from web
* Traffic_Sign_Classifier.py same code as Jupyter notebook but exported to .py and trimmed to run on server from console in background
* pretrained_model - pre-trained model (final version with paramters as set in the Jupyter notebook)
* results_charts/* - charts of loss/accuracy from various experiments

---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* **Training size:** 34,799
* **Validation size:** 4,410
* **Test size:** 12,630
* **Image shape:** 32x32x3, RGB (3 color channels)
* **Unique classes/labels count:** 43

#### 2. Include an exploratory visualization of the dataset.

Below chart is showing class distribution in training, validation and test sets. Some classes appear to be under-represented, however this trend seems to persist throughout all sets.

![alt text][image_1]

Throughout my exepriments I did not take any specific actions to remedy the above issue (e.g. balance dataset by removing over-represented cases; or add more augmented images of under-represented cases). However, this could be one potential next step for futher improvements.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

For image preprocessing at start I experimented with various augementations of train set images, I tested in following order:
* rotation
* flip up-down , flip left-right
* shear
* zoom

I started off testing each augmentation seperatly, one by one. Initially the reasults were not particularly good, I did not get much improvement with **rotation** applying (+45,-45) random rotations to all training set images, likewise **fliping** did not yield any significant improvements in accuracy. 

The first major improvements came with appying **shear** (by a small 0.2 factor). 

The **zoom**, I initially implemented 2x and randomly selecting a subsection of the image (ie. divided into 4 equal squares and selected one on random per training set image) - that did not provide any gains in accuracy. Subsequently, I tested a smaller zoom (1.5x) and into the center of the image. That gave better results and further helped to increase accuracy.

Finally, I tested simplyfing the input by turning it from color to grayscale, that did not seem to improve accuracy in a significant way however it did improve execution time, therefore I kept the grayscale transformation.

Concluding the data augmentation experiments, I went back to revised the initial experiments with unsuccessful augmentations modyfing parameters of transformations. For rotation I tried a smaller angle (+20, -20), which turned out to give some accuracy gains as well. In addition, I experimented with all sort of combinations of aforementioned augmentations. The ones that proved best jointly turned out to be: rotation (+20.-20), shear (0.2) and zoom (1.5x) applied to grayscale images.

Below can be seen some examples of the augmentations.

![alt text][image2]

As a last step, I normalized the image data because ...

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

My first modifications to the course lesson project did not involve pre-processing at all. I started off with hyperparameters like epoch/batch size as those are easier to modify. 

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


