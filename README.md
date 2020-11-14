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
[image_2]: ./results_charts/2_dataset_augmentation_rotate.png "Dataset Augmentation - Rotate"
[image_3]: ./results_charts/3_dataset_augmentation_shear.png "Dataset Augmentation - Shear"
[image_4]: ./results_charts/4_dataset_augmentation_zoom.png "Dataset Augmentation - Zoom"

[image_5]: ./results_charts/acc_loss_10epoch_128batch_2xconv.png "Experiment Results - Initial Results"
[image_6]: ./results_charts/acc_loss_60epoch_64batch.png "Experiment Results - Overfitting issues"
[image_9]: ./results_charts/acc_loss_120epoch_128batch_zoom1_5center_shear_rotate_3xconv_1xdropout_first_dense_grayscale.png "Experiment Results - Final version"


[image_7]: ./results_charts/5_dataset_new_test_signs.png "New Test Signs"

[image_8]: ./results_charts/6_dataset_test_prediction_top5.png "New test signs predictions, top 5 picks per each"





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

Prior to applying augmentations I did normalizaiton of data (all values between 0,1). Next, I started off testing each augmentation seperatly, one by one. Initially the reasults were not particularly good, I did not get much improvement with **rotation** applying (+45,-45) random rotations to all training set images, likewise **fliping** did not yield any significant improvements in accuracy. 

![alt text][image_2]
**Image Augmentation - Rotate**

The first major improvements came with appying **shear** (by a small 0.2 factor). 

![alt text][image_3]
**Image Augmentation - Shear**

The **zoom**, I initially implemented 2x and randomly selecting a subsection of the image (ie. divided into 4 equal squares and selected one on random per training set image) - that did not provide any gains in accuracy. Subsequently, I tested a smaller zoom (1.5x) and into the center of the image. That gave better results and further helped to increase accuracy.

![alt text][image_4]
**Image Augmentation - Zoom**

Finally, I tested simplyfing the input by turning it from color to grayscale, that did not seem to improve accuracy in a significant way however it did improve execution time, therefore I kept the grayscale transformation.

Concluding the data augmentation experiments, I went back to revised the initial experiments with unsuccessful augmentations modyfing parameters of transformations. For rotation I tried a smaller angle (+20, -20), which turned out to give some accuracy gains as well. In addition, I experimented with all sort of combinations of aforementioned augmentations. **The ones that proved best jointly turned out to be: rotation (+20.-20), shear (0.2) and zoom (1.5x) applied to grayscale images.**


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model did not differ much from LeNet-5 implemented during class. Only **modifications were**: to accomodate **different input image size**; adding **one more convolutional layer** following the initial ones from LeNet-5; adding **dropout** after the first dense layer. The final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution (1) 5x5  	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution (2) 5x5   | 1x1 stride, same padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16	  				|
| Convolution (3) 2x2   | 1x1 stride, same padding, outputs 4x4x100		|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 2x2x100	  				|
| Flatten	      		| Output size 400					  			|
| Fully connected		| Output size 120        						|
| RELU					|         										|
| Dropout         		| Set to 0.5 for training   					| 
| Fully connected		| Output size 84        						|
| RELU					|												|
| Fully connected		| Output size 43        						|
 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The main experiments I made were with number of epochs and batch size. I tried with 64, 128 and 256 batch sizes; and 10,20, 60 and 120 epochs. The final configuration was 128 and 120 epochs.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* Training set accuracy: 0.996
* Validation set accuracy: 0.970
* Test set accuracy: 0.954

Aside of basic normalization, my first modifications to the course lesson project did not involve data pre-processing at all. I started off with hyperparameters like epoch/batch size as those are easier to modify. 

I tried to modify the network hyper-parameters, architecture and data pre-processing iterativly. Below is a teble presenting experiments done (modifications to prior experiment shown in bold).

| No 	| Experiment Description       																					| accuracy 		| 
|:-----:|:--------------------------------------------------------------------------------------------------------------|:-------------:| 				
| 1 	| LeNet-5 with 10epoch / 128batch  																				| 0.906			|
| 2 	| 10 epoch / 128batch / **+1conv (3x conv)**  																	| 0.861			|
| 3 	| **20 epoch** / 128batch / +1conv (3x conv)  																	| 0.920			|
| 4 	| 20 epoch / **256batch** / +1conv (3x conv)  																	| 0.887			|
| 5 	| **60 epoch** / 256batch / +1conv (3x conv)  																	| 0.912			|
| 6 	| 60 epoch / **128batch** / +1conv (3x conv)  																	| 0.914			|
| 7 	| 60 epoch / 128batch / **(2x conv)**  																			| 0.922			|
| 8 	| 60 epoch / **64batch** / (2x conv)  																			| 0.920			|
| 9 	| 60 epoch / 64batch / **dropout** / **(3x conv)**		 														| 0.934			|
| 10 	| 60 epoch / 64batch / dropout / (3x conv) / **rotate** 														| 0.952			|
| 11 	| 60 epoch / 64batch / dropout / (3x conv) / **shear** 															| 0.940			|
| 12 	| 60 epoch / 64batch / dropout / (3x conv) / **flip l-r** 														| 0.899			|
| 13 	| 60 epoch / 64batch / dropout / (3x conv) / **flip u-d** 														| 0.926			|
| 14 	| 60 epoch / 64batch / dropout / (3x conv) / **zoom(2x)** 														| 0.868			|
| 15 	| 60 epoch / 64batch / dropout / (3x conv) / **zoom(1.5x)**														| 0.942			|
| 16 	| 60 epoch / 64batch / dropout / (3x conv) / **zoom(1.5x)	/ shear** 											| 0.954			|
| 17 	| 60 epoch / 64batch / **2x dropout** / (3x conv) / zoom(1.5x)	/ shear 										| 0.920			|
| 18 	| **120 epoch** / **128batch** / **dropout (first dense)** / (3x conv) / zoom(1.5x)	/ shear 					| 0.951			|
| 19 	| 120 epoch / 128batch / dropout (first dense) / (3x conv) / zoom(1.5x)	/ shear / **grayscale** 				| 0.946			|
| 20 	| 120 epoch / 128batch / dropout (first dense) / (3x conv) / zoom(1.5x)	/ shear / grayscale / **rotate** 		| **0.970**		|
| 21 	| 120 epoch / 128batch / dropout (first dense) / (3x conv) / zoom(1.5x)	/ shear / **color** / rotate 			| 0.953			|

**\*Related chart for loss and accuracy per each epoch for above experiments can be found [here](./results_charts)**

The initial results using LeNet-5 in unchanged form gave as expected accuracy around 0.9.

![alt text][image_5]
**Experiment (1) - first run with LeNet-5**

Initially increasing epoch size and later on adding extra convolutional layer seemed a good idea but when looking at loss chart it could be observed those modifications introduced quite significant overfitting.

![alt text][image_6]
**Experiments (2) - 60 epochs and 64 batch size. Increasing epoch count seemingly improved accuracy but increasing loss is sign of overfitting**

Adding drop-out seemed to have mitigated the problem. However the placement of drop-out later was not without meaning. When placed closer to the end of the network or when using more than single dropout the accuracy decreased significantly.

Picking the right data augmentation turned to be quite key as well as selecting the right parameters for augmentation (e.g. fliping did not seem to work particularly well perhaps due to some signs having differnt meaning when fliped, likewise rotation had to be tuned down).

![alt text][image_9]
**Experiments (20) - final version: input as grayscale, 120 epochs, 128batch size, dropout after first dense layer, 3 convolutional layers, training augmented with rotate, shear and zoom.**

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web (taken from https://github.com/Goddard/udacity-traffic-sign-classifier , to compare my work against other students):

![alt text][image_7]
**New test signs downloaded from the web**

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        	|     Prediction	        					| 
|:-------------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)  	| Speed limit (30km/h)							| 
| Bumpy road     			| Bumpy road 									|
| Ahead only				| Ahead only									|
| No vehicles	      		| No vehicles					 				|
| Go streight or left		| Go streight or left      						|
| General caution			| General caution      							|


The model was able to correctly guess all traffic signs, which gives an accuracy of 100%. Worth nothing however, there were no limitataion against the downlaoded traffic signs to the used images seem quite streightforward to classify. Potential future work could be to test with more challenging cases.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The top 5 picks and respectiev probabilities can be seen below. Model quite confidently (and accuractly) classified all test cases.

![alt text][image_8]
**Predictions for new test signs - top 5 per each sign**


#### Final Notes

Overall the bulk time spent in this project was on small iterative changes. Most of the changes were based on knowledge of deep learning networks and typical directions for improvement suggestion during the course.

The changes related to analysis of dataset were not major and could be potential subject of future work (e.g. class balancing or using more challenging images for testing).

Furthermore, I did not go much beyond the regular LeNet-5 architecture, this could also be subject of potential future work.

Those directions for improvement were not explored due to project deadline, normally I would consider this model as first cut and continue experiments far beyond it based on analysis of dataset and experimenting with other state of the art architecture as well as testing new architectures.

From a technical perspective, it was interesting to see the boost in calculation that can be achieved when using a discreet GPU. My first experiments with 10-20 epochs were done on regular laptop with CPU only. However, when moving to 60 or 120 epochs, increasing input size and network architecture complexity, made iterative experiments become not feasible. Moving to even an old laptop with GTX 1050 sped up each epoch calculation by about 10x, making the entire process a lot easier!
