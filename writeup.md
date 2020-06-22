# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/Dataset_Visualization.jpg "Dataset Visualization"
[image2]: ./writeup_images/example_images.png "Example Signs"
[image3]: ./writeup_images/original_image.jpg "Original Sign"
[image4]: ./writeup_images/augmented_image.jpg "Augmented Sign"
[image5]: ./writeup_images/preprocessed_image.jpg "Preprocessed Sign"
[image6]: ./test_examples/11_rigtoffway_atnextintersection_32x32x3.jpg
[image7]: ./test_examples/12_priority_road_32x32x3.jpg
[image8]: ./test_examples/13_yield.jpg
[image9]: ./test_examples/17_noentry_32x32x3.jpg
[image10]: ./test_examples/31_wildanimalscrossing_32x32x3.jpg
[image11]: ./test_examples/34_turn_left_ahead.jpg
[image12]: ./writeup_images/augmented_dataset.png
[image13]: ./writeup_images/softmax_probabillities.png
[image14]: ./writeup_images/mod_lenet_relu_augmentation.png


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

Here is a link to my [project code](https://github.com/28licaris/CarND-Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data data is distributed for each sign class.

![alt text][image1]
![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to augment my dataset to create more data for each sign class.Image augmentation is an important pre-processing step which helps expose the neural network to a wide variety of variations without having to collect and label more training data. My image augmentation functions are implemented in cell 5 and consist of the following techniques.

* 1. Rotation: +/- 10 degrees
* 2. Blur: kernel will be randomly chosen by random(0,1)
* 3. Shear: shear angle random -0.2 to 0.2
* 4. Translate: x, y +/- 2 pixels
* 5. Gamma Correction adjustment: 0.5 - 1.5

To augment the dataset I pick a random combination from 32 possible combinations. Ex: (1, 2, 3), (1, 2)
This chooses which functions to use for augmenting an image. Each function then randomly selects its
parameter. For example if the combo (1, 4) is selected the current image will be augmented
by rotating the image randomly between -10 and 10 degrees. The image will then be agumented by translating
the image in the x and y direction randomly between -2 and +2 pixels. I use a threshold of 2000 images per sign
class. So the augment_dataset() function will loop through every image in every sign class and create an augmented image until all sign classes have 2000 images for training.

Here is an example of an image before and after augmentation.
![alt text][image3]
![alt text][image4]

After augmenting the dataset I applied a preprocessing funciton which can be found in cell 4.
This function converts the image to grayscale and then normalizes the image between [0, 1].
This ensures that each pixel input into the neural network has a similar distribution. This
helps the model converge faster when training.

Here is an example of a traffic sign image after preprocessing.
![alt text][image5]

After data augmentation this is what the dataset looks like.
![alt text][image12]
 
#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

To come up with my final model I modified the existing convolution layers to make them deeper but I did not add any layers to the original LeNet model. Below is my final model:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image 						| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x20 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x20 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x90	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x90 				    |
| Flatten				| output = 2250        							|
| Fully connected		| output = 400        							|
| RELU					|												|
| Dropout				|											    |
| Fully Connected		| output = 120        							|        
| RELU					|												|
| Dropout				|											    |
| Fully Connected		| output = 43                                   |
| Softmax				|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.
To train my final model I used an iterative process. I adjusted my hyper parameters such as batch size, epoch, and learnning rate. I used the same optimizer as in the lab (Adam Optimizer). The final hyper parameters I used to train the model are below:

* Training Sample: 82469,
* EPOCHS: 25,
* BATCH_SIZE: 120,
* L_RATE: 0.001,
* BETA: 0.001,

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

After comparing the Original LeNet with L2 normalization and dropout with my modified LeNet which just consisted of adding more depth to the hidden layers i'm not sure this would be worth implementing. I did gain about 0.9% accuracy but each epoch took roughly 2 seconds longer. I think if this model were to be implemented in realtime I would have just used the original LeNet Model with L2 normalization and dropout.

My final model results for the architecture shown in 2:
* training set accuracy of 99.46% 
* validation set accuracy of 99.05%  
* test set accuracy of 96.09%

![alt text][image14]

#### What was the first architecture that was tried and why was it chosen?
First I chose to use the Original LeNet Model implemented in the lab previous to this project. I did this to get my pipeline working and to see what the architecture was capable of and to see how the hyper parameters effected the performance of the model. I used the original provided dataset with no image augmentation first. Below are the hyperparameters for the original architecture with no image augmentation that provided the best results:

* Training Samples: 34799,
* EPOCHS: 30,
* BATCH_SIZE: 128,
* L_RATE: 0.001,
* Training Accuracy: 0.9985631771027903,
* Training Loss: 0.005390240799801927,
* Validation Accuracy: 0.9349206346773506,
* Validation Loss: 0.40321589644265504,
* Test Accuracy: 0.9075217734700805,
* Test Loss: 0.8444955465251835

Next I decided to use the same architecture and hyper parameters but added in my image augmentation pipeline to see how this effects the performance of the model. My image augmentation pipeline loops through every image in each sign class and augments the image. This repeats until each sign class has 2000 images. Below are the training results using the original dataset with image augmentation added to each sign class.

* Training Samples: 82469,
* EPOCHS: 30,
* BATCH_SIZE: 128,
* L_RATE: 0.001,
* Training Accuracy: 0.9939856188385939,
* Training Loss: 0.01957852412566524,
* Validation Accuracy: 0.970974429896064,
* Validation Loss: 0.15089927786054547,
* Test Accuracy: 0.9197941410758622,
* Test Loss: 0.7054434086345814
 
As you can see the test accuracy improved slightly from 90.7% to 91.9%. Not as much as I would have expected but it is still an improvement.

#### What were some problems with the initial architecture?
The problem with this architecture is that there are no techiques implemented to prevent overfitting. You can see from the first run without augmentation there is a large discrepancy between the training accuracy and validation accuracy which can indicate overfitting.

#### How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I adjusted the original architecture by adding L2 normalization and droput in the fully connected layers to help prevent overfitting. After this I trained the model on the same dataset with augmentation, and got the following results:

* Training Samples: 82469,
* EPOCHS: 50,
* BATCH_SIZE": 128,
* L_RATE: 0.001,
* BETA: 0.001,
* Training Accuracy: 0.9834725775746038,
* Training Loss: 0.07407149015129047,
* Validation Accuracy: 0.9656761115087779,
* Validation Loss: 0.11709136493886789,
* Test Accuracy: 0.9511480603535215,
* Test Loss: 0.1915368856273751

Here we can see the training and validation accuracy does not have as large of a discrepancy indicating the model is not overfitting and generalizing better. This is also confirmed by a testing accuracy improvement from 91.9% to 95.1%.


#### Which parameters were tuned? How were they adjusted and why?
After getting the 95.1% accuracy with the original LeNet model with L2 normalization and dropout added, I modified the convolutional layers to make them deeper to see if I could get a better test accuracy. You can see the final architecture in part 2. I ended up getting 96% but the training obviously took longer because each layer had more depth. The only hyper parameters I tuned during this whole process where epochs, batch size, and learning rate. I did not mess with Beta which is the L2 normilaztion hyper parameter. Although this is a slight improvement from the previous architecuture, each epoch took about 2 seconds longer using the GPU's on my laptop graphics card.



#### What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

I decided not to try any other well known architectures but instead spend a lot of time with the LeNet by adjusting hyper paramters and also adjusting the depth of the LeNet model to help myself gain better intuition on the model. Next I would like to try implemening the GoogLeNet architecutre and train on this same dataset to compare the results. The concept of inception is pretty interesting to me and as I understand GoogLeNet is a good candidate for running in real time on an embedded system.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][image6] ![alt text][image7] ![alt text][image8] 
![alt text][image9] ![alt text][image10] ![alt text][image11]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			                            |     Prediction	        					| 
|:---------------------:                    |:---------------------------------------------:| 
| Right of Way at Intersection      		| Beware of ice/snow 						    | 
| Priority Road     			            | Priority Road									|
| Yield					                    | Yeild										    |
| No Entry	      		                    | No Entry				 				        |
| Wild Animal Crossing                      | Wild Animal Crossing                          |
| Turn Left Ahead			                | Turn Left Ahead      							|


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83%. This compares.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Beware of ice/snow (probability of 0.93) although this is the wrong prediction. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .93         			| Beware of ice/snow  							| 
| .06     				| Righ of way at next intersection				|
| .00					| Dangerous curve to the right					|
| .00	      			| Vehicles over 3.5 metric tons prohibited		|
| .00				    | Wild animal crossing     						|


For all 6 images here is what the softmax probabilities look like:
![alt text][image13]



