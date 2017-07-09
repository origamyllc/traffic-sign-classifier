#**Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

You're reading it! and here is a link to my [project code](https://github.com/origamyllc/traffic-sign-classifier/blob/master/classifier.ipynb)

### Data Set Summary & Exploration
I used numpy to do a expoloratory data analysis 
Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

 As a first step, I decided to convert the images to grayscale because this reduced the no of color channels to process to 1 then I normalized the image to smoothen the gradient this helps in better classification as images are more uniform 
while studying the data distribution in the histogram I realised that the data was not evenly distributed and this was causing the output result to be skewed towards traffic signs with higher frequency of occurence keeping this in mind I augmented the data for more uniformity for this purpouse I calculatefd the mean distribution and augmented images by rotating them and adding them back to the training set this gave me the below distribution 
The difference between the original data set and the augmented data set is the following ... 

![alt text][image1]

####  Model Architecture

My final model consisted of the following layers:
| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:|   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Fully connected		|Input = 400. Output = 120    									|
| Relu				| Input = 400. Output = 120   									|
|	Dropout					|	Input = 400. Output = 120											|
| Fully connected		|Input = 120. Output = 84.  									|
| Relu				| Input = 120. Output = 84.								|
|	Dropout					|	Input = 120. Output = 84.								|
| Fully connected		|Input = 84. Output = 43.    									|

 
To train the model, I used an Adam optimizer Adam offers several advantages over the simple tf.train.GradientDescentOptimizer. Foremost is that it uses moving averages of the parameters (momentum);Simply put, this enables Adam to use a larger effective step size, and the algorithm will converge to this step size without fine tuning.

The main down side of the algorithm is that Adam requires more computation to be performed for each parameter in each training step (to maintain the moving averages and variance, and calculate the scaled gradient); and more state to be retained for each parameter (approximately tripling the size of the model to store the average and variance for each parameter)

I chose the following hyper parameters 
EPOCHS = 10
BATCH_SIZE = 150
mu = 0
sigma = 0.1


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

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
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


