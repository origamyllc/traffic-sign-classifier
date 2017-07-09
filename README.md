#**Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


You're reading it! and here is a link to my [project code](https://github.com/origamyllc/traffic-sign-classifier/blob/master/classifier.ipynb)

### Data Set Summary & Exploration

The Dataset consists of the German traffic signs with different brightness,distortion etc and can be visualised here 

![Alt text](https://github.com/origamyllc/traffic-sign-classifier/blob/master/Screen%20Shot%202017-07-09%20at%201.34.37%20PM.png)
I used numpy to do a expoloratory data analysis 
Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![Alt text](https://github.com/origamyllc/traffic-sign-classifier/blob/master/Screen%20Shot%202017-07-09%20at%205.18.44%20AM.png)

### Design and Test a Model Architecture

 As a first step, I decided to convert the images to grayscale because this reduced the no of color channels to process to 1 then I normalized the image to smoothen the gradient this helps in better classification as images are more uniform 
while studying the data distribution in the histogram I realised that the data was not evenly distributed and this was causing the output result to be skewed towards traffic signs with higher frequency of occurence keeping this in mind I augmented the data for more uniformity for this purpouse I calculatefd the mean distribution and augmented images by rotating them and adding them back to the training set this gave me the below distribution 
The difference between the original data set and the augmented data set is the following ... 

![Alt text](https://github.com/origamyllc/traffic-sign-classifier/blob/master/Screen%20Shot%202017-07-09%20at%205.18.56%20AM.png)

####  Model Architecture

My final model consisted of the following layers:



| Layer         	|     Description        					| 
|:---------------------:|:---------------------------------------------:| 
| Convolution      	| 1x1 stride, same padding, outputs 32x32x64 	|
| Relu				| Input = 400. Output = 120   	
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 	    | Output = 10x10x16.      									|
| Relu				| Output = 10x10x16. 	
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

Epoch, learning rate, batch size, and drop out probability were all parameters tuned along with the number of random modifications to generate more image data was tuned. For Epoch the main reason I tuned this was after I started to get better accuracy early on I lowered the number once I had confidence I could reach my accuracy goals. The batch size I increased only slightly since starting once I increased the dataset size. The learning rate I think could of been left at .001 but I just wanted to try something different so .0005 was used

I chose the following hyper parameters 

EPOCHS = 10
BATCH_SIZE = 150
mu = 0
sigma = 0.1
learning rate = 0.005

this gave me 
Validation Accuracy = 0.965
Test Accuracy: 0.9072842597961426

this didnt seem right to me so I tweaked the model some to use 
EPOCHS = 27
BATCH_SIZE = 156
rate = 0.00097

this gave me 
Validation Accuracy = 0.990
Test Accuracy: 0.9356294274330139

### Test a Model on New Images
The given input images from the training set have the following qualities 

• Blurriness
• Noisiness
• Poor quality of the images
• Poor contrast with the background

some of the images are blurry and noisy causing image distortion this image distortion remains even if the images are converted to grey scale the normalization of the images helps in reducing the bluriness and noisiness of the image
normalize in such a way that the min value of dst is alpha and max value of dst is beta. :normalize does its magic using only scales and shifts (i.e. adding constants and multiplying by constants).

Here are the results of the prediction:
the performance on the new images when compared to the accuracy results of the test set seems to be low this may happen because there is not enough data in the originam dataset having the same charecteristics as the new image used the difference in error indicates that some charecteristics like 

• Blurriness
• Noisiness
• Poor quality of the images
• Poor contrast with the background

need to have enough variation in them so that every possible guess is correcly made . so more data with more variation might do the trick 

![Alt text](https://github.com/origamyllc/traffic-sign-classifier/blob/master/Screen%20Shot%202017-07-09%20at%205.35.15%20AM.png)
