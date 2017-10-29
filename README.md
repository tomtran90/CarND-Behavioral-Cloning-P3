**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report



[//]: # (Image References)

[image1]: ./examples/1.png "Recovery Image"
[image2]: ./examples/2.png "Recovery Image"
[image3]: ./examples/3.png "Center Lane Image"

---
#### 1. Files Submitted 
My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Drive
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Model Pipeline

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.


### Model Architecture and Training Strategy

#### 1. Model Architecture

I used the NVIDIA Architecture that was introduced in the lecture. It works and it's not too complicated.

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 90, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 43, 158, 24)       1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 20, 77, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 8, 37, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 6, 35, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 4, 33, 64)         36928     
_________________________________________________________________
flatten_1 (Flatten)          (None, 8448)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               844900    
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 981,819.0
Trainable params: 981,819.0
Non-trainable params: 0.0
```

#### 2. Training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I aslo made sure to combat the cases where the car drove to the lake, to the sand, or hit the curve.


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving in autonomous mode:

![alt text][image3]

I then recorded it driving in the other direction to reduce over fitting. To augment the data, I also flipped images and angles so that we would have twice the number of records. I also recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from the sides. 

These images show what a recovery looks like in autonomous mode where the car has learned the behavior:

![alt text][image1]
![alt text][image2]

After training with the data above, my car drove into the lake and the sand and hit the curve at certain points of the track. So I had to create more data to train the model better at these points of the track (not the entire track). The car finally stayed on track and did not hit anything.

My final data set had 37,152 data points. I preprocessed this data by cropping the unnecessary part of the image.
I finally randomly shuffled the data set and put 20% of the data into a validation set. 
I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by both training and validation errors decreasing and plateauing over 3 epochs. I used an Adam optimizer so that manually training the learning rate wasn't necessary.
