# **Behavioral Cloning** 

# CarND Behavioral Cloning

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model was based on NVIDIA's "End to End Learning for Self-Driving Cars" paper, consists of 5 convolution neural network, 5 fully-connected layer and 5 max-pooling layers.

The data was normalized in the model using a Keras lambda layer.

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road

Use fit_generator API of the Keras library for training our model.

### Architecture and Training Documentation

#### 1. Solution Design Approach


#### 2. Final Model Architecture

|Layer(type)                      |Output Shape          |Param       |Connected to           |
|:-------------------------------:|:--------------------:|:----------:|:---------------------:|
|lambda_1 (Lambda)                |(None, 160, 320, 3)   |0           |lambda_input_1[0][0]   |
|convolution2d_1 (Convolution2D)  |(None, 80, 160, 24)   |1824        |lambda_1[0][0]         |          
|activation_1 (Activation)        |(None, 80, 160, 24)   |0           |convolution2d_1[0][0]  |          
|maxpooling2d_1 (MaxPooling2D)    |(None, 79, 159, 24)   |0           |activation_1[0][0]     |          
|convolution2d_2 (Convolution2D)  |(None, 40, 80, 36)    |21636       |maxpooling2d_1[0][0]   |          
|activation_2 (Activation)        |(None, 40, 80, 36)    |0           |convolution2d_2[0][0]  |          
|maxpooling2d_2 (MaxPooling2D)    |(None, 39, 79, 36)    |0           |activation_2[0][0]     |          
|convolution2d_3 (Convolution2D)  |(None, 20, 40, 48)    |43248       |maxpooling2d_2[0][0]   |          
|activation_3 (Activation)        |(None, 20, 40, 48)    |0           |convolution2d_3[0][0]  |          
|maxpooling2d_3 (MaxPooling2D)    |(None, 19, 39, 48)    |0           |activation_3[0][0]     |          
|convolution2d_4 (Convolution2D)  |(None, 19, 39, 64)    |27712       |maxpooling2d_3[0][0]   |          
|activation_4 (Activation)        |(None, 19, 39, 64)    |0           |convolution2d_4[0][0]  |          
|maxpooling2d_4 (MaxPooling2D)    |(None, 18, 38, 64)    |0           |activation_4[0][0]     |          
|convolution2d_5 (Convolution2D)  |(None, 18, 38, 64)    |36928       |maxpooling2d_4[0][0]   |          
|activation_5 (Activation)        |(None, 18, 38, 64)    |0           |convolution2d_5[0][0]  |          
|maxpooling2d_5 (MaxPooling2D)    |(None, 17, 37, 64)    |0           |activation_5[0][0]     |          
|flatten_1 (Flatten)              |(None, 40256)         |0           |maxpooling2d_5[0][0]   |          
|dense_1 (Dense)                  |(None, 1164)          |46859148    |flatten_1[0][0]        |          
|activation_6 (Activation)        |(None, 1164)          |0           |dense_1[0][0]          |          
|dense_2 (Dense)                  |(None, 100)           |116500      |activation_6[0][0]     |          
|activation_7 (Activation)        |(None, 100)           |0           |dense_2[0][0]          |          
|dense_3 (Dense)                  |(None, 50)            |5050        |activation_7[0][0]     |          
|activation_8 (Activation)        |(None, 50)            |0           |dense_3[0][0]          |          
|dense_4 (Dense)                  |(None, 10)            |510         |activation_8[0][0]     |          
|activation_9 (Activation)        |(None, 10)            |0           |dense_4[0][0]          |          
|dense_5 (Dense)                  |(None, 1)             |11          |activation_9[0][0]     |          

- Total params: 47,112,567
- Trainable params: 47,112,567
- Non-trainable params: 0

#### 2. Creation of the Training Set & Training Process

I used the training data provided by Udacity.

1. Applied random shear on it in order to help the car to navigate in the training track.
2. Applied crop operation on the sheared images in order to remove the extra objects included trees and sky.
3. Applied random flip to avoid excessive left turning.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting.

- The number of epochs was 5.
- The number of samples per epoch was 20032
- The number of validation samples was 6400
