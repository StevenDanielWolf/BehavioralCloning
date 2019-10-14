# Behavioral Cloning Project


Overview
---
This repository contains a trained Convolutional Neural Network for a Behavioral Cloning Project.

Using an End-to-End Deep Learning approach I trained a network to clone human driving behavior.

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously
around the track by executing `python drive.py model.h5`

The `model.py` file contains the code for training and saving the convolution neural network. The file
shows the pipeline I used for training and validating the model, and it contains comments to explain
how the code works.

[//]: # (Image References)

[image1]: ./examples/architecture.png "1"



Repository contents
---
*model.py* The script used to create and train the model
*drive.py* The script to drive the car
*model.h5* The model weights
*video.py* Script used to capture output video
*run1.mp4* Output video



The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

Model Architecture
---

My model is based on the Nvidia End-to-End CNN. It consists of five convolutional layers followed by
a flattening layer and four fully connected layers. The network uses RELU activation functions to
introduce nonlinearities to the model. The input to the model is normalized using a lamda layer and
cropped to conserve some computational power. The last fully connected layers condense the data
stream to a single logit output which represents the desired steering angle.

The overall strategy for deriving a model architecture was to take the Nvidia End-to-End CNN and fit
it to our needs by changing the input size. With a network based on convolutions, this can easily be
done.
In order to gauge how well the model was working, I split my image and steering angle data into a
training and validation set. I found that my first model had a low mean squared error on the training
set but a high mean squared error on the validation set. This implied that the model was overfitting.
To combat overfitting, I increase the amount of training data by adding more laps and
augmenting the data by flipping it left to right.
The final step was to run the simulator to see how well the car was driving around track one. The
model performed astoundingly well with only the few tweaks mentioned before
At the end of the process, the vehicle is able to drive autonomously around the track without leaving
the road.


The final model architecture consisted of a convolution neural network with the following layers and
layer sizes:
* Normalization
* Cropping
* 3 5x5 Kernel Convolutions
* 2 3x3 Kernel Convolutions
* Flattening
* 3 Fully connected layers

![1][image1]  


Training
---

To capture good driving behavior, I first recorded two laps on track one using center lane driving.
I then recorded the vehicle recovering from the left side and right sides of the road back to center so
that the vehicle would learn to steer back to the middle.
To augment the data sat, I also flipped images and angles thinking that this would take out any left
turn biases from the circular race track. For example, here is an image that has then been flipped:
After the collection process, I had 58.950 number of data points. I then preprocessed this data by
normalizing in and then cropping off a top and bottom slice that didn’t contain useful information.
I finally randomly shuffled the data set and put 20% of the data into a validation set.
I used this training data for training the model. The validation set helped determine if the model was
over or under fitting. The ideal number of epochs was 5 as evidenced by trial and error. I used an
adam optimizer so that manually training the learning rate wasn't necessary.
