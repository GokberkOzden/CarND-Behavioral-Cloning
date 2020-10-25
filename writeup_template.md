# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model_Final.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model_Final.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I have used the NVIDIA architecture lines (81-94) I have only added another Dense layer at the end to drop the number to 1. I have also added Dropout after last convolution (0.3)

During the my trials I did not change the main architecture.

#### 2. Attempts to reduce overfitting in the model

I trained and valited the datas with 7:3 ratio on line 18. I have also added Dropout in my architecture.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

I couldn't manage to drive the car very well on simulator. So I used the provided data.

PS: I also couldn't get the datas for more than one lap. How can I record data for different driving styles?

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

At first try I did not use the generator. I did not use the dropout. The car went off track during first turn.
Later I have tried different training set where I drive form left to right always. The response was a little better than center driving. So I tried increasing the epoch to see if it will work. Then I have added augmentation then dropout.

Biggest challange was the turn with dirt road. I couldn't get the car to stay on lane. And I thought it was mainly because my traingn data. So I changed to Udacity data. Later I have realized an error where I did not add 0.2 offset to my steering and I have also switched to generator at this point. 
After the run I have seen that the created model was working on the track.


At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:

##
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
##

#### 3. Creation of the Training Set & Training Process

I have started with center driving, but the vehicle went off track on first turn. So I tried driving left to right.
After that I have did couple of runs on the track where I did center driving, left to right driving and also one where I drive the track reverse. But I couldn't collect the data from different laps. Recording only got one lap data and I couldn't figure out how to merge them. So I use Udacity data.

I finally randomly shuffled the data set and put 30% of the data into a validation set. 

Before using the generator I tried up to 15 epoch. But with the generator I did 5 epoch and it took a lot of time. So I did not try any other number of epoch.
