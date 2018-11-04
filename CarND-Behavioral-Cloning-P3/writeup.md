# **Behavioral Cloning**



**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./model.png "Model Visualization"
[image2]: ./examples/center_2018_11_02_15_46_07_436.jpg "center"
[image3]: ./examples/left_2018_11_02_15_46_07_069.jpg "left"
[image4]: ./examples/right_2018_11_02_15_46_07_069.jpg "right"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter and 3x3 filter sizes and depths between 32 and 64 (model.py lines 121-137)

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 121).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 123).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 151). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, the learning rate was tuned manually.
```py
adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='mse', optimizer=adam)
```

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.
```py
for line in lines:
    source_path = line[0]
    image = ndimage.imread(source_path)
    images.append(image)
    left_source_path = line[1]
    right_source_path = line[2]

    left_image = ndimage.imread(left_source_path)
    right_image = ndimage.imread(right_source_path)

    images.append(left_image)
    images.append(right_image)

    measurement = float(line[3])
    measurements.append(measurement)
    left_measurement = measurement-correction
    right_measurement = measurement+correction

    measurements.append(left_measurement)
    measurements.append(right_measurement)
```

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to building an end-to-end neural network

My first step was to use a convolution neural network model similar to the Lenet-5 I thought this model might be appropriate because
it works well in traffic sign recognition.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting,I add Dropout layers.
```py
model.add(Dropout(0.5))
```


Then I flip images and take the opposite sign of the steering measurement
```py
def img_flip(image, measurement):
    image_flipped = np.fliplr(image)
    images.append(image_flipped)
    measurement_flipped = -measurement
    measurements.append(measurement_flipped)
```

The final step was to run the simulator to see how well the car was driving around track one.
There were a few spots where the vehicle fell off the track to improve the driving behavior in these cases, I collect  enough data

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 116-148) consisted of a convolution neural network with the following layers and layer sizes.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center.

![alt text][image3]
![alt text][image2]
![alt text][image4]

Then I repeated this process on track two in order to get more data points.

After the collection process, I had X number of data points.

I finally randomly shuffled the data set and put 10% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10.

I first used an adam optimizer,but loss is unchanged,Prove that the model is not trained.so I change learning rate .
```py
adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='mse', optimizer=adam)
```