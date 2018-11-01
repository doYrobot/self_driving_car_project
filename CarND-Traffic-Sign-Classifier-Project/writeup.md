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

[image1]: ./cnn.png "cnn"
[image2]: ./web_image/web.png "tranffic sign"
[image3]: ./top5.png "tranffic sign"

./web_image/1.jpg
## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.


I used the numpy library to calculate summary statistics of the traffic
signs data set:
```python

Number of training examples = 34799
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43


```

#### 2. Include an exploratory visualization of the dataset.

```python
print('Label Counts: {}'.format(dict(zip(*np.unique(y_train, return_counts=True)))))
>>>
Label Counts: {0: 180, 1: 1980, 2: 2010, 3: 1260, 4: 1770, 5: 1650, 6: 360, 7: 1290, 8: 1260, 9: 1320, 10: 1800, 11: 1170, 12: 1890, 13: 1920, 14: 690, 15: 540, 16: 360, 17: 990, 18: 1080, 19: 180, 20: 300, 21: 270, 22: 330, 23: 450, 24: 240, 25: 1350, 26: 540, 27: 210, 28: 480, 29: 240, 30: 390, 31: 690, 32: 210, 33: 599, 34: 360, 35: 1080, 36: 330, 37: 180, 38: 1860, 39: 270, 40: 300, 41: 210, 42: 210}
```


### Design and Test a Model Architecture

#### 1. Preprocessed the image data.

```py
def normalize(x):
    return (x - 128.) / 128.

X_train = normalize(X_train)
X_valid = normalize(X_valid)
X_test = normalize(X_test)

X_train, y_train = shuffle(X_train, y_train)
```




#### 2.Model architecture.

My final model consisted of the following layers:

![alt text][image1]


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

**hyperparameters**

* rate : 0.0008
* EPOCHS : 50
* BATCH_SIZE : 256
```py
logits,conv1,conv2 = LeNet(x,is_training,keep_prob)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
```

#### 4. Approach taken for finding a solution and getting the validation set accuracy to be at least 0.93.

My final model results were:
*  Training set accuracy = 1.000
* Validation set accuracy = 0.950
* Test set accuracy = 0.944


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image2]


#### 2. Compare the results to predicting on the test set.

Here are the results of the prediction:



| Image			        |     Prediction	        					| Correct	    |
|:---------------------:|:---------------------:|:--------------------------------:|
| Misclassified image at index 0. | Predicted: Speed limit (100km/h) | Correct: Speed limit (50km/h) |
| Correctly classified image at index 1.|  Predicted: Keep left | Correct: Keep left |
| Correctly classified image at index 2. | Predicted: No entry  | Correct: No entry |
| Misclassified image at index 3. |  Predicted: Speed limit (80km/h) |  Correct: Speed limit (60km/h) |
| Correctly classified image at index 4.|  Predicted: Priority road |  Correct: Priority road |
| Correctly classified image at index 5. | Predicted: Turn left ahead |  Correct: Turn left ahead |

Accuracy of test images: 67%

#### 3. The top 5 softmax probabilities for each image along with the sign type of each probability.

![alt text][image3]
The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.



