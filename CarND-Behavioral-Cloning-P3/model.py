#!encoding:utf-8
import csv
from scipy import ndimage
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda
from keras.layers import Conv2D
from keras.layers import MaxPool2D,Dropout
from keras.layers import Cropping2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.utils import plot_model

lines = []
#samples = [] # for generator

with open('../../self-driving_car_engineer/car_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        # samples.append(line) # for generator

correction = 0.25
images = []
measurements = []


def img_flip(image, measurement):
    image_flipped = np.fliplr(image)
    images.append(image_flipped)
    measurement_flipped = -measurement
    measurements.append(measurement_flipped)

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

    img_flip(image, measurement)
    img_flip(left_image, left_measurement)
    img_flip(right_image, right_measurement)

X_train = np.array(images)
y_train = np.array(measurements)

# train_samples, validation_samples = train_test_split(samples, test_size=0.2) # for generator

# def generator(samples, batch_size=32):
#     num_samples = len(samples)
#     while 1:  # Loop forever so the generator never terminates
#         shuffle(samples)
#         for offset in range(0, num_samples, batch_size):
#             batch_samples = samples[offset:offset+batch_size]
#             images = []
#             measurements = []

#             def img_flip(image, measurement):
#                 image_flipped = np.fliplr(image)
#                 images.append(image_flipped)
#                 measurement_flipped = -measurement
#                 measurements.append(measurement_flipped)

#             for line in batch_samples:
#                 source_path = line[0]
#                 image = ndimage.imread(source_path)
#                 images.append(image)
#                 left_source_path = line[1]
#                 right_source_path = line[2]

#                 left_image = ndimage.imread(left_source_path)
#                 right_image = ndimage.imread(right_source_path)

#                 images.append(left_image)
#                 images.append(right_image)

#                 measurement = float(line[3])
#                 measurements.append(measurement)
#                 left_measurement = measurement-correction
#                 right_measurement = measurement+correction

#                 measurements.append(left_measurement)
#                 measurements.append(right_measurement)

#                 img_flip(image, measurement)
#                 img_flip(left_image, left_measurement)
#                 img_flip(right_image, right_measurement)

#             # trim image to only see section with road
#             X_train = np.array(images)
#             y_train = np.array(measurements)
#             yield shuffle(X_train, y_train)


# # compile and train the model using the generator function

# train_generator = generator(train_samples, batch_size=64) # for generator
# validation_generator = generator(validation_samples, batch_size=64) # for generator

model = Sequential()
model.add(Lambda(lambda x: (x/255.0-0.5), input_shape=(160, 320, 3)))  # 归一化层
model.add(Cropping2D(cropping=((45,15),(0,0)))) # 对图像高处剪裁45,底部剪裁15

#lenet-5

model.add(Conv2D(filters=24,kernel_size=(5,5),padding='valid',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(filters=36, kernel_size=(5, 5), padding='valid', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(filters=48, kernel_size=(5, 5),padding='valid', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(Dropout(0.5))

model.add(Conv2D(filters=64, kernel_size=(3, 3),padding='valid', activation='relu'))
model.add(Dropout(0.5))

model.add(Flatten()) # 展开
model.add(Dense(units=1164,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=10, activation='relu'))
model.add(Dense(1)) # 全连接层

plot_model(model, to_file='model.png', show_shapes=True)

adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model.compile(loss='mse', optimizer=adam)
history_object=model.fit(X_train, y_train, validation_split=0.1,shuffle=True, epochs=10, batch_size=256)

# for generator
# history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*6,
# validation_data=validation_generator, nb_val_samples=len(validation_samples)*6, nb_epoch=10, verbose=1)

model.save('model.h5')


## plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
