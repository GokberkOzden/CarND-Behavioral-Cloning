import os
import csv
import numpy as np
import cv2




lines = [] 

with open('./data_fromopt/driving_log.csv') as csvfile: 
    reader = csv.reader(csvfile)
    next(reader, None) #this is necessary to skip the first record as it contains the headings
    for line in reader:
        lines.append(line)
images = []
measurements = []
for line in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = './data_fromopt/IMG/' + filename
        image = cv2.imread(current_path)
        image_flipped = np.fliplr(image)
        images.append(image_flipped)
        measurement = float(line[3])
        measurement_flipped = -measurement
        measurements.append(measurement_flipped)
X_train = np.array(images)
Y_train = np.array(measurements)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
#NVIDIA Architecture
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
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

model.compile(loss = 'mse', optimizer='adam')
model.fit(X_train, Y_train, validation_split = 0.3, shuffle = True, nb_epoch = 15)

model.save('model.h5')
exit()
## First Run with Nvidia --had to change last dense from 10 to 1. Got an error I don't understand.
    #epoch 1 loss:0.0113 - val_loss:0.0109 | epoch2 loss:0.0099 - val_loss:0.0122
    # Off track after bridge - I will add data augmentation as in the example
## Second Run with Nvidia - flipped image and measurement
    #epoch 1 loss:0.0111 - val_loss:0.0110 | epoch2 loss:0.0098 - val_loss:0.0108
    # Go way too right, off track at first curvature.
## Third Run -- epoch - 15
    # epoch 15 loss:0.0044 - val_loss:0.0126
    # Off track after bridge - bit better than first run
##Fourth Run - Udacity Data
    #epoch 15 loss: 0.0151 - val_loss: 0.0535
##Fifth Run - New Training Set
    #epoch 15 loss:0:0128 - val_loss:0.0953
##Sixth Run - Dropout with 5 epochs
    #epoch5 loss:0.0476 - val_loss:0.0774 Stuck on a bridge lets try 15 epoc -- 0.0224/0.0917 failed
    #epoch 15 no dropout loss: 0.0155 - val_loss 0.0930
    #Get rid of flipped images and measurements - 0.0138 - 0.0962