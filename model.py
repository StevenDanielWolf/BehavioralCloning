# Import statements
import csv
import cv2
import sklearn
from math import ceil
import numpy as np
from scipy import ndimage
from sklearn.model_selection import train_test_split

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D


# Input .csv file containing training data
lines = []
with open('./testdata/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


def generator(samples, batch_size=32):
    # Function outputs batches of training data
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('/')[-1]

                # Get center, left and right camera images from file
                image_c = ndimage.imread(batch_sample[0])
                image_l = ndimage.imread(batch_sample[1])
                image_r = ndimage.imread(batch_sample[2])

                # Get center, left and right steering angle from file
                angle_c = float(batch_sample[3])
                correction = 0.2
                angle_l = angle_c + correction
                angle_r = angle_c - correction

                # Append images and flipped images to array
                images.extend((image_c, image_l, image_r))
                images.extend((np.fliplr(image_c), np.fliplr(image_l), np.fliplr(image_r)))

                # Append steering angles and flipped steering angles to array
                angles.extend((angle_c, angle_l, angle_r))
                angles.extend((angle_c * -1.0, angle_l * -1.0, angle_r * -1.0))


            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)



# Set batch size
batch_size=32

# Split data into training and validation sets
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# Generate training and validation data sets
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)



### KERAS Convolutional Neural Network based on Nvidia's End-to-End CNN ###

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
model.fit_generator(train_generator, steps_per_epoch=ceil(len(train_samples)/batch_size),
            validation_data=validation_generator,
            validation_steps=ceil(len(validation_samples)/batch_size),
            epochs=5, verbose=1)

# Save model in .h5 file
model.save('model.h5')
print("model saved")



    
