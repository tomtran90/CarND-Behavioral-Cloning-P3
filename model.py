import csv
import cv2
import numpy as np
import sklearn
from random import shuffle

# read data in
data_dir = '../data/'
lines = []
with open(data_dir + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


# random split fro train & valid
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)


def augment(images, angles):
	'''
	augment image with flip
	'''
    augmented_images, augmented_measurements = [], []
    for image, angle in zip(images, angles):
        augmented_images.append(image)
        augmented_measurements.append(angle)
        augmented_images.append(cv2.flip(image, 1))
        augmented_measurements.append(angle*-1.0)
    X_train, y_train = np.array(augmented_images), np.array(augmented_measurements)
    return X_train, y_train


# define generator

def generator(samples, batch_size=64):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    source_path = batch_sample[i]
                    filename = source_path.split('\\')[-1]
                    current_path = data_dir + 'IMG/' + filename
                    image = cv2.imread(current_path)
                    images.append(image)
                measurement = float(batch_sample[3])
				#change angle for left and right
                angles.extend([measurement, measurement + 0.2, measurement - 0.2])           
            X_train, y_train = augment(images, angles)
            yield sklearn.utils.shuffle(X_train, y_train)




train_generator = generator(train_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=64)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D


def nvidia_model():
	'''
	define nvidia model
	'''
    model = Sequential()
    model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
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
    return model

model = nvidia_model()
model.compile(loss = 'mse', optimizer = 'adam')


# fit model

model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=1)


# save model

model.save('model.h5')
