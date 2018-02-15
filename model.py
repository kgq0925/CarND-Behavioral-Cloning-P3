import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.misc
from keras.layers import Dense, Flatten, Lambda, Activation, MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.optimizers import Adam
from scipy.ndimage import rotate
from scipy.stats import bernoulli

number_of_epochs = 5
number_of_samples_per_epoch = 20032
number_of_validation_samples = 6400
learning_rate = 1e-4
activation_relu = 'relu'

# training data
DRIVING_LOG_FILE = './data/driving_log.csv'
IMG_PATH = './data/'
STEERING_COEFFICIENT = 0.2

def crop(image, top_percent, bottom_percent):
    return image[int(np.ceil(image.shape[0] * top_percent)):image.shape[0] - int(np.ceil(image.shape[0] * bottom_percent)), :]

def resize(image, new_dim):
    return scipy.misc.imresize(image, new_dim)

def random_flip(image, steering_angle, flipping_prob=0.5):
    head = bernoulli.rvs(flipping_prob)
    if head:
        return np.fliplr(image), -1 * steering_angle
    else:
        return image, steering_angle

def random_gamma(image):
    gamma = np.random.uniform(0.4, 1.5)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def random_shear(image, steering_angle, shear_range=200):
    rows, cols, ch = image.shape
    dx = np.random.randint(-shear_range, shear_range + 1)
    random_point = [cols / 2 + dx, rows / 2]
    pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
    pts2 = np.float32([[0, rows], [cols, rows], random_point])
    dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 6.0
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)
    steering_angle += dsteering

    return image, steering_angle

def preprocess_image(image, steering_angle, top_crop_percent=0.35, bottom_crop_percent=0.1,
                     resize_dim=(160, 320), do_shear_prob=0.9):
    head = bernoulli.rvs(do_shear_prob)
    if head == 1:
        image, steering_angle = random_shear(image, steering_angle)

    image = crop(image, top_crop_percent, bottom_crop_percent)

    image, steering_angle = random_flip(image, steering_angle)

    image = random_gamma(image)

    image = resize(image, resize_dim)

    return image, steering_angle

def get_image_files(batch_size=64):
    data = pd.read_csv(DRIVING_LOG_FILE)
    num_of_img = len(data)
    rnd_indices = np.random.randint(0, num_of_img, batch_size)

    image_files_and_angles = []
    for index in rnd_indices:
        rnd_image = np.random.randint(0, 3)
        if rnd_image == 0:
            img = data.iloc[index]['left'].strip()
            angle = data.iloc[index]['steering'] + STEERING_COEFFICIENT
            image_files_and_angles.append((img, angle))

        elif rnd_image == 1:
            img = data.iloc[index]['center'].strip()
            angle = data.iloc[index]['steering']
            image_files_and_angles.append((img, angle))
        else:
            img = data.iloc[index]['right'].strip()
            angle = data.iloc[index]['steering'] - STEERING_COEFFICIENT
            image_files_and_angles.append((img, angle))

    return image_files_and_angles

def generator(batch_size=64):
    while True:
        X_batch = []
        y_batch = []
        images = get_image_files(batch_size)
        for img_file, angle in images:
            raw_image = plt.imread(IMG_PATH + img_file)
            raw_angle = angle
            new_image, new_angle = preprocess_image(raw_image, raw_angle)
            X_batch.append(new_image)
            y_batch.append(new_angle)

        assert len(X_batch) == batch_size, 'len(X_batch) == batch_size should be True'

        yield np.array(X_batch), np.array(y_batch)

def cnn():
    # based on NVIDIA's "End to End Learning for Self-Driving Cars" paper
    model = Sequential()

    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))

    # starts with five convolutional and maxpooling layers
    model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2)))
    model.add(Activation(activation_relu))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
    model.add(Activation(activation_relu))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
    model.add(Activation(activation_relu))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
    model.add(Activation(activation_relu))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
    model.add(Activation(activation_relu))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Flatten())

    # Next, five fully connected layers
    model.add(Dense(1164))
    model.add(Activation(activation_relu))

    model.add(Dense(100))
    model.add(Activation(activation_relu))

    model.add(Dense(50))
    model.add(Activation(activation_relu))

    model.add(Dense(10))
    model.add(Activation(activation_relu))

    model.add(Dense(1))

    #model.summary()

    model.compile(optimizer=Adam(learning_rate), loss="mse", )

    return model

# create generators
train_gen = generator()
validation_gen = generator()

model = cnn()
model.fit_generator(train_gen,
                    samples_per_epoch=number_of_samples_per_epoch,
                    nb_epoch=number_of_epochs,
                    validation_data=validation_gen,
                    nb_val_samples=number_of_validation_samples,
                    verbose=1)

# finally save model
model.save('model.h5')