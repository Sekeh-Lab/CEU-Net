##########
#Author: Nicholas Soucy
#For: CEU-Net

#Purpose: This file contains the code for the
#feature reduction methods used in the paper:
#PCA, 2DCAE, and 3DCAE.
##########



#imports
import numpy as np
import keras
from tqdm import trange
import keras.utils
from keras.utils.np_utils import to_categorical
from keras import layers
from sklearn.model_selection import ShuffleSplit
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io as io
import pandas as pd
import time
import random
import keras.backend as K
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers import Conv2DTranspose, Add, Input, Concatenate, Layer, SeparableConv2D
from keras.layers import Dense, Activation, BatchNormalization, Dropout, LeakyReLU, Conv2D, Reshape
from keras.layers import Conv3D, Conv3DTranspose, PReLU, BatchNormalization, MaxPool3D, Flatten
from sklearn.decomposition import PCA
import sklearn.preprocessing as sp
import cv2
from operator import truediv


# Feature Reduction

# PCA

def PCA_DR(X, numComponents=30, return_model=False):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    if return_model == True:
        return newX, pca
    else:
        return newX


# 2D CAE

def CAE_2D(X, Y, num_epochs=25, dim=30):
    # reshape data
    x = np.reshape(X, (-1, X.shape[2]))
    y = Y

    dim_x, dim_y = x.shape
    # print(dim_x, dim_y)


    # build convolutional autoencoder
    x_new_new = x.reshape(x.shape[0], 1, 1, x.shape[1])

    input_img = keras.Input(shape=(1, 1, dim_y))

    # Reduce to 30

    x = layers.Conv2D(dim_y, (1, 1), activation='sigmoid', padding='same')(input_img)
    x = layers.MaxPooling2D((1, 1), padding='same')(x)
    x = layers.Conv2D(60, (1, 1), activation='sigmoid', padding='same')(x)
    x = layers.MaxPooling2D((1, 1), padding='same')(x)
    x = layers.Conv2D(dim, (1, 1), activation='sigmoid', padding='same')(x)
    encoded = layers.MaxPooling2D((1, 1), padding='same')(x)

    x = layers.Conv2D(dim, (1, 1), activation='sigmoid', padding='same')(encoded)
    x = layers.UpSampling2D((1, 1))(x)
    x = layers.Conv2D(60, (1, 1), activation='sigmoid', padding='same')(x)
    x = layers.UpSampling2D((1, 1))(x)
    x = layers.Conv2D(dim_y, (1, 1), activation='sigmoid')(x)
    x = layers.UpSampling2D((1, 1))(x)
    decoded = layers.Conv2D(dim_y, (1, 1), activation='sigmoid', padding='same')(x)

    autoencoder = keras.Model(input_img, decoded)
    # autoencoder.summary()
    autoencoder.compile(loss=keras.losses.MSE, optimizer=Adam(learning_rate=0.0001), metrics=['MSE'])

    cae = autoencoder.fit(x_new_new, x_new_new,
                          epochs=num_epochs,
                          batch_size=256,
                          verbose=1)

    plt.plot(cae.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    encoder = keras.Model(input_img, encoded)
    x_combined = encoder.predict(x_new_new)
    x_final = np.reshape(x_combined, (X.shape[0], X.shape[1], dim))

    return x_final, y


# 3D CAE

def CAE_3D(X, Y, num_epochs=25, dim=30):
    # reshape data
    x = np.reshape(X, (-1, X.shape[2]))
    y = Y

    dim_x, dim_y = x.shape

    # build convolutional autoencoder
    x_new_new = x.reshape(x.shape[0], 1, 1, 1, x.shape[1])

    input_img = keras.Input(shape=(1, 1, 1, dim_y))

    # reduce bands to 30
    x = layers.Conv3D(dim_y, (1, 1, 1), activation='sigmoid', padding='same')(input_img)
    x = layers.MaxPooling3D((1, 1, 1), padding='same')(x)
    x = layers.Conv3D(60, (1, 1, 1), activation='sigmoid', padding='same')(x)
    x = layers.MaxPooling3D((1, 1, 1), padding='same')(x)
    x = layers.Conv3D(dim, (1, 1, 1), activation='sigmoid', padding='same')(x)
    encoded = layers.MaxPooling3D((1, 1, 1), padding='same')(x)

    x = layers.Conv3D(dim, (1, 1, 1), activation='sigmoid', padding='same')(encoded)
    x = layers.UpSampling3D((1, 1, 1))(x)
    x = layers.Conv3D(60, (1, 1, 1), activation='sigmoid', padding='same')(x)
    x = layers.UpSampling3D((1, 1, 1))(x)
    x = layers.Conv3D(dim_y, (1, 1, 1), activation='sigmoid')(x)
    x = layers.UpSampling3D((1, 1, 1))(x)
    decoded = layers.Conv3D(dim_y, (1, 1, 1), activation='sigmoid', padding='same')(x)

    autoencoder = keras.Model(input_img, decoded)
    # autoencoder.summary()
    autoencoder.compile(loss=keras.losses.MSE, optimizer=Adam(learning_rate=0.0001), metrics=['MSE'])

    cae = autoencoder.fit(x_new_new, x_new_new,
                          epochs=num_epochs,
                          batch_size=256,
                          verbose=1)

    plt.plot(cae.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    encoder = keras.Model(input_img, encoded)
    x_combined = encoder.predict(x_new_new)
    x_final = np.reshape(x_combined, (X.shape[0], X.shape[1], dim))

    return x_final, y