##########
#Author: Nicholas Soucy
#For: CEU-Net

#Purpose: This file contains the code for the
#novel semantic segmentation methods used in the paper:
#Single U-Net and CEU-Net.
##########

#imports
import numpy as np
import keras
from tqdm import trange
import keras.utils
from keras.utils.np_utils import to_categorical
from keras import layers
from sklearn.model_selection import ShuffleSplit
import h5py
import spectral
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io as io
import pandas as pd
import time
import random
from sklearn.metrics import cohen_kappa_score as kappa
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import keras.backend as K
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers import Conv2DTranspose, Add, Input, Concatenate, Layer, SeparableConv2D
from keras.layers import Dense, Activation, BatchNormalization, Dropout, LeakyReLU, Conv2D, Reshape
from keras.layers import Conv3D, Conv3DTranspose, PReLU, BatchNormalization, MaxPool3D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import sklearn.preprocessing as sp
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
import cv2
from operator import truediv

#Patching Functions

def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
    if windowSize % 2 == 0:
        margin = int((windowSize) / 2)
    else:
        margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            if windowSize % 2 == 0:
                patch = zeroPaddedX[r - margin:r + margin, c - margin:c + margin]
            else:
                patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1
    return patchesData, patchesLabels


def Patch(data, height_index, width_index, ws=5):
    height_slice = slice(height_index, height_index + ws)
    width_slice = slice(width_index, width_index + ws)
    patch = data[height_slice, width_slice, :]

    return patch



#Semantic Segmentation Functions

# Single U-Net
# Unet based on: https://github.com/thatbrguy/Hyperspectral-Image-Segmentation

def UNet_Conv(data, gt, num_epochs=25, class_num=16, ws=10, return_all=False, folds=5):
    ## GLOBAL VARIABLES
    windowSize = ws  # Default 25

    X_SN, y_SN = createImageCubes(data, gt, windowSize=windowSize)
    train_patches = X_SN
    train_patches_gt = y_SN
    train_patches_gt = to_categorical(train_patches_gt, num_classes=class_num)

    # impliment cross validation

    k = folds  # number of k-folds
    acc_arr = [0] * k
    acc = 0

    ss = ShuffleSplit(n_splits=k, test_size=.15, random_state=0)

    train_i = [None] * k
    test_i = [None] * k

    ct = 0
    for train_index, test_index in ss.split(X=train_patches, y=train_patches_gt):
        train_i[ct] = train_index
        test_i[ct] = test_index
        ct += 1

    best_fold_acc = 0
    for q in range(k):
        x_train, x_test, y_train, y_test = train_patches[train_i[q]], train_patches[test_i[q]], train_patches_gt[
            train_i[q]], train_patches_gt[test_i[q]]

        class PixelSoftmax(Layer):
            """
            Pixelwise Softmax for Semantic Segmentation. Also known as
            4D Softmax in some sources. Applies Softmax along the last
            axis (-1 axis).
            """

            def __init__(self, axis=-1, **kwargs):
                self.axis = axis
                super(PixelSoftmax, self).__init__(**kwargs)

            def get_config(self):
                config = super().get_config().copy()
                return config

            def build(self, input_shape):
                pass

            def call(self, x, mask=None):
                e = K.exp(x - K.max(x, axis=self.axis, keepdims=True))
                s = K.sum(e, axis=self.axis, keepdims=True)
                return e / s

            def get_output_shape_for(self, input_shape):
                return input_shape

        class statsLogger(Callback):
            """
            Saving loss and accuracy details to an array
            """

            def __init__(self):
                self.logs = []

            def on_epoch_end(self, epoch, logs):
                logs['epoch'] = epoch
                self.logs.append(logs)

            def get_config(self):
                config = super().get_config().copy()
                return config

        input_shape = x_train.shape[1:]
        img = Input(shape=input_shape)

        x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv1_2', use_bias=False)(img)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)
        op1 = x

        x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv2_2', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)
        op2 = x

        x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv3_2', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)
        op3 = x

        x = Conv2DTranspose(256, kernel_size=(3, 3), strides=(1, 1), padding='same', name='deconv3', use_bias=False)(
            op3)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)
        x = Concatenate(axis=-1)([x, op2])

        x = Conv2DTranspose(128, kernel_size=(3, 3), strides=(1, 1), padding='same', name='deconv2', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)
        x = Concatenate(axis=-1)([x, op1])

        x = Conv2D(class_num, kernel_size=(3, 3), strides=(windowSize, windowSize), padding='same', name='deconv1')(x)
        x = Reshape((1, class_num))(x)

        x = PixelSoftmax(axis=-1)(x)
        model = Model(inputs=img, outputs=x)

        y_test = y_test.reshape(y_test.shape[0], 1, class_num)
        y_train = y_train.reshape(y_train.shape[0], 1, class_num)

        history = statsLogger()
        opt = Adam(learning_rate=0.0001, decay=1e-4)

        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'],
                      sample_weight_mode="temporal")

        hist = model.fit(x_train,
                         y_train,
                         batch_size=256,
                         epochs=num_epochs,
                         validation_data=(x_test, y_test))

        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

        acc_arr[q] = max(hist.history['val_accuracy'])

        if acc_arr[q] > best_fold_acc:
            best_x_train = x_train
            best_x_test = x_test
            best_y_train = y_train
            best_y_test = y_test
            best_model = model
            best_fold_acc = acc_arr[q]

    if (return_all):
        return (np.mean(acc_arr), best_x_train, best_x_test, best_y_train, best_y_test, best_model)
    else:
        return (np.mean(acc_arr))


# CEU-Net

# K-Means Ensemble UNet

def K_Ensemble_UNet(data, gt, ds, num_epochs=25, class_num=16, clusters=3, folds=5, ws=5, return_all=False):
    ## GLOBAL VARIABLES
    windowSize = ws  # Default 25

    def data_process(data, gt, tts='n'):

        X_SN, y_SN = createImageCubes(data, gt, windowSize=windowSize)
        train_patches = X_SN
        train_patches_gt = y_SN
        train_patches_gt = to_categorical(train_patches_gt, num_classes=class_num)

        x_train, x_test, y_train, y_test = train_test_split(train_patches, train_patches_gt, test_size=0.25,
                                                            random_state=42)
        if tts == 'y':
            return (x_train, x_test, y_train, y_test)
        else:
            return (train_patches, train_patches_gt)

    def UNet_Conv_mid(data, gt, num_epochs=25, class_num=16):
        x_train = data
        y_train = gt

        class PixelSoftmax(Layer):
            """
            Pixelwise Softmax for Semantic Segmentation. Also known as
            4D Softmax in some sources. Applies Softmax along the last
            axis (-1 axis).
            """

            def __init__(self, axis=-1, **kwargs):
                self.axis = axis
                super(PixelSoftmax, self).__init__(**kwargs)

            def get_config(self):
                config = super().get_config().copy()
                return config

            def build(self, input_shape):
                pass

            def call(self, x, mask=None):
                e = K.exp(x - K.max(x, axis=self.axis, keepdims=True))
                s = K.sum(e, axis=self.axis, keepdims=True)
                return e / s

            def get_output_shape_for(self, input_shape):
                return input_shape

        class statsLogger(Callback):
            """
            Saving loss and accuracy details to an array
            """

            def __init__(self):
                self.logs = []

            def on_epoch_end(self, epoch, logs):
                logs['epoch'] = epoch
                self.logs.append(logs)

            def get_config(self):
                config = super().get_config().copy()
                return config

        input_shape = x_train.shape[1:]
        img = Input(shape=input_shape)

        x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv1_2', use_bias=False)(img)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)
        op1 = x

        x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv2_2', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)
        op2 = x

        x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv3_2', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)
        op3 = x

        x = Conv2DTranspose(256, kernel_size=(3, 3), strides=(1, 1), padding='same', name='deconv3', use_bias=False)(
            op3)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)
        x = Concatenate(axis=-1)([x, op2])

        x = Conv2DTranspose(128, kernel_size=(3, 3), strides=(1, 1), padding='same', name='deconv2', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)
        x = Concatenate(axis=-1)([x, op1])

        x = Conv2D(class_num, kernel_size=(3, 3), strides=(windowSize, windowSize), padding='same', name='deconv1')(x)
        x = Reshape((1, class_num))(x)

        x = PixelSoftmax(axis=-1)(x)
        model = Model(inputs=img, outputs=x)

        y_train = y_train.reshape(y_train.shape[0], 1, class_num)

        filepath_name = "best-model.hdf5"
        history = statsLogger()
        opt = Adam(learning_rate=0.0001, decay=1e-4)

        ckpt = ModelCheckpoint(filepath=filepath_name,
                               save_best_only=True,
                               verbose=1,
                               monitor='loss')

        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'],
                      sample_weight_mode="temporal")

        hist = model.fit(x_train,
                         y_train,
                         batch_size=256,
                         epochs=num_epochs,
                         verbose=1)

        plt.plot(hist.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.show()

        train_acc = max(hist.history['accuracy'])

        return model, train_acc

    x_patched, Y_patched = data_process(data, gt)

    x_train_p, x_test_p, y_train_p, y_test_p = train_test_split(x_patched, Y_patched, test_size=0.15, random_state=42)
    y_train = y_train_p
    y_test = y_test_p
    x_train = x_train_p[:, 0, 0, :]
    x_test = x_test_p[:, 0, 0, :]

    # plt.scatter(data[:, 0], data[:, 1], cmap='viridis')
    # plt.title('Indian Pines')
    # plt.show()
    # Training
    K_model = KMeans(n_clusters=clusters)
    K_train_pred = K_model.fit_predict(x_train)
    centers = K_model.cluster_centers_
    # plt.scatter(x_train[:, 0], x_train[:, 1], c=K_train_pred, cmap='viridis')
    # plt.scatter(centers[:, 0], centers[:, 1], c='magenta', s=300, alpha=0.5)
    # plt.title('K-means Train')
    # plt.show()

    x_s = [None] * clusters
    y_s = [None] * clusters
    models = [None] * clusters
    accs = [None] * clusters

    for i in range(clusters):
        x_s[i] = np.zeros((x_train.shape[0], windowSize, windowSize, x_train.shape[1]))
        y_s[i] = np.zeros((y_train.shape[0], y_train.shape[1]))
        for j in range(y_train.shape[0]):
            if K_train_pred[j] == i:
                x_s[i][j] = x_train_p[j]
                y_s[i][j] = y_train_p[j]
        tmp_a = []
        for q in range(x_s[i].shape[0]):
            if np.all(x_s[i][q] == 0):
                tmp_a.append(q)
        x_s[i] = np.delete(x_s[i], tmp_a, 0)
        y_s[i] = np.delete(y_s[i], tmp_a, 0)

        models[i], accs[i] = UNet_Conv_mid(x_s[i], y_s[i], num_epochs=num_epochs, class_num=class_num)
        print("UNet Mid %2d Train Accuracy for Feature Count of %2d: %5.4f" % ((i + 1), 30, accs[i]))

    # Testing

    K_test_pred = K_model.predict(x_test)
    centers = K_model.cluster_centers_
    plt.scatter(x_test[:, 0], x_test[:, 1], c=K_test_pred, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='magenta', s=300, alpha=0.5)
    plt.title('K-means Test')
    plt.show()

    xt_s = [None] * clusters
    yt_s = [None] * clusters
    y_predicts = [None] * clusters

    for i in range(clusters):
        xt_s[i] = np.zeros((x_test.shape[0], windowSize, windowSize, x_test.shape[1]))
        yt_s[i] = np.zeros((y_test.shape[0], y_test.shape[1]))
        for j in range(y_test.shape[0]):
            if K_test_pred[j] == i:
                xt_s[i][j] = x_test_p[j]
                yt_s[i][j] = y_test_p[j]
        tmp_a = []
        for q in range(xt_s[i].shape[0]):
            if np.all(xt_s[i][q] == 0):
                tmp_a.append(q)
        xt_s[i] = np.delete(xt_s[i], tmp_a, 0)
        yt_s[i] = np.delete(yt_s[i], tmp_a, 0)

        y_predicts[i] = models[i].predict(xt_s[i])

    y_test_final_pred = y_predicts[0]
    y_test_final = yt_s[0]
    for i in range(1, clusters):
        y_test_final_pred = np.concatenate((y_test_final_pred, y_predicts[i]), axis=0)
        y_test_final = np.concatenate((y_test_final, yt_s[i]), axis=0)

    y_true = y_test_final.reshape(y_test_final.shape[0], y_test_final.shape[-1])
    y_pred = y_test_final_pred.reshape(y_test_final_pred.shape[0], y_test_final_pred.shape[-1])
    # print("y_true shape: ",y_true.shape)
    # print("y_pred shape: ",y_pred.shape)

    m = tf.keras.metrics.TopKCategoricalAccuracy(k=1)
    m.update_state(y_true, y_pred)
    # print("K-Means Clusters = %2d Ensemble UNet Test Accuracy for Feature Count of %2d: %5.4f" % (clusters,30,m.result().numpy()))

    best_y_pred = y_pred
    best_y_test = y_true

    if (return_all):
        return (m.result().numpy(), best_y_pred, best_y_test, K_model, models)
    else:
        return (m.result().numpy())


# GMM Ensemble UNet

def G_Ensemble_UNet(data, gt, ds, num_epochs=25, class_num=16, clusters=3, folds=5, ws=5, return_all=False):
    ## GLOBAL VARIABLES
    test_ratio = 0.75
    windowSize = ws  # Default 25

    def data_process(data, gt, tts='n'):

        X_SN, y_SN = createImageCubes(data, gt, windowSize=windowSize)
        train_patches = X_SN
        train_patches_gt = y_SN
        train_patches_gt = to_categorical(train_patches_gt, num_classes=class_num)

        x_train, x_test, y_train, y_test = train_test_split(train_patches, train_patches_gt, test_size=0.15,
                                                            random_state=42)
        if tts == 'y':
            return (x_train, x_test, y_train, y_test)
        else:
            return (train_patches, train_patches_gt)

    def UNet_Conv_mid(data, gt, num_epochs=25, class_num=16):
        x_train = data
        y_train = gt

        class PixelSoftmax(Layer):
            """
            Pixelwise Softmax for Semantic Segmentation. Also known as
            4D Softmax in some sources. Applies Softmax along the last
            axis (-1 axis).
            """

            def __init__(self, axis=-1, **kwargs):
                self.axis = axis
                super(PixelSoftmax, self).__init__(**kwargs)

            def get_config(self):
                config = super().get_config().copy()
                return config

            def build(self, input_shape):
                pass

            def call(self, x, mask=None):
                e = K.exp(x - K.max(x, axis=self.axis, keepdims=True))
                s = K.sum(e, axis=self.axis, keepdims=True)
                return e / s

            def get_output_shape_for(self, input_shape):
                return input_shape

        class statsLogger(Callback):
            """
            Saving loss and accuracy details to an array
            """

            def __init__(self):
                self.logs = []

            def on_epoch_end(self, epoch, logs):
                logs['epoch'] = epoch
                self.logs.append(logs)

            def get_config(self):
                config = super().get_config().copy()
                return config

        input_shape = x_train.shape[1:]
        img = Input(shape=input_shape)

        x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv1_2', use_bias=False)(img)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)
        op1 = x

        x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv2_2', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)
        op2 = x

        x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv3_2', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)
        op3 = x

        x = Conv2DTranspose(256, kernel_size=(3, 3), strides=(1, 1), padding='same', name='deconv3', use_bias=False)(
            op3)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)
        x = Concatenate(axis=-1)([x, op2])

        x = Conv2DTranspose(128, kernel_size=(3, 3), strides=(1, 1), padding='same', name='deconv2', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)
        x = Concatenate(axis=-1)([x, op1])

        x = Conv2D(class_num, kernel_size=(3, 3), strides=(windowSize, windowSize), padding='same', name='deconv1')(x)
        x = Reshape((1, class_num))(x)

        x = PixelSoftmax(axis=-1)(x)
        model = Model(inputs=img, outputs=x)

        y_train = y_train.reshape(y_train.shape[0], 1, class_num)

        filepath_name = "best-model.hdf5"
        history = statsLogger()
        opt = Adam(learning_rate=0.0001, decay=1e-4)

        ckpt = ModelCheckpoint(filepath=filepath_name,
                               save_best_only=True,
                               verbose=1,
                               monitor='loss')

        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'],
                      sample_weight_mode="temporal")

        hist = model.fit(x_train,
                         y_train,
                         batch_size=256,
                         epochs=num_epochs,
                         verbose=1)

        plt.plot(hist.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.show()

        train_acc = max(hist.history['accuracy'])

        return model, train_acc

    x_patched, Y_patched = data_process(data, gt)

    x_train_p, x_test_p, y_train_p, y_test_p = train_test_split(x_patched, Y_patched, test_size=0.25, random_state=42)
    y_train = y_train_p
    y_test = y_test_p
    x_train = x_train_p[:, 0, 0, :]
    x_test = x_test_p[:, 0, 0, :]

    # plt.scatter(data[:, 0], data[:, 1], cmap='viridis')
    # plt.title('Indian Pines')
    # plt.show()
    # Training
    G_model = GMM(n_components=clusters)
    G_train_pred = G_model.fit_predict(x_train)

    plt.show()

    x_s = [None] * clusters
    y_s = [None] * clusters
    models = [None] * clusters
    accs = [None] * clusters

    for i in range(clusters):
        x_s[i] = np.zeros((x_train.shape[0], windowSize, windowSize, x_train.shape[1]))
        y_s[i] = np.zeros((y_train.shape[0], y_train.shape[1]))
        for j in range(y_train.shape[0]):
            if G_train_pred[j] == i:
                x_s[i][j] = x_train_p[j]
                y_s[i][j] = y_train_p[j]
        tmp_a = []
        for q in range(x_s[i].shape[0]):
            if np.all(x_s[i][q] == 0):
                tmp_a.append(q)
        x_s[i] = np.delete(x_s[i], tmp_a, 0)
        y_s[i] = np.delete(y_s[i], tmp_a, 0)

        models[i], accs[i] = UNet_Conv_mid(x_s[i], y_s[i], num_epochs=num_epochs, class_num=class_num)
        print("UNet Mid %2d Train Accuracy for Feature Count of %2d: %5.4f" % ((i + 1), 30, accs[i]))

    # Testing

    G_test_pred = G_model.predict(x_test)

    xt_s = [None] * clusters
    yt_s = [None] * clusters
    y_predicts = [None] * clusters

    for i in range(clusters):
        xt_s[i] = np.zeros((x_test.shape[0], windowSize, windowSize, x_test.shape[1]))
        yt_s[i] = np.zeros((y_test.shape[0], y_test.shape[1]))
        for j in range(y_test.shape[0]):
            if G_test_pred[j] == i:
                xt_s[i][j] = x_test_p[j]
                yt_s[i][j] = y_test_p[j]
        tmp_a = []
        for q in range(xt_s[i].shape[0]):
            if np.all(xt_s[i][q] == 0):
                tmp_a.append(q)
        xt_s[i] = np.delete(xt_s[i], tmp_a, 0)
        yt_s[i] = np.delete(yt_s[i], tmp_a, 0)

        y_predicts[i] = models[i].predict(xt_s[i])

    y_test_final_pred = y_predicts[0]
    y_test_final = yt_s[0]
    for i in range(1, clusters):
        y_test_final_pred = np.concatenate((y_test_final_pred, y_predicts[i]), axis=0)
        y_test_final = np.concatenate((y_test_final, yt_s[i]), axis=0)

    y_true = y_test_final.reshape(y_test_final.shape[0], y_test_final.shape[-1])
    y_pred = y_test_final_pred.reshape(y_test_final_pred.shape[0], y_test_final_pred.shape[-1])
    # print("y_true shape: ",y_true.shape)
    # print("y_pred shape: ",y_pred.shape)

    m = tf.keras.metrics.TopKCategoricalAccuracy(k=1)
    m.update_state(y_true, y_pred)
    # print("GMM Clusters = %2d Ensemble UNet Test Accuracy for Feature Count of %2d: %5.4f" % (clusters,30,m.result().numpy()))

    best_y_pred = y_pred
    best_y_test = y_true

    if (return_all):
        return (m.result().numpy(), best_y_pred, best_y_test, G_model, models)
    else:
        return (m.result().numpy())

