
##########
#Author: Nicholas Soucy
#For: CEU-Net

#Purpose: This file is for loading datasets located
#in the Data folder with their specific subfolders.
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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv

#Dataloading functions
#Each function loads the dataset from their specific paths and
#returns the data, labels, class number, optimal clustering method,
#and optimal number of clusters.

def import_IP():
  #Number of Classes = 16
  #import indian Pines dataset
  dataset = io.loadmat('Data/Indian_Pines/Indian_pines_corrected.mat')
  data = dataset['indian_pines_corrected']

  groundtruth = io.loadmat('Data/Indian_Pines/Indian_pines_gt.mat')
  gt = groundtruth['indian_pines_gt']

  dim_x, dim_y, dim_z = data.shape
  dim = dim_x * dim_y

  x = np.zeros(shape=(dim, dim_z))

  cont = 0
  for i in range(dim_x):
      for j in range(dim_y):
          x[cont, :] = data[i, j, :]
          cont += 1

  #normalize x data
  scaler = sp.MinMaxScaler()
  x = scaler.fit_transform(x)

  X = x.reshape(dim_x, dim_y, dim_z)

  return X, gt, 16, 'k', 2

def import_Bot():
  #Number of Classes = 14
  #import Botswana dataset
  dataset = io.loadmat('Data/Botswana/Botswana.mat')
  data = dataset['Botswana']

  groundtruth = io.loadmat('Data/Botswana/Botswana_gt.mat')
  gt = groundtruth['Botswana_gt']

  dim_x, dim_y, dim_z = data.shape
  dim = dim_x * dim_y

  x = np.zeros(shape=(dim, dim_z))

  cont = 0
  for i in range(dim_x):
      for j in range(dim_y):
          x[cont, :] = data[i, j, :]
          cont += 1

  #normalize x data
  scaler = sp.MinMaxScaler()
  x = scaler.fit_transform(x)

  X = x.reshape(dim_x, dim_y, dim_z)

  return X, gt, 14, 'k', 3


def import_KSC():
  #Number of classes = 13
  #import Kennedy Space Center dataset
  dataset = io.loadmat('Data/Kennedy_Space_Center/KSC.mat')
  data = dataset['KSC']

  groundtruth = io.loadmat('Data/Kennedy_Space_Center/KSC_gt.mat')
  gt = groundtruth['KSC_gt']

  dim_x, dim_y, dim_z = data.shape
  dim = dim_x * dim_y

  x = np.zeros(shape=(dim, dim_z))

  cont = 0
  for i in range(dim_x):
      for j in range(dim_y):
          x[cont, :] = data[i, j, :]
          cont += 1

  #normalize x data
  scaler = sp.MinMaxScaler()
  x = scaler.fit_transform(x)

  X = x.reshape(dim_x, dim_y, dim_z)

  return X, gt, 13, 'k', 2


def import_Sal():
  #Number of classes = 16
  #import Salinas dataset
  dataset = io.loadmat('Data/Salinas/Salinas_corrected.mat')
  data = dataset['salinas_corrected']

  groundtruth = io.loadmat('Data/Salinas/Salinas_gt.mat')
  gt = groundtruth['salinas_gt']

  dim_x, dim_y, dim_z = data.shape
  dim = dim_x * dim_y

  x = np.zeros(shape=(dim, dim_z))

  cont = 0
  for i in range(dim_x):
      for j in range(dim_y):
          x[cont, :] = data[i, j, :]
          cont += 1

  #normalize x data
  scaler = sp.MinMaxScaler()
  x = scaler.fit_transform(x)

  X = x.reshape(dim_x, dim_y, dim_z)

  return X, gt, 16, 'g', 3


def import_PavU():
  #Number of classes = 9
  #import Pavia University dataset
  dataset = io.loadmat('Data/Pavia_University/PaviaU.mat')
  data = dataset['paviaU']

  groundtruth = io.loadmat('Data/Pavia_University/PaviaU_gt.mat')
  gt = groundtruth['paviaU_gt']

  dim_x, dim_y, dim_z = data.shape
  dim = dim_x * dim_y

  x = np.zeros(shape=(dim, dim_z))

  cont = 0
  for i in range(dim_x):
      for j in range(dim_y):
          x[cont, :] = data[i, j, :]
          cont += 1

  #normalize x data
  scaler = sp.MinMaxScaler()
  x = scaler.fit_transform(x)

  X = x.reshape(dim_x, dim_y, dim_z)

  return X, gt, 9, 'k', 2


def import_Houston():
  #Number of Classes = 15
  #import Houston dataset
  dataset = io.loadmat('Data/Houston/Houston.mat')
  data = dataset['Houston']

  groundtruth = io.loadmat('Data/Houston/Houston_gt.mat')
  gt = groundtruth['gt']

  dim_x, dim_y, dim_z = data.shape
  dim = dim_x * dim_y

  x = np.zeros(shape=(dim, dim_z))

  cont = 0
  for i in range(dim_x):
      for j in range(dim_y):
          x[cont, :] = data[i, j, :]
          cont += 1

  #normalize x data
  scaler = sp.MinMaxScaler()
  x = scaler.fit_transform(x)

  X = x.reshape(dim_x, dim_y, dim_z)

  return X, gt, 15, 'g', 2



#Data reshaping Functions

def remove_background(data, gt):
  tmp_a = []
  for i in range(data.shape[0]):
    if gt[i] == 0:
      tmp_a.append(i)

  x = np.delete(data,tmp_a,0)
  y = np.delete(gt,tmp_a,0)
  return x, y

def remove_background_full(data, gt):
  tmp_a = []
  for i in range(data.shape[0]):
    if gt[i] == 0:
      tmp_a.append(i)

  x = np.delete(data,tmp_a,0)
  y = np.delete(gt,tmp_a,0)
  return x, y,tmp_a

def flatten(data, gt):
  #reshape data
  dim_x, dim_y, dim_z = data.shape
  dim = dim_x * dim_y

  x = np.zeros(shape=(dim, dim_z))
  y = gt


  cont = 0
  for i in range(dim_x):
      for j in range(dim_y):
          x[cont, :] = data[i, j, :]
          cont += 1

  y = np.reshape(y,(dim))


  #normalize x data
  scaler = sp.MinMaxScaler()
  x = scaler.fit_transform(x)

  return x, y