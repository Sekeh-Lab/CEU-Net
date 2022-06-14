##########
#Paper by: Nicholas Soucy and Salimeh Yasaei Sekeh
#Author: Nicholas Soucy
#For: CEU-Net

#Purpose: This file contains the training and testing
#code for Single U-Net and CEU-Net.
##########


#imports
import numpy as np
from tqdm import trange
import spectral
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv

#custom imports
from data_loader import *
from feature_reduction import *
from networks import *



# Testing Functions


def AA_andEachClassAccuracy(confusion_matrix):
  counter = confusion_matrix.shape[0]
  list_diag = np.diag(confusion_matrix)
  list_raw_sum = np.sum(confusion_matrix, axis=1)
  each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
  average_acc = np.mean(each_acc)
  return each_acc, average_acc


def reports (X_test,y_test,name,model):
  #start = time.time()
  y_pred = model.predict(X_test)
  y_pred = y_pred.reshape(y_pred.shape[0],y_pred.shape[-1])
  y_test = y_test.reshape(y_test.shape[0],y_test.shape[-1])
  # y_pred = np.argmax(y_pred, axis=1)
  #end = time.time()
  #print(end - start)
  if name == 'Indian_Pines':
      target_names = ['Alfalfa', 'Corn Notill', 'Corn Mintill', 'Corn'
                      ,'Grass Pasture', 'Grass Trees', 'Grass Pasture Mowed', 
                      'Hay Windrowed', 'Oats', 'Soybean Notill', 'Soybean Mintill',
                      'Soybean Clean', 'Wheat', 'Woods', 'Buildings Grass Trees Drives',
                      'Stone Steel Towers']
  elif name == 'Salinas':
      target_names = ['Brocoli Green Weeds 1','Brocoli Green Weeds 2','Fallow','Fallow Rough Plow','Fallow Smooth',
                      'Stubble','Celery','Grapes Untrained','Soil Vinyard Develop','Corn Senesced Green Weeds',
                      'Lettuce Romaine 4wk','Lettuce Romaine 5wk','Lettuce Romaine 6wk','Lettuce Romaine 7wk',
                      'Vinyard Untrained','Vinyard Vertical Trellis']
  elif name == 'Pavia_University':
      target_names = ['Asphalt','Meadows','Gravel','Trees', 'Painted metal sheets','Bare Soil','Bitumen',
                      'Self-Blocking Bricks','Shadows']

  elif name == 'Kennedy_Space_Center':
      target_names = ['Scrub','Willow Swamp','CP Hammock', 'Slash Pine', 'Oak/Broadleaf','Hardwood',
                      'Swamp','Graminoid Marsh','Spartina Marsh','Cattail Marsh','Salt Marsh','Mud Flats','Water']
  
  elif name == 'Botswana':
      target_names = ['Water','Hippo Grass','Floodplain Grasses 1','Floodplain Grasses 2','Reeds 1','Riparian',
                      'Firescar 2','Island Interior','Acacia Woodlands','Acacia Shrublands','Acacia Grasslands',
                      'Short Mopane','Mixed Mopane','Exposed Soils']
    
  elif name == 'Houston':
      target_names = ['Healthy Grass','Stressed Grass','Artificial Turf','Trees','Soil','Water',
                      'Residential','Commercial','Roads','Highway','Railways','Parking Lot 1',
                      'Parking Lot 2','Tennis Court','Running Track']
  
  #handle different outputs from hybridSN vs U-Nets
  classification = classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), target_names=target_names)
  oa = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
  confusion = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
  confusion = pd.DataFrame(confusion, index=target_names, columns=target_names)
  each_acc, aa = AA_andEachClassAccuracy(confusion)
  kappa = cohen_kappa_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
  y_test = y_test.reshape(y_test.shape[0],1,y_test.shape[-1])
  score = model.evaluate(X_test, y_test, batch_size=32)
  Test_Loss =  score[0]*100
  Test_accuracy = score[1]*100
  
  return classification, confusion, Test_Loss, Test_accuracy, oa*100, each_acc*100, aa*100, kappa*100

def reports_CE (y_prediction,y_test,name):
  y_pred = y_prediction.reshape(y_prediction.shape[0],y_prediction.shape[-1])
  y_test = y_test.reshape(y_test.shape[0],y_test.shape[-1])
  # y_pred = np.argmax(y_pred, axis=1)
  #end = time.time()
  #print(end - start)
  if name == 'Indian_Pines':
      target_names = ['Alfalfa', 'Corn Notill', 'Corn Mintill', 'Corn'
                      ,'Grass Pasture', 'Grass Trees', 'Grass Pasture Mowed', 
                      'Hay Windrowed', 'Oats', 'Soybean Notill', 'Soybean Mintill',
                      'Soybean Clean', 'Wheat', 'Woods', 'Buildings Grass Trees Drives',
                      'Stone Steel Towers']
  elif name == 'Salinas':
      target_names = ['Brocoli Green Weeds 1','Brocoli Green Weeds 2','Fallow','Fallow Rough Plow','Fallow Smooth',
                      'Stubble','Celery','Grapes Untrained','Soil Vinyard Develop','Corn Senesced Green Weeds',
                      'Lettuce Romaine 4wk','Lettuce Romaine 5wk','Lettuce Romaine 6wk','Lettuce Romaine 7wk',
                      'Vinyard Untrained','Vinyard Vertical Trellis']
  elif name == 'Pavia_University':
      target_names = ['Asphalt','Meadows','Gravel','Trees', 'Painted metal sheets','Bare Soil','Bitumen',
                      'Self-Blocking Bricks','Shadows']

  elif name == 'Kennedy_Space_Center':
      target_names = ['Scrub','Willow Swamp','CP Hammock', 'Slash Pine', 'Oak/Broadleaf','Hardwood',
                      'Swamp','Graminoid Marsh','Spartina Marsh','Cattail Marsh','Salt Marsh','Mud Flats','Water']
  
  elif name == 'Botswana':
      target_names = ['Water','Hippo Grass','Floodplain Grasses 1','Floodplain Grasses 2','Reeds 1','Riparian',
                      'Firescar 2','Island Interior','Acacia Woodlands','Acacia Shrublands','Acacia Grasslands',
                      'Short Mopane','Mixed Mopane','Exposed Soils']
    
  elif name == 'Houston':
      target_names = ['Healthy Grass','Stressed Grass','Artificial Turf','Trees','Soil','Water',
                      'Residential','Commercial','Roads','Highway','Railways','Parking Lot 1',
                      'Parking Lot 2','Tennis Court','Running Track']


  classification = classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), target_names=target_names)
  oa = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
  confusion = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
  confusion = pd.DataFrame(confusion, index=target_names, columns=target_names)
  each_acc, aa = AA_andEachClassAccuracy(confusion)
  kappa = cohen_kappa_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
  
  return classification, confusion, oa*100, each_acc*100, aa*100, kappa*100



# Import data (for this example we import the Pavia University dataset).
# Change ds number to change dataset.
# Change patch_size to change size of patch for patching. If patch_size = 1 no patching is done.

dataset = ["Indian_Pines","Salinas","Pavia_University","Kennedy_Space_Center","Botswana","Houston"]
ds = 2

if ds == 0:
    X, Y, cls, en_type, clusters = import_IP()
elif ds == 1:
    X, Y, cls, en_type, clusters = import_Sal()
elif ds == 2:
    X, Y, cls, en_type, clusters = import_PavU()
elif ds == 3:
    X, Y, cls, en_type, clusters = import_KSC()
elif ds == 4:
    X, Y, cls, en_type, clusters = import_Bot()
elif ds == 5:
    X, Y, cls, en_type, clusters = import_Houston()

patch_size = 1   #1 means no patching

print(dataset[ds])
print(X.shape, Y.shape)


# Apply feature reduction on data.
# Uncomment which ever feature reduction method you wish to use.

#Number of features we wish to reduce to
feature_count = 30

print("***Starting Feature Reduction***")

#Apply PCA
x,pca_model = PCA_DR(X, feature_count,return_model = True)
y=Y

# #Apply 2D CAE
# x,y = CAE_2D(X,Y,num_epochs=100,dim=30)

# #Apply 3D CAE
# x,y = CAE_3D(X,Y,num_epochs=150,dim=30)


print("***Feature Reduction Finished***")
print(x.shape,y.shape)


# Training

# Train Single U-Net model

#UNet

print("***Starting Single U-Net***")

UNet_acc,best_x_train,best_x_test,best_y_train,best_y_test,best_model = UNet_Conv(x,y, num_epochs = 150, class_num = cls,return_all = True,ws = patch_size, folds = 1)

print("***Single U-Net Finished***")

# Train CEU-Net Model

print("***Starting CEU-Net***")

if en_type == 'k':
  #K Ensemble UNet Set
  K_Ensemble_UNet_acc,best_y_pred_CE,best_y_test_CE,cluster_model,class_models = K_Ensemble_UNet(x,y,ds,num_epochs = 200, class_num = cls, clusters = clusters,return_all = True,ws = patch_size)

elif en_type == 'g':
  #G Ensemble UNet Set
  G_Ensemble_UNet_acc,best_y_pred_CE,best_y_test_CE,cluster_model,class_models = G_Ensemble_UNet(x,y,ds,num_epochs = 200, class_num = cls, clusters = clusters,return_all = True,ws = patch_size)
  
print("***CEU-Net Finished***")

#Testing Results

# Single U-Net
# Get class-wise classification summary, overall accuracy (OA), average accuracy (AA) and kappa score for Single U-Net.
classification, confusion, Test_Loss, Test_accuracy, oa, each_acc, aa, kappa = reports(best_x_test,best_y_test,dataset[ds],best_model)

print("Testing Results for Single U-Net")
print(classification)
print("Overall Accuracy: ",oa)
print("Average Accuracy: ", aa)
print("Kappa Score: ",kappa)


# CEU-Net

# Get class-wise classification summary, overall accuracy (OA), average accuracy (AA) and kappa score for CEU-Net.
classification, confusion, oa, each_acc, aa, kappa = reports_CE(best_y_pred_CE,best_y_test_CE,dataset[ds])

print("Testing Results for CEU-Net")
print(classification)
print("Overall Accuracy: ",oa)
print("Average Accuracy: ", aa)
print("Kappa Score: ",kappa)


# CEU-Net Classification Map


# Pad original data for patching.
print("Pre Padding size: ",x.shape)
x_padded = padWithZeros(x, patch_size//2)
print("Padded Size: ",x_padded.shape)

# calculate the predicted image for multiple networks
height = y.shape[0]
width = y.shape[1]

outputs = np.zeros((height,width))
for i in trange(height):
    for j in range(width):
        target = int(y[i,j])
        if target == 0 :
            continue
        else :
            image_patch=Patch(x,i,j,patch_size) 
            # print(image_patch.shape)
            X_test_image = image_patch.reshape(1,image_patch.shape[0]* image_patch.shape[1]* image_patch.shape[2]).astype('float64')     
            # print(X_test_image.shape)   
            X_test_pred = cluster_model.predict(X_test_image)     
            # print(X_test_pred)                      
            X_test_image = image_patch.reshape(1,patch_size,patch_size, image_patch.shape[-1]).astype('float32')  
            prediction = class_models[X_test_pred[0]].predict(X_test_image)
            prediction = prediction.reshape(prediction.shape[-1])
            prediction = np.argmax(prediction, axis=0)
            outputs[i][j] = prediction+1


# Show classification map.
predicted_labels = spectral.imshow(classes = outputs.astype(int),figsize =(20,20))

