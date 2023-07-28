#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import datetime
import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import glob
import re
import scikitplot
# NEW
import tensorflow as tf
#from tensorflow.python.keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.python.layers import layers

# set logging level
logging.getLogger("tensorflow").setLevel(logging.ERROR)


physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("physical_devices-------------", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow.keras import initializers
from data_src import get_labels
freq_axis = 1
channel_axis = 3
batch_size = 8
num_epochs = 200

genre_list = get_labels()
col_list = ["file_id", "id"]
tags = np.array(genre_list)
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
import numpy as np
# NEW
import tensorflow as tf
from keras.models import model_from_json
from sklearn.metrics import confusion_matrix
import itertools
import pandas as pd
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import scikitplot as skplt

# set logging level
#logging.getLogger("tensorflow").setLevel(logging.ERROR)
# fold = genre_features.fold
# working_dir = "./"+ fold +"/"
# data_dir = working_dir+"result"+fold
# figures_dir =  working_dir+"figures"+fold
parser = argparse.ArgumentParser(description="GRU on GTZAN.")
parser.add_argument('--fold', default='TOTO')
parser.add_argument('--lr_decay',
                    # default=0.965,
                    default=0.975,
                    type=float)

args = parser.parse_args()
fold = args.fold
lrdecay = args.lr_decay
# working_dir = "./"+ fold +"_300_256/"
# working_dir = "./"+ fold +"_300_128October15/"
fold2 = fold #+"_4DATA_CNN"
working_dir = "./"+ fold2 +"/"
# working_dir = "./"+ fold +"/"
data_test_dir = "./"
data_dir = working_dir+"result"+fold+"_"+str(lrdecay).replace("0.","")
figures_dir =  working_dir+"figures"+fold+"_"+str(lrdecay).replace("0.","")
model_name = "CNN_8575"




img_rows = 50 #  1024 #512
img_cols = 50 # 128 #256
print ("Loading Data from CNN project.....")

def one_hot(Y_genre_strings):
    y_one_hot = np.zeros((Y_genre_strings.shape[0], len(genre_list)))
    for i, genre_string in enumerate(Y_genre_strings):
        index = genre_list.index(genre_string)
        y_one_hot[i, index] = 1
    return y_one_hot
valid_csv = pd.read_csv("VOICE_test_data.csv", usecols=col_list)
file_name = valid_csv["file_id"]
# print (filename[1])
genre = valid_csv["id"]
# in itertools.cycle(a)
print ("Loading test data---------------------")
file_path = "./npy/"
parameter_number = 50
total_file = 81 # Number of Test Files
# NFFT = 4096
timeseries_length = 50
data = np.zeros(
    (total_file, timeseries_length, parameter_number), dtype=np.float32)
target = []
i=0
for filename in enumerate(file_name):
    index,path=filename[0],filename[1]
    full_name='./npy/'+path
    if(index+1 > total_file):
      break

    data[i]   = np.load(full_name)
    if(data[i].shape !=(50,50)):
        print(data[i].shape,path)
        exit(0)
    target.append(genre[index])
    i = i + 1
   
X_test= data
Y_test= np.expand_dims(np.asarray(target), axis=1)
Y_test= one_hot(Y_test)






# X_test = np.load(data_test_dir+"Test_X_data.npy")
# Y_test = np.load(data_test_dir+"Test_Y_data.npy")
# No Normalization for LSTM and GRU
print ("Data Normalization............................")
X_scale = StandardScaler()
for i in range(0, X_test.shape[0]):
    Xtest = X_scale.fit_transform(X_test[i])
    X_test[i, :, :] = Xtest
print("Test X shape: " + str(X_test.shape))
print("Test Y shape: " + str(Y_test.shape))
# input_shape = (X_test.shape[1], X_test.shape[2])

# Formating X_test for CNN
X_test = X_test.reshape(X_test.shape[0],img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
# Load trained model
with open(figures_dir+"/modelCNN.json", "r") as model_file:
    trained_model = model_from_json(model_file.read())
# mylist = [f for f in glob.glob(data_dir+"/*.h5")]
# max_ep = -32767
# str_number = ""
# for filename in mylist:
#     name_file_only = filename.split("/")[-1].split(".")[0]
#     nepoch = int (name_file_only.split("-")[-1])
#     if nepoch >= max_ep:
#         max_ep = nepoch
#     if max_ep < 10:
#         str_number = '0'+str(max_ep)
#     else:
#         str_number = str(max_ep)
# print ("Max epoch = ", max_ep)
file_weight_max = data_dir + "/weights.h5"
trained_model.load_weights(file_weight_max)
#opt = Adam()
opt = "Adam"
trained_model.compile(
    loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
)
# datagen = ImageDataGenerator(
#     featurewise_center=True,
#     featurewise_std_normalization=True,
#     rotation_range=0, #20 #10
#     width_shift_range=0.1, # 0,2
#     height_shift_range=0.1, # 0.2
#     horizontal_flip=False) # True
# datagen = ImageDataGenerator(
#     featurewise_center=True,
#     featurewise_std_normalization=True,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=False)
# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
# datagen.fit(X_test)
# print (X_test.shape)
#
# (X_test, Y_test)= datagen.flow(X_test,Y_test, shuffle=False, batch_size=200).next()
print (X_test.shape)
# sys.exit()
predicted_labels = trained_model.predict(X_test,
                                        #steps_per_epoch=len(X_test) //batch_size,
                                        batch_size=batch_size, verbose=0)
predicted_labels_frames=np.argmax(predicted_labels, axis=1)
real_labels_frames = np.where(Y_test==1)[1]
cnf_matrix = confusion_matrix(real_labels_frames, predicted_labels_frames)
#-------------------------------------
cnfm_suma = cnf_matrix.sum(1)
cnfm_suma_matrix = np.repeat(cnfm_suma[:, None], cnf_matrix.shape[1], axis=1)
cnf_matrix = 10000 * cnf_matrix / cnfm_suma_matrix
cnf_matrix = np.round(cnf_matrix / (100 * 1.0),2)
print (cnf_matrix)
average_acc = round(np.trace(cnf_matrix)/float(len(tags)),2)
print ("Average Test Accuracy = ",average_acc, " %")
# f'{number:9.4f}'
fig = plt.figure()
plt.subplots_adjust(top=0.5)
plt.subplots_adjust(bottom=0.1)
cmap = plt.cm.get_cmap("Reds")
plt.imshow(cnf_matrix, interpolation='nearest', cmap=cmap)

plt.colorbar()
tick_marks = np.arange(len(tags))
plt.xticks(tick_marks, tags, rotation=45)
plt.yticks(tick_marks, tags)
thresh = cnf_matrix.max() / 2.
for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
    plt.text(j, i, cnf_matrix[i, j],
                     horizontalalignment="center",
                     color="white" if cnf_matrix[i, j] > thresh else "black")

plt.tight_layout()
title = "Confusion Matrix"
plt.title(title)
x = datetime.datetime.now()
time_now = x.strftime("%d%b%y%a") +"_"+ x.strftime("%I%p%M")+"_"
plt.ylabel('True label')
plt.xlabel('Predicted label')
file_name = figures_dir+"/"+time_now+"CFM"
np.savetxt(file_name+".txt",cnf_matrix,fmt="%6.2f")
fig.savefig(file_name,bbox_inches='tight')
#-------------------------------------
y_true =real_labels_frames # ground truth labels
# y_probas =trained_model.predict_proba(X_test) # predicted probabilities generated by sklearn classifier
y_probas = predicted_labels
# axes = skplt.metrics.plot_roc_curve(y_true, y_probas)
axes = scikitplot.metrics.plot_roc(y_true, y_probas)
#---------------------------------------
# scikitplot.metrics.plot_roc(y_true, y_probas, title='ROC Curves', plot_micro=True, plot_macro=True, classes_to_plot=None, ax=None, figsize=None, cmap='nipy_spectral', title_fontsize='large', text_fontsize='medium')
#---------------------------------------
fig_filename1 = figures_dir + "/"+model_name+"_ROC"#+'{:.2e}'.format(args.lr_decay)
axes.figure.savefig(fig_filename1 )
# sys.exit()
# NEW ACC
# cnf_matrix = confusion_matrix(real_labels_frames, predicted_labels_frames)
file_report = figures_dir+"/"+fold+"_4digits_REPORT_CNN_VoiceID.txt"
print(metrics.classification_report(real_labels_frames,
                                    predicted_labels_frames,
                                    digits=4), file=open(file_report, "a"))
auc_ = round(roc_auc_score(
    y_true,
    y_probas,
    average='macro',
    sample_weight=None,
    max_fpr=None,
    multi_class='ovr',
    labels=None
),3)
print ("auc_= ",auc_)
print("AUC             ", auc_, file=open(file_report, "a"))
# name_csv = model_name + "_AUC.csv"
# header = ["Folds","AUC"]
# with open(name_csv, 'a') as f:
#     if (fold == "CRV0"):
#         f.write(header[0] + ',' + header[1] + "\n")
#     f.write(args.fold + ',' +str(auc_)+"\n")
# # NEW ACC








#
#
#
#
#
#
#
#
#
# """
# els_frames = trained_model.predict_classes(X_test,
#                                                         #steps_per_epoch=len(X_test) //batch_size,
#                                                         batch_size=batch_size, verbose=0)
# real_labels_frames = np.where(Y_test==1)[1]
# cnf_matrix = confusion_matrix(real_labels_frames, predicted_labels_frames)
# cnfm_suma = cnf_matrix.sum(1)
# cnfm_suma_matrix = np.repeat(cnfm_suma[:, None], cnf_matrix.shape[1], axis=1)
#
# cnf_matrix = 10000 * cnf_matrix / cnfm_suma_matrix
# cnf_matrix = np.round(cnf_matrix / (100 * 1.0),2)
# print (cnf_matrix)
# average_acc = round(np.trace(cnf_matrix)/float(len(tags)),2)
# print ("Average Test Accuracy = ",average_acc, " %")
# # f'{number:9.4f}'
# fig = plt.figure()
# plt.subplots_adjust(top=0.5)
# plt.subplots_adjust(bottom=0.1)
# cmap = plt.cm.Reds
# plt.imshow(cnf_matrix, interpolation='nearest', cmap=cmap)
#
# plt.colorbar()
# tick_marks = np.arange(len(tags))
# plt.xticks(tick_marks, tags, rotation=45)
# plt.yticks(tick_marks, tags)
# thresh = cnf_matrix.max() / 2.
# for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
#     plt.text(j, i, cnf_matrix[i, j],
#                      horizontalalignment="center",
#                      color="white" if cnf_matrix[i, j] > thresh else "black")
#
# plt.tight_layout()
# title = "Confusion Matrix, "+"Average Test Accuracy = " + str(average_acc) + " %, Max Epoch = "+str_number
# plt.title(title)
# x = datetime.datetime.now()
# time_now = x.strftime("%d%b%y%a") +"_"+ x.strftime("%I%p%M")+"_"
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# file_name = figures_dir+"/"+time_now+"CFM_"+'{:.2e}'.format(average_acc).replace(".", "") + "_MxEp_"+str_number
# np.savetxt(file_name+".txt",cnf_matrix,fmt="%6.2f")
# fig.savefig(file_name,bbox_inches='tight')
# # with open('accuracy.csv', 'a') as f:
# #     f.write(args.fold +','+str(average_acc)+"\n")
# name_csv = "accuracy"+str(lrdecay).replace("0.","")+".csv"
# with open(name_csv, 'a') as f:
#     if (fold=="CRV0"):
#         f.write("lr_decay" + ',' + str(lrdecay)+"\n")
#     f.write(args.fold +','+str(average_acc)+"\n")
# if (fold=="CRV8"):
#     data = pd.read_csv(name_csv).values  # No reading 1st row
#     data = np.array(data[0:len(data), 1], dtype=float)
#     aver_acc = np.average(data)
#     with open(name_csv, 'a') as fcsv:
#         fcsv.write("aver_acc"+','+str(aver_acc)+'\n')
#
# #plt.show()
# # 4000 files: acc 78.75%, val_acc (100 files) = 93%