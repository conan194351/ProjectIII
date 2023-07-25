
import argparse
import logging
import os
import sys
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import re
from keras import callbacks
from keras.models import Sequential

from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping

import numpy as np
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Conv2D
# NEW
import tensorflow as tf
# from PIL import Image
# tf.config.gpu.set_per_process_memory_fraction(0.4)

# from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler
# from scikitplot.metrics import plot_confusion_matrix, plot_roc
# from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
# from keras.models import model_from_json
from keras.layers.activation import ELU
from keras.layers import BatchNormalization
# from sklearn.metrics import confusion_matrix
# from tensorflow.python import RandomUniform
from tensorflow.python.layers import layers
# import pandas as pd
# from keras.optimizers import SGD

# Preprocessing
from sklearn.model_selection import train_test_split

from data_src import get_labels
logging.getLogger("tensorflow").setLevel(logging.ERROR)


from Des_Models import (
    DefineModels,
)  # local python class with Audio feature extraction (librosa)
des_models = DefineModels()

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("physical_devices-------------", len(physical_devices))
#print ("NO GPU----------------------------------------")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.4)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #3: Filter out all messages


freq_axis = des_models.freq_axis
channel_axis = des_models.channel_axis
batch_size = 8

num_epochs = 400

img_rows = 215
img_cols = 215 




# GTZAN Dataset Tags
tagstring = [0,1,2,3,4,5]
tags = np.array(tagstring)
parser = argparse.ArgumentParser(description="CNN on Training VoiceID.")

parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--batch_size', default=10, type=int)
parser.add_argument('--lr', default=0.001, type=float,
                    help="Initial learning rate")
parser.add_argument('--lr_decay',
                    default=0.965,
                    type=float)
parser.add_argument('--save_dir', default="TOTO")
parser.add_argument('--debug', action='store_true',
                    help="Save weights by TensorBoard")
parser.add_argument('--fold', default='TOTO')


args = parser.parse_args()
fold = args.fold
lrdecay = args.lr_decay
fold2 = fold
working_dir = "./"+ fold +"/"
data_dir = working_dir+"result"+fold+"_"+str(lrdecay).replace("0.","")
args.save_dir = data_dir
figures_dir =  working_dir+"figures"+fold+"_"+str(lrdecay).replace("0.","")
newpath1 = data_dir
if not os.path.exists(newpath1):
    os.makedirs(newpath1)
newpath2 = figures_dir
if not os.path.exists(newpath2):
    os.makedirs(newpath2)
data_working_dir = "./" + fold +"/"

print ("Load validation and training data................")
file_path = ".//npy/"
parameter_number = 215
total_file = 81

timeseries_length = 215
data = np.zeros(
    (total_file, timeseries_length, parameter_number), dtype=np.float32)
genre_list =get_labels()
col_list = ["file_id", "id"]
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
def one_hot(Y_genre_strings):
    y_one_hot = np.zeros((Y_genre_strings.shape[0], len(genre_list)))
    for i, genre_string in enumerate(Y_genre_strings):
        index = genre_list.index(genre_string)
        y_one_hot[i, index] = 1
    return y_one_hot
valid_csv = pd.read_csv("VOICE_valid_data_"+ fold +".csv", usecols=col_list)
file_name = valid_csv["file_id"]
# print (filename[1])
genre = valid_csv["id"]
# in itertools.cycle(a)
print ("Loading valid data---------------------")
target = []
i=0
for filename in enumerate(file_name):
    
    index,path=filename[0],filename[1]
    full_name='./npy/'+path
    if(index+1 > total_file):
      break

    data[i]   = np.load(full_name)
    if(data[i].shape !=(215,215)):
        print(data[i].shape,path)
        exit(0)
    target.append(genre[index])
    i = i + 1

X_val= data
Y_val = np.expand_dims(np.asarray(target), axis=1)
Y_val = one_hot(Y_val)

# print (Y_val)
# sys.exit()

# LOAD TRAINING DATA===================================
total_file_training = 641 # Number of Train Files
data_train = np.zeros(
    (total_file_training, timeseries_length, parameter_number), dtype=np.float32)
train_csv = pd.read_csv("VOICE_train_data_"+ fold +".csv", usecols=col_list)
file_name = train_csv["file_id"]
# print (filename[1])
genre = train_csv["id"]
# in itertools.cycle(a)
print ("Loading train data---------------------")
target = []
i=0
for filename in enumerate(file_name):
    index,path=filename[0],filename[1]
    full_name='./npy/'+path
    if(index+1 > total_file_training):
      break
      
    # data[i]   = np.load(full_name)
   
    # target.append(genre[index])
    # i = i + 1
    data_train[i, :, 0:parameter_number]   = np.load(full_name)
    if(data_train[i].shape !=(215,215)):
        print(data[i].shape,path)
        exit(0)
    target.append(genre[index])
    i = i + 1

    
X_train = data_train
Y_train = np.expand_dims(np.asarray(target), axis=1)
Y_train = one_hot(Y_train)

#=======================================================
# For CNN only
X_scale = StandardScaler()
for i in range(0, X_train.shape[0]):
    Xtrain = X_scale.fit_transform(X_train[i])
    X_train[i, :, :] = Xtrain
for i in range(0, X_val.shape[0]):
    Xval = X_scale.fit_transform(X_val[i])
    X_val[i, :, :] = Xval
print("Training X shape: " + str(X_train.shape))
print("Training Y shape: " + str(Y_train.shape))

print("Valid X shape: " + str(X_val.shape))
print("Valid Y shape: " + str(Y_val.shape))

#sys.exit()
# X_train = genre_features.X_train
# X_val= genre_features.X_val

# Formating data for CNN
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols,1)
X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols,1)

input_shape = (img_rows, img_cols,1) # CNN
# For LSTM
# input_shape = (X_train.shape[1], X_train.shape[2])
# ORG---------------------------------------------------
print("input shape = ", np.shape(input_shape))
print("Build CNN model ...")
# ---GRU
model,name_model = des_models.modelCNN_8575(input_shape,Y_train.shape[1])
#model.build(input_shape)
print("Compiling ...")
opt = "Adam"
# opt = SGD()
# opt = "adadelta"
# opt = "RMSprop" #acc 0.7, 0.9
# opt ="Adagrad"
# opt = "Adamax"
# opt = "Nadam"
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])  # metrics=["categorical_accuracy"])
# # model.compile(loss="SparseCategoricalCrossentropy", optimizer=opt,metrics=["accuracy"])
#
print (" Model compiled.........................")
# sys.exit()
from contextlib import redirect_stdout
with open(figures_dir + '/modelsummary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()
model.summary()
#sys.exit()
print("Training ...")

# log = callbacks.CSVLogger(args.save_dir + '/log.csv')
# tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
#                            batch_size=batch_size, histogram_freq=int(args.debug))
checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights.h5', monitor='val_accuracy',
                                       save_best_only=True, save_weights_only=True, verbose=1)
lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))
es = EarlyStopping(monitor='val_accuracy',mode="max", verbose=1, patience=200)
args = parser.parse_args()

seqModel = model.fit(X_train, Y_train,
                    validation_data=(X_val,Y_val),
                    validation_steps=len(X_val) // batch_size,
                    validation_batch_size=batch_size,
                    batch_size=batch_size,
                    epochs=num_epochs,
                    # callbacks=[log, tb, es, checkpoint, lr_decay])
                    callbacks=[es, checkpoint, lr_decay])


# visualizing losses and accuracy
train_loss = seqModel.history['loss']
train_acc = seqModel.history['accuracy']
val_loss = seqModel.history['val_loss']
val_acc = seqModel.history['val_accuracy']
xc = range(len(train_loss))

model_json = model.to_json()
with open(figures_dir+"/modelCNN.json", "w") as json_file:
    json_file.write(model_json)
fig = plt.figure()
fig.subplots_adjust(hspace=0.5)
plt.subplot(211)
plt.plot(xc, train_loss)
plt.plot(xc, val_loss)
plt.legend(["train_loss", "val_loss"], loc="upper right")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.subplot(212)

plt.plot(xc, train_acc)
plt.plot(xc, val_acc)
plt.legend(["train_acc", "val_acc"], loc="lower right")
max_val_acc = max(val_acc)
plt.title("Max Validation Accuracy = "+str(max_val_acc))
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
x = datetime.datetime.now()
time_now = x.strftime("%d%b%y%a") +"_"+ x.strftime("%I%p%M")+"_"
fig_filename1 = figures_dir + "/"+time_now + "TVA_"+name_model+"_" + str(X_train.shape[1]) + "_" + str(X_train.shape[2])  # +"_"+'{:.2e}'.format(args.lr_decay)
fig.savefig(fig_filename1)
fig_filename2 = figures_dir + "/"+time_now + "TVA_"+name_model+"_"+opt+"_" + str(X_train.shape[1]) + "_" + str(X_train.shape[2]) + "_" + '{:.2e}'.format(
    args.lr_decay).replace(".", "")+"_"+'{:.2e}'.format(max_val_acc).replace(".", "")

fig.savefig(fig_filename2)
#plt.show()

