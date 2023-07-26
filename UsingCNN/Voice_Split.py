import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
import sys
import pandas as pd
import csv, ast
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_validate
# Randomly choose 1/10 for test files and 9/10 for train-valid files for each class from 0.7.9.14
# genre_list = ["['0']",
#               "['7']",
#               "['9']",
#               "['14']"
#         ]
label_list = [0,1,2,3,4,5]
col_list = ["file", "label"]
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate

train_ratio = 0.9   # Train and Valid
test_ratio = 0.1    # Testing

# For all classes
data_full = pd.read_csv("npy_label.csv", usecols=col_list)
# import random
# from sklearn.utils import shuffle
# random.shuffle(data_full)
X_data_full = data_full["file"]
Y_data_full = data_full["label"]

from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
sss.get_n_splits(X_data_full, Y_data_full)

# print(sss)
# # StratifiedShuffleSplit(n_splits=5, random_state=0, ...)
for train_index, test_index in sss.split(X_data_full, Y_data_full):
    # print("TRAIN:", train_index, "TEST:",test_index)
    X_train, X_test = X_data_full[train_index], X_data_full[test_index]
    y_train, y_test = Y_data_full[train_index], Y_data_full[test_index]

# Save test and training in CSV format
df = pd.DataFrame({"file_id" : X_test, "id" : y_test})
df.to_csv("VOICE_test_data.csv", index=False)
df = pd.DataFrame({"file_id" : X_train, "id" : y_train})
df.to_csv("VOICE_trainvalid_data.csv", index=False)