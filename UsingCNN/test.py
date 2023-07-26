import os

import librosa
import cv2
import numpy as np
from sklearn.model_selection import StratifiedKFold
import sys
import pandas as pd
import re
import csv, ast
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_validate
import librosa.display
import scipy.signal as sig
import scipy.io.wavfile as wf
import json
import shutil

parameter_number = 215
# total_file = 100 # Number of Test Files
timeseries_length = 215
num_mel = parameter_number
data = []
nfft = 1024
hlength = nfft//2
sr = 22050
path_npy = "./npy/"
os.makedirs(path_npy,exist_ok=True)
with open('./data_src.json') as json_file:
        folders = np.array(json.load(json_file), dtype=object)


               
# list_subfolders_with_paths = [f.path for f in os.scandir(path_files) if f.is_dir()]
with open('./npy_label.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['file','label'])

        for folder in folders:
            ss = folder['name']
            print(ss)
            list_files =os.listdir(folder['data_dir'])
            for k in range (0,len(list_files)):
                file_name =  str(list_files[k])
                full_name_wav = folder['data_dir'] +'/'+ file_name
                print (full_name_wav)
                pet_img = cv2.imread(full_name_wav, 0)
                try:
                    pet_img= cv2.resize(pet_img,(50,50))
                    image = np.array(pet_img).flatten()
                    data = image
                except Exception as e:
                    pass
                file_name_npy = '{}-{}.npy'.format(folder['id'],k)
                # print (file_name_npy)
                full_name_npy = path_npy + file_name_npy
                # print (full_name_npy)
                csv_writer.writerow([file_name_npy,folder['name']])
                with open(full_name_npy, "wb") as f:
                    np.save(full_name_npy, ss)
