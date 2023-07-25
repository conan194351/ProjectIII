import os

import librosa
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
data = np.zeros(
    (timeseries_length, parameter_number), dtype=np.float32)
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
            list_files =os.listdir(folder['training_set'])
            for k in range (0,len(list_files)):
                file_name =  str(list_files[k])
                full_name_wav = folder['training_set'] +'/'+ file_name
                print (full_name_wav)
                y, sr = librosa.load(full_name_wav, mono=True, offset=0, duration=(timeseries_length - 1) * hlength / sr, sr=sr)
                y = 0.1 * y * 10.0
                y = sig.lfilter([1, -0.97], [1.0, 0], y)
                S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=nfft, hop_length=hlength, window="hamming",
                                                n_mels=num_mel, fmax=sr // 2)
                S_dB = librosa.power_to_db(S, ref=np.max)
                #print("shape S_dB = ", S_dB.shape)
                data[:, 0:parameter_number] = S_dB.T[0:timeseries_length, :]
                file_name_npy = '{}-{}.npy'.format(folder['id'],k)
                # print (file_name_npy)
                full_name_npy = path_npy + file_name_npy
                # print (full_name_npy)
                csv_writer.writerow([file_name_npy,folder['name']])
                with open(full_name_npy, "wb") as f:
                    np.save(f, data)
