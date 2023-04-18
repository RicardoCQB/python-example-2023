# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 19:01:15 2023

@author: crdru
"""

from helper_code import *
import numpy as np, os, sys
import mne
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from scipy import signal
from scipy.integrate import simps
import joblib
import pywt
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import shutil

data_folder=r'C:\Users\crdru\Desktop\python-example-2023\training_data'
patient_ids = find_data_folders(data_folder)
num_patients = len(patient_ids)
num_recordings=72
threshold=1

for i in range(num_patients):

    # Load data.
    patient_id = patient_ids[i]
    _ , recording_metadata, _ = load_challenge_data(data_folder, patient_id)

    #Get quality score
    quality_score_array=np.zeros([num_recordings])
    quality_score_array=get_quality_scores(recording_metadata)
    good_quality_indexes=np.where(quality_score_array >= threshold)
    if good_quality_indexes[0].size >0:
        original = rf'C:\Users\crdru\Desktop\python-example-2023\training_data\{patient_id}'
        target = rf'C:\Users\crdru\Desktop\python-example-2023\traning_data_clean\{patient_id}'
        shutil.move(original, target)