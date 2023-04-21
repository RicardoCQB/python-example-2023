#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries and functions. You can change or remove them.
#
################################################################################

from helper_code import *
import numpy as np, os, sys
import mne
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import coherence, hann
from scipy import signal
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.impute import KNNImputer
import joblib
import pywt
from scipy.integrate import simps
from collections import Counter

# from scipy.stats import entropy

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose >= 1:
        print('Finding the Challenge data...')

    patient_ids = find_data_folders(data_folder)
    num_patients = len(patient_ids)

    if num_patients==0:
        raise FileNotFoundError('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    features = list()
    outcomes = list()
    cpcs = list()

    for i in range(num_patients):
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patients))

        # Load data.
        patient_id = patient_ids[i]
        patient_metadata, recording_metadata, recording_data = load_challenge_data(data_folder, patient_id)

        # # Extract features.
        current_features = get_features(patient_metadata, recording_metadata, recording_data)
        features.append(current_features)

        # Extract labels.
        current_outcome = get_outcome(patient_metadata)
        outcomes.append(current_outcome)
        current_cpc = get_cpc(patient_metadata)
        cpcs.append(current_cpc)
        
    features = np.vstack(features)
    outcomes = np.vstack(outcomes)
    cpcs = np.vstack(cpcs)
  

    # Train the models.
    if verbose >= 1:
        print('Training the Challenge models on the Challenge data...')

    # Define parameters for random forest classifier and regressor.
    n_estimators   = 1000  # Number of trees in the forest. 1000
    max_leaf_nodes = 25  # Maximum number of leaf nodes in each tree.25
    random_state   = 42  # Random state; set for reproducibility.
    
    # Impute any missing features; use the mean value by default.
    # imputer2 = SimpleImputer().fit(features)
    imputer2 = KNNImputer().fit(features)
    # Save imputer
    joblib.dump(imputer2, "imputer2.pkl")

    # Train the models.
    features = imputer2.transform(features)
    outcome_model = RandomForestClassifier(
        n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, outcomes.ravel())
    cpc_model = RandomForestRegressor(
        n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, cpcs.ravel())

    # Save the models.
    save_challenge_model(model_folder, imputer2, outcome_model, cpc_model)

    if verbose >= 1:
        print('Done.')

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_models(model_folder, verbose):
    filename = os.path.join(model_folder, 'models.sav')
    return joblib.load(filename)

# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_models(models, data_folder, patient_id, verbose):
    imputer = models['imputer']
    outcome_model = models['outcome_model']
    cpc_model = models['cpc_model']

    # Load data.
    patient_metadata, recording_metadata, recording_data = load_challenge_data(data_folder, patient_id)

    # Extract features.
    features = get_features(patient_metadata, recording_metadata, recording_data)    
    # Convert the feature list to a NumPy array
    features = features.reshape(-1, 1) 
    features = features.T

    features = imputer.transform(features)

    # Apply models to features.
    outcome = outcome_model.predict(features)[0]
    outcome_probability = outcome_model.predict_proba(features)[0, 1]
    cpc = cpc_model.predict(features)[0]

    # Ensure that the CPC score is between (or equal to) 1 and 5.
    cpc = np.clip(cpc, 1, 5)

    return outcome, outcome_probability, cpc

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
def save_challenge_model(model_folder, imputer, outcome_model, cpc_model):
    d = {'imputer': imputer, 'outcome_model': outcome_model, 'cpc_model': cpc_model}
    filename = os.path.join(model_folder, 'models.sav')
    joblib.dump(d, filename, protocol=0)
    

def compute_regularity(signal_data, sampling_frequency):
    
    window_size = 0.5 * sampling_frequency  # assume sampling_rate is known
    weights = np.ones(int(window_size)) / int(window_size)

    # compute regularity feature for each channel

    # Step 1: Square the signal
    signal_squared = np.square(signal_data)
    # print(signal_data)
    # print(signal_squared)
    # Step 2: Apply a moving-average filter with a window of 0.5 seconds
    signal_averaged = np.convolve(signal_squared, weights, mode='same')
    # signal_averaged = np.nan_to_num(signal_averaged, nan=0.0) #check what happens, how does it change
    # print(signal_averaged)
    

    # Step 3: Sort the values of the smoothed signal in descending order
    q = np.sort(signal_averaged)[::-1]
    # plt.plot(q)
    # plt.show()
    # Step 4: Compute the normalized standard deviation of the sorted signal
    N = len(signal_squared)
    q_sum = sum(q)
    REG = math.sqrt((sum(i**2 * q[i-1] for i in range(1, N+1))) / ((1/3) * N**2 * q_sum)) #if the warning of this code continues, cut the equation into parts!
    
    return REG

# Extract features from the data.
def get_features(patient_metadata, recording_metadata, recording_data):
    # Extract features from the patient metadata.
    age = get_age(patient_metadata)
    sex = get_sex(patient_metadata)
    rosc = get_rosc(patient_metadata)
    #ohca = get_ohca(patient_metadata)
    #vfib = get_vfib(patient_metadata)
    ttm = get_ttm(patient_metadata)

    # Use one-hot encoding for sex; add more variables
    if sex == 'Female':
        female = 1
        male   = 0
        other  = 0
    elif sex == 'Male':
        female = 0
        male   = 1
        other  = 0
    else:
        female = 0
        male   = 0
        other  = 1
        
    if ttm == 33:
        ttm_33 = 1
        ttm_36   = 0
        ttm_none  = 0
    elif ttm == 36:
        ttm_33 = 0
        ttm_36   = 1
        ttm_none  = 0
    else:
        ttm_33 = 0
        ttm_36   = 0
        ttm_none  = 1

    # Combine the patient features.
    patient_features = np.array([age, female, male, other, rosc, ttm_33, ttm_36, ttm_none])

    # Extract features from the recording data and metadata.
    channels = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'Fp1-F3',
                'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fz-Cz', 'Cz-Pz']
    num_channels = len(channels)
    num_recordings = len(recording_data)
    
    # Compute mean and standard deviation for each channel for each recording.
    available_signal_data = list()
    mean_absolute_values_A = list()
    mean_absolute_values_D4 = list()
    mean_absolute_values_D3 = list()
    mean_absolute_values_D2 = list()
    mean_absolute_values_D1 = list()
    average_power_A = list()
    average_power_D4 = list()
    average_power_D3 = list()
    average_power_D2 = list()
    average_power_D1 = list()
    sd_A = list()
    sd_D4 = list()
    sd_D3 = list()
    sd_D2 = list()
    sd_D1 = list()
    
    MAEratio_D4_A = list()
    MAEratio_D3_D4 = list()
    MAEratio_D2_D3 = list()
    MAEratio_D1_D2 = list()
    
    entropy_A=list()
    entropy_D4=list()
    entropy_D3=list()
    entropy_D2=list()
    entropy_D1=list()

    # Compute mean and standard deviation for each channel for each recording.
    available_signal_data = list()
    for i in range(num_recordings):
        signal_data, sampling_frequency, signal_channels = recording_data[i]
        if signal_data is not None:
            signal_data = reorder_recording_channels(signal_data, signal_channels, channels) # Reorder the channels in the signal data, as needed, for consistency across different recordings.
            available_signal_data.append(signal_data)

    if len(available_signal_data) > 0:
        available_signal_data = np.hstack(available_signal_data)
        signal_mean = np.nanmean(available_signal_data, axis=1)
        signal_std  = np.nanstd(available_signal_data, axis=1)
        signal_power = np.square(signal_std)
    else:
        signal_mean = float('nan') * np.ones(num_channels)
        signal_std  = float('nan') * np.ones(num_channels)
        signal_power  = float('nan') * np.ones(num_channels)

     # Compute the power spectral density for the delta, theta, alpha, and beta frequency bands for each channel of the most
     # recent recording with quality score between 0 and 1.
    index = None
     
    found_index = False
    for i in reversed(range(num_recordings)):
        signal_data, sampling_frequency, signal_channels = recording_data[i]
        if signal_data is not None:
            quality_score = get_quality_scores(recording_metadata)[i]
            if quality_score > 0.0 and quality_score <=1:
                index = i
                found_index = True
                break
    

    if index is not None:
        signal_data, sampling_frequency, signal_channels = recording_data[index]
        signal_data = reorder_recording_channels(signal_data, signal_channels, channels) # Reorder the channels in the signal data, as needed, for consistency across different recordings.
        
        delta_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency,  fmin=0.5,  fmax=4.0, verbose=False) #array (18,9)
        theta_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency,  fmin=4.0,  fmax=8.0, verbose=False) #array (18,10)
        alpha_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency,  fmin=8.0, fmax=12.0, verbose=False) #array (18,10)
        beta_psd,  _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency, fmin=12.0, fmax=30.0, verbose=False) #array (18,46)
        
        delta_psd_mean = np.nanmean(delta_psd, axis=1) #mean per channel, array(18,1)
        theta_psd_mean = np.nanmean(theta_psd, axis=1)
        alpha_psd_mean = np.nanmean(alpha_psd, axis=1)
        beta_psd_mean  = np.nanmean(beta_psd,  axis=1)
        
        # std
        delta_psd_std = np.nanstd(delta_psd_mean, axis=0) #should be 1 value now
        theta_psd_std = np.nanstd(theta_psd_mean, axis=0)
        alpha_psd_std = np.nanstd(alpha_psd_mean, axis=0)
        beta_psd_std  = np.nanstd(beta_psd_mean,  axis=0)
        
        # Compute the alpha-delta ratio
        alpha_delta_ratio = alpha_psd_mean/delta_psd_mean 
        alpha_delta_ratio_mean = np.nanmean(alpha_delta_ratio, axis=0) # 1 value
        alpha_delta_ratio_std = np.nanstd(alpha_delta_ratio, axis=0) 
   
        # Compute Shannon entropy for each channel 
        shannon_entropies = np.zeros(num_channels)
        reg_chan = np.zeros(num_channels)
        for j in range(num_channels):
            # Compute the probability distribution of the EEG signal for this channel
            hist, bin_edges = np.histogram(signal_data[j,:], bins=100, density=True)
            pdf = hist / np.sum(hist)
            eps = 1e-12 # small constant to add to denominator
            # Compute the Shannon entropy of the PDF
            shannon_entropies[j] = -np.sum(pdf * np.log2(pdf+eps)) #array(18,1)
            # Plot the window
            # plt.figure()
            # plt.plot(shannon_entropies)
            reg_chan[j] = compute_regularity(signal_data[j,:], sampling_frequency) #array(18,1)
        
        # Compute average Shannon entropy over channels
        mean_shannon_entropy = np.nanmean(shannon_entropies, axis=0) 
        mean_shannon_entropy_std = np.nanstd(shannon_entropies, axis=0)    
        # Compute the regularity feature over channels 
        mean_reg = np.nanmean(reg_chan, axis=0)
        mean_reg_std = np.nanstd(reg_chan, axis=0)
        
   
        quality_score = get_quality_scores(recording_metadata)[index] #one value
        
        # Compute the coherence in delta band
        window_length = int(4 * sampling_frequency)  # in samples
        window_overlap = int (2 * sampling_frequency)  # in samples
        window = hann(window_length, sym=True)
        delta_band = [0.5, 4]
            
        all_coherences = np.zeros((18, 18))
        window = hann(signal_data.shape[1], sym=False)
        windowed_data = signal_data * window
   
        for channel1 in range(signal_data.shape[0]):
            # apply a Hanning window to the signal
            windowed_data[channel1, :] = signal_data[channel1, :] * window
    
            # loop over all channels again to get coherence values between all possible channel combinations
            for channel2 in range(channel1+1, signal_data.shape[0]):
                f, Cxy = coherence(windowed_data[channel1], windowed_data[channel2], fs=sampling_frequency, nperseg=window_length,
                       window='hann', noverlap=window_overlap, nfft=None, detrend='constant', axis=-1)
    
                # calculate the average coherence in the delta band
                delta_coherence = np.mean(Cxy[(f >= delta_band[0]) & (f <= delta_band[1])], axis=0)
    
                # save coherence values for all channel combinations
                all_coherences[channel1, channel2] += delta_coherence
                all_coherences[channel2, channel1] += delta_coherence
           
           # average the coherence values over epochs and all channel combinations
        mean_coherence = np.mean(all_coherences, axis=(0, 1))
        mean_coherence_std = np.std(all_coherences, axis=(0, 1))
        
        #Cristian's code
        sos = signal.butter(6,[0.56, 40], 'bandpass', fs=100, output='sos')
        signal_data = signal.sosfilt(sos, signal_data)
        coeff=pywt.wavedec(signal_data,'db4',mode='symmetric',level=4,axis=1) #coeff[subfreq][channels]
        
        mean_abs_val_A=np.nanmean(abs(coeff[0]))
        mean_abs_val_D4=np.nanmean(abs(coeff[1]))
        mean_abs_val_D3=np.nanmean(abs(coeff[2]))
        mean_abs_val_D2=np.nanmean(abs(coeff[3]))
        mean_abs_val_D1=np.nanmean(abs(coeff[4]))
        
        ratio_MAE_D4_A=mean_abs_val_D4/mean_abs_val_A
        ratio_MAE_D3_D4=mean_abs_val_D3/mean_abs_val_D4
        ratio_MAE_D2_D3=mean_abs_val_D2/mean_abs_val_D3
        ratio_MAE_D1_D2=mean_abs_val_D1/mean_abs_val_D2
        
        
        mean_absolute_values_A.append(mean_abs_val_A)
        mean_absolute_values_D4.append(mean_abs_val_D4)
        mean_absolute_values_D3.append(mean_abs_val_D3)
        mean_absolute_values_D2.append(mean_abs_val_D2)
        mean_absolute_values_D1.append(mean_abs_val_D1)
        
        MAEratio_D4_A.append(ratio_MAE_D4_A)
        MAEratio_D3_D4.append(ratio_MAE_D3_D4)
        MAEratio_D2_D3.append(ratio_MAE_D2_D3)
        MAEratio_D1_D2.append(ratio_MAE_D1_D2)
        #available_signal_data.append(signal_data)
        
        standard_deviation_A = np.nanmean(np.nanstd(coeff[0],axis=1))
        standard_deviation_D4 = np.nanmean(np.nanstd(coeff[1],axis=1))
        standard_deviation_D3 = np.nanmean(np.nanstd(coeff[2],axis=1))
        standard_deviation_D2 = np.nanmean(np.nanstd(coeff[3],axis=1))
        standard_deviation_D1 = np.nanmean(np.nanstd(coeff[4],axis=1))
        
        sd_A.append(standard_deviation_A)
        sd_D4.append(standard_deviation_D4)
        sd_D3.append(standard_deviation_D3)
        sd_D2.append(standard_deviation_D2)
        sd_D1.append(standard_deviation_D1)
        
        freqs, psd = signal.welch(coeff[0][:][:], sampling_frequency/8)
        avg_power_A=average_power(psd,freqs,num_channels)
        freqs, psd = signal.welch(coeff[1][:][:], sampling_frequency/8)
        avg_power_D4=average_power(psd,freqs,num_channels)
        freqs, psd = signal.welch(coeff[2][:][:], sampling_frequency/4)
        avg_power_D3=average_power(psd,freqs,num_channels)
        freqs, psd = signal.welch(coeff[3][:][:], sampling_frequency/2)
        avg_power_D2=average_power(psd,freqs,num_channels)
        freqs, psd = signal.welch(coeff[4][:][:], sampling_frequency)
        avg_power_D1=average_power(psd,freqs,num_channels)
        
        average_power_A.append(avg_power_A)
        average_power_D4.append(avg_power_D4)
        average_power_D3.append(avg_power_D3)
        average_power_D2.append(avg_power_D2)
        average_power_D1.append(avg_power_D1)
        
        entr_A=calculate_entropy(coeff[0],num_channels)
        entr_D4=calculate_entropy(coeff[1],num_channels)
        entr_D3=calculate_entropy(coeff[2],num_channels)
        entr_D2=calculate_entropy(coeff[3],num_channels)
        entr_D1=calculate_entropy(coeff[4],num_channels)
        
        entropy_A.append(entr_A)
        entropy_D4.append(entr_D4)
        entropy_D3.append(entr_D3)
        entropy_D2.append(entr_D2)
        entropy_D1.append(entr_D1)

# mean_absolute_values_A=np.nanmean(np.hstack(mean_absolute_values_A))
# mean_absolute_values_D4=np.nanmean(np.hstack(mean_absolute_values_D4))
# mean_absolute_values_D3=np.nanmean(np.hstack(mean_absolute_values_D3))
# mean_absolute_values_D2=np.nanmean(np.hstack(mean_absolute_values_D2))
# mean_absolute_values_D1=np.nanmean(np.hstack(mean_absolute_values_D1))

# MAEratio_D4_A=np.nanmean(np.hstack(MAEratio_D4_A))
# MAEratio_D3_D4=np.nanmean(np.hstack(MAEratio_D3_D4))
# MAEratio_D2_D3=np.nanmean(np.hstack(MAEratio_D2_D3))
# MAEratio_D1_D2=np.nanmean(np.hstack(MAEratio_D1_D2))

# average_power_A=np.nanmean(np.hstack(average_power_A))
# average_power_D4=np.nanmean(np.hstack(average_power_D4))
# average_power_D3=np.nanmean(np.hstack(average_power_D3))
# average_power_D2=np.nanmean(np.hstack(average_power_D2))
# average_power_D1=np.nanmean(np.hstack(average_power_D1))

# sd_A=np.nanmean(np.hstack(sd_A))
# sd_D4=np.nanmean(np.hstack(sd_D4))
# sd_D3=np.nanmean(np.hstack(sd_D3))
# sd_D2=np.nanmean(np.hstack(sd_D2))
# sd_D1=np.nanmean(np.hstack(sd_D1))

# entropy_A=np.nanmean(np.hstack(entropy_A))
# entropy_D4=np.nanmean(np.hstack(entropy_D4))
# entropy_D3=np.nanmean(np.hstack(entropy_D3))
# entropy_D2=np.nanmean(np.hstack(entropy_D2))
# entropy_D1=np.nanmean(np.hstack(entropy_D1))
        
        recording_features = np.hstack((signal_mean, signal_std, signal_power, delta_psd_mean, theta_psd_mean, alpha_psd_mean, 
                                        beta_psd_mean, quality_score, mean_shannon_entropy, mean_coherence, mean_reg, alpha_delta_ratio, 
                                        delta_psd_std, theta_psd_std, alpha_psd_std, beta_psd_std,alpha_delta_ratio_std, mean_shannon_entropy_std, 
                                        mean_reg_std, mean_coherence_std, mean_absolute_values_A,mean_absolute_values_D4,mean_absolute_values_D3,
                                        mean_absolute_values_D2,mean_absolute_values_D1,average_power_A,average_power_D4,average_power_D3,average_power_D2,
                                        average_power_D1,sd_A,sd_D4,sd_D3,sd_D2,sd_D1,MAEratio_D4_A,MAEratio_D3_D4,MAEratio_D2_D3,MAEratio_D1_D2,entropy_A,
                                        entropy_D4,entropy_D3,entropy_D2,entropy_D1))

    else:

        recording_features=np.empty(180) 
        recording_features[:]=np.nan
        

    features = np.hstack((patient_features, recording_features))

    return features

#Function to compute power
def average_power(psd,freqs,num_channels):
    avg_power=np.zeros(num_channels)
    for channel in range(num_channels):
        avg_power[channel] = simps(psd[channel,:], dx=freqs[1] - freqs[0])
    avg_power=np.nanmean(avg_power)    
    return avg_power

def calculate_entropy(list_values,num_channels):
    entropy=np.zeros(num_channels)
    for channel in range(num_channels):
        counter_values = Counter(np.around(list_values[channel,:])).most_common()
        probabilities = [elem[1]/len(list_values) for elem in counter_values]
        entropy[channel]=scipy.stats.entropy(probabilities)
    entropy=np.nanmean(entropy)
    return entropy