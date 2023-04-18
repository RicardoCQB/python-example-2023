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
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from scipy import signal
from scipy.integrate import simps
import joblib
import pywt
import matplotlib.pyplot as plt
from collections import Counter

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
        print(patient_id)
        patient_metadata, recording_metadata, recording_data = load_challenge_data(data_folder, patient_id)

        # Extract features.
        current_features, quality_score_array = get_features(patient_metadata, recording_metadata, recording_data)

        # If the maximum quality index score is 1, then it uses the features of this patient.
        good_quality_indexes = np.where(quality_score_array == 1)

        if len(good_quality_indexes[0]) > 0:
            features.append(current_features)
            # Extract labels.
            current_outcome = get_outcome(patient_metadata)
            outcomes.append(current_outcome)
            current_cpc = get_cpc(patient_metadata)
            cpcs.append(current_cpc)
        else:
            if verbose >= 2:
                print('    Patient {} was not used because the quality index score was not 1.'.format(patient_id))

    features = np.vstack(features)
    outcomes = np.vstack(outcomes)
    cpcs = np.vstack(cpcs)

    # Train the models.
    if verbose >= 1:
        print('Training the Challenge models on the Challenge data...')

    # Define parameters for random forest classifier and regressor.
    n_estimators   = 100  # Number of trees in the forest.
    max_leaf_nodes = 25  # Maximum number of leaf nodes in each tree.
    random_state   = 42  # Random state; set for reproducibility.

    # Impute any missing features; use the mean value by default.
    imputer = KNNImputer().fit(features)

    # Train the models.
    features = imputer.transform(features)
    outcome_model = RandomForestClassifier(
        n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, outcomes.ravel())
    cpc_model = RandomForestRegressor(
        n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, cpcs.ravel())

    # Save the models.
    save_challenge_model(model_folder, imputer, outcome_model, cpc_model)

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
    features, quality_score_array = get_features(patient_metadata, recording_metadata, recording_data)
    features = features.reshape(1, -1)

    # Impute missing data.
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
    num_channels = len(channels)        #channels - 18
    num_recordings = len(recording_data) #hours - 72

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
    #Get quality score
    quality_score_array=np.zeros([num_recordings])

    for i in range(num_recordings):
        quality_score_array[i] = get_quality_scores(recording_metadata)[i]

    good_quality_indexes = np.where(quality_score_array == np.nanmax(quality_score_array))

    for i in np.nditer(good_quality_indexes[0]):
        signal_data, sampling_frequency, signal_channels = recording_data[i]
        if signal_data is not None:
            signal_data = reorder_recording_channels(signal_data, signal_channels, channels) # Reorder the channels in the signal data, as needed, for consistency across different recordings.
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

    mean_absolute_values_A=np.nanmean(np.hstack(mean_absolute_values_A))
    mean_absolute_values_D4=np.nanmean(np.hstack(mean_absolute_values_D4))
    mean_absolute_values_D3=np.nanmean(np.hstack(mean_absolute_values_D3))
    mean_absolute_values_D2=np.nanmean(np.hstack(mean_absolute_values_D2))
    mean_absolute_values_D1=np.nanmean(np.hstack(mean_absolute_values_D1))
    
    MAEratio_D4_A=np.nanmean(np.hstack(MAEratio_D4_A))
    MAEratio_D3_D4=np.nanmean(np.hstack(MAEratio_D3_D4))
    MAEratio_D2_D3=np.nanmean(np.hstack(MAEratio_D2_D3))
    MAEratio_D1_D2=np.nanmean(np.hstack(MAEratio_D1_D2))
    
    average_power_A=np.nanmean(np.hstack(average_power_A))
    average_power_D4=np.nanmean(np.hstack(average_power_D4))
    average_power_D3=np.nanmean(np.hstack(average_power_D3))
    average_power_D2=np.nanmean(np.hstack(average_power_D2))
    average_power_D1=np.nanmean(np.hstack(average_power_D1))
    
    sd_A=np.nanmean(np.hstack(sd_A))
    sd_D4=np.nanmean(np.hstack(sd_D4))
    sd_D3=np.nanmean(np.hstack(sd_D3))
    sd_D2=np.nanmean(np.hstack(sd_D2))
    sd_D1=np.nanmean(np.hstack(sd_D1))
    
    entropy_A=np.nanmean(np.hstack(entropy_A))
    entropy_D4=np.nanmean(np.hstack(entropy_D4))
    entropy_D3=np.nanmean(np.hstack(entropy_D3))
    entropy_D2=np.nanmean(np.hstack(entropy_D2))
    entropy_D1=np.nanmean(np.hstack(entropy_D1))
    # if len(available_signal_data) > 0:
    #     available_signal_data = np.hstack(available_signal_data)
    #     signal_mean = np.nanmean(available_signal_data, axis=1)
    #     signal_std  = np.nanstd(available_signal_data, axis=1)
    # else:
    #     signal_mean = float('nan') * np.ones(num_channels)
    #     signal_std  = float('nan') * np.ones(num_channels)

    # Compute the power spectral density for the delta, theta, alpha, and beta frequency bands for each channel of the most
    # recent recording.
    # index = None
    # quality_reversed=good_quality_indexes[0][::-1]
    # for i in np.nditer(good_quality_indexes[0]):
    #     signal_data, sampling_frequency, signal_channels = recording_data[i]
    #     if signal_data is not None:
    #         index = i
    #         break

    # if index is not None:
    #     signal_data, sampling_frequency, signal_channels = recording_data[index]
    #     signal_data = reorder_recording_channels(signal_data, signal_channels, channels) # Reorder the channels in the signal data, as needed, for consistency across different recordings.

    #     delta_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency,  fmin=0.5,  fmax=4.0, verbose=False)
    #     theta_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency,  fmin=4.0,  fmax=8.0, verbose=False)
    #     alpha_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency,  fmin=8.0, fmax=12.0, verbose=False)
    #     beta_psd,  _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency, fmin=12.0, fmax=30.0, verbose=False)

    #     delta_psd_mean = np.nanmean(delta_psd, axis=1)
    #     theta_psd_mean = np.nanmean(theta_psd, axis=1)
    #     alpha_psd_mean = np.nanmean(alpha_psd, axis=1)
    #     beta_psd_mean  = np.nanmean(beta_psd,  axis=1)

    #     quality_score = get_quality_scores(recording_metadata)[index]
    # else:
    #     delta_psd_mean = theta_psd_mean = alpha_psd_mean = beta_psd_mean = float('nan') * np.ones(num_channels)
    #     quality_score = float('nan')

    recording_features = np.hstack((mean_absolute_values_A,mean_absolute_values_D4,mean_absolute_values_D3,
                                    mean_absolute_values_D2,mean_absolute_values_D1,average_power_A,average_power_D4,
                                    average_power_D3,average_power_D2,average_power_D1,sd_A,sd_D4,sd_D3,sd_D2,sd_D1,
                                    MAEratio_D4_A,MAEratio_D3_D4,MAEratio_D2_D3,MAEratio_D1_D2,entropy_A,entropy_D4,
                                    entropy_D3,entropy_D2,entropy_D1))#signal_mean, signal_std, delta_psd_mean, theta_psd_mean, alpha_psd_mean, beta_psd_mean, quality_score))

    # Combine the features from the patient metadata and the recording data and metadata.
    features = np.hstack((patient_features, recording_features))

    return features, quality_score_array

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