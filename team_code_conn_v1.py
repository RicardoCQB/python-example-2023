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
from mne_connectivity import spectral_connectivity_time

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
    available_timepoints = list()
    conn_features=list()
    recording_features=list()
    clinical_record=list()
    channels = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'Fp1-F3',
            'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fz-Cz', 'Cz-Pz']

    for i in range(num_patients):
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patients))

        # Load data.
        patient_id = patient_ids[i]
        patient_metadata, recording_metadata, recording_data = load_challenge_data(data_folder, patient_id)

        # Extract features from the recording data and metadata.
        num_channels = len(channels)
        num_recordings = len(recording_data)
        train_or_test='train'

        # Extract available and non-zero signal data
        available_signal_data, available_timepoints, sampling_frequency=get_signal_data(num_channels, num_recordings, channels,
                                                                                        recording_metadata, recording_data, train_or_test)
        if not available_signal_data: # If list is empty - didn't find any signal that met conditions, just skip this subject
            continue 
            
        # Directly selection last available timepoint
        #last_available_timepoint=available_times[-1]
        #signal_data, sampling_frequency, signal_channels = recording_data[last_available_timepoint]
        #last_available_signal=reorder_recording_channels(signal_data, signal_channels, channels)
        
        # Function to select specific timepoints # 12h, 24h, 48, 72h...
        # Now is selecting last available: closest to 72h
        last_available_signal, t_last = select_time_points(available_signal_data, available_timepoints)
        
        # Compute Patient Features
        clinical_feat=patient_features(patient_metadata)
        clinical_record.append(clinical_feat)
        
        # Compute Coherence 
        mean_coh_total, mean_coh_by_channel=connectivity_features(last_available_signal)
        mean_coh_by_channel=np.hstack(mean_coh_by_channel) # from (6,18,1) features to (108,1) features

        conn_features.append(np.concatenate((mean_coh_total, mean_coh_by_channel)))
        
        # Compute Signal Features
        signal_feat=signal_features(recording_metadata, available_signal_data, last_available_signal, 
                                       t_last, num_channels, sampling_frequency)
        recording_features.append(signal_feat)


        # Extract labels.
        current_outcome = get_outcome(patient_metadata)
        outcomes.append(current_outcome)
        current_cpc = get_cpc(patient_metadata)
        cpcs.append(current_cpc)

    recording_features=np.stack(recording_features)
    conn_features=np.stack(conn_features)
    clinical_record=np.stack(clinical_record)
    
    features=np.hstack((clinical_record,recording_features, conn_features))
    outcomes = np.vstack(outcomes)
    cpcs = np.vstack(cpcs)

    # Train the models.
    if verbose >= 1:
        print('Training the Challenge models on the Challenge data...')

    # Define parameters for random forest classifier and regressor.
    n_estimators   = 123  # Number of trees in the forest.
    max_leaf_nodes = 456  # Maximum number of leaf nodes in each tree.
    random_state   = 789  # Random state; set for reproducibility.

    # Impute any missing features; use the mean value by default.
    imputer = SimpleImputer().fit(features)

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
    # CHANGE HERE!
    
    # Extract features from the recording data and metadata.
    channels = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'Fp1-F3',
                'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fz-Cz', 'Cz-Pz']
    num_channels = len(channels)
    num_recordings = len(recording_data)
    train_or_test='test'
    #train_or_test='train'
    
    # Extract available and non-zero signal data
    available_signal_data, available_timepoints, sampling_frequency=get_signal_data(num_channels, 
                                                                                    num_recordings, channels, recording_metadata, 
                                                                                    recording_data, train_or_test)
    # Directly selection last available timepoint
    #last_available_timepoint=available_times[-1]
    #signal_data, sampling_frequency, signal_channels = recording_data[last_available_timepoint]
    #last_available_signal=reorder_recording_channels(signal_data, signal_channels, channels)
        
    # Function to select specific timepoints # 12h, 24h, 48, 72h...
    # Now is selecting last available: closest to 72h
    last_available_signal, t_last = select_time_points(available_signal_data, available_timepoints)
        
    # Compute Patient Features
    clinical_feat=patient_features(patient_metadata)
        
    # Compute Coherence 
    mean_coh_total, mean_coh_by_channel=connectivity_features(last_available_signal)
    mean_coh_by_channel=np.hstack(mean_coh_by_channel) # from (6,18,1) features to (108,1) features

        
    # Compute Signal Features
    signal_feat=signal_features(recording_metadata, available_signal_data, last_available_signal, 
                                       t_last, num_channels, sampling_frequency)

    features=np.hstack((clinical_feat,signal_feat,mean_coh_total, mean_coh_by_channel)).reshape(1, -1)
    
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

def get_signal_data(num_channels, num_recordings, channels, recording_metadata, recording_data, train_or_test):
    
    available_signal_data = list()
    available_timepoints = list()
    
    available_times=[i for i,tup in enumerate(recording_data) if all(elem is not None for elem in tup)]
    for i in available_times:
        signal_data, sampling_frequency, signal_channels = recording_data[i]
        
        if train_or_test=='train':
            quality_score = get_quality_scores(recording_metadata)[i] # quality score of last available
            
            if (signal_data is not None) & (np.sum(signal_data)!=0) & (quality_score>0.5):
                signal_data = reorder_recording_channels(signal_data, signal_channels, channels) # Reorder the channels in the signal data, as needed, for consistency across different recordings.
                available_signal_data.append(signal_data)
                available_timepoints.append(i)
        
        elif train_or_test=='test':
            if (signal_data is not None) & (np.sum(signal_data)!=0):
                signal_data = reorder_recording_channels(signal_data, signal_channels, channels) # Reorder the channels in the signal data, as needed, for consistency across different recordings.
                available_signal_data.append(signal_data)
                available_timepoints.append(i)
            
    return available_signal_data, available_timepoints, sampling_frequency

    
def select_time_points(available_signal_data, available_time_point):
    """
    # Check what is the maximum time_point available for the recordings
    
    max_hour_recorded=np.argmax(np.array(available_time_point))
    
    if max_hour_recorded > 48: # we can have a time point corresponding to the 3rd day
        # we can define all time_points
        
    elif max_hour_recorded>24: # The last available time point is not bigger than 48h but we know that there is at least hour 25th - second day
        # we can define time_point 12h, 24h, 48h, and set 72h to zero(?)
        
    elif max_hour_recorded>12: # the last available time point is not after 24h, but there is signal during the second half of first day
        # we can define the 12h and 24h, - set 48h and 72h to zero or nan (?)
    
    elif max_hour_recorded<12: # the last available time point only allows to have signal during the first half of the first day
        # we define the 12h - set, 24h, 48h, and 72h to (?)
    """
    # I select the available at 12h, 24h, 48h, 72h
    
    # TO DO - IF IT'S NOT ZERO
    
    #t_12=np.argmin(np.abs(np.array(available_time_point)-12))
    #t_24=np.argmin(np.abs(np.array(available_time_point)-24))
    #t_48=np.argmin(np.abs(np.array(available_time_point)-48))
    t_72=np.argmin(np.abs(np.array(available_time_point)-72)) # The last available point
    
    #Check is the time_points are different from each other
    
    four_signal_data=list()
    #four_signal_data.append(available_signal_data[t_12])
    #four_signal_data.append(available_signal_data[t_24])
    #four_signal_data.append(available_signal_data[t_48])
    four_signal_data.append(available_signal_data[t_72])
    
    return np.stack(four_signal_data), t_72 # np.stack(four_signal_data) for (4, 18, 30000) for 1 subject here 

# re-reference EEG?

def connectivity_features(four_signal_data):
    #PLV Theta CZ, PLV theta T3, PLV alpha Fp2, ciCOH Beta CZ, PLV Delta F3
    # ciCoh PLV
    eeg_filter_bands=list()
    mean_coh=list()
    mean_coh_by_channel=list()
    
    filter_params ={'delta': (1, 4),'theta': (4, 8),'alpha': (8, 12),'beta1': (12, 18),'beta2': (18, 25),'gamma': (25, 45)}
    
    for i, signal in enumerate(four_signal_data):
        # Time point i
        
        for band_name, (low, high) in filter_params.items(): # 6 frequency bands for 4 Time points (18, 30000)
            
            signal = np.array(signal, dtype=np.float64) 
            signal_band=mne.filter.filter_data(signal, 100, low, high, method='fir', 
                                               fir_design='firwin', filter_length='auto', phase='zero-double', verbose=False)
            
            #from scipy import signal
            #sos = signal.butter(6,[0.56, 40], 'bandpass', fs=100, output='sos')
            #signal_data = signal.sosfilt(sos, signal_data)
            
            #fir_window='chebyshev'
            #signal_band=mne.filter.filter_data(signal, sfreq=100, l_freq=low, h_freq=high, method='fir', l_trans_bandwidth='auto', h_trans_bandwidth='auto', filter_length=1025)
            
            #eeg_filter_bands.append(mne.filter.filter_data(signal, sfreq=100, l_freq=low, h_freq=high, method='fir',fir_window='chebyshev', l_trans_bandwidth='auto', h_trans_bandwidth='auto', filter_length=1025))
            
            # Signal is (18, 30000) filter for freq_band
            ten_segments=np.stack(np.split(signal_band, 30, axis=-1)) # (30, 18, 1000)
            freqs=np.array([low, high])
            n_cycles=freqs / 2
            con=spectral_connectivity_time(ten_segments, freqs=freqs, method='coh', average=True, indices=None, sfreq=100,
                                           fmin=low, fmax=high, fskip=0, faverage=True, 
                                           sm_times=0, sm_freqs=1,sm_kernel='hanning', padding=0, mode='cwt_morlet', 
                                           mt_bandwidth=None,n_cycles=n_cycles, decim=1, n_jobs=1, verbose=None)
            
            # Get a symmetric connectivity matrix
            
            #con.get_data(output="dense")[:,:,1].T (18,18,n_freq)
            
            con_matrix=con.get_data().reshape(18,18).T # [:,1]-> where axis 1 is n_frequency bands; each row is channel i -> all others (1,2,3,...)
            #print(con_matrix)
            r,c=np.tril_indices(18, -1) # the lower triangle index
            con_matrix[r,c]= con_matrix.T[r,c] # copy the upper triangle to the lower triangle into a symmetric matrix
            
            # Average COH by channel
            mean_coh_by_channel.append(np.nanmean(np.where(con_matrix!=0,con_matrix,np.nan),axis=0)) # Do the mean over channel ignoring diagonal
            mean_coh.append(np.nanmean(np.where(con_matrix!=0,con_matrix,np.nan))) # Total mean connectivity for 5min epoch for freqband for timepoint ?
            
        mean_coh=np.stack(mean_coh) #(x,6,1)

        mean_coh_by_channel=np.stack(mean_coh_by_channel) #(6, 18, 1)
        print(mean_coh_by_channel.shape) # shape is (6, 18, 1)
        
    return mean_coh, mean_coh_by_channel #mean_coh[0], mean_coh[1], mean_conh[2], mean_coh[3], mean_coh[4], mean_coh[5]

def patient_features(patient_metadata):
    # Extract features from the patient metadata.
    age = get_age(patient_metadata)
    sex = get_sex(patient_metadata)
    rosc = get_rosc(patient_metadata)
    ohca = get_ohca(patient_metadata)
    vfib = get_vfib(patient_metadata)
    ttm = get_ttm(patient_metadata)

    # Use one-hot encoding for sex; add more variables
    sex_features = np.zeros(2, dtype=int)
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
    patient_features = np.array([age, female, male, other, rosc, ttm_33, ttm_36, ttm_none]) #age, female, male, other, rosc, ohca, vfib, ttm
    
    return patient_features

# Extract features from the data.
def signal_features(recording_metadata, available_signal_data, last_available_signal, t_last, num_channels, sampling_frequency):
    

    if len(available_signal_data) > 0:
        available_signal_data = np.hstack(available_signal_data) # It concatenates all 5min epochs 
        signal_mean = np.nanmean(available_signal_data, axis=1)
        signal_std  = np.nanstd(available_signal_data, axis=1)      
    else:
        print('Patient {} has no available data '.format(patient_id))
        signal_mean = float('nan') * np.ones(num_channels)
        signal_std  = float('nan') * np.ones(num_channels)
        
    last_available_signal=np.hstack(last_available_signal)
    delta_psd, _ = mne.time_frequency.psd_array_welch(last_available_signal, sfreq=sampling_frequency,  fmin=0.5,  fmax=4.0, verbose=False)
    theta_psd, _ = mne.time_frequency.psd_array_welch(last_available_signal, sfreq=sampling_frequency,  fmin=4.0,  fmax=8.0, verbose=False)
    alpha_psd, _ = mne.time_frequency.psd_array_welch(last_available_signal, sfreq=sampling_frequency,  fmin=8.0, fmax=12.0, verbose=False)
    beta_psd,  _ = mne.time_frequency.psd_array_welch(last_available_signal, sfreq=sampling_frequency, fmin=12.0, fmax=30.0, verbose=False)

    delta_psd_mean = np.nanmean(delta_psd, axis=1)
    theta_psd_mean = np.nanmean(theta_psd, axis=1)
    alpha_psd_mean = np.nanmean(alpha_psd, axis=1)
    beta_psd_mean  = np.nanmean(beta_psd,  axis=1)

    quality_score = get_quality_scores(recording_metadata)[t_last] # quality score of last available
    
    # For get_features function it needs to output the features for 1 patient at a time

    recording_features=np.hstack((signal_mean, signal_std, delta_psd_mean, theta_psd_mean, alpha_psd_mean, beta_psd_mean, quality_score))

    return recording_features
