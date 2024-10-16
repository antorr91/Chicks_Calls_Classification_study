import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from scipy.io import wavfile
import librosa 
from matplotlib.colors import LinearSegmentedColormap
from math import pi
import seaborn as sns
import itertools
import soundfile as sf
from sklearn.preprocessing import StandardScaler
import cv2
import librosa as lb
import librosa.display
from scipy.signal import correlate2d
import torch
import torch.nn.functional as F


def load_audio_data(audio_path, all_data, call_id, onset, offset, sr=44100):
    """
    Load the audio data, extract the specific segment corresponding to the onset and offset, 
    and generate the spectrogram for the specified call_id.

    Parameters:
    - audio_path (str): Path to the directory containing audio files.
    - all_data (DataFrame): DataFrame containing the call metadata, including 'call_id', 'onsets_sec', and 'offsets_sec'.
    - call_id (str): Identifier of the call.
    - onset (float): Start time of the call in seconds.
    - offset (float): End time of the call in seconds.
    - sr (int): Sample rate for audio loading.

    Returns:
    - dict: A dictionary with 'call_id', 'audio', and 'spectrogram'.
    """
    # Filter the DataFrame for the specific call_id
    audio_data = all_data[all_data['call_id'] == call_id]

    if audio_data.empty:
        raise ValueError(f"Call ID {call_id} not found in the provided DataFrame.")
    
    # Assuming 'recording' column contains the audio file names
    recording_file = audio_data.iloc[0]['recording'] + '.wav'
    file_path = os.path.join(audio_path, recording_file)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file {file_path} not found.")

    # Load the entire audio file
    data, sr = lb.load(file_path, sr=sr)
    
    # Convert onset and offset from seconds to samples
    onset_sample = int(onset * sr)
    offset_sample = int(offset * sr)
    
    # Extract the audio segment corresponding to the onset and offset
    audio_segment = data[onset_sample:offset_sample]
    
    # Compute the mel spectrogram for the segment and use the magnitude in dB
    S = lb.feature.melspectrogram(y=audio_segment, sr=sr, n_mels=128, fmin=2000, fmax=10000, power=1.0)
    S_dB = lb.power_to_db(S, ref=np.max)

    # Plot the spectrogram
    plt.figure(figsize=(10, 5))
    lb.display.specshow(S, sr=sr, x_axis='time', y_axis='mel')
    plt.title(f'Spectrogram for Call ID: {call_id}')
    plt.tight_layout()
    # plt.show()
    plt.close()

    # Return the result as a dictionary
    return {
        'call_id': call_id,
        'audio': audio_segment,
        'spectrogram': S
    }



def plot_compared_spectrograms(test_spectrogram, rep_spectrogram, similarity, test_call_id, rep_dict_id, test_cluster_membership, rep_cluster_membership, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Comparison between Test Call {test_call_id} (Cluster {test_cluster_membership}) and Representative Call {rep_dict_id} (Cluster {rep_cluster_membership})\nSimilarity: {similarity:.4f}", fontsize=14)

    # Plot test spectrogram
    axes[0].imshow(test_spectrogram, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title(f"Test Call {test_call_id} (Cluster {test_cluster_membership})")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Frequency")

    # Plot representative spectrogram
    axes[1].imshow(rep_spectrogram, aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title(f"Representative Call {rep_dict_id} (Cluster {rep_cluster_membership})")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Frequency")

    # Save and show the plot
    plot_save_path = os.path.join(save_path, f"comparison_test_{test_call_id}_cluster_{test_cluster_membership}_rep_{rep_dict_id}_cluster_{rep_cluster_membership}.png")
    plt.savefig(plot_save_path)
    print(f"Saved comparison plot to: {plot_save_path}")





def load_audio_data_with_pcen(audio_path, all_data, call_id, onset, offset, sr=44100, apply_pcen=True):
    """
    Load the audio data, extract the specific segment corresponding to the onset and offset, 
    and generate the spectrogram or PCEN for the specified call_id using a streaming approach.

    Parameters:
    - audio_path (str): Path to the directory containing audio files.
    - all_data (DataFrame): DataFrame containing the call metadata, including 'call_id', 'onsets_sec', and 'offsets_sec'.
    - call_id (str): Identifier of the call.
    - onset (float): Start time of the call in seconds.
    - offset (float): End time of the call in seconds.
    - sr (int): Sample rate for audio loading.
    - apply_pcen (bool): Whether to apply PCEN (Per-Channel Energy Normalization) instead of the regular mel spectrogram.

    Returns:
    - dict: A dictionary with 'call_id', 'audio', 'spectrogram', and optionally 'PCEN' if apply_pcen is True.
    """
    # Filter the DataFrame for the specific call_id
    audio_data = all_data[all_data['call_id'] == call_id]

    if audio_data.empty:
        raise ValueError(f"Call ID {call_id} not found in the provided DataFrame.")
    
    # Assuming 'recording' column contains the audio file names
    recording_file = audio_data.iloc[0]['recording'] + '.wav'
    file_path = os.path.join(audio_path, recording_file)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file {file_path} not found.")

    # Load the specific audio segment
    audio_segment, sr = librosa.load(file_path, sr=sr, offset=onset, duration=offset-onset)

    # Set up STFT parameters
    n_fft = 2048
    hop_length = 512

    # Create a stream for the loaded audio segment
    stream = librosa.util.frame(audio_segment, frame_length=n_fft, hop_length=hop_length).T

    # Initialize variables for PCEN streaming
    pcen_blocks = []
    zi = None
    D = None

    for y_block in stream:
        # Compute the STFT
        D = librosa.stft(y_block, n_fft=n_fft, hop_length=hop_length, center=False, out=D)

        if apply_pcen:
            # Compute PCEN on the magnitude spectrum
            P, zi = librosa.pcen(np.abs(D), sr=sr, hop_length=hop_length, zi=zi, return_zf=True)
            pcen_blocks.extend(P.T)
        else:
            # Compute mel spectrogram
            S = librosa.feature.melspectrogram(S=np.abs(D), sr=sr, n_mels=128, fmin=2000, fmax=10000, power=1.0)
            S_dB = librosa.power_to_db(S, ref=np.max)
            pcen_blocks.extend(S_dB.T)

    # Convert to numpy array
    spectrogram = np.array(pcen_blocks).T

    # Plot the spectrogram or PCEN
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='mel', fmin=2000, fmax=10000)
    plt.colorbar(format='%+2.0f dB')
    title = f'PCEN for Call ID: {call_id}' if apply_pcen else f'Spectrogram for Call ID: {call_id}'
    plt.title(title)
    plt.tight_layout()
    # plt.show()

    # Return the result as a dictionary
    return {
        'call_id': call_id,
        'audio': audio_segment,
        'spectrogram': spectrogram
    }



def split_data_recordings_stratified(all_data, train_recordings=None, group_col='recording', label_col='label', test_ratio=0.3, min_test_samples_per_cluster=6):
    """
    Split data into train and test sets with stratified sampling to ensure balanced clusters in the test set,
    while controlling the proportion of samples in training vs testing.

    Parameters:
    - all_data: DataFrame containing all data.
    - train_recordings: List of recordings to be included in the training set.
    - group_col: Column to group by (e.g., 'recording').
    - label_col: Column containing the cluster labels.
    - test_ratio: Proportion of the data to allocate to the test set (default is 30%).
    - min_test_samples_per_cluster: Minimum number of samples per cluster in the test set.

    Returns:
    - train_data: Training dataset.
    - test_data: Stratified and balanced test dataset.
    """
    # If no recordings are provided for training, use the default list
    if train_recordings is None:
        train_recordings = ['chick32_d0', 'chick34_d0', 'chick41_d0', 
                            'chick85_d0', 'chick87_d0', 'chick89_d0', 'chick91_d0']

    # Split data into training and testing based on recordings
    train_data = all_data[all_data[group_col].isin(train_recordings)]
    test_data = all_data[~all_data[group_col].isin(train_recordings)]

    # Stratified test data collection
    stratified_test_data = []

    # Ensure a minimum number of samples per cluster in the test set
    for cluster in train_data[label_col].unique():
        cluster_samples = test_data[test_data[label_col] == cluster]
        total_cluster_samples = len(cluster_samples)

        # Determine the number of test samples for this cluster
        num_test_samples = max(min_test_samples_per_cluster, int(total_cluster_samples * test_ratio))
        
        # If there aren't enough samples in the test set for this cluster, use all available samples
        selected_samples = cluster_samples.sample(n=min(num_test_samples, total_cluster_samples), random_state=42)
        stratified_test_data.append(selected_samples)

    # Combine the stratified test data into a single DataFrame
    stratified_test_data = pd.concat(stratified_test_data)

    # Remove the selected test samples from the original test_data to avoid duplication
    final_test_data = test_data.drop(stratified_test_data.index)
    
    # Ensure the train and test sets are mutually exclusive
    final_test_data = pd.concat([final_test_data, stratified_test_data])

    return train_data, test_data


def split_data_recordings(all_data, train_recordings=None, group_col='recording', label_col='label'):
    """
    Split data into train and test sets based on predefined recordings for the training set,
    while ensuring a more balanced distribution of samples in the test set across clusters.

    Parameters:
    - all_data: DataFrame containing all data.
    - train_recordings: List of recordings to be included in the training set.
    - group_col: Column to group by (e.g., 'recording').
    - label_col: Column containing the cluster labels.

    Returns:
    - train_data: Training dataset.
    - test_data: Test dataset.
    """
    # If no recordings are provided for training, use the default list
    if train_recordings is None:
        train_recordings = ['chick32_d0', 'chick34_d0','chick39_d0', 'chick41_d0', 'chick85_d0', 'chick87_d0', 'chick89_d0', 'chick91_d0']

    # Split data into training and testing based on recordings
    train_data = all_data[all_data[group_col].isin(train_recordings)]
    test_data = all_data[~all_data[group_col].isin(train_recordings)]

    # Now balance the test data based on the distribution of clusters
    test_data_balanced = []
    
    # Calculate the proportion of each cluster in the training data
    cluster_proportions = train_data[label_col].value_counts(normalize=True)

    # For each cluster, take a proportional number of samples in the test data
    for cluster, proportion in cluster_proportions.items():
        cluster_samples = test_data[test_data[label_col] == cluster]
        n_samples = int(proportion * len(test_data))  # Proportional to the test set size
        if len(cluster_samples) > 0:
            # Select a random sample of proportional size for this cluster in the test set
            sampled_cluster = cluster_samples.sample(n=min(n_samples, len(cluster_samples)), random_state=42)
            test_data_balanced.append(sampled_cluster)

    # Concatenate all the balanced test samples back into a single DataFrame
    test_data_balanced = pd.concat(test_data_balanced, axis=0)

    return train_data, test_data_balanced







def sliding_window(test_spectrogram, rep_spectrogram):
    """
    Create sliding windows over the test spectrogram starting from the middle 
    and sliding toward the end for comparison.
    
    Parameters:
    - test_spectrogram: The spectrogram of the test call (longer one).
    - rep_spectrogram: The spectrogram of the representative call (shorter one).
    
    Returns:
    - windows: List of spectrogram slices (windows) from the test spectrogram.
    """
    
    window_size = rep_spectrogram.shape[1]  # Width of the representative call spectrogram
    test_size = test_spectrogram.shape[1]   # Width of the test call spectrogram

    windows = []
    
    # Start sliding from 30% into the test call (emphasizing the beginning and middle)
    start_point = int(test_size * 0.3)
    
    for start_idx in range(start_point, test_size - window_size + 1):
        end_idx = start_idx + window_size
        window = test_spectrogram[:, start_idx:end_idx]
        windows.append(window)
    
    return windows









def per_channel_standardization(spectrogram):   

    # Compute the mean and standard deviation along the frequency axis
    mean = np.mean(spectrogram, axis=0)
    std = np.std(spectrogram, axis=0)

    # Perform standardization for each channel
    standardized_spectrogram = (spectrogram - mean) / (std + 1e-6)

    return standardized_spectrogram



def plot_and_save_single_audio_segment(representative_calls, audio_path, save_path, cluster_label):
    """
    Extract, plot, and save audio segments for representative calls of a cluster.
    
    Args:
    representative_calls (DataFrame): DataFrame containing representative calls for a cluster
    audio_path (str): Path to the directory containing audio files
    save_path (str): Path to save the results
    cluster_label (str): Label for the cluster
    """
    # Create directory for saving audio files and plots
    cluster_audio_dir = os.path.join(save_path, 'audio')
    os.makedirs(cluster_audio_dir, exist_ok=True)
    
    for idx, (_, call) in enumerate(representative_calls.iterrows()):
        audio_file = os.path.join(audio_path, call['recording'] + '.wav')
        if os.path.exists(audio_file):
            data, sr = lb.load(audio_file, sr=44100)
            S = lb.feature.melspectrogram(y=data, sr=sr, n_mels=128, fmin=2000, fmax=10000)
            log_S = lb.power_to_db(S, ref=np.max)

            # Use your existing segment_spectrogram function
            calls_S = segment_spectrogram(log_S, [call['onsets_sec']], [call['offsets_sec']], sr=sr)
            call_S = calls_S[0]

            # Extract audio segment
            onset_samples = int(call['onsets_sec'] * sr)
            offset_samples = int(call['offsets_sec'] * sr)
            call_audio = data[onset_samples:offset_samples]

            # Plot the spectrogram of the individual call
            plt.figure(figsize=(5, 2))
            plt.imshow(call_S, aspect='auto', origin='lower', cmap='magma')
            plt.title(f'Call {idx + 1} of {call["recording"]} \n {cluster_label}', fontsize=8)
            plt.xlabel('Time', fontsize=7)
            plt.ylabel('Frequency', fontsize=7)
            plt.colorbar(format='%+2.0f dB')
            plt.tight_layout()

            # Save the spectrogram plot
            plot_filename = f'call_{idx + 1}_{call["recording"]}_{cluster_label}.png'
            plt.savefig(os.path.join(cluster_audio_dir, plot_filename))
            plt.close()

            # Save the audio file
            audio_filename = os.path.join(cluster_audio_dir, f'call_{idx + 1}_{call["recording"]}_{cluster_label}.wav')
            sf.write(audio_filename, call_audio, sr)

            print(f'Saved audio and plot for call {idx + 1} of {call["recording"]}')
        else:
            print(f'Audio file {audio_file} not found')

    print(f'All representative calls for {cluster_label} have been processed and saved.')