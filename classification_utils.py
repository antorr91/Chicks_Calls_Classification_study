import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from scipy.io import wavfile
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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
import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from matplotlib_venn import venn2
from sklearn.model_selection import PredefinedSplit, GridSearchCV, train_test_split

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

    all_data['call_id'] = all_data['call_id'].str.replace(r'_call_', '_', regex=True)

    call_id = call_id.replace('_call_', '_')

    audio_data = all_data[all_data['call_id'] == call_id]
    # the audio data is given by the recording column
    # audio_data = all_data[all_data['recording'] == call_id]

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
    S = lb.feature.melspectrogram(y=audio_segment, sr=sr, n_mels=128, fmin=2000, fmax=12600, power=1.0)
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



# def load_audio_data(audio_path, all_data, call_id, onset, offset, sr=44100):
#     """
#     Load the audio data, extract the specific segment corresponding to the onset and offset, 
#     and generate the spectrogram for the specified call_id.
#     """

#     # Rimuovo '_call_' per uniformare il formato, se necessario
#     formatted_call_id = call_id.replace('_call_', '_')

#     # Filtro il DataFrame per trovare la riga corretta
#     audio_data = all_data[all_data['call_id'] == formatted_call_id]

#     if audio_data.empty:
#         raise ValueError(f"Call ID {call_id} not found in the provided DataFrame.")

#     # Estraggo il nome del file di registrazione corrispondente
#     recording_name = audio_data.iloc[0]['recording']
#     recording_file = f"{recording_name}.wav"

#     file_path = os.path.join(audio_path, recording_file)

#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"Audio file {file_path} not found.")

#     # Carico il file audio
#     data, sr = lb.load(file_path, sr=sr)

#     # Converti onset e offset in campioni
#     onset_sample = int(onset * sr)
#     offset_sample = int(offset * sr)

#     # Estraggo il segmento di audio desiderato
#     audio_segment = data[onset_sample:offset_sample]

#     # Calcolo lo spettrogramma mel senza conversione in dB
#     S = lb.feature.melspectrogram(y=audio_segment, sr=sr, n_mels=128, fmin=2000, fmax=12600, power=1.0)

#     # Plotta lo spettrogramma (opzionale)
#     plt.figure(figsize=(10, 5))
#     lb.display.specshow(S, sr=sr, x_axis='time', y_axis='mel')
#     plt.title(f'Spectrogram for Call ID: {call_id}')
#     plt.tight_layout()
#     plt.close()

#     return {
#         'call_id': call_id,
#         'audio': audio_segment,
#         'spectrogram': S
#     }






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




def split_data_recordings_stratified(
    all_data, train_recordings=None, group_col='recording', label_col='label', 
    test_ratio=0.35, min_test_samples_per_cluster=50, validation_ratio=0.10):
    """
    Split data into train, validation, and test sets with stratified sampling 
    to ensure balanced clusters in each set, while controlling the proportion of samples 
    in training, validation, and testing.

    Parameters:
    - all_data: DataFrame containing all data.
    - train_recordings: List of recordings to be included in the training set.
    - group_col: Column to group by (e.g., 'recording').
    - label_col: Column containing the cluster labels.
    - test_ratio: Proportion of the data to allocate to the test set (default is 35%).
    - min_test_samples_per_cluster: Minimum number of samples per cluster in the test set.
    - validation_ratio: Proportion of the training data to allocate to the validation set (default is 10%).

    Returns:
    - new_train_data: Training dataset (after splitting).
    - validation_data: Validation dataset.
    - final_test_data: Stratified and balanced test dataset.
    """
    # If no recordings are provided for training, use the default list
    if train_recordings is None:
        train_recordings = ['chick32_d0', 'chick34_d0', 'chick39_d0', 'chick41_d0', 
                            'chick85_d0', 'chick87_d0', 'chick89_d0', 'chick91_d0']

    # Split data into training and testing based on recordings
    train_data = all_data[all_data[group_col].isin(train_recordings)]
    test_data = all_data[~all_data[group_col].isin(train_recordings)]

    # Limit calls from chick39_d0 to 35% excluding the 5th percentile
    if 'chick39_d0' in train_recordings:
        chick39_data = train_data[train_data[group_col] == 'chick39_d0']
        
        # Calculate the 5th percentile threshold
        fifth_percentile_threshold = chick39_data['distance_to_center'].quantile(0.05)
        
        # Filter out the calls below the 5th percentile
        chick39_filtered = chick39_data[chick39_data['distance_to_center'] <= fifth_percentile_threshold]
        
        # Get the remaining calls excluding the 5th percentile
        chick39_remaining = chick39_data[chick39_data['distance_to_center'] > fifth_percentile_threshold]
        
        # Calculate the limit for the remaining calls (35% of the total remaining calls)
        remaining_limit = int(len(chick39_remaining) * 0.25)
        
        # Sample from the remaining calls
        if remaining_limit > 0:
            chick39_limited = chick39_remaining.sample(n=remaining_limit, random_state=42)
        else:
            chick39_limited = pd.DataFrame(columns=chick39_remaining.columns)  # Empty DataFrame if limit is 0

        # Combine the filtered 5th percentile calls with the limited remaining calls
        chick39_final_train = pd.concat([chick39_filtered, chick39_limited])
        
        # Remove the old chick39 calls from the train_data and add the limited version
        train_data = train_data[train_data[group_col] != 'chick39_d0']
        train_data = pd.concat([train_data, chick39_final_train])

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

    # Now split the training data into new training and validation sets
    stratified_train_data = []
    stratified_validation_data = []

    for cluster in train_data[label_col].unique():
        cluster_samples = train_data[train_data[label_col] == cluster]
        total_cluster_samples = len(cluster_samples)

        # Calculate the number of validation samples for this cluster
        num_validation_samples = int(total_cluster_samples * validation_ratio)
        
        # Sample validation data for this cluster
        validation_samples = cluster_samples.sample(n=num_validation_samples, random_state=42)
        train_samples = cluster_samples.drop(validation_samples.index)
        
        stratified_train_data.append(train_samples)
        stratified_validation_data.append(validation_samples)

    # Combine the stratified training and validation data into DataFrames
    new_train_data = pd.concat(stratified_train_data)
    validation_data = pd.concat(stratified_validation_data)

    return new_train_data, validation_data, final_test_data







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


def segment_spectrogram(spectrogram, onsets, offsets, sr=44100):
    # Initialize lists to store spectrogram slices
    calls_S = []
    # Loop through each onset and offset pair
    for onset, offset in zip(onsets, offsets):
        # Convert time (in seconds) to sample indices
        onset_frames = lb.time_to_frames(onset, sr=sr)
        offset_frames = lb.time_to_frames(offset, sr=sr)

        call_spec = spectrogram[:, onset_frames: offset_frames]
        # call_audio = audio_data[onset_frames: offset_frames]

        # Append the scaled log-spectrogram slice to the calls list
        calls_S.append(call_spec)
        # calls_audio.append(call_audio)
    
    return calls_S

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



def count_calls_by_cluster(data, group_col='recording', label_col='label'):
    """
    Counts the number of calls for each cluster in each recording.

    Parameters:
    - data: DataFrame containing the dataset with recordings and cluster labels.
    - group_col: Column name for the recordings.
    - label_col: Column name for the cluster labels.

    Returns:
    - A DataFrame showing the count of calls for each cluster in each recording.
    """
    # Group the data by the recording and cluster label, then count the occurrences
    call_counts = data.groupby([group_col, label_col]).size().reset_index(name='count')
    
    # Pivot the DataFrame so that clusters are columns
    call_counts_pivot = call_counts.pivot(index=group_col, columns=label_col, values='count').fillna(0)
    
    # Rename the columns to make them more understandable (Cluster 0, Cluster 1, etc.)
    call_counts_pivot.columns = [f'Cluster {int(col)}' for col in call_counts_pivot.columns]
    
    return call_counts_pivot




def count_calls_by_cluster_in_train_test(data_dict, group_col='recording', label_col='label'):
    """
    Counts the number of calls for each cluster in each recording for both training and testing sets.

    Parameters:
    - data_dict: Dictionary containing 'train' and 'test' DataFrames with recordings and cluster labels.
    - group_col: Column name for the recordings.
    - label_col: Column name for the cluster labels.

    Returns:
    - A dictionary containing two DataFrames: counts for training and testing sets.
    """
    counts = {}

    for key, data in data_dict.items():
        # Group the data by the recording and cluster label, then count the occurrences
        call_counts = data.groupby([group_col, label_col]).size().reset_index(name='count')
        
        # Pivot the DataFrame so that clusters are columns
        call_counts_pivot = call_counts.pivot(index=group_col, columns=label_col, values='count').fillna(0)
        
        # Rename the columns to make them more understandable (Cluster 0, Cluster 1, etc.)
        call_counts_pivot.columns = [f'Cluster {int(col)}' for col in call_counts_pivot.columns]
        
        # Store the result in the counts dictionary
        counts[key] = call_counts_pivot

    return counts


def print_and_export_metrics(y_test, y_pred, model_name, output_file=None):
    """
    Print performance metrics and export classification report to LaTeX.
    
    Parameters:
    -----------
    y_test : array-like
        True labels
    y_pred : array-like
        Predicted labels
    model_name : str
        Name of the model (e.g., 'MLP' or 'SVM')
    output_file : str, optional
        Path to output LaTeX file. If None, uses 'model_name_results.tex'
    """
    
    # Imposta il nome del file di output di default basato sul nome del modello
    if output_file is None:
        output_file = f"{model_name.lower()}_results.tex"
    
    # Print metrics to console
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Classifier Results for {model_name}:")
    print(f"Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Convert classification report to DataFrame
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    
    # Crea il DataFrame con i metriche desiderate
    metrics_dict = {key: value for key, value in report_dict.items() if isinstance(value, dict)}
    df = pd.DataFrame(metrics_dict).transpose()
    
    # Rinomina colonne per maggiore leggibilità
    df.columns = ['Precision', 'Recall', 'F1-Score', 'Support']
    df['Support'] = df['Support'].astype(int)
    
    # Crea tabella in formato LaTeX
    latex_table = df.to_latex(
        float_format=lambda x: '{:0.3f}'.format(x) if isinstance(x, (float, np.float64)) else '{:d}'.format(x),
        caption=f'{model_name} Classification Results',
        label=f'tab:{model_name.lower()}_results',
        escape=False
    )
    
    # Contenuto del documento LaTeX
    latex_content = r"""\documentclass{article}
\usepackage{booktabs}
\usepackage{caption}
\begin{document}

\section{""" + f"{model_name} Classification Results" + r"""}
\subsection{Performance Metrics}
Overall Accuracy: """ + f"{accuracy:.3f}" + r"""

\subsection{Classification Report}
""" + latex_table + r"""

\end{document}
"""
    
    # Scrive nel file di output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(latex_content)
    
    print(f"\nResults exported to {output_file}")



def analyse_misclassifications(y_true, y_pred, original_data, model_name, features):
    """
    Analyze misclassified samples and return detailed information
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    original_data : pandas.DataFrames
        Original dataset with all features and metadata
    model_name : str
        Name of the model (e.g., 'MLP' or 'SVM')
    features : list
        List of feature names used for classification
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing misclassified samples with detailed information
    """
    # Create mask and indices for misclassified samples
    misclassified_mask = y_true != y_pred
    misclassified_indices = np.where(misclassified_mask)[0]
    
    # Create DataFrame with misclassification details
    misclassified_df = pd.DataFrame({
        'True_Label': y_true[misclassified_mask],
        'Predicted_Label': y_pred[misclassified_mask],
        'Index': misclassified_indices
    })
    
    call_ids = original_data['call_id'].values


    features_data =  original_data.drop(['Call Number', 'onsets_sec', 'offsets_sec', 'recording', 'call_id', 'cluster_membership', 'distance_to_center', 'label'], axis=1)

    features_data_scaled = StandardScaler().fit_transform(features_data)
    # Extract call IDs for misclassified samples

    # Align misclassified details from the original data
    misclassified_details = original_data.iloc[misclassified_indices].reset_index(drop=True)
    misclassified_df = pd.concat([misclassified_df.reset_index(drop=True), misclassified_details], axis=1)
    
    # Save misclassified samples to CSV
    output_path = f'misclassified_{model_name.lower()}.csv'
    misclassified_df.to_csv(output_path, index=False)
    print(f"Misclassified samples saved to {output_path}")
    
    # # Visualization: Confusion Matrix for Misclassifications :: this is unnecessary now
    # plt.figure(figsize=(12, 6))
    # confusion = confusion_matrix(y_true[misclassified_mask], y_pred[misclassified_mask])
    # sns.heatmap(confusion, annot=True, fmt='d',
    #             xticklabels=['Class 0', 'Class 1', 'Class 2'],
    #             yticklabels=['Class 0', 'Class 1', 'Class 2'])
    # plt.title(f'Confusion Matrix of Misclassifications - {model_name}')
    # plt.xlabel('Predicted Label')
    # plt.ylabel('True Label')
    # plt.savefig(f'misclassification_patterns_{model_name.lower()}.png')
    # plt.close()
    # Feature distribution analysis for misclassified samples
# Feature distribution analysis for misclassified samples
    plt.figure(figsize=(15, 5))
    for i, feature in enumerate(features[:5]):  # Plot the first 5 features
        plt.subplot(1, 5, i+1)
        
        # Drop NaN values to prevent plotting errors
        feature_data = misclassified_df[['True_Label', feature]].dropna()
        
        # Check if there’s data to plot
        if not feature_data.empty:
            sns.boxplot(data=feature_data, x='True_Label', y=feature)
            plt.title(f'{feature} Distribution')
            plt.xticks(rotation=45)
        else:
            print(f"No data available for feature '{feature}' to plot.")
            plt.text(0.5, 0.5, 'No Data', ha='center')

    plt.tight_layout()
    plt.savefig(f'feature_distribution_misclassified_{model_name.lower()}.png')
    plt.close()
    
    # Print summary statistics
    print(f"\nMisclassification Analysis for {model_name}:")
    print(f"Total misclassified samples: {len(misclassified_df)}")
    print("\nMisclassification patterns:")
    for true_label in sorted(misclassified_df['True_Label'].unique()):
        mask = misclassified_df['True_Label'] == true_label
        print(f"\nTrue Class {true_label}:")
        print(misclassified_df[mask]['Predicted_Label'].value_counts())
    
    # Plot audio segments of misclassified samples
    plot_and_save_audio_segments(misclassified_df, original_data, 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Data\\high_quality_dataset', model_name)

    return misclassified_df

def plot_and_save_audio_segments(misclassified_df, original_data, audio_path, model_name):
    """
    Plot and save audio segments for misclassified samples.
    
    Parameters:
    -----------
    misclassified_df : pandas.DataFrame
        DataFrame containing misclassified samples
    original_data : pandas.DataFrame
        Original dataset with all features and metadata
    audio_path : str
        Path to the directory containing audio files
    model_name : str
        Name of the model (e.g., 'MLP' or 'SVM')
    """
    # Use formatted string to include model_name in the path
    save_path = f'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Clustering_\\Distance_based_hac_clustering_and_classification\\misclassified_audio_segments_{model_name}\\'
    
    os.makedirs(save_path, exist_ok=True)

    for _, call in misclassified_df.iterrows():
        audio_file = os.path.join(audio_path, call['recording'] + '.wav')
        if os.path.exists(audio_file):
            data, sr = lb.load(audio_file, sr=44100)
            onset_samples = int(call['onsets_sec'] * sr)
            offset_samples = int(call['offsets_sec'] * sr)
            call_audio = data[onset_samples:offset_samples]

            # Save the audio file
            audio_filename = os.path.join(save_path, f'misclassified_call_{call["Index"]}.wav')
            sf.write(audio_filename, call_audio, sr)

            # Optional: Plot the spectrogram for the individual call
            S = lb.feature.melspectrogram(y=call_audio, sr=sr, n_mels=128, fmin=2000, fmax=10000)
            log_S = lb.power_to_db(S, ref=np.max)

            plt.figure(figsize=(3, 2))
            plt.imshow(log_S, aspect='auto', origin='lower', cmap='magma')
            plt.title(f'Misclassified Call: {call["recording"]} (Index: {call["Index"]})', fontsize=8)
            plt.xlabel('Time', fontsize=7)
            plt.ylabel('Frequency', fontsize=7)
            plt.colorbar(format='%+2.0f dB')
            plt.tight_layout()

            # Save the spectrogram plot
            plot_filename = os.path.join(save_path, f'misclassified_call_{call["Index"]}_spectrogram.png')
            plt.savefig(plot_filename)
            plt.close()

            print(f'Saved audio segment and plot for misclassified call: {audio_filename}')
        else:
            print(f'Audio file {audio_file} not found')




def compare_model_errors(mlp_errors, svm_errors, original_data, save_directory):
    """
    Compare misclassifications between MLP and SVM models.
    
    Parameters:
    -----------
    mlp_errors : pandas.DataFrame
        DataFrame with MLP misclassifications.
    svm_errors : pandas.DataFrame
        DataFrame with SVM misclassifications.
    original_data : pandas.DataFrame
        Original dataset containing 'call_id' information.
    save_directory : str
        Directory where the Venn diagram will be saved.
    """

    # Ensure we have 'call_id' in mlp_errors and svm_errors using the original dataset.
    mlp_errors['call_id'] = original_data['call_id'].loc[mlp_errors['Index'].values].values
    svm_errors['call_id'] = original_data['call_id'].loc[svm_errors['Index'].values].values

    # Create sets of call_ids for each model's errors.
    mlp_call_ids = set(mlp_errors['call_id'])
    svm_call_ids = set(svm_errors['call_id'])

    # Identify common misclassifications between both models.
    common_errors = mlp_call_ids.intersection(svm_call_ids)

    print("\nError Analysis Comparison:")
    print(f"Total MLP errors: {len(mlp_call_ids)}")
    print(f"Total SVM errors: {len(svm_call_ids)}")
    print(f"Common errors: {len(common_errors)}")
    print(f"Unique to MLP: {len(mlp_call_ids - svm_call_ids)}")
    print(f"Unique to SVM: {len(svm_call_ids - mlp_call_ids)}")

    # Create the Venn diagram to visualise the error sets.
    plt.figure(figsize=(10, 6))
    venn2([mlp_call_ids, svm_call_ids], ('MLP Errors', 'SVM Errors'))

    # Add labels (call_id) for common errors with jitter to avoid overlap.
    for call_id in common_errors:
        # Generate a small random offset (jitter) for x and y positions.
        jitter_x = np.random.uniform(-0.13, 0.13)
        jitter_y = np.random.uniform(-0.13, 0.13)
        plt.text(0 + jitter_x, 0 + jitter_y, call_id, ha='center', va='center', fontsize=8)

    # Optional: Uncomment the following lines if you wish to label unique errors.
    # for call_id in mlp_call_ids - svm_call_ids:
    #     plt.text(-0.5, 0, call_id, ha='center', va='center', fontsize=8)
    # for call_id in svm_call_ids - mlp_call_ids:
    #     plt.text(0.5, 0, call_id, ha='center', va='center', fontsize=8)

    plt.title('Comparison of Model Errors')
    
    # Ensure the save directory exists.
    os.makedirs(save_directory, exist_ok=True)

    # Save the Venn diagram to the specified directory.
    output_path = os.path.join(save_directory, 'model_errors_comparison.png')
    plt.savefig(output_path)
    plt.close()

    print(f"Venn diagram saved at: {output_path}")


def surrogate_model_analysis(X_train_scaled, y_train, X_val_scaled, y_val, 
                           y_pred_train, y_pred_val, feature_names, 
                           model_name="Model"):
    """
    Complete surrogate model analysis including tree visualization and fidelity metrics.
    
    Parameters:
    -----------
    X_train_scaled, X_val_scaled : scaled feature matrices
    y_train, y_val : true labels (used only for reference)
    y_pred_train, y_pred_val : black-box model predictions
    feature_names : list of feature names
    model_name : name of the black-box model being analyzed
    """
    # Combine datasets
    X_combined = np.vstack([X_train_scaled, X_val_scaled])
    y_combined = np.concatenate([y_pred_train, y_pred_val])
    
    # Create split for cross-validation
    test_fold = np.zeros(len(X_combined))
    test_fold[:len(X_train_scaled)] = -1
    ps = PredefinedSplit(test_fold)
    
    print(f"\n{'='*20} Analysis for {model_name} {'='*20}")
    
    # 1. Simple Decision Tree for Interpretability
    simple_dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    simple_dt.fit(X_combined, y_combined)
    
    # Visualize simple tree
    plt.figure(figsize=(20, 10))
    plot_tree(simple_dt, feature_names=feature_names, 
             class_names=np.unique(y_train).astype(str), 
             filled=True, rounded=True, fontsize=10)
    plt.title(f'Simple Decision Tree Surrogate (depth=3) - {model_name}')
    plt.savefig(f'simple_tree_{model_name.lower()}.png', bbox_inches='tight')
    # plt.showtx()
    
    # Calculate fidelity for simple tree
    simple_predictions = simple_dt.predict(X_combined)
    simple_fidelity = accuracy_score(y_combined, simple_predictions)
    print(f"\nSimple Decision Tree Fidelity: {simple_fidelity:.3f}")
    

#         # 2. Optimized Decision Tree (your original grid search)
#     dt_param_grid = {
#     'max_depth': [3, 5, 7, 10, 15],  # None lets the tree grow until pure leaves
#     'min_samples_split': [2, 5, 10],  # Minimum samples required to split
#     'min_samples_leaf': [2, 4, 6, 8],  # Minimum samples in leaf nodes
#     'max_leaf_nodes': [2, 3, 5, 10],  # Maximum number of leaf nodes
#     'criterion': ['gini', 'entropy'],  # Splitting criterion
#     'max_features': ['sqrt', 'log2'],  # Number of features to consider at each split
#     'class_weight': ['balanced', {0: 0.35, 1: 0.45, 2: 0.20}, {0: 0.44, 1: 0.41, 2: 0.13}],
# }
    # 2. Optimized Decision Tree (your original grid search)
    dt_param_grid = {
    'max_depth': [3, 5, 7, 10, 15],  # None lets the tree grow until pure leaves
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split
    'min_samples_leaf': [2, 4, 6, 8],  # Minimum samples in leaf nodes
    'max_leaf_nodes': [2, 3, 5, 10],  # Maximum number of leaf nodes
    'criterion': ['gini', 'entropy'],  # Splitting criterion
    'max_features': ['sqrt', 'log2'],  # Number of features to consider at each split
    'class_weight': ['balanced', {0: 0.35, 1: 0.45, 2: 0.20}, {0: 0.44, 1: 0.41, 2: 0.13}],
}
    
    dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=42), 
                          dt_param_grid, cv=ps, 
                          scoring='f1_weighted', n_jobs=-1)
    dt_grid.fit(X_combined, y_combined)
    best_dt = dt_grid.best_estimator_
    
    # Visualize optimized tree
    plt.figure(figsize=(25, 10))
    plot_tree(best_dt, feature_names=feature_names, 
             class_names=np.unique(y_train).astype(str), 
             filled=True, rounded=True, fontsize=7)
    plt.title(f'Optimized Decision Tree Surrogate - {model_name}')
    plt.savefig(f'optimized_tree_{model_name.lower()}.png', bbox_inches='tight')
    # plt.show()
    
    print(f"\nOptimized Decision Tree:")
    print(f"Best Parameters: {dt_grid.best_params_}")
    print(f"Fidelity Score: {accuracy_score(y_combined, best_dt.predict(X_combined)):.3f}")


        
    # 3. Random Forest Analysis (your original implementation)
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [2, 4, 8],
        'max_features': ['sqrt', 'log2'],
        'max_leaf_nodes': [10, 15],  # Maximum number of leaf nodes
        'class_weight': ['balanced', {0: 0.35, 1: 0.45, 2: 0.20}, {0: 0.44, 1: 0.41, 2: 0.13}],
        'criterion': ['gini', 'entropy']
    }
    
    # 3. Random Forest Analysis (your original implementation)
    rf_param_grid = {
        'n_estimators': [50],
        'max_depth': [3, 5, 10, 15],
        'min_samples_split': [2],
        'min_samples_leaf': [2, 4, 8],
        'max_features': ['sqrt', 'log2'],
        'max_leaf_nodes': [10, 15],  # Maximum number of leaf nodes
        'class_weight': ['balanced', {0: 0.35, 1: 0.45, 2: 0.20}, {0: 0.44, 1: 0.41, 2: 0.13}],
        'criterion': ['gini', 'entropy']
    }
    
    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), 
                          rf_param_grid, cv=ps, 
                          scoring='f1_weighted', n_jobs=-1)
    rf_grid.fit(X_combined, y_combined)
    best_rf = rf_grid.best_estimator_
    
    print(f"\nRandom Forest:")
    print(f"Best Parameters: {rf_grid.best_params_}")
    print(f"Fidelity Score: {accuracy_score(y_combined, best_rf.predict(X_combined)):.3f}")
    
    # 4. Feature Importance Comparison
    # Plot top 20 feature importances with a colorblind-friendly scheme
    plt.figure(figsize=(15, 10))

    feature_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'DT_Importance': best_dt.feature_importances_,
        'RF_Importance': best_rf.feature_importances_
    })
    
    # Ordina per l'importanza del Decision Tree
    feature_imp_df_dt = feature_imp_df.sort_values('DT_Importance', ascending=False)

    # Plot delle importanze del Decision Tree
    plt.figure(figsize=(12, 6))

    # Ordina per l'importanza del Random Forest
    feature_imp_df_rf = feature_imp_df.sort_values('RF_Importance', ascending=False)

    plt.barh(feature_imp_df_rf['Feature'], feature_imp_df_rf['RF_Importance'], color='tab:green')
    plt.xlabel('Importance')
    plt.title(f'RF Feature Importance - {model_name}')
    plt.gca().invert_yaxis()  # Inverti l'asse per mostrare le feature più importanti in alto
    # Layout e salvataggio
    plt.tight_layout()
    plt.savefig(f'feature_importance_dt_rf_{model_name.lower()}.png', bbox_inches='tight')
    # plt.show()

    return best_dt, best_rf



    