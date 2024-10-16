import os
import librosa
import librosa.display as lb
import numpy as np
from skimage.feature import match_template
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import pandas as pd
import librosa.display as ld    
from clustering_utils import segment_spectrogram

from skimage.feature import match_template
import numpy as np

def template_matching_and_evaluation(representative_call_df, test_call_df, save_path, audio_path):
    """
    Perform template matching between representative calls and test calls, and save the results.

    Args:
    representative_call_df (DataFrame): DataFrame containing a representative call (with its onsets/offsets).
    test_call_df (DataFrame): DataFrame containing a test call (with its onsets/offsets).
    save_path (str): Path to save the results.
    audio_path (str): Path where audio files are stored.
    """
    # Extract the representative call
    rep_call = representative_call_df.iloc[0]  # Assume only one representative call in DataFrame
    rep_audio_file = os.path.join(audio_path, rep_call['recording'] + '.wav')
    
    if not os.path.exists(rep_audio_file):
        print(f"Audio file for representative call {rep_audio_file} not found.")
        return
    
    # Load representative call audio and compute its mel-spectrogram
    rep_audio_data, rep_sr = lb.load(rep_audio_file, sr=44100)
    rep_onset_sample = int(rep_call['onsets_sec'] * rep_sr)
    rep_offset_sample = int(rep_call['offsets_sec'] * rep_sr)
    rep_call_audio = rep_audio_data[rep_onset_sample:rep_offset_sample]

    rep_melspec = lb.feature.melspectrogram(y=rep_call_audio, sr=rep_sr, n_mels=128, fmin=2000, fmax=10000)
    rep_log_melspec = lb.power_to_db(rep_melspec, ref=np.max)

    # Extract the test call
    test_call = test_call_df.iloc[0]  # Assume single test call at a time
    test_audio_file = os.path.join(audio_path, test_call['recording'] + '.wav')
    
    if not os.path.exists(test_audio_file):
        print(f"Audio file for test call {test_audio_file} not found.")
        return

    # Load test call audio and compute its mel-spectrogram
    test_audio_data, test_sr = lb.load(test_audio_file, sr=44100)
    test_onset_sample = int(test_call['onsets_sec'] * test_sr)
    test_offset_sample = int(test_call['offsets_sec'] * test_sr)
    test_call_audio = test_audio_data[test_onset_sample:test_offset_sample]

    test_melspec = lb.feature.melspectrogram(y=test_call_audio, sr=test_sr, n_mels=128, fmin=2000, fmax=10000)
    test_log_melspec = lb.power_to_db(test_melspec, ref=np.max)

    # Perform template matching using match_template
    result = match_template(test_log_melspec, rep_log_melspec)

    # Calculate the similarity score as the max value of the match result
    similarity_score = np.max(result)
    print(f"Similarity score between representative call and test call: {similarity_score}")

    # Save the result of the template matching
    results_path = os.path.join(save_path, f'template_matching_results_{test_call["recording"]}.txt')
    with open(results_path, 'w') as f:
        f.write(f'Similarity score between representative call and test call: {similarity_score}\n')

    print(f'Template matching result saved to {results_path}')


def plot_and_save_audio_segments(representative_calls, audio_path, save_path, cluster_label):
    """
    Extract, plot, and save audio segments for representative calls of a cluster.
    
    Args:
    representative_calls (DataFrame): DataFrame containing representative calls for a cluster.
    audio_path (str): Path to the directory containing audio files.
    save_path (str): Path to save the results.
    cluster_label (str): Label for the cluster.
    """
    num_calls = len(representative_calls)
    
    # Prepare for subplots
    fig, axes = plt.subplots(1, num_calls, figsize=(2 * num_calls, 2))
    fig.suptitle(f'{cluster_label} Audio Segments')
    
    # Ensure axes is always a list, even if there's only one call
    if num_calls == 1:
        axes = [axes]

    # Directory to save audio
    cluster_audio_dir = os.path.join(save_path, 'audio')
    os.makedirs(cluster_audio_dir, exist_ok=True)

    # Iterate over the representative calls
    for idx in range(num_calls):
        call = representative_calls.iloc[idx]
        audio_file = os.path.join(audio_path, call['recording'] + '.wav')
        

        if os.path.exists(audio_file):
            data, sr = lb.load(audio_file, sr=44100)
            
            # Compute the mel spectrogram
            S = lb.feature.melspectrogram(y=data, sr=sr, n_mels=128, fmin=2000, fmax=10000)
            log_S = lb.power_to_db(S, ref=np.max)

            # Segment the spectrogram
            calls_S = segment_spectrogram(log_S, [call['onsets_sec']], [call['offsets_sec']], sr=sr)
            call_S = calls_S[0]

            # Extract the audio segment
            onset_samples = int(call['onsets_sec'] * sr)
            offset_samples = int(call['offsets_sec'] * sr)
            call_audio = data[onset_samples:offset_samples]

            # Plot the spectrogram
            img = axes[idx].imshow(call_S, aspect='auto', origin='lower', cmap='magma')
            axes[idx].set_title(f'Call {idx + 1} of {call["recording"]} \n {cluster_label}', fontsize=6)
            axes[idx].set_xlabel('Time', fontsize=5)
            axes[idx].set_ylabel('Frequency', fontsize=5)
            fig.colorbar(img, ax=axes[idx])

            # Save the audio file
            audio_filename = os.path.join(cluster_audio_dir, f'call_{idx + 1}_{call["recording"]}.wav')
            sf.write(audio_filename, call_audio, sr)
        else:
            print(f'Audio file {audio_file} not found')

    # Save and close the plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename = f'{cluster_label}.png'
    plt.savefig(os.path.join(save_path, plot_filename))
    plt.close(fig)
