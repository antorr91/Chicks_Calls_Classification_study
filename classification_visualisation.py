import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from scipy.io import wavfile
import librosa 
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import silhouette_score, silhouette_samples, classification_report
from math import pi
import seaborn as sns
import itertools
import soundfile as sf
import cv2
import librosa as lb
import librosa.display
from scipy.signal import correlate2d
import torch
import torch.nn.functional as F


# Function to plot comparison bar chart with different colors for each metric in separate subplots
def plot_comparison_bar(metrics, mlp_values, svm_values, title='Comparison of Classifiers', save_path=None):    
    """Generate separate bar plots for MLP and SVM classifiers with different colors for each metric."""
    bar_width = 0.40
    x = np.arange(len(metrics))
    
    # Generate a color palette based on the number of metrics
    colors = sns.color_palette("husl", len(metrics))
    
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)
    
    # Plot MLP values
    axes[0].bar(x, mlp_values, width=bar_width, color=colors)
    axes[0].set_title('MLP Classifier Metrics')
    axes[0].set_ylabel('Scores')
    
    # Plot SVM values
    axes[1].bar(x, svm_values, width=bar_width, color=colors)
    axes[1].set_title('SVM Classifier Metrics')
    axes[1].set_xlabel('Metrics')
    axes[1].set_ylabel('Scores')
    
    # Set x-ticks and labels
    plt.xticks(x, metrics, rotation=45, ha='right')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.suptitle(title, y=1.02)
    plt.savefig('comparison_bar_chart.png')
    plt.show()

def plot_confusion_matrix(cm, labels, title='Confusion Matrix', model_label='', results_path='results'):
    """Plot a confusion matrix using a heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(title)
    plt.savefig(os.path.join(results_path, 'confusion_matrix_' + model_label + '.png'))
    plt.show()



# Function to plot comparison bar chart for MLP and SVM
def plot_comparison_bar_metrics(metrics, mlp_values, svm_values, title='Comparison of MLP and SVM Classifiers', save_path=None):
    """Generate a bar plot with MLP and SVM classifiers side by side for each metric."""
    
    bar_width = 0.30  # Width of the bars
    x = np.arange(len(metrics))  # Positions on the x-axis
    
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot MLP bars (shifted to the left by bar_width / 2)
    ax.bar(x - bar_width / 2, mlp_values, width=bar_width, label='MLP', color='skyblue')
    
    # Plot SVM bars (shifted to the right by bar_width / 2)
    ax.bar(x + bar_width / 2, svm_values, width=bar_width, label='SVM', color='orange')

    # Add labels and title
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Scores')
    ax.set_title(title)
    
    # Set x-ticks with the metric names
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    
    # Add legend
    ax.legend()

    # Show plot
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def parse_classification_report(report):
    """Parse the classification report to extract precision, recall, and f1-score for each class."""
    metrics = ['accuracy']
    metrics_values = [report['accuracy']]
    
    for label, metrics_dict in report.items():
        if label not in ['accuracy', 'macro avg', 'weighted avg']:  # Ignore these summary rows
            metrics.append(f'precision_{label}')
            metrics.append(f'recall_{label}')
            metrics.append(f'f1_{label}')
            metrics_values.extend([metrics_dict['precision'], metrics_dict['recall'], metrics_dict['f1-score']])
    
    return metrics, metrics_values


def generate_metrics_from_report(y_true, y_pred):
    """Generate classification metrics and return a dictionary of results."""
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics, metrics_values = parse_classification_report(report)
    return dict(zip(metrics, metrics_values))


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

def plot_model_comparison(y_true, y_pred_model1, y_pred_model2, model1_name="Model 1", model2_name="Model 2"):
    """
    Confronta due modelli di classificazione in termini di Precision, Recall e F1-score complessivi.
    
    Parametri:
        - y_true: array dei veri valori delle classi
        - y_pred_model1: array delle predizioni del primo modello
        - y_pred_model2: array delle predizioni del secondo modello
        - model1_name: nome del primo modello (default: "Model 1")
        - model2_name: nome del secondo modello (default: "Model 2")
    """
    
    # Calcolo delle metriche globali (weighted-averaged)
    model1_metrics = [
        precision_score(y_true, y_pred_model1, average='weighted'),
        recall_score(y_true, y_pred_model1, average='weighted'),
        f1_score(y_true, y_pred_model1, average='weighted')
    ]

    model2_metrics = [
        precision_score(y_true, y_pred_model2, average='weighted'),
        recall_score(y_true, y_pred_model2, average='weighted'),
        f1_score(y_true, y_pred_model2, average='weighted')
    ]

    # Definizione delle metriche
    metrics = ['Precision', 'Recall', 'F1-score']
    
    # Creazione del grafico a barre
    bar_width = 0.30  # Larghezza delle barre
    x = np.arange(len(metrics))  # Posizioni sull'asse x

    fig, ax = plt.subplots(figsize=(8, 5))

    # Barre per il primo modello (sinistra)
    ax.bar(x - bar_width / 2, model1_metrics, width=bar_width, label=model1_name, color='skyblue')
    
    # Barre per il secondo modello (destra)
    ax.bar(x + bar_width / 2, model2_metrics, width=bar_width, label=model2_name, color='orange')

    # Label e titolo
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Scores')
    ax.set_ylim(0, 1)  # Range da 0 a 1 per chiarezza
    ax.set_title('Comparison of Classifier Performance')

    # Impostazione delle etichette sull'asse x
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)

    # Aggiunta della legenda
    ax.legend()

    # Mostra il grafico
    plt.tight_layout()
    plt.show()



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



# Funzione per tracciare e salvare i segmenti audio rappresentativi
def plot_and_save_audio_segments(representative_calls, audio_path, save_path, cluster_label):
    """
    Estrai, traccia e salva i segmenti audio delle calls rappresentative di un cluster.
    
    Args:
    representative_calls (DataFrame): DataFrame contenente le calls rappresentative per un cluster.
    audio_path (str): Percorso alla directory contenente i file audio.
    save_path (str): Percorso per salvare i risultati.
    cluster_label (str): Etichetta per il cluster.
    """
    fig, axes = plt.subplots(1, len(representative_calls), figsize=(2 * len(representative_calls), 2))
    fig.suptitle(f'{cluster_label} Audio Segments')
    
    if len(representative_calls) == 1:
        axes = [axes]

    # Creazione della directory per salvare i file audio
    cluster_audio_dir = os.path.join(save_path, 'audio')
    os.makedirs(cluster_audio_dir, exist_ok=True)

    for idx, (_, call) in enumerate(representative_calls.iterrows()):
        audio_file = os.path.join(audio_path, call['recording'] + '.wav')
        if os.path.exists(audio_file):
            data, sr = lb.load(audio_file, sr=44100)
            S = lb.feature.melspectrogram(y=data, sr=sr, n_mels=128, fmin=2000, fmax=10000)
            log_S = lb.power_to_db(S, ref=np.max)

            # Usa la funzione esistente per segmentare lo spettrogramma
            calls_S = segment_spectrogram(log_S, [call['onsets_sec']], [call['offsets_sec']], sr=sr)
            call_S = calls_S[0]

            # Estrai il segmento audio
            onset_samples = int(call['onsets_sec'] * sr)
            offset_samples = int(call['offsets_sec'] * sr)
            call_audio = data[onset_samples:offset_samples]

            # Plot dello spettrogramma della call rappresentativa
            img = axes[idx].imshow(call_S, aspect='auto', origin='lower', cmap='magma')
            axes[idx].set_title(f'Call {idx + 1} of {call["call_id"]} \n {cluster_label}', fontsize=6)
            axes[idx].set_xlabel('Time', fontsize=5)
            axes[idx].set_ylabel('Frequency', fontsize=5)
            fig.colorbar(img, ax=axes[idx])

            # Salva il file audio
            audio_filename = os.path.join(cluster_audio_dir, f'call_{idx + 1}_{call["recording"]}.wav')
            sf.write(audio_filename, call_audio, sr)
        else:
            print(f'Audio file {audio_file} not found')

    # Salva il plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename = f'{cluster_label}.png'
    plt.savefig(os.path.join(save_path, plot_filename))
    plt.close(fig)

