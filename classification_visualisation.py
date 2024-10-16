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
def plot_comparison_bar(metrics, mlp_values, svm_values, title='Comparison of Classifiers'):
    """Generate separate bar plots for MLP and SVM classifiers with different colors for each metric."""
    bar_width = 0.40
    x = np.arange(len(metrics))
    
    # Generate a color palette based on the number of metrics
    colors = sns.color_palette("husl", len(metrics))
    
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), sharex=True)
    
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
    plt.show()

def plot_confusion_matrix(cm, labels, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(title)
    plt.show()


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

