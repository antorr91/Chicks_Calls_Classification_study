import os
import glob
import pandas as pd
import numpy as np
import librosa as lb
import librosa.display
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from skimage.feature import match_template
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import torch
import torch.nn.functional as F
import gc
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.cluster import AgglomerativeClustering
from skimage.feature import match_descriptors
from skimage.transform import resize
from template_matching_utils import template_matching_and_evaluation, plot_and_save_audio_segments
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from clustering_utils import plot_and_save_single_audio_segment, split_data_recordings
from classification_utils import load_audio_data,load_audio_data_with_pcen, crop_test_call, padding_test_call, sliding_window, plot_compared_spectrograms
# Path settings
features_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_result_high_quality_dataset_without_jtfs'

clusters_results_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Clustering_\\Classification_hac_template_matching\\hierarchical_clustering_3_distance_membership.csv' 
metadata_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_results_high_quality_dataset_meta\\high_quality_dataset_metadata.csv'
audio_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Data\\high_quality_dataset'
classification_results_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Classification\\Classification_hac_template_matching'
clusterings_results_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Clustering_\\Classification_hac_template_matching'

dictionary_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Clustering_\\Dictionary'
save_path='C:\\Users\\anton\\Chicks_Onset_Detection_project\\aip_examples'



# Create the results directory if it doesn't exist
if not os.path.exists(classification_results_path):
    os.makedirs(classification_results_path)


# Load all data
list_files = glob.glob(os.path.join(features_path, '*.csv'))
all_data = pd.concat([pd.read_csv(f) for f in list_files], ignore_index=True)
metadata = pd.read_csv(metadata_path)

# Drop NaN values
all_data = all_data.dropna()

# Scale data with StandardScaler
scaler = StandardScaler()
features = all_data.drop(['Call Number', 'onsets_sec', 'offsets_sec', 'recording', 'call_id'], axis=1)
features_scaled = scaler.fit_transform(features)

# Clustering setup
n_clusters = 3
agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', compute_distances=True)
cluster_membership = agg.fit_predict(features_scaled)
all_data['cluster_membership'] = cluster_membership

# Save cluster membership
all_data.to_csv(os.path.join(clusterings_results_path, f'hierarchical_clustering_{n_clusters}_membership.csv'), index=False)

# Compute cluster centers and distances
cluster_centers = np.array([features_scaled[all_data['cluster_membership'] == i].mean(axis=0) for i in range(n_clusters)])
distances_to_centers = cdist(features_scaled, cluster_centers, 'euclidean')
all_data['distance_to_center'] = [distances_to_centers[i, cluster] for i, cluster in enumerate(cluster_membership)]
all_data.to_csv(os.path.join(clusterings_results_path, f'hierarchical_clustering_{n_clusters}_distance_membership.csv'), index=False)

# Select representative calls (5th percentile)
excluded_recordings = set()  # To keep track of recordings to exclude from the testing set


# Select representative calls (5th percentile)
for cluster in range(n_clusters):
    cluster_data = all_data[all_data['cluster_membership'] == cluster]
    cluster_data = cluster_data.sort_values('distance_to_center', ascending=True)
    percentile_value = np.percentile(cluster_data['distance_to_center'], 5)
    representative_calls = cluster_data[cluster_data['distance_to_center'] <= percentile_value].head(10)

    save_path = os.path.join(clusterings_results_path, f'cluster_{cluster}_5_percentile')
    os.makedirs(save_path, exist_ok=True)
    representative_calls.to_csv(os.path.join(save_path, f'representative_calls_cluster_{cluster}_5_percentile.csv'), index=False)


    # Add recordings to the exclusion list
    excluded_recordings.update(representative_calls['recording'].tolist())

    # plot_and_save_single_audio_segment(representative_calls, audio_path, dictionary_path, f'cluster_{cluster}_5_percentile')

    # create a dictionary with the representative calls
    # representative_calls.to_csv(os.path.join(clusterings_results_path, f'dictionary_cluster_{cluster}_5_percentile.csv'), index=False)


print("Selection of the calls at fifth percentile completed.")



# Filter out the recordings in the exclusion list
filtered_all_data = all_data[~all_data['recording'].isin(excluded_recordings)]

# Now sample 10 calls from the filtered data
dataset_of_testing_calls = filtered_all_data.sample(n=5)

# Debug: Check the sampled data
print("Sampled test calls:", dataset_of_testing_calls[['call_id', 'onsets_sec', 'offsets_sec', 'Duration_call']])
# Load the metadata
audio_data = pd.read_csv(clusters_results_path)


#Initialize lists to store all results
all_similarity_results = []
all_template_matching_results = []

# Batch size and tqdm for progress tracking
batch_size = 2  # Number of calls to process per batch
# Lista per salvare i risultati del template matching su rep_calls stesse
# self_similarity_results = []

# # Test sulle rep_calls per se stesse (Template Matching intra-cluster)
# for cluster in range(n_clusters):
#     # Carica le chiamate rappresentative per il cluster corrente
#     representative_calls = pd.read_csv(os.path.join(dictionary_path, f'dictionary_cluster_{cluster}_5_percentile.csv'))

#     # Itera su ogni chiamata rappresentativa
#     for _, rep_call in representative_calls.iterrows():
#         rep_call_id = rep_call['call_id']
#         rep_onset = rep_call['onsets_sec']
#         rep_offset = rep_call['offsets_sec']
#         cluster_membership_rep = rep_call['cluster_membership']
#         rep_duration = rep_call['Duration_call']

#         # Carica lo spettrogramma della chiamata rappresentativa
#         rep_result = load_audio_data(audio_path, all_data, rep_call_id, rep_onset, rep_offset)
#         rep_spectrogram = rep_result['spectrogram']

#         # Manteniamo l'originale lunghezza dello spettrogramma prima del padding/cropping
#         original_length = rep_spectrogram.shape[1]

#         # Padding o cropping: Assicuriamo che lo spettrogramma si confronti con una versione modificata di se stesso
#         if rep_spectrogram.shape[1] < original_length:
#             # Pad rep_spectrogram per eguagliare la sua lunghezza originale (caso ipotetico)
#             pad_width = original_length - rep_spectrogram.shape[1]
#             rep_spectrogram = np.pad(rep_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
#             print(f"Padded rep_spectrogram to shape: {rep_spectrogram.shape}")
#         elif rep_spectrogram.shape[1] > original_length:
#             # Crop rep_spectrogram per eguagliare la lunghezza originale
#             rep_spectrogram = rep_spectrogram[:, :original_length]
#             print(f"Cropped rep_spectrogram to shape: {rep_spectrogram.shape}")

#         # Confronta la chiamata rappresentativa con se stessa
#         similarity = match_template(rep_spectrogram, rep_spectrogram)
#         max_similarity = np.max(similarity)

#         # Salva i risultati del template matching
#         self_similarity_results.append({
#             'rep_call_id': rep_call_id,
#             'similarity_with_self': max_similarity,
#             'cluster_membership_rep': cluster_membership_rep
#         })

#         # Visualizza la somiglianza massima (dovrebbe essere molto alta)
#         print(f'Similarity of rep_call {rep_call_id} with itself: {max_similarity:.4f}')

# # Crea un DataFrame per i risultati del self-matching delle rep_calls
# self_similarity_results_df = pd.DataFrame(self_similarity_results)

# # Salva i risultati in un file CSV
# self_similarity_results_df.to_csv(os.path.join(classification_results_path, 'self_similarity_results.csv'), index=False)

# print("Template matching of representative calls with themselves completed.")

# Inizializza liste per i risultati del template matching inter-cluster
inter_similarity_results = []

# Ciclo su tutte le chiamate di test
for _, call in dataset_of_testing_calls.iterrows():
    test_call_id = call['call_id']
    test_onset = call['onsets_sec']
    test_offset = call['offsets_sec']
    test_cluster_membership = call['cluster_membership']
    test_call_duration = call['Duration_call']

    # Carica lo spettrogramma della chiamata di test
    test_result = load_audio_data(audio_path, all_data, test_call_id, test_onset, test_offset)
    test_spectrogram = test_result['spectrogram']

    best_match = {'similarity': -1, 'rep_call_id': None, 'cluster_membership_rep': None}

    # Itera su tutti i cluster e tutte le chiamate rappresentative
    for cluster in range(n_clusters):
        representative_calls = pd.read_csv(os.path.join(dictionary_path, f'dictionary_cluster_{cluster}_5_percentile.csv'))

        for _, rep_call in representative_calls.iterrows():
            rep_call_id = rep_call['call_id']
            rep_onset = rep_call['onsets_sec']
            rep_offset = rep_call['offsets_sec']
            cluster_membership_rep = rep_call['cluster_membership']
            rep_duration = rep_call['Duration_call']
            rep_dict_id = rep_call['dictionary_id']

            # Carica lo spettrogramma della chiamata rappresentativa
            rep_result = load_audio_data(audio_path, all_data, rep_call_id, rep_onset, rep_offset)
            rep_spectrogram = rep_result['spectrogram']



            # Carica i dati audio
            rep_result = load_audio_data(audio_path, all_data, rep_call_id, rep_onset, rep_offset)

            # check if the audio have same length
            if test_spectrogram.shape[1] < rep_spectrogram.shape[1]:
                pad_width = rep_spectrogram.shape[1] - test_spectrogram.shape[1]
                test_spectrogram = np.pad(test_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
            elif test_spectrogram.shape[1] > rep_spectrogram.shape[1]:
                test_spectrogram = test_spectrogram[:, :rep_spectrogram.shape[1]]

            # Applica il template matching
            similarity = match_template(test_spectrogram, rep_spectrogram)
            max_similarity = np.max(similarity)
          

            plot_compared_spectrograms(
                test_spectrogram, 
                rep_spectrogram, 
                max_similarity, 
                test_call_id, 
                rep_dict_id, 
                test_cluster_membership,  # membership del test_call
                cluster_membership_rep,    # membership del rep_call
                clusterings_results_path                    # path per il salvataggio
            )
            # Salva i risultati di similarità
            inter_similarity_results.append({
                'test_call_id': test_call_id,
                'rep_call_id': rep_call_id,
                'similarity': max_similarity,
                'cluster_membership_rep': cluster_membership_rep,
                'cluster_membership_test': test_cluster_membership
            })

            # Aggiorna il miglior match
            if max_similarity > best_match['similarity']:
                best_match.update({
                    'similarity': max_similarity,
                    'rep_call_id': rep_call_id,
                    'cluster_membership_rep': cluster_membership_rep
                })

# Crea il DataFrame dei risultati di similarità inter-cluster
inter_similarity_df = pd.DataFrame(inter_similarity_results)
inter_similarity_df.to_csv(os.path.join(classification_results_path, 'inter_similarity_results.csv'), index=False)

print("Template matching inter-cluster completato e risultati salvati.")

# Valutazione delle performance del clustering
y_true_inter = inter_similarity_df['cluster_membership_test']


# Ensure the similarity max is calculated within each group
y_pred_inter = inter_similarity_df.groupby('test_call_id').apply(lambda x: x.loc[x['similarity'].idxmax(), 'cluster_membership_rep'])


# Calcolo delle metriche
accuracy_inter = accuracy_score(y_true_inter, y_pred_inter)
precision_inter = precision_score(y_true_inter, y_pred_inter, average='weighted')
recall_inter = recall_score(y_true_inter, y_pred_inter, average='weighted')
f1_inter = f1_score(y_true_inter, y_pred_inter, average='weighted')

print(f'Inter-cluster Accuracy: {accuracy_inter:.4f}')
print(f'Inter-cluster Precision: {precision_inter:.4f}')
print(f'Inter-cluster Recall: {recall_inter:.4f}')
print(f'Inter-cluster F1 Score: {f1_inter:.4f}')

# Salva il report di classificazione
classification_report_inter_df = classification_report(y_true_inter, y_pred_inter, output_dict=True)
classification_report_inter_df = pd.DataFrame(classification_report_inter_df).transpose()
classification_report_inter_df.to_csv(os.path.join(classification_results_path, 'inter_classification_report.csv'), index=True)

# Confusion Matrix
confusion_mtx = confusion_matrix(y_true_inter, y_pred_inter)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='Blues', xticklabels=range(n_clusters), yticklabels=range(n_clusters))
plt.title('Confusion Matrix - Inter-Cluster')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig(os.path.join(classification_results_path, 'confusion_matrix_inter_cluster.png'))
plt.show()

print("Inter-cluster classification performance evaluated.")


























# # Initialize tqdm progress bar
# for i in tqdm(range(0, len(dataset_of_testing_calls), batch_size), desc="Processing batches", unit="batch"):
#     batch_calls = dataset_of_testing_calls.iloc[i:i+batch_size]

#     for _, call in batch_calls.iterrows():
#         test_call_id = call['call_id']
#         test_onset = call['onsets_sec']
#         test_offset = call['offsets_sec']
#         test_cluster_membership = call['cluster_membership']
#         test_call_duration = call['Duration_call']

#         # Ensure values are valid
#         if pd.isnull(test_call_id) or pd.isnull(test_onset) or pd.isnull(test_offset):
#             raise ValueError(f"Missing data for test call {test_call_id}")

#         # Load audio data and spectrogram for the test call
#         test_result = load_audio_data(audio_path, all_data, test_call_id, test_onset, test_offset)

#         # Ensure the test_result is valid
#         if test_result is None or 'spectrogram' not in test_result:
#             raise ValueError(f"Failed to load spectrogram for test call {test_call_id}")

#         # Extract the test spectrogram
#         test_spectrogram = test_result['spectrogram']
#         print(f"Test spectrogram shape for call {test_call_id}: {test_spectrogram.shape}")

#         best_match = {'similarity': -1, 'rep_call_id': None, 'cluster_membership_rep': None}

#         # Iterate over representative calls from all clusters
#         for cluster in range(n_clusters):
#             # Load representative calls for the current cluster
#             representative_calls = pd.read_csv(os.path.join(dictionary_path, f'dictionary_cluster_{cluster}_5_percentile.csv'))

#             # Check if 'call_id' column exists in representative_calls DataFrame
#             if 'call_id' not in representative_calls.columns:
#                 raise KeyError("Column 'call_id' not found in the DataFrame")

#             # Iterate over all representative calls in the current cluster
#             for _, rep_call in representative_calls.iterrows():
#                 rep_call_id = rep_call['call_id']
#                 rep_onset = rep_call['onsets_sec']
#                 rep_offset = rep_call['offsets_sec']
#                 cluster_membership_rep = rep_call['cluster_membership']
#                 rep_duration = rep_call['Duration_call']

#                 # Load representative call audio data
#                 rep_result = load_audio_data(audio_path, all_data, rep_call_id, rep_onset, rep_offset)
#                 rep_spectrogram = rep_result['spectrogram']     

#                 # Initialize max_similarity for this pair
#                 max_similarity = -1
#                 # Store the original length of the test_spectrogram before padding
#                 original_length = test_spectrogram.shape[1]



#                 # Ensure the spectrogram dimensions match
#                 # if the test call is shorter than the representative call
#                 if test_spectrogram.shape[1] < rep_spectrogram.shape[1]:
#                     # Pad test_spectrogram to match rep_spectrogram's length (columns)
#                     pad_width = rep_spectrogram.shape[1] - test_spectrogram.shape[1]
#                     test_spectrogram = np.pad(test_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
#                     print(f"Padded test spectrogram to shape: {test_spectrogram.shape}")
#                 # if the test call is longer than the representative call
#                 elif test_spectrogram.shape[1] > rep_spectrogram.shape[1]:
#                     # Crop test_spectrogram to match rep_spectrogram's length (columns)
#                     test_spectrogram = test_spectrogram[:, :rep_spectrogram.shape[1]]
#                     print(f"Cropped test spectrogram to shape: {test_spectrogram.shape}")
#                 # if the test call is the same length as the representative call
#                 else:
#                     # Do nothing
#                     pass
#                 # After ensuring both spectrograms are the same size, apply match_template
#                 similarity = match_template(test_spectrogram, rep_spectrogram)
#                 max_similarity = np.max(similarity)

#                 # Store the similarity results
#                 all_similarity_results.append({
#                     'test_call_id': test_call_id,
#                     'rep_call_id': rep_call_id,
#                     'similarity': max_similarity,
#                     'cluster_membership_rep': cluster_membership_rep,
#                     'cluster_membership_test': test_cluster_membership
#                 })

#                 # Visualize the key matched features
#                 fig, axs = plt.subplots(1, 2, figsize=(12, 6))

#                 # Plot test spectrogram
#                 axs[0].imshow(test_spectrogram, aspect='auto', cmap='inferno')
#                 axs[0].set_title(f'Test Call: {test_call_id}')
#                 axs[0].set_xlabel('Time')
#                 axs[0].set_ylabel('Frequency')

#                 # Plot representative spectrogram
#                 axs[1].imshow(rep_spectrogram, aspect='auto', cmap='inferno')
#                 axs[1].set_title(f'Rep Call: {rep_call_id}')
#                 axs[1].set_xlabel('Time')

#                 plt.savefig(f"matching_features_{test_call_id}_{rep_call_id}.png")
#                 plt.close(fig)

#                 # Update best match if current one is the most similar
#                 if max_similarity > best_match['similarity']:
#                     best_match.update({
#                         'similarity': max_similarity,
#                         'rep_call_id': rep_call_id,
#                         'cluster_membership_rep': cluster
#                     })

#         # Store the best match for the current test call
#         all_template_matching_results.append({
#             'test_call_id': test_call_id,
#             'rep_call_id': best_match['rep_call_id'],
#             'similarity': best_match['similarity'],
#             'cluster_membership_rep': best_match['cluster_membership_rep'],  # La membership della chiamata rappresentativa
#             'cluster_membership_test': test_cluster_membership,  # La membership della chiamata test originale
#             'new_matched_cluster_membership_test': best_match['cluster_membership_rep']  # Nuova assegnazione di membership
#         })

#     # Clear memory
#     gc.collect()




# Create final DataFrames
similarity_results_df = pd.DataFrame(all_similarity_results)
template_matching_df = pd.DataFrame(all_template_matching_results)

# Save final results
similarity_results_df.to_csv(os.path.join(classification_results_path, 'similarity_results.csv'), index=False)
template_matching_df.to_csv(os.path.join(classification_results_path, 'template_matching_results.csv'), index=False)

print("Template matching process completed. Final results saved.")

# Evaluate classification performance
y_true = template_matching_df['cluster_membership_test']
y_pred = template_matching_df['new_matched_cluster_membership_test']

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

classification_report_df = classification_report(y_true, y_pred, output_dict=True)
classification_report_df = pd.DataFrame(classification_report_df).transpose()
classification_report_df.to_csv(os.path.join(classification_results_path, 'classification_report.csv'), index=True)


# plot the results
plt.figure(figsize=(10, 5))
plt.plot(template_matching_df['test_call_id'], template_matching_df['similarity'], 'o')
plt.xlabel('Test Call ID')
plt.ylabel('Similarity Score')
plt.title('Template Matching Results')
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(classification_results_path, 'template_matching_results_plot.png'))
# plt.show()
plt.close()


# Compute and plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(np.unique(y_true)))
plt.xticks(tick_marks, np.unique(y_true))
plt.yticks(tick_marks, np.unique(y_true))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(os.path.join(classification_results_path, 'confusion_matrix.png'))
plt.close()

# Error analysis
# errors= similarity_results_df[similarity_results_df['cluster_membership_test'] != similarity_results_df['cluster_membership_rep']]
errors = template_matching_df[template_matching_df['cluster_membership_test'] != template_matching_df['new_matched_cluster_membership_test']]
print("Number of misclassifications:", len(errors))
# save in a csv all the misclassifications
errors.to_csv(os.path.join(classification_results_path, 'misclassifications.csv'), index=False)
print("\nMisclassified instances:")
print(errors)


# # Compute ROC curve and ROC area for each class
# n_classes = 3  # Specifichiamo esplicitamente che abbiamo 3 classi
# y_test_bin = label_binarize(y_true, classes=np.unique(y_true))
# y_score = label_binarize(y_pred, classes=np.unique(y_true))

# # Compute ROC curve and ROC area for each class
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])

# # Compute micro-average ROC curve and ROC area
# fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# # Plot ROC curves
# plt.figure(figsize=(10, 8))
# plt.plot(fpr["micro"], tpr["micro"],
#          label=f'micro-average ROC curve (area = {roc_auc["micro"]:0.2f})',
#          color='deeppink', linestyle=':', linewidth=4)

# colors = ['aqua', 'darkorange', 'cornflowerblue']
# for i, color in zip(range(n_classes), colors):
#     plt.plot(fpr[i], tpr[i], color=color, lw=2,
#              label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')

# plt.plot([0, 1], [0, 1], 'k--', lw=2)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve for 3 Classes')
# plt.legend(loc="lower right")
# plt.savefig(os.path.join(classification_results_path, 'roc_curve.png'))
# plt.close()


# print("Template matching completed and results saved.")


# # Sort similarity results by similarity in descending order
# similarity_results_df = pd.DataFrame(similarity_results)
# similarity_results_df = similarity_results_df.sort_values('similarity', ascending=False)

# # Save the sorted similarity results
# similarity_results_df.to_csv(os.path.join(classification_results_path, 'similarity_results_sorted.csv'), index=False)

# # Define percentiles to select examples (e.g., 90th, 75th, 50th percentiles)
# percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
# percentile_calls = {}

# # Extract few examples for each percentile
# for percentile in percentiles:
#     percentile_value = np.percentile(similarity_results_df['similarity'], percentile)
#     selected_calls = similarity_results_df[similarity_results_df['similarity'] >= percentile_value].head(10)
#     percentile_calls[percentile] = selected_calls
#     plot_and_save_audio_segments(selected_calls, audio_path, classification_results_path, f'percentile_{percentile}')

#     # Save the percentile calls to CSV
#     selected_calls.to_csv(os.path.join(classification_results_path, f'percentile_{percentile}_calls.csv'), index=False)

# # Introduce a threshold for similarity to accept classification
# similarity_threshold = 0.7  # You can adjust this value based on your needs

# # Apply the threshold to filter similarity results
# accepted_calls = similarity_results_df[similarity_results_df['similarity'] >= similarity_threshold]

# # Save accepted calls based on threshold
# accepted_calls.to_csv(os.path.join(classification_results_path, 'accepted_calls_above_threshold.csv'), index=False)


# plot_and_save_audio_segments(accepted_calls, audio_path, classification_results_path, 'accepted_calls_above_threshold')


# # Update template_matching_results based on the threshold
# template_matching_filtered_results = [row for row in template_matching_results if row['similarity'] >= similarity_threshold]

# # Convert results to DataFrame and save filtered results
# template_matching_filtered_df = pd.DataFrame(template_matching_filtered_results)
# template_matching_filtered_df.to_csv(os.path.join(classification_results_path, 'template_matching_results_filtered.csv'), index=False)


print("Percentile selection and thresholding completed.")