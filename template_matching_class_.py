import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from skimage.feature import match_template
from tqdm import tqdm
from classification_utils import load_audio_data

# Path settings
audio_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Data\\high_quality_dataset'
classification_results_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Classification\\Classification_hac_template_matching\\trial'
clusterings_results_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Clustering_\\Classification_hac_template_matching'
dictionary_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Clustering_\\Dictionary'

# Create the results directory if it doesn't exist
os.makedirs(classification_results_path, exist_ok=True)

# Load the clustering results
cluster_results = pd.read_csv(os.path.join(clusterings_results_path, 'hierarchical_clustering_3_distance_membership.csv'))

# Load representative calls and exclude their recordings from test call selection
excluded_recordings = set()

# Load representative calls and add to the exclusion set
for cluster in range(3):  # Assuming 3 clusters
    rep_calls = pd.read_csv(os.path.join(dictionary_path, f'dictionary_cluster_{cluster}_5_percentile.csv'))
    excluded_recordings.update(rep_calls['recording'].tolist())

# Filter out the test calls that are part of the excluded recordings
test_calls = cluster_results[~cluster_results['recording'].isin(excluded_recordings)] #.sample(n=20)

# Number of clusters
n_clusters = 3

# Initialize lists for inter-cluster and intra-cluster results
inter_similarity_results = []
intra_similarity_results = []

cross_correlation_outputs = [] 

# Iterate over test calls
for _, call in tqdm(test_calls.iterrows(), total=len(test_calls)):
    test_call_id = call['call_id']
    test_onset = call['onsets_sec']
    test_offset = call['offsets_sec']
    test_cluster_membership = call['cluster_membership']

    test_result = load_audio_data(audio_path, cluster_results, test_call_id, test_onset, test_offset)
    test_spectrogram = test_result['spectrogram']

    best_inter_match = {'similarity': -1, 'rep_call_id': None, 'cluster_membership_rep': None}
    best_intra_match = {'similarity': -1, 'rep_call_id': None, 'cluster_membership_rep': None}

    # Iterate over clusters for inter-cluster comparison
    for cluster in range(n_clusters):
        representative_calls = pd.read_csv(os.path.join(dictionary_path, f'dictionary_cluster_{cluster}_5_percentile.csv'))

        for _, rep_call in representative_calls.iterrows():
            rep_call_id = rep_call['call_id']
            rep_onset = rep_call['onsets_sec']
            rep_offset = rep_call['offsets_sec']
            cluster_membership_rep = rep_call['cluster_membership']

            rep_result = load_audio_data(audio_path, cluster_results, rep_call_id, rep_onset, rep_offset)
            rep_spectrogram = rep_result['spectrogram']

            # Ensure spectrograms have the same dimensions
            if test_spectrogram.shape[1] < rep_spectrogram.shape[1]:
                pad_width = rep_spectrogram.shape[1] - test_spectrogram.shape[1]
                test_spectrogram = np.pad(test_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
            elif test_spectrogram.shape[1] > rep_spectrogram.shape[1]:
                test_spectrogram = test_spectrogram[:, :rep_spectrogram.shape[1]]

            # Compute similarity
            similarity = match_template(test_spectrogram, rep_spectrogram)
            max_similarity = np.max(similarity)


            # Store cross-correlation result
            cross_correlation_outputs.append({
                'test_call_id': test_call_id,
                'rep_call_id': rep_call_id,
                'max_similarity': max_similarity,
                'cluster_membership_rep': cluster_membership_rep,
                'cluster_membership_test': test_cluster_membership
            })

            # Inter-cluster matching
            if test_cluster_membership != cluster_membership_rep:
                inter_similarity_results.append({
                    'test_call_id': test_call_id,
                    'rep_call_id': rep_call_id,
                    'similarity': max_similarity,
                    'cluster_membership_rep': cluster_membership_rep,
                    'cluster_membership_test': test_cluster_membership
                })

                if max_similarity > best_inter_match['similarity']:
                    best_inter_match.update({
                        'similarity': max_similarity,
                        'rep_call_id': rep_call_id,
                        'cluster_membership_rep': cluster_membership_rep
                    })

            # Intra-cluster matching
            if test_cluster_membership == cluster_membership_rep:
                intra_similarity_results.append({
                    'test_call_id': test_call_id,
                    'rep_call_id': rep_call_id,
                    'similarity': max_similarity,
                    'cluster_membership_rep': cluster_membership_rep,
                    'cluster_membership_test': test_cluster_membership
                })

                if max_similarity > best_intra_match['similarity']:
                    best_intra_match.update({
                        'similarity': max_similarity,
                        'rep_call_id': rep_call_id,
                        'cluster_membership_rep': cluster_membership_rep
                    })


# Save the cross-correlation results to a CSV file
cross_correlation_df = pd.DataFrame(cross_correlation_outputs)
cross_correlation_df.to_csv(os.path.join(classification_results_path, 'cross_correlation_results.csv'), index=False)

# Save results
inter_similarity_df = pd.DataFrame(inter_similarity_results)
intra_similarity_df = pd.DataFrame(intra_similarity_results)

inter_similarity_df.to_csv(os.path.join(classification_results_path, 'inter_similarity_results.csv'), index=False)
intra_similarity_df.to_csv(os.path.join(classification_results_path, 'intra_similarity_results.csv'), index=False)

# Evaluation for Inter-cluster
# y_true_inter = inter_similarity_df['cluster_membership_test']

# Ottieni il ground truth corrispondente solo per i test_call unici
y_true_inter = inter_similarity_df.groupby('test_call_id').first()['cluster_membership_test'].values

# Previsione rimane la stessa
y_pred_inter = inter_similarity_df.groupby('test_call_id').apply(lambda x: x.loc[x['similarity'].idxmax(), 'cluster_membership_rep']).values


# Group by test_call_id and find the rep_call with max similarity for each test_call
# y_pred_inter = inter_similarity_df.groupby('test_call_id').apply(lambda x: x.loc[x['similarity'].idxmax(), 'cluster_membership_rep']).values

accuracy_inter = accuracy_score(y_true_inter, y_pred_inter)
precision_inter = precision_score(y_true_inter, y_pred_inter, average='weighted')
recall_inter = recall_score(y_true_inter, y_pred_inter, average='weighted')
f1_inter = f1_score(y_true_inter, y_pred_inter, average='weighted')

print(f'Inter-cluster Accuracy: {accuracy_inter:.4f}')
print(f'Inter-cluster Precision: {precision_inter:.4f}')
print(f'Inter-cluster Recall: {recall_inter:.4f}')
print(f'Inter-cluster F1 Score: {f1_inter:.4f}')

# Evaluation for Intra-cluster
y_true_intra = intra_similarity_df['cluster_membership_test']

# Group by test_call_id and find the row with the highest similarity
# y_pred_intra = intra_similarity_df.loc[intra_similarity_df.groupby('test_call_id')['similarity'].idxmax(), 'cluster_membership_rep'].values




# y_pred_intra = intra_similarity_df.groupby('test_call_id').apply(lambda x: x.loc[x['similarity'].idxmax(), 'cluster_membership_rep']).values


# print(f"Length of y_true_intra: {len(y_true_intra)}")
# print(f"Length of y_pred_intra: {len(y_pred_intra)}")



# accuracy_intra = accuracy_score(y_true_intra, y_pred_intra)
# precision_intra = precision_score(y_true_intra, y_pred_intra, average='weighted')
# recall_intra = recall_score(y_true_intra, y_pred_intra, average='weighted')
# f1_intra = f1_score(y_true_intra, y_pred_intra, average='weighted')

# print(f'Intra-cluster Accuracy: {accuracy_intra:.4f}')
# print(f'Intra-cluster Precision: {precision_intra:.4f}')
# print(f'Intra-cluster Recall: {recall_intra:.4f}')
# print(f'Intra-cluster F1 Score: {f1_intra:.4f}')


# Controlla il numero di righe in intra_similarity_df
print(f"Number of rows in intra_similarity_df: {intra_similarity_df.shape[0]}")

# Estrazione della verità di terra (ground truth)
y_true_intra = intra_similarity_df.groupby('test_call_id').first()['cluster_membership_test'].values
print(f"Length of y_true_intra: {len(y_true_intra)}")

# Estrazione delle previsioni
y_pred_intra = intra_similarity_df.groupby('test_call_id').apply(lambda x: x.loc[x['similarity'].idxmax(), 'cluster_membership_rep']).values
print(f"Length of y_pred_intra: {len(y_pred_intra)}")

# Calcolo delle metriche e gestione degli errori
try:
    accuracy_intra = accuracy_score(y_true_intra, y_pred_intra)
    precision_intra = precision_score(y_true_intra, y_pred_intra, average='weighted')
    recall_intra = recall_score(y_true_intra, y_pred_intra, average='weighted')
    f1_intra = f1_score(y_true_intra, y_pred_intra, average='weighted')

    print(f'Intra-cluster Accuracy: {accuracy_intra:.4f}')
    print(f'Intra-cluster Precision: {precision_intra:.4f}')
    print(f'Intra-cluster Recall: {recall_intra:.4f}')
    print(f'Intra-cluster F1 Score: {f1_intra:.4f}')
except ValueError as e:
    print(f"Error: {e}")
    print(f"y_true_intra length: {len(y_true_intra)}")
    print(f"y_pred_intra length: {len(y_pred_intra)}")


# Confusion Matrix for Inter-cluster
confusion_mtx_inter = confusion_matrix(y_true_inter, y_pred_inter)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mtx_inter, annot=True, fmt='d', cmap='Blues', xticklabels=range(n_clusters), yticklabels=range(n_clusters))
plt.title('Confusion Matrix - Inter-Cluster')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig(os.path.join(classification_results_path, 'confusion_matrix_inter_cluster.png'))
plt.show()

# Confusion Matrix for Intra-cluster
confusion_mtx_intra = confusion_matrix(y_true_intra, y_pred_intra)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mtx_intra, annot=True, fmt='d', cmap='Blues', xticklabels=range(n_clusters), yticklabels=range(n_clusters))
plt.title('Confusion Matrix - Intra-Cluster')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig(os.path.join(classification_results_path, 'confusion_matrix_intra_cluster.png'))
plt.show()

print("Template matching and classification performance evaluation for both inter-cluster and intra-cluster comparisons completed.")
