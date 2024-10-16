import os
import glob
import pandas as pd
import numpy as np
import umap.umap_ as umap
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from itertools import cycle

# Define the file paths
features_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_result_high_quality_dataset_without_jtfs'
metadata_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_results_high_quality_dataset_meta\\high_quality_dataset_metadata.csv'
audio_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Data\\high_quality_dataset'
clusterings_results_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Clustering_\\Distance_based_kmeans_clustering_classification'

# Create the results directory if it doesn't exist
if not os.path.exists(clusterings_results_path):
    os.makedirs(clusterings_results_path)

# Get a list of all CSV files in the directory
list_files = glob.glob(os.path.join(features_path, '*.csv'))

# Read and concatenate all CSV files into a single dataframe
all_data = pd.concat([pd.read_csv(f) for f in list_files], ignore_index=True)
metadata = pd.read_csv(metadata_path)

# Save the concatenated dataframe with unique call_id to a CSV file
all_data.to_csv(os.path.join(clusterings_results_path, 'all_data.csv'), index=False)

# Drop NaN values
all_data = all_data.dropna()

# Scale data with StandardScaler on used features only
scaler = StandardScaler()
features = all_data.drop(['recording', 'Call Number', 'onsets_sec', 'offsets_sec', 'call_id'], axis=1)
features_scaled = scaler.fit_transform(features)

best_n_clusters = 3
# Perform K-Means clustering with the determined number of clusters
kmeans = KMeans(n_clusters=best_n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
kmeans.fit(features_scaled)

cluster_membership = kmeans.labels_

# Add cluster membership to the dataframe
all_data['cluster_membership'] = cluster_membership

# Calculate distances for all points to their respective cluster centroids
distances = np.zeros(len(all_data))
for i in range(len(all_data)):
    cluster = all_data.iloc[i]['cluster_membership']
    distances[i] = np.linalg.norm(features_scaled[i] - kmeans.cluster_centers_[cluster])
all_data['distance_to_centroid'] = distances

# Save the results with cluster membership and distances
all_data.to_csv(os.path.join(clusterings_results_path, 'all_data_with_distances.csv'), index=False)

# Relabeling based on distance thresholds
thresholds = [1.90, 1.85, 4.30]

# Reassign clusters based on these thresholds
new_labels = []

for i, row in all_data.iterrows():
    cluster = row['cluster_membership']
    if row['distance_to_centroid'] > thresholds[cluster]:
        new_labels.append(-1)  # -1 indicates an outlier or unassigned
    else:
        new_labels.append(cluster)

# Add new labels to the dataframe
all_data['new_cluster_membership'] = new_labels

# Save the updated data with new cluster labels
all_data.to_csv(os.path.join(clusterings_results_path, f'kmeans_clustering_{best_n_clusters}_relabeled.csv'), index=False)

print("K-Means clustering: Relabeled data saved.")

# Train and evaluate classifiers using the relabeled data

# Prepare the data for classification
X = features_scaled
y = all_data['new_cluster_membership']

# Filter out the outliers (label -1)
X_filtered = X[y != -1]
y_filtered = y[y != -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42)

# SVM Classifier
svm = SVC(kernel='rbf', probability=True, random_state=42)  # Using radial basis function kernel
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
y_proba_svm = svm.predict_proba(X_test)

# MLP Classifier
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
y_proba_mlp = mlp.predict_proba(X_test)

# Evaluate SVM performance
print("SVM Classifier Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Precision:", precision_score(y_test, y_pred_svm, average='weighted'))
print("Recall:", recall_score(y_test, y_pred_svm, average='weighted'))
print("F1-score:", f1_score(y_test, y_pred_svm, average='weighted'))

# Evaluate MLP performance
print("\nMLP Classifier Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_mlp))
print("Precision:", precision_score(y_test, y_pred_mlp, average='weighted'))
print("Recall:", recall_score(y_test, y_pred_mlp, average='weighted'))
print("F1-score:", f1_score(y_test, y_pred_mlp, average='weighted'))

# Confusion Matrix for SVM
cm_svm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_filtered), yticklabels=np.unique(y_filtered))
plt.title("Confusion Matrix - SVM")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Confusion Matrix for MLP
cm_mlp = confusion_matrix(y_test, y_pred_mlp)
sns.heatmap(cm_mlp, annot=True, fmt="d", cmap="Greens", xticklabels=np.unique(y_filtered), yticklabels=np.unique(y_filtered))
plt.title("Confusion Matrix - MLP")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Bar plots for accuracy, precision, recall, F1-score
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
svm_scores = [accuracy_score(y_test, y_pred_svm),
              precision_score(y_test, y_pred_svm, average='weighted'),
              recall_score(y_test, y_pred_svm, average='weighted'),
              f1_score(y_test, y_pred_svm, average='weighted')]

mlp_scores = [accuracy_score(y_test, y_pred_mlp),
              precision_score(y_test, y_pred_mlp, average='weighted'),
              recall_score(y_test, y_pred_mlp, average='weighted'),
              f1_score(y_test, y_pred_mlp, average='weighted')]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, svm_scores, width, label='SVM')
rects2 = ax.bar(x + width/2, mlp_scores, width, label='MLP')

ax.set_ylabel('Scores')
ax.set_title('Classifier Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

fig.tight_layout()
plt.show()

# ROC Curve
y_test_bin = label_binarize(y_test, classes=np.unique(y_filtered))
n_classes = y_test_bin.shape[1]

# Compute ROC curve and ROC area for each class
fpr_svm = dict()
tpr_svm = dict()
roc_auc_svm = dict()
fpr_mlp = dict()
tpr_mlp = dict()
roc_auc_mlp = dict()

for i in range(n_classes):
    fpr_svm[i], tpr_svm[i], _ = roc_curve(y_test_bin[:, i], y_proba_svm[:, i])
    roc_auc_svm[i] = auc(fpr_svm[i], tpr_svm[i])
    fpr_mlp[i], tpr_mlp[i], _ = roc_curve(y_test_bin[:, i], y_proba_mlp[:, i])
    roc_auc_mlp[i] = auc(fpr_mlp[i], tpr_mlp[i])

# Plot all ROC curves
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
plt.figure()
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr_svm[i], tpr_svm[i], color=color, lw=2, label=f'SVM ROC curve (area = {roc_auc_svm[i]:0.2f}) for class {i}')
    plt.plot(fpr_mlp[i], tpr_mlp[i], color=color, lw=2, linestyle='--', label=f'MLP ROC curve (area = {roc_auc_mlp[i]:0.2f}) for class {i}')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc="lower right")
plt.show()

# UMAP Visualization
reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(features_scaled)

plt.figure(figsize=(10, 7))
plt.scatter(embedding[:, 0], embedding[:, 1], c=all_data['new_cluster_membership'], cmap='Spectral', s=5)
plt.colorbar(boundaries=np.arange(-1, best_n_clusters + 1) - 0.5).set_ticks(np.arange(-1, best_n_clusters))
plt.title('UMAP projection of the data')
plt.show()

# Reassign clusters based on these thresholds
new_labels = []

for i, row in all_data.iterrows():
    cluster = row['cluster_membership']
    if row['distance_to_centroid'] > thresholds[cluster]:
        new_labels.append(-1)  # -1 could indicate an outlier or unassigned
    else:
        new_labels.append(cluster)

# Add new labels to the dataframe
all_data['new_cluster_membership'] = new_labels

# Save the updated data with new cluster labels
all_data.to_csv(os.path.join(clusterings_results_path, f'kmeans_clustering_{best_n_clusters}_relabeled.csv'), index=False)

print("K-Means clustering: Relabeled data saved.")

# Train and evaluate an SVM classifier using the relabeled data
X = features_scaled
y = all_data['new_cluster_membership']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an SVM classifier
svm = SVC(kernel='rbf', random_state=42)  # Using radial basis function kernel
svm.fit(X_train, y_train)

# Make predictions
y_pred = svm.predict(X_test)

# Evaluate performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("F1-score:", f1_score(y_test, y_pred, average='weighted'))

# UMAP Visualization Before and After Relabeling
umap_reducer = umap.UMAP(n_neighbors=20, n_components=3, min_dist=0.7, random_state=42)
standard_embedding = umap_reducer.fit_transform(features_scaled)

# Transform the KMeans cluster centers using the UMAP reducer
umap_centroids = umap_reducer.transform(kmeans.cluster_centers_)

# Plot the UMAP embedding before relabeling
fig = plt.figure(figsize=(12, 6))

ax = fig.add_subplot(121, projection='3d')
colors = ['lightgreen', 'lightskyblue', 'lightpink', 'navajowhite', 'lightseagreen']

for i in range(best_n_clusters):
    ax.scatter(standard_embedding[cluster_membership == i, 0], standard_embedding[cluster_membership == i, 1],
               standard_embedding[cluster_membership == i, 2], c=colors[i], label=f'Cluster {i}', alpha=0.1)

ax.scatter(umap_centroids[:, 0], umap_centroids[:, 1], umap_centroids[:, 2], color='crimson', marker='x', s=80, label='Centroids')
for j in range(best_n_clusters):
    ax.text(umap_centroids[j, 0], umap_centroids[j, 1], umap_centroids[j, 2], str(j+1), color='k', fontsize=10, fontweight='bold')

ax.set_title(f'KMeans Clustering Before Relabeling ({best_n_clusters} clusters)')
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_zlabel('UMAP 3')
ax.legend()

# Plot the UMAP embedding after relabeling
ax2 = fig.add_subplot(122, projection='3d')

for i in range(best_n_clusters):
    mask = (all_data['new_cluster_membership'] == i)
    ax2.scatter(standard_embedding[mask, 0], standard_embedding[mask, 1], standard_embedding[mask, 2], c=colors[i], label=f'Cluster {i}', alpha=0.1)

ax2.scatter(umap_centroids[:, 0], umap_centroids[:, 1], umap_centroids[:, 2], color='crimson', marker='x', s=80, label='Centroids')
for j in range(best_n_clusters):
    ax2.text(umap_centroids[j, 0], umap_centroids[j, 1], umap_centroids[j, 2], str(j+1), color='k', fontsize=10, fontweight='bold')

ax2.set_title(f'KMeans Clustering After Relabeling ({best_n_clusters} clusters)')
ax2.set_xlabel('UMAP 1')
ax2.set_ylabel('UMAP 2')
ax2.set_zlabel('UMAP 3')
ax2.legend()

# Save and show the figure
plt.tight_layout()
plt.savefig(os.path.join(clusterings_results_path, f'umap_embedding_comparison_{best_n_clusters}_clusters.png'))
plt.show()

# # Get random samples
# random_samples = get_random_samples(all_data, 'new_cluster_membership', num_samples=5)
# plot_and_save_audio_segments(random_samples, audio_path, clusterings_results_path, 'new_cluster_membership')

# # Generate a statistical report using the new cluster memberships
# stats = statistical_report(all_data, all_data['new_cluster_membership'], best_n_clusters, metadata, clusterings_results_path)

# # Create radar plots for the statistical report
# radar_results = create_statistical_report_with_radar_plots(all_data, all_data['new_cluster_membership'], best_n_clusters, metadata, clusterings_results_path)

# print(stats)
# print(radar_results)
