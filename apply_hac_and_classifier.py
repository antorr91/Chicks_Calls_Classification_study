import os
import glob
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from classification_utils import  split_data_recordings_stratified
from classification_visualisation import generate_metrics_from_report, plot_comparison_bar, plot_confusion_matrix


# Define paths
features_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_result_high_quality_dataset_without_jtfs'
metadata_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_results_high_quality_dataset_meta\\high_quality_dataset_metadata.csv'
audio_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Data\\high_quality_dataset'

# Path to save the results
clusterings_results_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Clustering_\\Distance_based_hac_clustering_and_classification'
if not os.path.exists(clusterings_results_path):
    os.makedirs(clusterings_results_path)

# Load data
list_files = glob.glob(os.path.join(features_path, '*.csv'))
all_data = pd.concat([pd.read_csv(f) for f in list_files], ignore_index=True)
metadata = pd.read_csv(metadata_path)

# Drop NaN values
all_data = all_data.dropna()

# Scale data
scaler = StandardScaler()
features = all_data.drop(['Call Number', 'onsets_sec', 'offsets_sec', 'recording', 'call_id'], axis=1)
features_scaled = scaler.fit_transform(features)

# Clustering
n_clusters = 3
agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', compute_distances=True)
cluster_membership = agg.fit_predict(features_scaled)
all_data['cluster_membership'] = cluster_membership

# Linkage matrix for dendrogram
linkage_matrix = linkage(features_scaled, method='ward')

# Calculate cluster centers
cluster_centers = np.array([features_scaled[all_data['cluster_membership'] == i].mean(axis=0) for i in range(n_clusters)])

# Calculate distances to cluster centers
distances_to_centers = cdist(features_scaled, cluster_centers, 'euclidean')
all_data['distance_to_center'] = [distances_to_centers[i, cluster] for i, cluster in enumerate(cluster_membership)]

# Define thresholds for clusters
cluster_thresholds = {
    0: 1.91,  # Threshold for cluster 0
    1: 1.79,  # Threshold for cluster 1
    2: 1.80   # Threshold for cluster 2
}

# Assign labels based on thresholds
def assign_labels(row):
    cluster = row['cluster_membership']
    distance = row['distance_to_center']
    if distance > cluster_thresholds[cluster]:
        return -1  # Outlier
    else:
        return cluster  # Original cluster label

all_data['label'] = all_data.apply(assign_labels, axis=1)

# Filter out outliers (-1)
filtered_data = all_data[all_data['label'] != -1]


# count the number of samples in each cluster
n_samples = filtered_data['label'].value_counts().sort_index()
print("Number of samples in each cluster:")

for cluster, count in n_samples.items():
    print(f"Cluster {cluster}: {count} samples")

print("\n")

train_data, test_data = split_data_recordings_stratified(filtered_data, test_ratio=0.3, min_test_samples_per_cluster=6)






# Print out the distribution of clusters in the training and test sets
print(f"Training set has {len(train_data)} samples.")
print(f"Balanced test set has {len(test_data)} samples.")

# count the number of samples in each cluster in the training and test set
train_cluster_counts = train_data['label'].value_counts().sort_index()
test_cluster_counts = test_data['label'].value_counts().sort_index()

print("Number of samples in each cluster in the training set:")
for cluster, count in train_cluster_counts.items():
    print(f"Cluster {cluster}: {count} samples")

print("\nNumber of samples in each cluster in the balanced test set:")
for cluster, count in test_cluster_counts.items():
    print(f"Cluster {cluster}: {count} samples")

# Print the results to verify
print(f"Training set has {len(train_data)} samples.")
print(f"Test set has {len(test_data)} samples.")

# count the number of samples in each cluster in the training set
n_samples_train = train_data['label'].value_counts().sort_index()
print("Number of samples in each cluster in the training set:")

for cluster, count in n_samples_train.items():
    print(f"Cluster {cluster}: {count} samples")



# count the number of samples in each cluster in the testing set

n_samples_test = test_data['label'].value_counts().sort_index()
print("Number of samples in each cluster in the testing set:")
for cluster, count in n_samples_test.items():
    print(f"Cluster {cluster}: {count} samples")



# export in a csv file the training and testing data with the labels

train_data.to_csv(os.path.join(clusterings_results_path, 'train_data.csv'), index=False)
test_data.to_csv(os.path.join(clusterings_results_path, 'test_data.csv'), index=False)

print("Training and testing data exported to CSV files.")


print("\n")
# Separate features and labels for training and testing
X_train = train_data.drop(['Call Number', 'onsets_sec', 'offsets_sec', 'recording', 'call_id', 'cluster_membership', 'distance_to_center', 'label'], axis=1)
y_train = train_data['label']

X_test = test_data.drop(['Call Number', 'onsets_sec', 'offsets_sec', 'recording', 'call_id', 'cluster_membership', 'distance_to_center', 'label'], axis=1)
y_test = test_data['label']

# Train MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=300, alpha=1e-4,
                    solver='adam', random_state=42, learning_rate_init=.001)

mlp.fit(X_train, y_train)

# Train SVM classifier
svm = SVC(kernel='rbf', random_state=42, C=1.0, gamma='scale')
svm.fit(X_train, y_train)

# Make predictions with MLP
y_pred_mlp = mlp.predict(X_test)

# Make predictions with SVM
y_pred_svm = svm.predict(X_test)

# Print performance metrics for MLP
print("MLP Classifier Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_mlp))
print("Classification Report:\n", classification_report(y_test, y_pred_mlp))

# Print performance metrics for SVM
print("SVM Classifier Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm))

# Save the classifier models
import joblib
mlp_model_path = os.path.join(clusterings_results_path, 'mlp_classifier_model.pkl')
svm_model_path = os.path.join(clusterings_results_path, 'svm_classifier_model.pkl')
joblib.dump(mlp, mlp_model_path)
joblib.dump(svm, svm_model_path)
print(f"MLP model saved to {mlp_model_path}")
print(f"SVM model saved to {svm_model_path}")




# Generate metrics for both classifiers
mlp_results = generate_metrics_from_report(y_test, y_pred_mlp)
svm_results = generate_metrics_from_report(y_test, y_pred_svm)

# Prepare data for plotting
metrics = list(mlp_results.keys())
mlp_values = [mlp_results[metric] for metric in metrics]
svm_values = [svm_results[metric] for metric in metrics]

# Plot comparison
# plot_comparison_bar([mlp_values, svm_values], ['MLP', 'SVM'], 'Comparison of Classifiers')
plot_comparison_bar(metrics, mlp_values, svm_values, 'Comparison of MLP and SVM Classifiers')



# Get confusion matrices
cm_mlp = confusion_matrix(y_test, y_pred_mlp, labels=[0, 1, 2])
cm_svm = confusion_matrix(y_test, y_pred_svm, labels=[0, 1, 2])

# Plot confusion matrices
plot_confusion_matrix(cm_mlp, labels=['0', '1', '2'], title='MLP Classifier Confusion Matrix')
plot_confusion_matrix(cm_svm, labels=['0', '1', '2'], title='SVM Classifier Confusion Matrix')

# ROC Curve for each class (binarized)
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
y_pred_mlp_bin = label_binarize(y_pred_mlp, classes=[0, 1, 2])
y_pred_svm_bin = label_binarize(y_pred_svm, classes=[0, 1, 2])

# ROC curve plots
plt.figure(figsize=(12, 8))
for i in range(y_test_bin.shape[1]):
    fpr_mlp, tpr_mlp, _ = roc_curve(y_test_bin[:, i], y_pred_mlp_bin[:, i])
    roc_auc_mlp = auc(fpr_mlp, tpr_mlp)
    plt.plot(fpr_mlp, tpr_mlp, label=f'MLP Class {i} (AUC = {roc_auc_mlp:.2f})')

    fpr_svm, tpr_svm, _ = roc_curve(y_test_bin[:, i], y_pred_svm_bin[:, i])
    roc_auc_svm = auc(fpr_svm, tpr_svm)
    plt.plot(fpr_svm, tpr_svm, linestyle='--', label=f'SVM Class {i} (AUC = {roc_auc_svm:.2f})')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()