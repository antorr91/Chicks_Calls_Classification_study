import os
import glob
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, classification_report, 
    confusion_matrix, accuracy_score, f1_score, roc_auc_score, 
    ConfusionMatrixDisplay
)

from classification_utils import (
    split_data_recordings_stratified, count_calls_by_cluster, 
    count_calls_by_cluster_in_train_test, print_and_export_metrics, 
    analyse_misclassifications, compare_model_errors
)
from classification_visualisation import (
    generate_metrics_from_report, plot_comparison_bar, plot_confusion_matrix, 
    plot_and_save_audio_segments, plot_comparison_bar_metrics
)




# =====================================================================
# Data Loading and Pre-processing
# =====================================================================

# Load the data ( my Labelled dataset)
# (Loading the filtered data from the specified classification results path)
classification_results_path = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Classification\\MLP_SVM'
labelled_dataset = pd.read_csv(os.path.join(classification_results_path, 'labelled_dataset.csv'))

scaler = StandardScaler()
# Extract features by dropping columns that are not used for modelling.
features = labelled_dataset.drop(['Call Number', 'onsets_sec', 'offsets_sec', 'recording', 'call_id'], axis=1)
features_scaled = scaler.fit_transform(features)

# --- Splitting the data into training, validation and testing sets using a stratified split
train_data, validation_data, test_data = split_data_recordings_stratified(
    all_data=labelled_dataset,                         # The complete dataset
    train_recordings=['chick32_d0', 'chick34_d0', 'chick39_d0', 'chick41_d0',  # List of recordings for the training set
                      'chick85_d0', 'chick87_d0', 'chick89_d0', 'chick91_d0'],  
    group_col='recording',                     # Column that contains the names of the recordings
    label_col='label',                         # Column that contains the cluster labels
    test_ratio=0.35,                          # Proportion of data to allocate to the test set
    min_test_samples_per_cluster=50,            # Minimum number of samples per cluster in the test set
    validation_ratio=0.15                    # Proportion of data to allocate to the validation set
)

# =====================================================================
# Data Exploration: Cluster Counts and Distribution
# =====================================================================

# count the number of samples in each cluster in the training and test set
train_cluster_counts = train_data['label'].value_counts().sort_index()
test_cluster_counts = test_data['label'].value_counts().sort_index()
valid_cluster_counts = validation_data['label'].value_counts().sort_index()

print("\n")
print(f"Training set has {len(train_data)} samples.")
print("\n")
print("\n")
# Check the cluster representation in the training set
print("Training set cluster distribution:")
print(train_data['label'].value_counts(normalize=True) * 100)
print("\n")
print("\n")
print("Number of samples in each cluster in the training set:")
for cluster, count in train_cluster_counts.items():
    print(f"Cluster {cluster}: {count} samples")
print("\n")
print(f"°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°")
print("\n")
print("\n")
print("\n")
print(f"Test set has {len(test_data)} samples.")
print("\n")
print("\n")
# Check the cluster representation in the test set
print("\nTesting set cluster distribution:")
print(test_data['label'].value_counts(normalize=True) * 100)
print("\n")
print("\n")
print("Number of samples in each cluster in the balanced test set:")
for cluster, count in test_cluster_counts.items():
    print(f"Cluster {cluster}: {count} samples")
print("\n")
print(f"The validation set has {len(validation_data)} samples.")
print("\n")
print("\n")
# Check the cluster representation in the validation set
print("\nValidation set cluster distribution:")
print(validation_data['label'].value_counts(normalize=True) * 100)
print("\n")
print("\n")
print("Number of samples in each cluster in the balanced validation set:")
for cluster, count in valid_cluster_counts.items():
    print(f"Cluster {cluster}: {count} samples")
print("\n")

# =====================================================================
# Save the Data Splits to CSV Files
# =====================================================================

# Save the training and testing data to a dictionary for further processing.
data_dict = {
    'train': train_data,
    'test': test_data,
    'validation': validation_data
}

# Call the function to count calls by cluster in train, test and validation sets.
call_counts = count_calls_by_cluster_in_train_test(data_dict, group_col='recording', label_col='label')

# Accessing the results from the call counts.
train_counts = call_counts['train']
test_counts = call_counts['test']
val_counts = call_counts['validation']

print("Training Call Counts:")
print(train_counts)

print("\nTesting Call Counts:")
print(test_counts)

print("\nValidation Call Counts:")
print(val_counts)

# Export training, testing and validation data to CSV files.
train_data.to_csv(os.path.join(classification_results_path, 'train_data.csv'), index=False)
test_data.to_csv(os.path.join(classification_results_path, 'test_data.csv'), index=False)
validation_data.to_csv(os.path.join(classification_results_path, 'validation_data.csv'), index=False)

print("Training and testing data exported to CSV files.")

# =====================================================================
# Feature and Label Separation and Scaling for Model Training
# =====================================================================

# Separate features and labels for training, testing and validation.
X_train = train_data.drop(['Call Number', 'onsets_sec', 'offsets_sec', 'recording', 'call_id', 
                           'cluster_membership', 'distance_to_center', 'label'], axis=1)
y_train = train_data['label']

# print head of X_train
# print(X_train.head(3))

X_test = test_data.drop(['Call Number', 'onsets_sec', 'offsets_sec', 'recording', 'call_id', 
                         'cluster_membership', 'distance_to_center', 'label'], axis=1)
y_test = test_data['label']

# print(X_test.head(3))

X_val = validation_data.drop(['Call Number', 'onsets_sec', 'offsets_sec', 'recording', 'call_id', 
                              'cluster_membership', 'distance_to_center', 'label'], axis=1)
y_val = validation_data['label']

# print(X_val.head(3))

# Scale the training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform validation and test data using the same scaler.
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# =====================================================================
# Prepare Combined Data for Predefined Split (Validation Setup)
# =====================================================================

# Combine training and validation sets for model selection using PredefinedSplit.
X_combined = np.vstack([X_train_scaled, X_val_scaled])
y_combined = np.concatenate([y_train, y_val])
test_fold = np.zeros(len(X_combined))
# Mark training samples with -1 so that they are always in the training fold.
test_fold[:len(X_train_scaled)] = -1
ps = PredefinedSplit(test_fold)


# =====================================================================
# Grid Search for MLP Classifier
# =====================================================================

# Grid search parameters for MLP  
# First trial parameters:  MLPClassifier(alpha=0.1, batch_size=48, early_stopping=True, hidden_layer_sizes=(300, 200, 100), learning_rate_init=0.05, max_iter=1500)
# Second trial parameters: {'activation': 'relu', 'alpha': 0.1, 'batch_size': 20, 'early_stopping': True, 'hidden_layer_sizes': (100, 50), 'learning_rate': 'constant', 'learning_rate_init': 0.005, 'solver': 'adam'}

mlp_params = {
    'hidden_layer_sizes': [  
        (200, 100), (200, 150, 50), (250, 150, 50), (300, 150, 50), (300, 200, 100), (300, 200, 150), (300, 300, 100)
    ],
    'activation': ['relu', 'tanh'],
    'solver': ['adam'],
    'alpha': [0.1, 0.05, 0.01, 0.5, 1.0],
    'learning_rate_init': [0.05, 0.01, 0.001, 0.005],
    'early_stopping': [True],
    'batch_size': [10, 20, 32, 48, 64, 128],
    'learning_rate': ['adaptive', 'invscaling', 'constant']
}

# Grid search for MLP using the PredefinedSplit
mlp_grid = GridSearchCV(MLPClassifier(max_iter=1500), mlp_params, scoring='f1_weighted', cv=ps, n_jobs=-1, verbose=2)
mlp_grid.fit(X_combined, y_combined)

# Print best parameters and performance for MLP
print("Best parameters for MLP with validation set:", mlp_grid.best_params_)
print("Best score for MLP with validation set:", mlp_grid.best_score_)
print("Best estimator for MLP with validation set:", mlp_grid.best_estimator_)

# Save the grid search results to CSV files.
results_mlp = pd.DataFrame(mlp_grid.cv_results_)
results_mlp.to_csv(os.path.join(classification_results_path, 'mlp_grid_search_results.csv'), index=False)

# =====================================================================

print("°" * 80)
print("\n")

# =====================================================================
# Grid Search for SVM Classifier
# =====================================================================

# Grid search parameters for SVM
# First trial parameters: {'C': 5, 'class_weight': {0: 0.35, 1: 0.45, 2: 0.2}, 'gamma': 'scale', 'kernel': 'rbf'}
# Second trial parameters: {'C': 0.01, 'class_weight': 'balanced', 'gamma': 0.1, 'kernel': 'rbf'}

svm_params = {
    'C': [0.1, 1, 5, 10, 20, 25],  # Reduced range of C values.
    'gamma': ['scale', 0.1, 0.05],
    'kernel': ['rbf'],
    'class_weight': [{0: 0.35, 1: 0.45, 2: 0.20}, {0: 0.44, 1: 0.41, 2: 0.13}]
}

# Grid search for SVM using the PredefinedSplit.
svm_grid = GridSearchCV(SVC(max_iter=2000), svm_params, scoring='f1_weighted', cv=ps, n_jobs=-1, verbose=2)
svm_grid.fit(X_combined, y_combined)

# Print best parameters and performance for SVM.
print("Best parameters for SVM with validation set:", svm_grid.best_params_)
print("Best score for SVM with validation set:", svm_grid.best_score_)
print("Best estimator for SVM with validation set:", svm_grid.best_estimator_)

print("°" * 80)
print("\n")
# Save the grid search results to CSV files.
results_svm = pd.DataFrame(svm_grid.cv_results_)
results_svm.to_csv(os.path.join(classification_results_path, 'svm_grid_search_results.csv'), index=False)

# =====================================================================
# Final Training of Selected Models
# =====================================================================

# Train again the models using the best parameters obtained from grid search.

# Train the final MLP model.
final_model_mlp = MLPClassifier(**mlp_grid.best_params_, max_iter=1500)
final_model_mlp.fit(X_train_scaled, y_train)

# Train the final SVM model.
final_model_svm = SVC(**svm_grid.best_params_, max_iter=2000, probability=True)
final_model_svm.fit(X_train_scaled, y_train)

# =====================================================================
# Evaluate Final Models on Test Set and Export Metrics
# =====================================================================

# Final test predictions for both models.
y_pred_mlp_test = final_model_mlp.predict(X_test_scaled)
y_pred_svm_test = final_model_svm.predict(X_test_scaled)

# Print performance metrics for MLP.
print("MLP Classifier Results:")
print("MLP Test Accuracy:", accuracy_score(y_test, y_pred_mlp_test))
print("MLP Classification Report:\n", classification_report(y_test, y_pred_mlp_test))
# print_and_export_metrics(y_test, y_pred_mlp_test, "MLP")

# Print performance metrics for SVM.
print("SVM Classifier Results:")
print("SVM Test Accuracy:", accuracy_score(y_test, y_pred_svm_test))
print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm_test))
# print_and_export_metrics(y_test, y_pred_svm_test, "SVM")

print("\n")
print("°" * 80)
print("\n")
print("°" * 80)
print("\n")


# =====================================================================
# Generate Comparison Metrics and Confusion Matrices
# =====================================================================

# Generate metrics from the classification reports for both models.
mlp_results = generate_metrics_from_report(y_test, y_pred_mlp_test)
svm_results = generate_metrics_from_report(y_test, y_pred_svm_test)

# Prepare data for plotting: obtain a list of metric names and corresponding values.
metrics = list(mlp_results.keys())
mlp_values = [mlp_results[metric] for metric in metrics]
svm_values = [svm_results[metric] for metric in metrics]

# Plot a comparison bar chart for both classifiers.
plot_comparison_bar(metrics, mlp_values, svm_values, 'Comparison of MLP and SVM Classifiers')

# Plot a comparison bar chart for weighted metrics.
plot_comparison_bar_metrics(metrics, mlp_values, svm_values, title='Comparison of MLP and SVM Classifiers (Weighted Metrics)')

# Obtain confusion matrices for both models.
cm_mlp = confusion_matrix(y_test, y_pred_mlp_test, labels=[0, 1, 2])
cm_svm = confusion_matrix(y_test, y_pred_svm_test, labels=[0, 1, 2])

# Plot the confusion matrices.
plot_confusion_matrix(cm_mlp, labels=['0', '1', '2'], title='MLP Classifier Confusion Matrix')
plot_confusion_matrix(cm_svm, labels=['0', '1', '2'], title='SVM Classifier Confusion Matrix')


# =====================================================================
# ROC Curve for Each Class (Binarised)
# =====================================================================

# Binarise the true and predicted labels for ROC curve analysis.
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
y_pred_mlp_bin = label_binarize(y_pred_mlp_test, classes=[0, 1, 2])
y_pred_svm_bin = label_binarize(y_pred_svm_test, classes=[0, 1, 2])

# Plot the ROC curves for each class.
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
plt.savefig(os.path.join(classification_results_path, 'roc_curve.png'))
plt.show()















