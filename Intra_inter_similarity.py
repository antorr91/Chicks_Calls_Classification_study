import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, mannwhitneyu
import os

# File paths
dataset_path = r'C:\Users\anton\Chicks_Onset_Detection_project\Results_Classification\Template_matching\trial_no_cropping'
intra_file = os.path.join(dataset_path, 'intra_similarity_results.csv')
inter_file = os.path.join(dataset_path, 'inter_similarity_results.csv')

# Load the data
intra_similarity_df = pd.read_csv(intra_file)
inter_similarity_df = pd.read_csv(inter_file)

print(f"Dimensions of intra_similarity_df: {intra_similarity_df.shape}")
print(f"Dimensions of inter_similarity_df: {inter_similarity_df.shape}")

# Verify columns
print("Columns in intra_similarity_df:", intra_similarity_df.columns.tolist())
print("Columns in inter_similarity_df:", inter_similarity_df.columns.tolist())

# 1. General analysis of intra vs inter cluster
print("\n--- General analysis of intra vs inter cluster ---")

intra_mean = intra_similarity_df['similarity'].mean()
intra_std = intra_similarity_df['similarity'].std()
inter_mean = inter_similarity_df['similarity'].mean()
inter_std = inter_similarity_df['similarity'].std()

print(f"Intra-cluster similarity: mean={intra_mean:.4f}, std={intra_std:.4f}, n={len(intra_similarity_df)}")
print(f"Inter-cluster similarity: mean={inter_mean:.4f}, std={inter_std:.4f}, n={len(inter_similarity_df)}")

# General statistical test
stat, p_value = mannwhitneyu(
    intra_similarity_df['similarity'], 
    inter_similarity_df['similarity'], 
    alternative='greater'
)
print(f"Mann-Whitney U general test: statistic={stat:.4f}, p-value={p_value:.8f}")
if p_value < 0.05:
    print("Result: The difference is statistically significant!")
else:
    print("Result: No significant difference.")

# Visualisation of distributions
plt.figure(figsize=(10, 6))
combined_data = pd.concat([
    intra_similarity_df.assign(type='Intra-cluster')[['similarity', 'type']],
    inter_similarity_df.assign(type='Inter-cluster')[['similarity', 'type']]
])
sns.boxplot(x='type', y='similarity', data=combined_data)
plt.title('General comparison of intra vs inter cluster similarity')
plt.ylabel('Similarity')
plt.tight_layout()
plt.savefig(os.path.join(dataset_path, 'general_similarity_comparison.png'))
plt.show()


# 2. Analysis for each test cluster (cluster_membership_test)
plt.figure(figsize=(10, 5))
sns.histplot(intra_similarity_df['similarity'], kde=True, color='blue', label='Intra-cluster')
sns.histplot(inter_similarity_df['similarity'], kde=True, color='red', label='Inter-cluster', alpha=0.7)
plt.xlabel("Template Matching Similarity")
plt.ylabel("Frequency")
plt.legend()
plt.title("Template Matching Similarity Distribution")
plt.savefig(os.path.join(dataset_path, 'general_similarity_comparison_histogram.png'))
plt.show()

# 1. Analysis of similarity within each cluster (intra_similarity_df)
plt.figure(figsize=(12, 6))
for cluster in intra_similarity_df['cluster_membership_rep'].unique():
    subset = intra_similarity_df[intra_similarity_df['cluster_membership_rep'] == cluster]
    sns.histplot(subset['similarity'], kde=True, label=f'Cluster {cluster}', alpha=0.6)
plt.xlabel("Template Matching Similarity")
plt.ylabel("Frequency")
plt.title("Similarity Distribution Intra-Cluster")
plt.legend()
plt.show()

# 2. Analysis for each test cluster (cluster_membership_test)
print("\n--- Analysis by test cluster ---")

# Identification of unique clusters in test data
test_clusters = sorted(pd.concat([
    intra_similarity_df['cluster_membership_test'],
    inter_similarity_df['cluster_membership_test']
]).unique())

print(f"Test clusters found: {test_clusters}")

# Create a figure for boxplot comparisons
plt.figure(figsize=(15, 8))

# Initialise dataframe for collecting statistics
cluster_stats = pd.DataFrame(columns=['Cluster_Test', 'Type', 'Mean', 'Std', 'N', 'p-value'])

# For each test cluster
for i, test_cluster in enumerate(test_clusters):
    print(f"\nAnalysis for Test Cluster {test_cluster}:")
    
    # Filter intra-cluster data for this specific test cluster
    intra_this_test = intra_similarity_df[
        intra_similarity_df['cluster_membership_test'] == test_cluster
    ].copy()
    
    # Filter inter-cluster data for this specific test cluster
    inter_this_test = inter_similarity_df[
        inter_similarity_df['cluster_membership_test'] == test_cluster
    ].copy()
    
    print(f"  Intra-cluster samples for test {test_cluster}: {len(intra_this_test)}")
    print(f"  Inter-cluster samples for test {test_cluster}: {len(inter_this_test)}")
    
    if len(intra_this_test) == 0 or len(inter_this_test) == 0:
        print(f"  WARNING: Insufficient data for test cluster {test_cluster}")
        continue
    
    # Calculate statistics
    intra_mean = intra_this_test['similarity'].mean()
    intra_std = intra_this_test['similarity'].std()
    inter_mean = inter_this_test['similarity'].mean()
    inter_std = inter_this_test['similarity'].std()
    
    print(f"  Intra-cluster similarity {test_cluster}: mean={intra_mean:.4f}, std={intra_std:.4f}")
    print(f"  Inter-cluster similarity {test_cluster}: mean={inter_mean:.4f}, std={inter_std:.4f}")
    
    # Mann-Whitney U statistical test
    try:
        stat, p_value = mannwhitneyu(
            intra_this_test['similarity'], 
            inter_this_test['similarity'], 
            alternative='greater'
        )
        print(f"  Mann-Whitney U test: statistic={stat:.4f}, p-value={p_value:.8f}")
        
        if p_value < 0.05:
            print("  Result: The difference is statistically significant!")
        else:
            print("  Result: No significant difference.")
    except ValueError as e:
        print(f"  Error in statistical test: {e}")
        p_value = np.nan
    
    # Add to statistics dataframe
    new_stats = pd.DataFrame({
        'Cluster_Test': [test_cluster, test_cluster],
        'Type': ['Intra', 'Inter'],
        'Mean': [intra_mean, inter_mean],
        'Std': [intra_std, inter_std],
        'N': [len(intra_this_test), len(inter_this_test)],
        'p-value': [p_value, p_value]
    })
    cluster_stats = pd.concat([cluster_stats, new_stats], ignore_index=True)
    
    # Prepare data for boxplot
    intra_this_test['type'] = f'Intra-{test_cluster}'
    inter_this_test['type'] = f'Inter-{test_cluster}'
    
    # Add to subplot for boxplots
    plt.subplot(1, len(test_clusters), i+1)
    this_data = pd.concat([
        intra_this_test[['similarity', 'type']],
        inter_this_test[['similarity', 'type']]
    ])
    
    # Ensure data is ordered so that Intra comes before Inter
    order = [f'Intra-{test_cluster}', f'Inter-{test_cluster}']
    sns.boxplot(x='type', y='similarity', data=this_data, order=order)
    
    plt.title(f'Test Cluster {test_cluster}')
    plt.xlabel('')
    plt.ylabel('Similarity' if i == 0 else '')
    plt.ylim(0, 1)  # Assuming similarity is between 0 and 1

plt.tight_layout()
plt.savefig(os.path.join(dataset_path, 'cluster_test_similarity_comparison.png'))
plt.show()

# 3. Summary visualisation of statistics by test cluster
plt.figure(figsize=(12, 6))
cluster_stats_pivot = cluster_stats.pivot(index='Cluster_Test', columns='Type', values='Mean')
cluster_stats_pivot.plot(kind='bar', yerr=cluster_stats.pivot(index='Cluster_Test', columns='Type', values='Std'))
plt.title('Mean similarity by test cluster')
plt.ylabel('Mean similarity')
plt.ylim(0, 1)  # Assuming similarity is between 0 and 1
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(os.path.join(dataset_path, 'cluster_test_similarity_summary.png'))
plt.show()

# 4. Analysis of similarity between specific clusters (similarity matrix)
print("\n--- Similarity matrix between clusters ---")

# Get all unique clusters (rep and test)
all_clusters = sorted(pd.concat([
    intra_similarity_df['cluster_membership_rep'],
    intra_similarity_df['cluster_membership_test'],
    inter_similarity_df['cluster_membership_rep'],
    inter_similarity_df['cluster_membership_test']
]).unique())

# Create an empty matrix for mean similarity between clusters
similarity_matrix = pd.DataFrame(
    index=all_clusters, 
    columns=all_clusters, 
    dtype=float
)

# Combine all data
all_data = pd.concat([
    intra_similarity_df,
    inter_similarity_df
])

# Calculate mean similarity for each pair of clusters
for rep_cluster in all_clusters:
    for test_cluster in all_clusters:
        subset = all_data[
            (all_data['cluster_membership_rep'] == rep_cluster) & 
            (all_data['cluster_membership_test'] == test_cluster)
        ]
        
        if len(subset) > 0:
            similarity_matrix.loc[rep_cluster, test_cluster] = subset['similarity'].mean()
        else:
            similarity_matrix.loc[rep_cluster, test_cluster] = np.nan

print("Mean similarity matrix between clusters:")
print(similarity_matrix)

# Visualisation of similarity matrix
plt.figure(figsize=(10, 8))
sns.heatmap(similarity_matrix, annot=True, fmt=".3f", cmap="Blues", vmin=0, vmax=1)
plt.title('Mean similarity matrix between clusters')
plt.xlabel('Test Cluster')
plt.ylabel('Rep Cluster')
plt.tight_layout()
plt.savefig(os.path.join(dataset_path, 'cluster_similarity_matrix.png'))
plt.show()

# Save results
print("\nSummary table:")
print(cluster_stats.to_string(index=False))

# Save results to CSV
cluster_stats.to_csv(os.path.join(dataset_path, 'cluster_similarity_stats.csv'), index=False)
similarity_matrix.to_csv(os.path.join(dataset_path, 'cluster_similarity_matrix.csv'))
print(f"\nStatistics saved to: {os.path.join(dataset_path, 'cluster_similarity_stats.csv')}")
print(f"Similarity matrix saved to: {os.path.join(dataset_path, 'cluster_similarity_matrix.csv')}")