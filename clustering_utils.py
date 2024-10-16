import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from scipy.io import wavfile
import librosa as lb
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import skfuzzy as fuzz
from math import pi
import seaborn as sns
import itertools
import soundfile as sf
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler







def split_data_recordings(all_data, test_size_per_group=2, group_col='recording'):
    """
    Split data into train and test sets based on a grouping column (e.g., recording).
    Ensures that no overlap exists between the train and test sets based on the grouping.

    Parameters:
    - all_data: DataFrame containing all data.
    - test_size_per_group: Number of groups (e.g., recordings) to include in the test set.
    - group_col: Column to group by (e.g., 'recording').

    Returns:
    - train_data: Training dataset.
    - test_data: Test dataset.
    """

    # change the code to have in the training set the recqording chick32_d0, chick34_d0, chick41_d0, chick85_d0, chick87_d0, chick89_d0, chick91_d0
    # then in the test set the reaiming recordings



    unique_groups = all_data[group_col].unique()
    test_groups = np.random.choice(unique_groups, test_size_per_group, replace=False)
    
    test_data = all_data[all_data[group_col].isin(test_groups)]
    train_data = all_data[~all_data[group_col].isin(test_groups)]
    
    return train_data, test_data










def plot_dendrogram(model, num_clusters=None, **kwargs):
    """
    Plot a dendrogram for hierarchical clustering.

    Args:
        model: The hierarchical clustering model (e.g., from scikit-learn).
        num_clusters (int): The number of clusters desired. If provided, a threshold line will be drawn on the dendrogram to indicate where to cut it.
        **kwargs: Additional keyword arguments to pass to the dendrogram function.

    Returns:
        threshold (float): The distance threshold at which to cut the dendrogram.
        linkage_matrix (numpy.ndarray): The linkage matrix used to construct the dendrogram.
        counts (numpy.ndarray): Counts of samples under each node in the dendrogram.
        n_samples (int): Total number of samples.
        labels (numpy.ndarray): Labels assigned to each sample by the clustering model.
    """
    
    # Create linkage matrix and then plot the dendrogram
    counts = np.zeros(model.children_.shape[0])  # Initialize counts of samples under each node
    n_samples = len(model.labels_)  # Total number of samples
    
    # Iterate over merges to calculate counts
    for i, merge in enumerate(model.children_):
        current_count = 0
        # Iterate over children of merge
        for child_idx in merge:
            if (child_idx < n_samples):
                current_count += 1  # Leaf node
            else:
                current_count += counts[child_idx - n_samples]  # Non-leaf node
        counts[i] = current_count  # Update counts
        
    # Construct the linkage matrix
    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    
    # Debug: stampa le dimensioni degli array
    print(f"linkage_matrix shape: {linkage_matrix.shape}")
    print(f"linkage_matrix: {linkage_matrix}")

    # Plot the dendrogram
    dendrogram(linkage_matrix, **kwargs, leaf_rotation= 90. , leaf_font_size=5.0, truncate_mode='level', p=4, above_threshold_color='gray')
    #     p : int, optional
    #     The ``p`` parameter for ``truncate_mode``.
    # truncate_mode : str, optional
    #     The dendrogram can be hard to read when the original
    #     observation matrix from which the linkage is derived is
    #     large. Truncation is used to condense the dendrogram. There
    #     are several modes:

    #     ``None``
    #       No truncation is performed (default).
    #       Note: ``'none'`` is an alias for ``None`` that's kept for
    #       backward compatibility.
   
    # Plot the threshold line if num_clusters is specified
    if num_clusters is not None:
        max_d = np.max(model.distances_)
        threshold = max_d / (num_clusters - 1)
        plt.axhline(y=threshold, color='crimson', linestyle='--', label=f'{num_clusters} clusters', linewidth=1.5)
        plt.legend()
    
    plt.xlabel('Sample index or Cluster size', fontsize=10, fontfamily='Palatino Linotype')
    plt.ylabel('Distance', fontsize=10, fontfamily='Palatino Linotype')
    plt.title('Hierarchical Clustering Dendrogram', fontsize=13, fontfamily='Palatino Linotype')
    plt.show()
    plt.savefig('hierarchical_clustering_dendrogram.png')

    return threshold, linkage_matrix, counts, n_samples, model.labels_
    




# Define the function to find the elbow point
def find_elbow_point(scores):
    n_points = len(scores)
    all_coord = np.vstack((range(n_points), scores)).T
    first_point = all_coord[0]
    line_vec = all_coord[-1] - all_coord[0]
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    vec_from_first = all_coord - first_point
    scalar_product = np.sum(vec_from_first * np.tile(line_vec_norm, (n_points, 1)), axis=1)
    vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)
    vec_to_line = vec_from_first - vec_from_first_parallel
    dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
    best_index = np.argmax(dist_to_line)
    return best_index + 2  



# Function to get 5 random samples for each cluster
def get_random_samples(df, cluster_col, num_samples=5):
    random_samples = {}
    for cluster in df[cluster_col].unique():
        cluster_df = df[df[cluster_col] == cluster]
        if len(cluster_df) >= num_samples:
            random_samples[cluster] = cluster_df.sample(num_samples)
        else:
            random_samples[cluster] = cluster_df
    return random_samples




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

         

# Function to extract and plot audio segments
def plot_audio_segments(samples_dict, audio_path, clusterings_results_path, cluster_membership_label):
    for cluster, samples in samples_dict.items():
        fig, axes = plt.subplots(1, len(samples), figsize=(2 * len(samples), 2))
        fig.suptitle(f'Cluster {cluster} Audio Segments')
        if len(samples) == 1:
            axes = [axes]

        for idx, (i, sample) in enumerate(samples.iterrows()):
            audio_file = os.path.join(audio_path, sample['recording'] + '.wav')
            if os.path.exists(audio_file):
                # Load the audio file with librosa
                data, sr = lb.load(audio_file, sr=44100)
                
                # Compute the mel spectrogram
                S = lb.feature.melspectrogram(y=data, sr=sr, n_mels=128, fmin=2000, fmax=10000)
                log_S = lb.power_to_db(S, ref=np.max)

                # Segment the spectrogram
                calls_S = segment_spectrogram(log_S, [sample['onsets_sec']], [sample['offsets_sec']], sr=sr)
                call_S = calls_S[0]

                # Convert onset seconds with decimals to readable format
                # onset_sec = sample['onsets_sec']
                # if onset_sec < 60:
                #     onset_time = f"{onset_sec:.2f} sec"
                # else:
                #     minutes = int(onset_sec // 60)
                #     seconds = onset_sec % 60
                #     onset_time = f"{minutes} min & {seconds:.2f} sec"

                # Plot the audio segment
                img= axes[idx].imshow(call_S, aspect='auto', origin='lower', cmap='magma')
                axes[idx].set_title(f'Call {idx + 1} of {sample["recording"]} \n cluster {cluster}', fontsize=6)

                axes[idx].set_xlabel('Time', fontsize=5)
                axes[idx].set_ylabel('Frequency', fontsize=5)
                fig.colorbar(img, ax=axes[idx])
            else:
                print(f'Audio file {audio_file} not found')

        # Save the plot
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_filename = f'cluster_{cluster}_{cluster_membership_label}.png'
        plt.savefig(os.path.join(clusterings_results_path, plot_filename))


def plot_and_save_audio_segments(representative_calls, audio_path, save_path, cluster_label):
    """
    Extract, plot, and save audio segments for representative calls of a cluster.
    
    Args:
    representative_calls (DataFrame): DataFrame containing representative calls for a cluster
    audio_path (str): Path to the directory containing audio files
    save_path (str): Path to save the results
    cluster_label (str): Label for the cluster
    """
    fig, axes = plt.subplots(1, len(representative_calls), figsize=(2 * len(representative_calls), 2))
    fig.suptitle(f'{cluster_label} Audio Segments')
    if len(representative_calls) == 1:
        axes = [axes]

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

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename = f'{cluster_label}.png'
    plt.savefig(os.path.join(save_path, plot_filename))
    plt.close(fig)





def silhouette_visualizer(data, n_clusters, title, method='gmm', **kwargs):
    """
    Visualizza il coefficiente silhouette per vari metodi di clustering.
    
    Args:
        data (array-like): Dati da clusterizzare.
        n_clusters (int): Numero di cluster desiderati.
        title (str): Titolo del grafico.
        method (str): Metodo di clustering da usare ('fcm', 'gmm', 'dbscan', 'agglomerative').
        **kwargs: Parametri aggiuntivi per il metodo di clustering scelto.
    """
    # Esegui il clustering in base al metodo specificato
    if method == 'fcm':
        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            data.T, n_clusters, 2, error=0.005, maxiter=1000, init=None
        )
        cluster_labels = np.argmax(u, axis=0)
    elif method == 'gmm':
        model = GaussianMixture(n_components=n_clusters, **kwargs)
        cluster_labels = model.fit_predict(data)
    elif method == 'dbscan':
        model = DBSCAN(**kwargs)
        cluster_labels = model.fit_predict(data)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    elif method == 'agglomerative':
        model = AgglomerativeClustering(n_clusters=n_clusters, **kwargs)
        cluster_labels = model.fit_predict(data)
    else:
        raise ValueError(f"Unsupported method: {method}")

    # Calcola il coefficiente silhouette
    silhouette_avg = silhouette_score(data, cluster_labels)
    sample_silhouette_values = silhouette_samples(data, cluster_labels)
    
    print(f"Average silhouette score: {silhouette_avg}")

    # Crea il grafico del coefficiente silhouette
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(9, 7)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(data) + (n_clusters + 1) * 10])

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        # Usa colori più chiari
        color = sns.color_palette("pastel", n_colors=n_clusters)[i]
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )
        
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax1.set_title(title)
    ax1.set_xlabel("Silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])
    ax1.set_xticks(np.arange(-0.1, 1.1, 0.2))

    plt.tight_layout()
    plt.show()

# Esempio d'uso:
# silhouette_visualizer(data_scaled, n_clusters=3, title='Silhouette Plot', method='gmm', covariance_type='full', random_state=42)



def statistical_report(all_data, cluster_membership, n_clusters, metadata, output_folder):
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Add cluster membership to all_data
    all_data['cluster_membership'] = cluster_membership

    # Drop specified columns
    all_data = all_data.drop(['recording', 'Call Number', 'onsets_sec', 'offsets_sec', 'call_id'], axis=1)
    
    # Group by cluster membership and calculate the mean of each feature
    statistical_report_df = all_data.groupby('cluster_membership').mean().reset_index()
    
    # Add the number of samples in each cluster
    n_samples = all_data['cluster_membership'].value_counts().sort_index()
    statistical_report_df['n_samples'] = n_samples

    # Save the statistical report to a CSV file
    csv_file_path = os.path.join(output_folder, 'statistical_report.csv')
    statistical_report_df.to_csv(csv_file_path, index=False)

    # Convert and export the statistical report to LaTeX
    latex_file_path = os.path.join(output_folder, 'statistical_report.tex')
    statistical_report_df.to_latex(latex_file_path, index=False)

    # Colors for clusters
    colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
    color_map = {i: color for i, color in enumerate(colors[:n_clusters])}
    color_map[-1] = 'slategray'  # Color for the noise cluster

    # Separate plots for groups of features
    features = list(all_data.columns[:-1])  # Exclude 'cluster_membership'
    num_features = len(features)
    features_per_plot = 4  # Number of features per plot
    num_plots = (num_features + features_per_plot - 1) // features_per_plot  # Calculate the number of plots needed

    for i in range(num_plots):
        start_idx = i * features_per_plot
        end_idx = min(start_idx + features_per_plot, num_features)
        plot_features = features[start_idx:end_idx]

        fig, axs = plt.subplots(1, len(plot_features), figsize=(20, 5))

        # Ensure axs is a list even if it contains a single subplot
        if len(plot_features) == 1:
            axs = [axs]

        for j, feature in enumerate(plot_features):
            data = [all_data[all_data['cluster_membership'] == k][feature] for k in range(-1, n_clusters)]

            bplot = axs[j].boxplot(data, patch_artist=True, showfliers=False)
            for patch, k in zip(bplot['boxes'], range(-1, n_clusters)):
                patch.set_alpha(0.1)
                patch.set_facecolor(color_map.get(k, 'slategray'))

            # Overlay scatterplot
            for cluster in range(-1, n_clusters):
                cluster_data = all_data[all_data['cluster_membership'] == cluster]
                axs[j].scatter([cluster + 2] * len(cluster_data), cluster_data[feature], 
                               alpha=0.3, c=color_map.get(cluster, 'slategray'), edgecolor=color_map.get(cluster, 'slategray'), s=13, label=f'Cluster {cluster}')
                
                axs[j].set_title(f'{feature} per cluster', size=7)
                axs[j].set_xlabel('Cluster', size=7)
                axs[j].set_ylabel('Value', size=7)

            # Set x-tick labels to show cluster numbers including -1
            axs[j].set_xticks(range(1, n_clusters + 2))
            axs[j].set_xticklabels(['-1'] + [str(i) for i in range(n_clusters)])

        plt.tight_layout()

        # Save the plot to a file
        plot_file_path = os.path.join(output_folder, f'statistical_report_part_{i+1}.png')
        plt.savefig(plot_file_path)
        # plt.show()

    return statistical_report_df

    # # To plot all features in a single plot
    # fig, axs = plt.subplots(nrows=6, ncols=5, figsize=(40, 25))  # Creiamo una griglia di subplot 6x5 (per un totale di 30 spazi)

    # # Appiattiamo axs per un facile accesso
    # axs = axs.flatten()

    # for j, feature in enumerate(features):
    #     data = [all_data[all_data['cluster_membership'] == k][feature] for k in range(n_clusters)]
        
    #     bplot = axs[j].boxplot(data, patch_artist=True, showfliers= True)
    #     for patch, k in zip(bplot['boxes'], range(n_clusters)):
    #         patch.set_facecolor(color_map[k])
    #         patch.set_alpha(0.1)

    #         # Scatterplot sovrapposto
    #     for cluster in range(n_clusters):
    #         cluster_data = all_data[all_data['cluster_membership'] == cluster]
    #         axs[j].scatter([cluster + 1] * len(cluster_data), cluster_data[feature], 
    #                         alpha=0.3, c=color_map[cluster], edgecolor='k', s=13, label=f'Cluster {cluster}')
            
    #         axs[j].set_title(f'{feature}', size=7)
    #         axs[j].set_xlabel('Clusters', size=5)
    #         axs[j].set_ylabel('Value', size=5)

    
    # # Rimuove assi vuoti se ce ne sono meno di 30
    # for j in range(len(features), len(axs)):
    #     fig.delaxes(axs[j])

    # plt.tight_layout()

    # # Salva il plot su file
    # plot_file_path = os.path.join(output_folder, 'statistical_report_all_features.png')
    # plt.savefig(plot_file_path)
    # plt.show()

    # return statistical_report_df





def create_statistical_report_with_radar_plots(all_data, cluster_membership, n_clusters, metadata, output_folder):
  
    all_data = all_data.drop(['recording', 'Call Number', 'onsets_sec', 'offsets_sec', 'call_id'], axis=1)

    # Separare la colonna 'cluster_membership' dalle altre feature
    cluster_membership_col = all_data['cluster_membership']
    features = all_data.drop(['cluster_membership'], axis=1)

    # Applicare lo scaling alle feature
    scaled_features = StandardScaler().fit_transform(features)

    # Convertire l'array scalato in DataFrame
    scaled_features_df = pd.DataFrame(scaled_features, columns=features.columns)

    # Riaggiungere la colonna 'cluster_membership'
    scaled_features_df['cluster_membership'] = cluster_membership_col.values

    # Raggruppa per appartenenza al cluster e calcola la media di ogni caratteristica
    statistical_report_df = scaled_features_df.groupby('cluster_membership').mean().reset_index()

    # Aggiungi il numero di campioni in ogni cluster
    n_samples = scaled_features_df['cluster_membership'].value_counts().sort_index()
    statistical_report_df['n_samples'] = n_samples.values


    # Salva il report statistico su file CSV
    csv_file_path = os.path.join(output_folder, 'statistical_report.csv')
    statistical_report_df.to_csv(csv_file_path, index=False)

    # Converti ed esporta il report statistico in LaTeX
    latex_file_path = os.path.join(output_folder, 'statistical_report.tex')
    statistical_report_df.to_latex(latex_file_path, index=False)

    # Creare radar plot per visualizzare le variazioni delle feature per cluster
    features = list(all_data.columns[:-3])  # Escludi 'cluster_membership'
    num_features = len(features)
    num_clusters = len(statistical_report_df)

    # Definire colori distinti per ogni cluster
    colors = plt.cm.get_cmap('tab10', n_clusters)

    # Funzione per creare un singolo radar plot
    def create_radar_plot(data, title, color, ax):
        categories = list(data.keys())
        values = list(data.values())
        values += values[:1]  # Chiudi il cerchio

        angles = [n / float(num_features) * 2 * pi for n in range(num_features)]
        angles += angles[:1]

        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, color='dimgrey', size=4.5, fontweight='bold')
        
        ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
        ax.fill(angles, values, color= color, alpha=0.3)

        ax.set_title(title, size=10, color=color, y=1.1)

    # Creare una griglia di radar plot
    num_cols = 3
    num_rows = int(np.ceil(num_clusters / num_cols))
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4), subplot_kw=dict(polar=True))

    axs = axs.flatten()

    for i, row in statistical_report_df.iterrows():
        data = row[features].to_dict()
        color = colors(i)
        create_radar_plot(data, f'Cluster {int(row["cluster_membership"])}', color, axs[i])

    for j in range(num_clusters, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plot_file_path = os.path.join(output_folder, 'radar_plots_clusters.png')
    plt.savefig(plot_file_path)
    # plt.show()

    return statistical_report_df


def radarplot_individual(all_data, output_folder):
    # Rimuovere colonne non necessarie
    all_data = all_data.drop(['Call Number', 'onsets_sec', 'offsets_sec', 'call_id'], axis=1)

    # Applicare lo scaling alle feature
    scaled_features = StandardScaler().fit_transform(all_data.drop(['recording'], axis=1))

    # Convertire l'array scalato in DataFrame
    scaled_features_df = pd.DataFrame(scaled_features, columns=all_data.columns[:-1])

    # Riaggiungere la colonna 'recording'
    scaled_features_df['recording'] = all_data['recording'].values

    # Raggruppa per recording e calcola la media di ogni caratteristica
    statistical_report_df = scaled_features_df.groupby('recording').mean().reset_index()

    # Aggiungi il numero di campioni in ogni recording
    n_samples = scaled_features_df['recording'].value_counts().sort_index()
    statistical_report_df['n_samples'] = n_samples.values

    # Salva il report statistico su file CSV
    csv_file_path = os.path.join(output_folder, 'statistical_report.csv')
    statistical_report_df.to_csv(csv_file_path, index=False)

    # Converti ed esporta il report statistico in LaTeX
    latex_file_path = os.path.join(output_folder, 'statistical_report.tex')
    statistical_report_df.to_latex(latex_file_path, index=False)

    # Creare radar plot per visualizzare le variazioni delle feature per recording
    features = list(all_data.columns[:-1])  # Escludi 'recording'
    num_features = len(features)
    num_files = len(statistical_report_df)

    custom_colors = [
        "steelblue", "darkcyan", "mediumseagreen", 
        "indianred", "goldenrod", "orchid", 
        "lightskyblue", "limegreen", "tomato", 
        "mediumslateblue", "darkolivegreen", "cornflowerblue"
    ]

    # Funzione per creare un singolo radar plot
    def create_radar_plot(data, title, color, ax):
        categories = list(data.keys())
        values = list(data.values())
        values += values[:1]  # Chiudi il cerchio

        angles = [n / float(num_features) * 2 * pi for n in range(num_features)]
        angles += angles[:1]

        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, color='dimgrey', size=5, fontweight='bold')
        
        ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
        ax.fill(angles, values, color=color, alpha=0.3)

        ax.set_title(title, size=12, color=color, y=1.1)

    # Creare una griglia di radar plot
    num_cols = 3
    num_rows = int(np.ceil(num_files / num_cols))
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 9, num_rows * 9), subplot_kw=dict(polar=True))

    axs = axs.flatten()

    for i, row in statistical_report_df.iterrows():
        data = row[features].to_dict()
        color = custom_colors[i % len(custom_colors)]
        
        # Controllo per gestire il numero variabile di plot
        if i < len(axs) and axs[i] is not None:
            create_radar_plot(data, f'Recording of {row["recording"]}', color, axs[i])

    # Rimuovere gli assi extra se non ci sono dati sufficienti per riempire la griglia
    for j in range(len(statistical_report_df), len(axs)):
        if axs[j] is not None:
            fig.delaxes(axs[j])

    plt.tight_layout()
    plot_file_path = os.path.join(output_folder, 'radar_plots_recordings.png')
    plt.savefig(plot_file_path)
    # plt.show()

    # Aggiungi la gestione per plottare 6 radar plot per griglia
    if num_files > 6:
        for k in range(2):
            fig, axs = plt.subplots(2, 3, figsize=(18, 12), subplot_kw=dict(polar=True))
            axs = axs.flatten()
            
            start_idx = k * 6
            end_idx = min(start_idx + 6, num_files)
            
            for idx in range(start_idx, end_idx):
                data = statistical_report_df.iloc[idx][features].to_dict()
                color = custom_colors[idx % len(custom_colors)]
                
                create_radar_plot(data, f'Recording of {statistical_report_df.iloc[idx]["recording"]}', color, axs[idx - start_idx])
            
            plt.tight_layout()
            plot_file_path = os.path.join(output_folder, f'radar_plots_recordings_{k + 1}.png')
            plt.savefig(plot_file_path)

    return statistical_report_df



def plot_and_save_extreme_calls(audio_data, audio_path, clusterings_results_path):
    """
    Extract, plot, and save spectrograms and audio of selected calls.

    Args:
    audio_data (pd.DataFrame): DataFrame containing audio metadata and cluster probabilities
    audio_path (str): Path to the directory containing audio files
    clusterings_results_path (str): Path to save the results
    """
    for idx, sample in audio_data.iterrows():
        try:
            audio_file = os.path.join(audio_path, sample['call_id'].split('_')[0] + '_d0.wav')

            if os.path.exists(audio_file):
                data, sr = lb.load(audio_file, sr=44100)

                # Extract the specific call
                onset_samples = int(sample['onsets_sec'] * sr)
                offset_samples = int(sample['offsets_sec'] * sr)
                call_audio = data[onset_samples:offset_samples]

                # Compute mel spectrogram
                S = lb.feature.melspectrogram(y=call_audio, sr=sr, n_mels=128, fmin=2000, fmax=10000)
                log_S = lb.power_to_db(S, ref=np.max)

                # Plot the spectrogram
                fig, ax = plt.subplots(figsize=(10, 5))
                img = ax.imshow(log_S, aspect='auto', origin='lower', cmap='magma')
                ax.set_title(f'{sample["call_id"]}_clustered in:_{sample["cluster_membership"]}', fontsize=14)
                ax.set_xlabel('Time (s)', fontsize=12)
                ax.set_ylabel('Frequency (Hz)', fontsize=12)
                fig.colorbar(img, ax=ax, format='%+2.0f dB')
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])

                # Save the individual spectrogram
                spectrogram_filename = os.path.join(clusterings_results_path, f'{sample["call_id"]}_clustered_in_{sample["cluster_membership"]}_as_{sample["point_type"]}.png')
                plt.savefig(spectrogram_filename)
                plt.close(fig)  # Close the figure to free up memory

                # Save the audio
                audio_filename = os.path.join(clusterings_results_path, f'top_call_{sample["call_id"]}_clustered_in_{sample["cluster_membership"]}_as_{sample["point_type"]}.wav')
                sf.write(audio_filename, call_audio, sr)

                print(f"Processed call {sample['call_id']}")
            else:
                print(f'Audio file {audio_file} not found')

        except Exception as e:
            print(f"Error processing call {sample['call_id']}: {str(e)}")

    print(f"Processed all {len(audio_data)} calls.")


    # Function to get representative calls by percentile
def get_representative_calls_by_percentile(cluster_data, percentiles, n_calls=10):
    total_calls = len(cluster_data)
    percentile_indices = [int(np.percentile(range(total_calls), p)) for p in percentiles]

    representative_calls = []
    for idx in percentile_indices:
        start_idx = max(0, idx - n_calls//2)
        end_idx = min(total_calls, idx + n_calls//2)
        representative_calls.append(cluster_data.iloc[start_idx:end_idx])
    
    return representative_calls



def get_representative_calls_by_threshold(cluster_data, cluster_num, audio_path, results_path, n_calls=25):
    """
    Select representative calls for a given cluster, and determine a threshold based on the distance to the cluster center.

    Parameters:
    - cluster_data: DataFrame with all data related to the current cluster.
    - cluster_num: The cluster number being analyzed.
    - audio_path: Path to the directory containing the audio files.
    - results_path: Path to the directory where results will be saved.
    - n_calls: Number of calls to select per batch (default is 25).

    Returns:
    - A threshold rank and distance for the cluster, if found.
    """
    # Ensure results path exists
    cluster_results_path = os.path.join(results_path, f'cluster_{cluster_num}')
    os.makedirs(cluster_results_path, exist_ok=True)
    
    # Sort data by distance to center (from closest to farthest)
    cluster_data = cluster_data.sort_values('distance_to_center', ascending=False)
    start_rank = 0
    threshold_found = False
    
    while not threshold_found:
        # Select the next batch of calls
        representative_calls = cluster_data.iloc[start_rank:start_rank + n_calls]
        
        if representative_calls.empty:
            print(f"Reached the end of calls in cluster {cluster_num} without finding a threshold.")
            break
        
        # Save path for the current batch
        save_path = os.path.join(cluster_results_path, f'rank_{start_rank + 1}_{start_rank + len(representative_calls)}')
        os.makedirs(save_path, exist_ok=True)
        
        # Save representative calls
        representative_calls.to_csv(os.path.join(save_path, f'representative_calls_rank_{start_rank + 1}_{start_rank + len(representative_calls)}.csv'), index=False)

        # Plot and save audio segments
        plot_and_save_audio_segments(representative_calls, audio_path, save_path, f'cluster_{cluster_num}')
        
        print(f"\nAnalysing Cluster {cluster_num}, Calls {start_rank + 1} to {start_rank + len(representative_calls)}:")
        print(representative_calls[['recording', 'call_id', 'distance_to_center']])
        print("\nPlease analyse these calls.")
        response = input("Have you found the threshold in this batch? (yes/no): ")
    
        if response.lower() == 'yes':
            threshold_rank = int(input("Enter the rank number where you found the threshold: "))
            threshold_distance = cluster_data.iloc[threshold_rank - 1]['distance_to_center']
            print(f"Threshold found for cluster {cluster_num} at rank {threshold_rank}, distance {threshold_distance:.4f}")
            threshold_found = True
        else:
            start_rank += len(representative_calls)
    
    # After finding threshold, save it
    if threshold_found:
        threshold_path = os.path.join(cluster_results_path, 'threshold.txt')
        with open(threshold_path, 'w') as f:
            f.write(f"Threshold for cluster {cluster_num}: rank {threshold_rank}\n")
            f.write(f"This corresponds to a distance_to_center of {threshold_distance:.4f}\n")
        return threshold_rank, threshold_distance
    else:
        return None, None




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



