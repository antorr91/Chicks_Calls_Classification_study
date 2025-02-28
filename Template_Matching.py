import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from skimage.feature import match_template
from tqdm import tqdm
from classification_utils import load_audio_data

def main():
    # Setup dei percorsi
    paths = {
        'audio': 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Data\\High_quality_dataset',
        'classification_results': 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Classification\\Template_matching\\trial_no_cropping',
        'clusterings_results': 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Classification\\MLP_SVM',
        'dictionary': 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_Classification\\Template_Matching\\Dictionary_Selection\\closer_centroid'
    }
    
    # Crea la directory dei risultati se non esiste
    os.makedirs(paths['classification_results'], exist_ok=True)
    
    # Carica i dati
    data = load_data(paths)
    
    # Calcola la similarità intra-cluster tra chiamate rappresentative
    intracluster_df = calculate_intracluster_similarity(paths, data)
    
    # Calcola l'auto-similarità per le chiamate rappresentative
    auto_similarity_df = calculate_auto_similarity(paths, data)
    
    # Calcola la similarità inter-cluster tra chiamate di test e chiamate rappresentative
    inter_similarity_df, intra_similarity_df = calculate_intercluster_similarity(paths, data)
    
    # Calcola e stampa le metriche di valutazione (senza generare grafici)
    calculate_metrics(inter_similarity_df, intra_similarity_df)
    
    print("Template matching e calcolo dei risultati completati.")

def calculate_intracluster_similarity(paths, data):
    """Confronta ogni chiamata rappresentativa con le altre dello stesso cluster"""
    print("Calcolo della similarità intra-cluster tra chiamate rappresentative...")
    
    intracluster_results = []
    
    # Itera su ogni cluster
    for cluster in range(data['n_clusters']):
        rep_calls = data['representative_calls'][cluster]
        cluster_correlations = []
        
        # Esegui la cross-correlazione tra tutte le combinazioni di rep-calls nel cluster
        for i, rep_call_1 in rep_calls.iterrows():
            rep_call_id_1 = rep_call_1['call_id']
            rep_onset_1 = rep_call_1['onsets_sec']
            rep_offset_1 = rep_call_1['offsets_sec']
            rep_call_membership_1 = rep_call_1['cluster_membership']
            
            # Carica lo spettrogramma della prima rep-call
            rep_result_1 = load_audio_data(paths['audio'], data['cluster_results'], rep_call_id_1, rep_onset_1, rep_offset_1)
            rep_spectrogram_1 = rep_result_1['spectrogram']
            
            for j, rep_call_2 in rep_calls.iterrows():
                # Evita di calcolare la correlazione tra la stessa chiamata (i == j)
                if i >= j:
                    continue
                
                rep_call_id_2 = rep_call_2['call_id']
                rep_onset_2 = rep_call_2['onsets_sec']
                rep_offset_2 = rep_call_2['offsets_sec']
                rep_call_membership_2 = rep_call_2['cluster_membership']
                
                # Carica lo spettrogramma della seconda rep-call
                rep_result_2 = load_audio_data(paths['audio'], data['cluster_results'], rep_call_id_2, rep_onset_2, rep_offset_2)
                rep_spectrogram_2 = rep_result_2['spectrogram']
                
                # Assicura che gli spettrogrammi abbiano la stessa lunghezza
                if rep_spectrogram_1.shape[1] < rep_spectrogram_2.shape[1]:
                    pad_width = rep_spectrogram_2.shape[1] - rep_spectrogram_1.shape[1]
                    rep_spectrogram_1 = np.pad(rep_spectrogram_1, ((0, 0), (0, pad_width)), mode='constant')
                elif rep_spectrogram_1.shape[1] > rep_spectrogram_2.shape[1]:
                    # rep_spectrogram_2 = rep_spectrogram_2[:, :rep_spectrogram_1.shape[1]]
                    pass
                
                # Calcola la similarità usando la cross-correlazione
                similarity = match_template(rep_spectrogram_1, rep_spectrogram_2)
                max_similarity = np.max(similarity)
                
                # Salva il risultato
                cluster_correlations.append({
                    'rep_call_id_1': rep_call_id_1,
                    'rep_call_id_2': rep_call_id_2,
                    'rep_call_membership_1': rep_call_membership_1,
                    'rep_call_membership_2': rep_call_membership_2,
                    'similarity': max_similarity
                })
        
        # Aggiungi i risultati del cluster alla lista complessiva
        intracluster_results.extend(cluster_correlations)
        
        # Salva i risultati di cross-correlazione per questo cluster
        cluster_corr_df = pd.DataFrame(cluster_correlations)
        cluster_corr_df.to_csv(os.path.join(paths['classification_results'], f'rep_calls_cross_correlation_cluster_{cluster}.csv'), index=False)
    
    # Salva tutti i risultati
    intracluster_df = pd.DataFrame(intracluster_results)
    intracluster_df.to_csv(os.path.join(paths['classification_results'], 'rep_calls_cross_correlation_all_clusters.csv'), index=False)
    
    print("Cross-correlazione delle chiamate rappresentative intra-cluster completata e salvata.")
    return intracluster_df

def calculate_auto_similarity(paths, data):
    """Verifica che ogni chiamata rappresentativa sia coerente con sé stessa (controllo interno)"""
    print("Calcolo dell'auto-similarità per le chiamate rappresentative...")
    
    auto_similarity_results = []
    
    # Itera su ogni cluster
    for cluster in range(data['n_clusters']):
        rep_calls = data['representative_calls'][cluster]
        
        # Calcola l'auto-similarità per ogni chiamata rappresentativa
        for _, rep_call in rep_calls.iterrows():
            rep_call_id = rep_call['call_id']
            rep_onset = rep_call['onsets_sec']
            rep_offset = rep_call['offsets_sec']
            cluster_membership = rep_call['cluster_membership']
            
            # Carica lo spettrogramma della chiamata
            rep_result = load_audio_data(paths['audio'], data['cluster_results'], rep_call_id, rep_onset, rep_offset)
            rep_spectrogram = rep_result['spectrogram']
            
            # Calcola l'auto-similarità
            similarity = match_template(rep_spectrogram, rep_spectrogram)
            max_similarity = np.max(similarity)
            
            # Salva il risultato
            auto_similarity_results.append({
                'rep_call_id': rep_call_id,
                'cluster_membership': cluster_membership,
                'auto_similarity': max_similarity
            })
    
    # Salva i risultati
    auto_similarity_df = pd.DataFrame(auto_similarity_results)
    auto_similarity_df.to_csv(os.path.join(paths['classification_results'], 'rep_calls_auto_similarity.csv'), index=False)
    
    print("Calcolo dell'auto-similarità delle chiamate rappresentative completato e salvato.")
    return auto_similarity_df

def calculate_intercluster_similarity(paths, data):
    """Confronta ogni chiamata di test con tutte le chiamate rappresentative di tutti i cluster"""
    print("Calcolo della similarità inter-cluster tra chiamate di test e chiamate rappresentative...")
    
    # Inizializza liste per i risultati
    inter_similarity_results = []
    intra_similarity_results = []
    cross_correlation_outputs = []
    
    # Itera sulle chiamate di test
    for _, call in tqdm(data['test_calls'].iterrows(), total=len(data['test_calls'])):
        test_call_id = call['call_id']
        test_onset = call['onsets_sec']
        test_offset = call['offsets_sec']
        test_cluster_membership = call['cluster_membership']
        
        # Carica lo spettrogramma della chiamata di test
        test_result = load_audio_data(paths['audio'], data['cluster_results'], test_call_id, test_onset, test_offset)
        test_spectrogram = test_result['spectrogram']
        
        best_inter_match = {'similarity': -1, 'rep_call_id': None, 'cluster_membership_rep': None}
        best_intra_match = {'similarity': -1, 'rep_call_id': None, 'cluster_membership_rep': None}
        
        # Itera sui cluster per il confronto inter-cluster
        for cluster in range(data['n_clusters']):
            representative_calls = data['representative_calls'][cluster]
            
            for _, rep_call in representative_calls.iterrows():
                rep_call_id = rep_call['call_id']
                rep_onset = rep_call['onsets_sec']
                rep_offset = rep_call['offsets_sec']
                cluster_membership_rep = rep_call['cluster_membership']
                
                # Carica lo spettrogramma della chiamata rappresentativa
                rep_result = load_audio_data(paths['audio'], data['cluster_results'], rep_call_id, rep_onset, rep_offset)
                rep_spectrogram = rep_result['spectrogram']
                
                # Assicura che gli spettrogrammi abbiano le stesse dimensioni
                if test_spectrogram.shape[1] < rep_spectrogram.shape[1]:
                    pad_width = rep_spectrogram.shape[1] - test_spectrogram.shape[1]
                    test_spectrogram = np.pad(test_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
                elif test_spectrogram.shape[1] > rep_spectrogram.shape[1]:
                    # test_spectrogram = test_spectrogram[:, :rep_spectrogram.shape[1]]
                    pass
                # Calcola la similarità
                similarity = match_template(test_spectrogram, rep_spectrogram)
                max_similarity = np.max(similarity)
                
                # Memorizza il risultato della cross-correlazione
                cross_correlation_outputs.append({
                    'rep_call_id': rep_call_id,
                    'test_call_id': test_call_id,
                    'max_similarity': max_similarity,
                    'cluster_membership_rep': cluster_membership_rep,
                    'cluster_membership_test': test_cluster_membership
                })
                
                # Confronto inter-cluster (chiamata di test vs chiamata rappresentativa di un cluster diverso)
                if test_cluster_membership != cluster_membership_rep:
                    inter_similarity_results.append({
                        'rep_call_id': rep_call_id,
                        'test_call_id': test_call_id,
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
                
                # Confronto intra-cluster (chiamata di test vs chiamata rappresentativa dello stesso cluster)
                if test_cluster_membership == cluster_membership_rep:
                    intra_similarity_results.append({
                        'rep_call_id': rep_call_id,
                        'test_call_id': test_call_id,
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
    
    # Salva i risultati della cross-correlazione
    cross_correlation_df = pd.DataFrame(cross_correlation_outputs)
    cross_correlation_df.to_csv(os.path.join(paths['classification_results'], 'cross_correlation_results.csv'), index=False)
    
    # Salva i risultati di similarità inter-cluster e intra-cluster
    inter_similarity_df = pd.DataFrame(inter_similarity_results)
    intra_similarity_df = pd.DataFrame(intra_similarity_results)
    
    inter_similarity_df.to_csv(os.path.join(paths['classification_results'], 'inter_similarity_results.csv'), index=False)
    intra_similarity_df.to_csv(os.path.join(paths['classification_results'], 'intra_similarity_results.csv'), index=False)
    
    print("Calcolo della similarità inter-cluster completato e salvato.")
    return inter_similarity_df, intra_similarity_df

def calculate_metrics(inter_similarity_df, intra_similarity_df):
    """Calcola le metriche di valutazione (senza generare grafici)"""
    print("Calcolo delle metriche di valutazione...")
    
    # Valutazione per Inter-cluster
    y_true_inter = inter_similarity_df.groupby('test_call_id').first()['cluster_membership_test'].values
    y_pred_inter = inter_similarity_df.groupby('test_call_id').apply(lambda x: x.loc[x['similarity'].idxmax(), 'cluster_membership_rep']).values
    
    accuracy_inter = accuracy_score(y_true_inter, y_pred_inter)
    precision_inter = precision_score(y_true_inter, y_pred_inter, average='weighted')
    recall_inter = recall_score(y_true_inter, y_pred_inter, average='weighted')
    f1_inter = f1_score(y_true_inter, y_pred_inter, average='weighted')
    
    print(f'Inter-cluster Accuracy: {accuracy_inter:.4f}')
    print(f'Inter-cluster Precision: {precision_inter:.4f}')
    print(f'Inter-cluster Recall: {recall_inter:.4f}')
    print(f'Inter-cluster F1 Score: {f1_inter:.4f}')
    
    # Valutazione per Intra-cluster
    try:
        y_true_intra = intra_similarity_df.groupby('test_call_id').first()['cluster_membership_test'].values
        y_pred_intra = intra_similarity_df.groupby('test_call_id').apply(lambda x: x.loc[x['similarity'].idxmax(), 'cluster_membership_rep']).values
        
        accuracy_intra = accuracy_score(y_true_intra, y_pred_intra)
        precision_intra = precision_score(y_true_intra, y_pred_intra, average='weighted')
        recall_intra = recall_score(y_true_intra, y_pred_intra, average='weighted')
        f1_intra = f1_score(y_true_intra, y_pred_intra, average='weighted')
        
        print(f'Intra-cluster Accuracy: {accuracy_intra:.4f}')
        print(f'Intra-cluster Precision: {precision_intra:.4f}')
        print(f'Intra-cluster Recall: {recall_intra:.4f}')
        print(f'Intra-cluster F1 Score: {f1_intra:.4f}')
        
    except ValueError as e:
        print(f"Error in intra-cluster evaluation: {e}")
        print(f"y_true_intra length: {len(y_true_intra)}")

def load_data(paths):
    """Carica i dati necessari per l'analisi"""
    # Carica i risultati del clustering
    cluster_results = pd.read_csv(os.path.join(paths['clusterings_results'], 'labelled_dataset.csv'))
    cluster_results['call_id'] = cluster_results['call_id'].str.replace(r'_call_', '_', regex=True)
    
    # Carica le chiamate rappresentative ed escludi le loro registrazioni dalla selezione delle chiamate di test
    excluded_recordings = set()
    representative_calls = {}
    
    n_clusters = 3  # Numero di cluster
    
    for cluster in range(n_clusters):
        rep_calls = pd.read_csv(os.path.join(paths['dictionary'], f'dictionary_cluster_{cluster}.csv'))
        rep_calls['call_id'] = rep_calls['call_id'].str.replace(r'_call_', '_', regex=True)
        
        # Aggiorna l'insieme delle registrazioni escluse
        excluded_recordings.update(rep_calls['recording'].tolist())
        representative_calls[cluster] = rep_calls
        
    # Filtra le chiamate di test, escludendo quelle che fanno parte delle registrazioni escluse
    test_calls = cluster_results[~cluster_results['recording'].isin(excluded_recordings)]
    
    return {
        'cluster_results': cluster_results,
        'representative_calls': representative_calls,
        'test_calls': test_calls,
        'n_clusters': n_clusters
    }

if __name__ == "__main__":
    main()