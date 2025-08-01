from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import hdbscan
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np

def run_pca(data, n_components=0.85):
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(data)
    return reduced, pca

def run_kmeans(data, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    return labels, kmeans

def run_hdbscan(data, min_cluster_size=10):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(data)
    return labels, clusterer

def evaluate_kmeans_clusters(X, k_range=range(3, 11)):
    scores = {}
    for k in k_range:
        try:
            km = KMeans(n_clusters=k, random_state=42).fit(X)
            score = silhouette_score(X, km.labels_)
            scores[k] = score
        except:
            continue
    return scores

def get_cluster_centroids(kmeans_model, scaler, feature_names):
    return pd.DataFrame(
        scaler.inverse_transform(kmeans_model.cluster_centers_),
        columns=feature_names
    )

def classify_playstyle(row, scaler, kmeans_model, cluster_names):
    duration_mins = row['duration'] / 60
    kda = (row['kills'] + row['assists']) / (row['deaths'] + 1)
    dpm = row['damage_to_champ'] / (duration_mins + 1e-6)
    vision_pm = row['vision_score'] / (duration_mins + 1e-6)
    survivability = 1 / (1 + (row['deaths'] / (duration_mins + 1e-6)))
    team_obj = (
        row['team_towerKills'] + row['team_dragonKills'] +
        row['team_baronKills'] + row['team_riftHeraldKills'] +
        row['team_inhibitorKills']
    )
    objective_score = row['kill_participation'] * team_obj

    fighting = np.mean([kda, row['kill_participation'], dpm])
    farming = row['gold_per_min']
    feature_vec = pd.DataFrame([[fighting, farming, vision_pm, objective_score, survivability]],
                               columns=['fighting_score', 'farming_score', 'vision_score_pm', 'objective_score', 'survivability_score'])
    vec_scaled = scaler.transform(feature_vec)
    cid = int(kmeans_model.predict(vec_scaled)[0])
    return {
        "cluster": cid,
        "label": cluster_names.get(cid, f"Cluster {cid}"),
        "features": feature_vec.iloc[0].to_dict()
    }
