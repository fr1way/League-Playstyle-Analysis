from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import hdbscan

def run_pca(data, n_components=0.85):
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(data)
    return reduced, pca

def run_kmeans(data, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    return labels, kmeans

def run_hdbscan(data):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
    labels = clusterer.fit_predict(data)
    return labels, clusterer
