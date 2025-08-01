import matplotlib.pyplot as plt
import numpy as np

def plot_clusters(reduced_data, labels):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='tab10')
    plt.title("Cluster Visualization")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.colorbar(scatter)
    return plt.gcf()

def plot_radar(df_centroids, features, titles=None):
    N = len(features)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, len(df_centroids), figsize=(4*len(df_centroids), 4), subplot_kw=dict(polar=True))
    if len(df_centroids) == 1:
        axes = [axes]

    for ax, (idx, row) in zip(axes, df_centroids.iterrows()):
        values = row[features].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(features, fontsize=9)
        ax.set_yticklabels([])
        title = titles[0] if titles else f"Cluster {idx}"
        ax.set_title(title, pad=15)

    plt.tight_layout()
    return plt.gcf()
