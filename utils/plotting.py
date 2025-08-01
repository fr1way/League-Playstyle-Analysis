import matplotlib.pyplot as plt

def plot_clusters(reduced_data, labels):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='tab10')
    plt.title("Cluster Visualization")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.colorbar(scatter)
    return plt.gcf()
