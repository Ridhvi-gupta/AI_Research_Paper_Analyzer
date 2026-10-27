from sklearn.cluster import KMeans

def cluster_papers(matrix, n_clusters: int = 3):
    k = max(1, min(n_clusters, matrix.shape[0]))
    model = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = model.fit_predict(matrix)
    return model, labels