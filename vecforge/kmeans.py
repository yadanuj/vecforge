import numpy as np
from .utils import normalize_vectors

def kmeans(X: np.ndarray, n_clusters: int, max_iter: int = 20, seed: int = 42) -> np.ndarray:
    """Spherical K-means clustering using only numpy.
    
    Args:
        X: (N, dim) matrix of unit-normalized vectors
        n_clusters: number of clusters
        max_iter: max iterations
        seed: random seed
    
    Returns:
        centroids: (n_clusters, dim) unit-normalized centroids
    
    Algorithm:
    1. Initialize centroids by randomly selecting n_clusters vectors from X
    2. Repeat until convergence or max_iter:
       a. Assign each vector to nearest centroid (max dot product)
       b. Recompute centroids as mean of assigned vectors
       c. Re-normalize centroids to unit length
       d. If assignments didn't change, stop early
    3. Handle empty clusters by reinitializing with random vectors
    """
    if len(X) < n_clusters:
        raise ValueError("Number of vectors must be greater than or equal to n_clusters")
        
    rng = np.random.default_rng(seed)
    
    # Initialize centroids by randomly selecting from X
    indices = rng.choice(len(X), size=n_clusters, replace=False)
    centroids = X[indices].copy()
    
    prev_assignments = None
    
    for _ in range(max_iter):
        # Assign each vector to nearest centroid
        similarities = np.dot(X, centroids.T)
        assignments = np.argmax(similarities, axis=1)
        
        # Check convergence
        if prev_assignments is not None and np.array_equal(assignments, prev_assignments):
            break
        prev_assignments = assignments
        
        # Recompute centroids
        for k in range(n_clusters):
            cluster_points = X[assignments == k]
            if len(cluster_points) > 0:
                centroids[k] = np.mean(cluster_points, axis=0)
            else:
                # Handle empty cluster by selecting a random point
                centroids[k] = X[rng.choice(len(X))]
                
        # Re-normalize centroids
        centroids = normalize_vectors(centroids)
        
    return centroids
