import pytest
import numpy as np
from vecforge.kmeans import kmeans
from vecforge.utils import normalize_vectors

def test_basic_clustering():
    rng = np.random.default_rng(42)
    
    # Create two well-separated groups
    group1 = rng.normal(loc=[10.0, 10.0], scale=1.0, size=(50, 2)).astype(np.float32)
    group2 = rng.normal(loc=[-10.0, -10.0], scale=1.0, size=(50, 2)).astype(np.float32)
    
    X = np.vstack([group1, group2])
    X = normalize_vectors(X)
    
    centroids = kmeans(X, n_clusters=2, max_iter=20, seed=42)
    
    assert centroids.shape == (2, 2)
    
    # Centroids should be somewhat close to the normalized centers
    dot_prod = np.dot(centroids[0], centroids[1])
    assert dot_prod < 0  # They should be pointing in opposite directions

def test_correct_shape():
    rng = np.random.default_rng(42)
    dim = 16
    X = rng.standard_normal((100, dim)).astype(np.float32)
    X = normalize_vectors(X)
    
    n_clusters = 5
    centroids = kmeans(X, n_clusters=n_clusters, seed=42)
    
    assert centroids.shape == (n_clusters, dim)

def test_normalized_centroids():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 8)).astype(np.float32)
    X = normalize_vectors(X)
    
    centroids = kmeans(X, n_clusters=4, seed=42)
    
    norms = np.linalg.norm(centroids, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5)

def test_reproducibility():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 8)).astype(np.float32)
    X = normalize_vectors(X)
    
    centroids1 = kmeans(X, n_clusters=3, seed=123)
    centroids2 = kmeans(X, n_clusters=3, seed=123)
    
    np.testing.assert_array_equal(centroids1, centroids2)
