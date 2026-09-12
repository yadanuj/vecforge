import pytest
import numpy as np
from vecforge import IVFIndex
from vecforge.utils import normalize_vectors

def test_train_and_search():
    rng = np.random.default_rng(42)
    dim = 32
    index = IVFIndex(dim=dim, n_clusters=4, seed=42)
    
    train_vectors = normalize_vectors(rng.standard_normal((1000, dim)).astype(np.float32))
    index.train(train_vectors)
    
    ids = [f"doc_{i}" for i in range(1000)]
    index.insert_batch(ids, train_vectors)
    assert len(index) == 1000
    
    query = train_vectors[42]
    results = index.search(query, k=10, nprobe=1)
    
    found_ids = [r[0] for r in results]
    assert "doc_42" in found_ids

def test_nprobe_effect():
    rng = np.random.default_rng(42)
    dim = 32
    index = IVFIndex(dim=dim, n_clusters=16, seed=42)
    
    vectors = normalize_vectors(rng.standard_normal((500, dim)).astype(np.float32))
    index.train(vectors)
    ids = [f"id_{i}" for i in range(500)]
    index.insert_batch(ids, vectors)
    
    query = normalize_vectors(rng.standard_normal((1, dim)).astype(np.float32))[0]
    
    res_low = index.search(query, k=50, nprobe=1)
    res_high = index.search(query, k=50, nprobe=8)
    
    assert len(res_high) >= len(res_low)

def test_nprobe_max_is_exact():
    rng = np.random.default_rng(42)
    dim = 32
    n_clusters = 4
    index = IVFIndex(dim=dim, n_clusters=n_clusters, seed=42)
    
    vectors = normalize_vectors(rng.standard_normal((100, dim)).astype(np.float32))
    index.train(vectors)
    ids = [f"id_{i}" for i in range(100)]
    index.insert_batch(ids, vectors)
    
    query = vectors[0]
    res_all = index.search(query, k=100, nprobe=n_clusters)
    
    # It should return all vectors because we search all clusters
    assert len(res_all) == 100

def test_delete():
    dim = 16
    index = IVFIndex(dim=dim, n_clusters=2, seed=42)
    rng = np.random.default_rng(42)
    vectors = normalize_vectors(rng.standard_normal((10, dim)).astype(np.float32))
    
    index.train(vectors)
    index.insert("1", vectors[0])
    
    assert len(index) == 1
    deleted = index.delete("1")
    assert deleted is True
    assert len(index) == 0
    
    res = index.search(vectors[0], k=1)
    assert len(res) == 0

def test_untrained_raises():
    index = IVFIndex(dim=16, n_clusters=2)
    vec = np.zeros(16, dtype=np.float32)
    vec[0] = 1.0
    
    with pytest.raises(Exception):
        index.insert("1", vec)
        
    assert index.search(vec, k=1) == []
