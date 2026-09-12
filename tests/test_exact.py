import pytest
import numpy as np
from vecforge import ExactIndex
from vecforge.utils import normalize_vectors

def test_insert_and_search():
    rng = np.random.default_rng(42)
    dim = 64
    index = ExactIndex(dim=dim)
    
    vectors = rng.standard_normal((100, dim)).astype(np.float32)
    vectors = normalize_vectors(vectors)
    
    for i in range(100):
        index.insert(f"doc_{i}", vectors[i])
        
    assert len(index) == 100
    
    # Search for vector 42
    results = index.search(vectors[42], k=5)
    assert len(results) == 5
    assert results[0][0] == "doc_42"
    assert np.isclose(results[0][1], 1.0, atol=1e-5)

def test_insert_batch():
    rng = np.random.default_rng(42)
    dim = 64
    index = ExactIndex(dim=dim)
    
    vectors = rng.standard_normal((1000, dim)).astype(np.float32)
    vectors = normalize_vectors(vectors)
    ids = [f"batch_{i}" for i in range(1000)]
    
    index.insert_batch(ids, vectors)
    assert len(index) == 1000
    
    results = index.search(vectors[100], k=1)
    assert results[0][0] == "batch_100"
    
def test_delete():
    dim = 16
    index = ExactIndex(dim=dim)
    rng = np.random.default_rng(42)
    
    vec1 = normalize_vectors(rng.standard_normal((1, dim)).astype(np.float32))[0]
    vec2 = normalize_vectors(rng.standard_normal((1, dim)).astype(np.float32))[0]
    
    index.insert("1", vec1)
    index.insert("2", vec2)
    
    assert len(index) == 2
    res = index.delete("1")
    assert res is True
    assert len(index) == 1
    
    results = index.search(vec1, k=2)
    assert len(results) == 1
    assert results[0][0] == "2"
    
def test_delete_nonexistent():
    dim = 16
    index = ExactIndex(dim=dim)
    vec = normalize_vectors(np.random.default_rng(42).standard_normal((1, dim)).astype(np.float32))[0]
    index.insert("1", vec)
    
    res = index.delete("unknown")
    assert res is False
    assert len(index) == 1

def test_search_empty():
    index = ExactIndex(dim=32)
    vec = np.zeros(32, dtype=np.float32)
    vec[0] = 1.0
    
    results = index.search(vec, k=5)
    assert results == []

def test_ground_truth():
    dim = 2
    index = ExactIndex(dim=dim)
    
    # Unit vectors
    v1 = np.array([1.0, 0.0], dtype=np.float32)
    v2 = np.array([0.707, 0.707], dtype=np.float32)
    v3 = np.array([0.0, 1.0], dtype=np.float32)
    
    index.insert("v1", v1)
    index.insert("v2", v2)
    index.insert("v3", v3)
    
    # Query with v1
    results = index.search(v1, k=3)
    
    assert len(results) == 3
    assert results[0][0] == "v1"
    assert results[1][0] == "v2"
    assert results[2][0] == "v3"
