import pytest
import numpy as np
from vecforge import LSHIndex
from vecforge.utils import normalize_vectors

def test_insert_and_search():
    rng = np.random.default_rng(42)
    dim = 64
    index = LSHIndex(dim=dim, num_tables=5, num_bits=8, seed=42)
    
    vectors = rng.standard_normal((1000, dim)).astype(np.float32)
    vectors = normalize_vectors(vectors)
    ids = [f"doc_{i}" for i in range(1000)]
    
    index.insert_batch(ids, vectors)
    assert len(index) == 1000
    
    query = vectors[42]
    results = index.search(query, k=10)
    
    # LSH is approximate, but the exact vector should almost certainly be found
    found_ids = [r[0] for r in results]
    assert "doc_42" in found_ids

def test_recall_improves_with_tables():
    rng = np.random.default_rng(42)
    dim = 64
    vectors = normalize_vectors(rng.standard_normal((500, dim)).astype(np.float32))
    ids = [f"id_{i}" for i in range(500)]
    
    query = vectors[0]
    
    # Low tables
    index_low = LSHIndex(dim=dim, num_tables=1, num_bits=16, seed=42)
    index_low.insert_batch(ids, vectors)
    res_low = index_low.search(query, k=50)
    
    # High tables
    index_high = LSHIndex(dim=dim, num_tables=20, num_bits=16, seed=42)
    index_high.insert_batch(ids, vectors)
    res_high = index_high.search(query, k=50)
    
    assert len(res_high) > len(res_low) or (len(res_high) == len(res_low) and len(res_high) > 0)
    
def test_delete():
    dim = 16
    index = LSHIndex(dim=dim, num_tables=2, num_bits=4, seed=42)
    rng = np.random.default_rng(42)
    
    vec1 = normalize_vectors(rng.standard_normal((1, dim)).astype(np.float32))[0]
    
    index.insert("1", vec1)
    assert len(index) == 1
    
    res = index.search(vec1, k=1)
    assert len(res) == 1
    
    deleted = index.delete("1")
    assert deleted is True
    assert len(index) == 0
    
    res2 = index.search(vec1, k=1)
    assert len(res2) == 0

def test_search_empty():
    index = LSHIndex(dim=32, num_tables=1, num_bits=1)
    vec = np.zeros(32, dtype=np.float32)
    vec[0] = 1.0
    results = index.search(vec, k=5)
    assert results == []

def test_batch_insert():
    dim = 16
    index1 = LSHIndex(dim=dim, num_tables=2, num_bits=4, seed=42)
    index2 = LSHIndex(dim=dim, num_tables=2, num_bits=4, seed=42)
    
    rng = np.random.default_rng(42)
    vectors = normalize_vectors(rng.standard_normal((10, dim)).astype(np.float32))
    ids = [str(i) for i in range(10)]
    
    for i in range(10):
        index1.insert(ids[i], vectors[i])
        
    index2.insert_batch(ids, vectors)
    
    assert len(index1) == len(index2) == 10
    
    q = vectors[0]
    res1 = index1.search(q, k=3)
    res2 = index2.search(q, k=3)
    
    assert res1 == res2
