import numpy as np
from collections import defaultdict
from typing import List, Tuple
from .utils import normalize_vectors, cosine_similarity

class LSHIndex:
    """Approximate nearest neighbor search using Random Hyperplane LSH.
    
    Uses multi-table (AND-OR) construction:
    - L independent hash tables, each with b random hyperplanes
    - A vector hashes to a binary string per table: sign(hyperplanes @ vector)
    - Query: collect candidates from all L tables, re-rank by exact cosine
    
    Tunable knob: num_tables (L)
    - More tables = higher recall, slower search
    - Fewer tables = lower recall, faster search
    """
    
    def __init__(self, dim: int, num_tables: int = 10, num_bits: int = 12, seed: int = 42):
        self.dim = dim
        self.num_tables = num_tables  # L: THE tunable knob
        self.num_bits = num_bits      # b: bits per hash
        
        # Generate random hyperplanes: L sets of b hyperplanes each
        rng = np.random.default_rng(seed)
        self.hyperplanes = [rng.standard_normal((num_bits, dim)).astype(np.float32) for _ in range(num_tables)]
        
        # Hash tables: table_idx -> {hash_key: set of doc_ids}
        self.tables = [defaultdict(set) for _ in range(num_tables)]
        
        # Reverse index for deletion: doc_id -> list of (table_idx, hash_key)
        self._doc_to_hashes = {}
        
        # Vector storage for re-ranking
        self.vectors = {}  # doc_id -> normalized vector
    
    def _hash(self, vector: np.ndarray, table_idx: int) -> int:
        """Compute integer hash key for a vector in a specific table.
        Project onto hyperplanes, take sign, convert binary to integer."""
        projections = np.dot(self.hyperplanes[table_idx], vector)
        bits = (projections >= 0).astype(int)
        
        # Convert binary array to integer
        powers_of_two = 1 << np.arange(self.num_bits - 1, -1, -1)
        return int(np.dot(bits, powers_of_two))
    
    def insert(self, doc_id: str, vector: np.ndarray) -> None:
        """Insert vector into all L hash tables."""
        if doc_id in self.vectors:
            raise ValueError(f"Document ID {doc_id} already exists.")
            
        vec_norm = normalize_vectors(vector)
        self.vectors[doc_id] = vec_norm
        
        hashes = []
        for i in range(self.num_tables):
            h = self._hash(vec_norm, i)
            self.tables[i][h].add(doc_id)
            hashes.append((i, h))
            
        self._doc_to_hashes[doc_id] = hashes
    
    def insert_batch(self, doc_ids: list, vectors: np.ndarray) -> None:
        """Batch insert."""
        if len(doc_ids) != len(vectors):
            raise ValueError("Number of IDs must match number of vectors.")
            
        vecs_norm = normalize_vectors(vectors)
        
        for i, doc_id in enumerate(doc_ids):
            if doc_id in self.vectors:
                raise ValueError(f"Document ID {doc_id} already exists.")
                
            vec_norm = vecs_norm[i]
            self.vectors[doc_id] = vec_norm
            
            hashes = []
            for j in range(self.num_tables):
                h = self._hash(vec_norm, j)
                self.tables[j][h].add(doc_id)
                hashes.append((j, h))
                
            self._doc_to_hashes[doc_id] = hashes
    
    def search(self, query: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Approximate search:
        1. Hash query in all L tables
        2. Collect candidate IDs from matching buckets
        3. Re-rank candidates by exact cosine similarity
        4. Return top-k
        Returns list of (doc_id, similarity_score)"""
        if not self.vectors:
            return []
            
        q_norm = normalize_vectors(query)
        candidates = set()
        
        for i in range(self.num_tables):
            h = self._hash(q_norm, i)
            candidates.update(self.tables[i][h])
            
        if not candidates:
            return []
            
        # Re-rank candidates
        candidate_list = list(candidates)
        cand_vectors = np.array([self.vectors[doc_id] for doc_id in candidate_list])
        
        scores = cosine_similarity(q_norm, cand_vectors)
        
        k = min(k, len(candidate_list))
        idx = np.argpartition(scores, -k)[-k:]
        top_k_idx = idx[np.argsort(-scores[idx])]
        
        return [(candidate_list[i], float(scores[i])) for i in top_k_idx]
    
    def delete(self, doc_id: str) -> bool:
        """Remove from all hash tables. Uses reverse index for O(L) deletion."""
        if doc_id not in self._doc_to_hashes:
            return False
            
        hashes = self._doc_to_hashes.pop(doc_id)
        for table_idx, h in hashes:
            self.tables[table_idx][h].discard(doc_id)
            
        del self.vectors[doc_id]
        return True
    
    def __len__(self) -> int:
        return len(self.vectors)
