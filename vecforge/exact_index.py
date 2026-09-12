import numpy as np
from typing import List, Tuple
from .utils import normalize_vectors, cosine_similarity, top_k_indices

class ExactIndex:
    """Brute-force exact nearest neighbor search. This is ground truth."""
    
    def __init__(self, dim: int):
        self.dim = dim
        self.vectors = np.empty((0, dim), dtype=np.float32)  # Matrix of all vectors
        self.ids = []              # List of string IDs
        self._id_to_idx = {}       # ID -> row index mapping
    
    def insert(self, doc_id: str, vector: np.ndarray) -> None:
        """Insert a single vector. Normalizes it."""
        if doc_id in self._id_to_idx:
            raise ValueError(f"Document ID {doc_id} already exists.")
        
        vec_norm = normalize_vectors(vector)
        self._id_to_idx[doc_id] = len(self.ids)
        self.ids.append(doc_id)
        self.vectors = np.vstack([self.vectors, vec_norm])
    
    def insert_batch(self, doc_ids: list, vectors: np.ndarray) -> None:
        """Insert multiple vectors at once. Much faster than repeated insert."""
        if len(doc_ids) != len(vectors):
            raise ValueError("Number of IDs must match number of vectors.")
        
        for doc_id in doc_ids:
            if doc_id in self._id_to_idx:
                raise ValueError(f"Document ID {doc_id} already exists.")
        
        vecs_norm = normalize_vectors(vectors)
        start_idx = len(self.ids)
        for i, doc_id in enumerate(doc_ids):
            self._id_to_idx[doc_id] = start_idx + i
            self.ids.append(doc_id)
            
        self.vectors = np.vstack([self.vectors, vecs_norm])
    
    def search(self, query: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Return list of (doc_id, similarity_score) for top-k nearest.
        Query is normalized internally."""
        if len(self.ids) == 0:
            return []
            
        q_norm = normalize_vectors(query)
        scores = cosine_similarity(q_norm, self.vectors)
        top_indices = top_k_indices(scores, k)
        
        return [(self.ids[idx], float(scores[idx])) for idx in top_indices]
    
    def delete(self, doc_id: str) -> bool:
        """Delete a vector by ID. Returns True if found and deleted.
        Must update the matrix and all index mappings."""
        if doc_id not in self._id_to_idx:
            return False
            
        idx_to_delete = self._id_to_idx[doc_id]
        last_idx = len(self.ids) - 1
        
        if idx_to_delete != last_idx:
            # Swap with last element
            last_id = self.ids[last_idx]
            self.vectors[idx_to_delete] = self.vectors[last_idx]
            self.ids[idx_to_delete] = last_id
            self._id_to_idx[last_id] = idx_to_delete
            
        # Truncate
        self.vectors = self.vectors[:-1]
        self.ids.pop()
        del self._id_to_idx[doc_id]
        
        return True
    
    def __len__(self) -> int:
        return len(self.ids)
    
    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._id_to_idx
