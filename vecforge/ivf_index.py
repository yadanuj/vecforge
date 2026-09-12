import numpy as np
from typing import List, Tuple
from .utils import normalize_vectors, cosine_similarity
from .kmeans import kmeans

class IVFIndex:
    """Approximate nearest neighbor search using Inverted File Index.
    
    Uses k-means to partition space into Voronoi cells.
    At query time, searches only the nprobe nearest cells.
    
    Tunable knob: nprobe (passed at query time)
    - nprobe=1: fastest, lowest recall
    - nprobe=n_clusters: exact search (checks everything)
    """
    
    def __init__(self, dim: int, n_clusters: int = 64, seed: int = 42):
        self.dim = dim
        self.n_clusters = n_clusters
        self.seed = seed
        self.centroids = None          # (n_clusters, dim) after training
        self.trained = False
        
        # Inverted lists: cluster_id -> list of doc_ids
        self.inverted_lists = [[] for _ in range(n_clusters)]
        
        # Vector storage
        self.vectors = {}              # doc_id -> normalized vector
        
        # Reverse index for deletion
        self._doc_to_cluster = {}      # doc_id -> cluster_id
    
    def train(self, vectors: np.ndarray) -> None:
        """Train centroids using spherical k-means from vecforge.kmeans.
        Must be called before insert."""
        vecs_norm = normalize_vectors(vectors)
        self.centroids = kmeans(vecs_norm, self.n_clusters, seed=self.seed)
        self.trained = True
    
    def insert(self, doc_id: str, vector: np.ndarray) -> None:
        """Assign vector to nearest centroid and add to its inverted list.
        Must be trained first."""
        if not self.trained:
            raise RuntimeError("Index must be trained before insertion.")
        if doc_id in self.vectors:
            raise ValueError(f"Document ID {doc_id} already exists.")
            
        vec_norm = normalize_vectors(vector)
        self.vectors[doc_id] = vec_norm
        
        # Find nearest centroid
        similarities = np.dot(self.centroids, vec_norm)
        cluster_id = int(np.argmax(similarities))
        
        self.inverted_lists[cluster_id].append(doc_id)
        self._doc_to_cluster[doc_id] = cluster_id
    
    def insert_batch(self, doc_ids: list, vectors: np.ndarray) -> None:
        """Batch insert — vectorized centroid assignment."""
        if not self.trained:
            raise RuntimeError("Index must be trained before insertion.")
        if len(doc_ids) != len(vectors):
            raise ValueError("Number of IDs must match number of vectors.")
            
        vecs_norm = normalize_vectors(vectors)
        
        for doc_id in doc_ids:
            if doc_id in self.vectors:
                raise ValueError(f"Document ID {doc_id} already exists.")
                
        # Vectorized centroid assignment
        similarities = np.dot(vecs_norm, self.centroids.T)
        cluster_ids = np.argmax(similarities, axis=1)
        
        for i, doc_id in enumerate(doc_ids):
            self.vectors[doc_id] = vecs_norm[i]
            cluster_id = int(cluster_ids[i])
            self.inverted_lists[cluster_id].append(doc_id)
            self._doc_to_cluster[doc_id] = cluster_id
    
    def search(self, query: np.ndarray, k: int = 10, nprobe: int = 1) -> List[Tuple[str, float]]:
        """Search nprobe nearest clusters.
        1. Find nprobe closest centroids to query
        2. Collect all vectors in those clusters
        3. Re-rank by exact cosine
        4. Return top-k as list of (doc_id, similarity_score)"""
        if not self.trained or not self.vectors:
            return []
            
        q_norm = normalize_vectors(query)
        
        # Find nprobe closest centroids
        centroid_sims = np.dot(self.centroids, q_norm)
        nprobe = min(nprobe, self.n_clusters)
        
        top_cluster_ids = np.argpartition(centroid_sims, -nprobe)[-nprobe:]
        
        candidate_ids = []
        for cluster_id in top_cluster_ids:
            candidate_ids.extend(self.inverted_lists[cluster_id])
            
        if not candidate_ids:
            return []
            
        # Unique candidates (in case of overlap, though theoretically impossible here)
        candidate_ids = list(set(candidate_ids))
        
        # Re-rank
        cand_vectors = np.array([self.vectors[doc_id] for doc_id in candidate_ids])
        scores = cosine_similarity(q_norm, cand_vectors)
        
        k = min(k, len(candidate_ids))
        idx = np.argpartition(scores, -k)[-k:]
        top_k_idx = idx[np.argsort(-scores[idx])]
        
        return [(candidate_ids[i], float(scores[i])) for i in top_k_idx]
    
    def delete(self, doc_id: str) -> bool:
        """Remove from inverted list using reverse index."""
        if doc_id not in self._doc_to_cluster:
            return False
            
        cluster_id = self._doc_to_cluster.pop(doc_id)
        self.inverted_lists[cluster_id].remove(doc_id)
        del self.vectors[doc_id]
        
        return True
    
    def __len__(self) -> int:
        return len(self.vectors)
