import numpy as np

def normalize_vectors(X: np.ndarray) -> np.ndarray:
    """L2-normalize vectors. X shape (N, dim) or (dim,). Returns same shape, float32."""
    X = np.asarray(X, dtype=np.float32)
    is_1d = X.ndim == 1
    if is_1d:
        X = X[np.newaxis, :]
    
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = X / norms
    
    if is_1d:
        return normalized[0]
    return normalized

def cosine_similarity(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between a query (dim,) and vectors (N, dim).
    Both must be pre-normalized. Returns (N,) similarities.
    Uses dot product since vectors are unit-normalized."""
    return np.dot(vectors, query)

def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Return indices of top-k highest scores using argpartition for efficiency."""
    k = min(k, len(scores))
    if k == 0:
        return np.array([], dtype=int)
    # argpartition puts the k smallest items at the beginning if we use negative scores
    # or k largest at the end if we use normal scores.
    # To get top k highest scores, we partition by -k
    idx = np.argpartition(scores, -k)[-k:]
    # sort the top k indices by score descending
    top_k_idx = idx[np.argsort(-scores[idx])]
    return top_k_idx
