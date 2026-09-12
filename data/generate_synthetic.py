import os
import json
import numpy as np
from rich.console import Console
from rich.progress import track
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vecforge import ExactIndex, normalize_vectors

console = Console()

SEED = 42
N_VECTORS = 50000
N_QUERIES = 500
DIM = 384
N_CLUSTERS = 100

def main():
    np.random.seed(SEED)
    console.print("[bold green]Generating Synthetic Data...[/bold green]")
    
    # 1. Create a global "cone axis" (anisotropy bias)
    cone_axis = np.random.randn(DIM).astype(np.float32)
    cone_axis = normalize_vectors(cone_axis.reshape(1, -1))[0]
    
    # 2. Generate 100 cluster centers on unit sphere, biased toward cone axis
    bias_strength = 0.5
    raw_centers = np.random.randn(N_CLUSTERS, DIM).astype(np.float32)
    biased_centers = raw_centers + bias_strength * cone_axis
    cluster_centers = normalize_vectors(biased_centers)
    
    # 3. Dirichlet distribution for imbalanced cluster sizes
    cluster_probs = np.random.dirichlet(np.ones(N_CLUSTERS) * 0.5)
    
    def generate_vectors(n_samples):
        # Pick clusters
        cluster_assignments = np.random.choice(N_CLUSTERS, size=n_samples, p=cluster_probs)
        base_vectors = cluster_centers[cluster_assignments]
        # Add noise
        noise = np.random.normal(scale=0.08, size=(n_samples, DIM)).astype(np.float32)
        vectors = base_vectors + noise
        vectors = normalize_vectors(vectors)
        return vectors, cluster_assignments

    console.print(f"Generating {N_VECTORS} database vectors...")
    db_vectors, db_labels = generate_vectors(N_VECTORS)
    
    console.print(f"Generating {N_QUERIES} query vectors...")
    query_vectors, _ = generate_vectors(N_QUERIES)
    
    console.print("Computing ground truth using ExactIndex...")
    exact_index = ExactIndex(DIM)
    
    db_ids = list(range(N_VECTORS))
    exact_index.insert_batch(db_ids, db_vectors)
    
    ground_truth = {}
    for i in track(range(N_QUERIES), description="Computing exact nearest neighbors"):
        results = exact_index.search(query_vectors[i], k=10)
        ground_truth[str(i)] = [res[0] for res in results]
        
    os.makedirs('d:/itgeeks/data', exist_ok=True)
    np.save('d:/itgeeks/data/synthetic_vectors.npy', db_vectors)
    np.save('d:/itgeeks/data/synthetic_labels.npy', db_labels)
    np.save('d:/itgeeks/data/query_vectors.npy', query_vectors)
    
    with open('d:/itgeeks/data/ground_truth.json', 'w') as f:
        json.dump(ground_truth, f)
        
    console.print("[bold green]Done![/bold green] Data saved to data/")

if __name__ == "__main__":
    main()
