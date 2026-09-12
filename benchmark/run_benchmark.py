import os
import json
import time
import numpy as np
from rich.console import Console
from rich.table import Table
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vecforge import ExactIndex, LSHIndex, IVFIndex

console = Console()

def measure_qps_and_recall(index, queries, ground_truth, nprobe=None):
    # Warm-up
    for i in range(10):
        if nprobe is not None:
            index.search(queries[i], k=10, nprobe=nprobe)
        else:
            index.search(queries[i], k=10)
            
    start_time = time.perf_counter()
    recalls = []
    
    for i, q in enumerate(queries):
        if nprobe is not None:
            results = index.search(q, k=10, nprobe=nprobe)
        else:
            results = index.search(q, k=10)
            
        retrieved = set([res[0] for res in results])
        true_neighbors = set(ground_truth[str(i)])
        
        recall = len(retrieved.intersection(true_neighbors)) / 10.0
        recalls.append(recall)
        
    end_time = time.perf_counter()
    qps = len(queries) / (end_time - start_time)
    mean_recall = np.mean(recalls)
    return qps, mean_recall

def main():
    console.print("[bold cyan]Loading Synthetic Data for Benchmarking...[/bold cyan]")
    
    db_vectors = np.load('d:/itgeeks/data/synthetic_vectors.npy')
    query_vectors = np.load('d:/itgeeks/data/query_vectors.npy')
    with open('d:/itgeeks/data/ground_truth.json', 'r') as f:
        ground_truth = json.load(f)
        
    N_VECTORS, DIM = db_vectors.shape
    db_ids = list(range(N_VECTORS))
    
    benchmark_results = {
        'exact': {},
        'lsh': [],
        'ivf': []
    }
    
    console.print("[bold green]Benchmarking ExactIndex...[/bold green]")
    exact_index = ExactIndex(DIM)
    exact_index.insert_batch(db_ids, db_vectors)
    
    qps, recall = measure_qps_and_recall(exact_index, query_vectors, ground_truth)
    benchmark_results['exact'] = {'qps': qps, 'recall': recall}
    console.print(f"ExactIndex - QPS: {qps:.2f}, Recall@10: {recall:.4f}")
    
    console.print("[bold green]Benchmarking LSHIndex...[/bold green]")
    lsh_tables = [1, 2, 4, 6, 8, 10, 14, 18, 22, 26, 30]
    
    for t in lsh_tables:
        lsh_index = LSHIndex(DIM, num_tables=t, num_bits=16, seed=42)
        lsh_index.insert_batch(db_ids, db_vectors)
        qps, recall = measure_qps_and_recall(lsh_index, query_vectors, ground_truth)
        benchmark_results['lsh'].append({'num_tables': t, 'qps': qps, 'recall': recall})
        console.print(f"LSHIndex (tables={t}) - QPS: {qps:.2f}, Recall@10: {recall:.4f}")
        
    console.print("[bold green]Benchmarking IVFIndex...[/bold green]")
    ivf_index = IVFIndex(DIM, n_clusters=256, seed=42)
    console.print("Training IVFIndex...")
    ivf_index.train(db_vectors)
    ivf_index.insert_batch(db_ids, db_vectors)
    
    nprobes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    for np_val in nprobes:
        qps, recall = measure_qps_and_recall(ivf_index, query_vectors, ground_truth, nprobe=np_val)
        benchmark_results['ivf'].append({'nprobe': np_val, 'qps': qps, 'recall': recall})
        console.print(f"IVFIndex (nprobe={np_val}) - QPS: {qps:.2f}, Recall@10: {recall:.4f}")
        
    console.print("[bold magenta]Testing Deletion...[/bold magenta]")
    delete_ids = list(np.random.choice(N_VECTORS, 100, replace=False))
    
    # We will test deletion on exact index as asked, and also IVF and LSH
    # Rebuild LSH with a single setting for test
    lsh_index_del = LSHIndex(DIM, num_tables=4, num_bits=16, seed=42)
    lsh_index_del.insert_batch(db_ids, db_vectors)
    
    for d_id in delete_ids:
        exact_index.delete(int(d_id))
        ivf_index.delete(int(d_id))
        lsh_index_del.delete(int(d_id))
    
    console.print("Verifying deletions...")
    del_success = True
    for d_id in delete_ids:
        vec = db_vectors[d_id]
        res = exact_index.search(vec, k=10)
        if d_id in [r[0] for r in res]:
            del_success = False
            
    if del_success:
        console.print("[bold green]Deletion successful![/bold green]")
    else:
        console.print("[bold red]Deletion failed![/bold red]")
        
    os.makedirs('d:/itgeeks/benchmark/results', exist_ok=True)
    with open('d:/itgeeks/benchmark/results/benchmark_results.json', 'w') as f:
        json.dump(benchmark_results, f, indent=4)
        
    table = Table(title="VecForge Benchmark Results")
    table.add_column("Index Type", style="cyan")
    table.add_column("Parameters", style="magenta")
    table.add_column("Recall@10", style="green")
    table.add_column("QPS", style="yellow")
    
    table.add_row("Exact", "-", f"{benchmark_results['exact']['recall']:.4f}", f"{benchmark_results['exact']['qps']:.2f}")
    
    for res in benchmark_results['lsh']:
        table.add_row("LSH", f"tables={res['num_tables']}", f"{res['recall']:.4f}", f"{res['qps']:.2f}")
        
    for res in benchmark_results['ivf']:
        table.add_row("IVF", f"nprobe={res['nprobe']}", f"{res['recall']:.4f}", f"{res['qps']:.2f}")
        
    console.print(table)

if __name__ == "__main__":
    main()
