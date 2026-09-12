import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    results_path = 'd:/itgeeks/benchmark/results/benchmark_results.json'
    if not os.path.exists(results_path):
        print(f"Results file not found at {results_path}")
        return
        
    with open(results_path, 'r') as f:
        results = json.load(f)
        
    exact_qps = results['exact']['qps']
    lsh_data = results['lsh']
    ivf_data = results['ivf']
    
    plt.style.use('default')
    
    # 1. LSH Tradeoff
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')
    
    lsh_recalls = [r['recall'] for r in lsh_data]
    lsh_qps = [r['qps'] for r in lsh_data]
    lsh_labels = [r['num_tables'] for r in lsh_data]
    
    ax.plot(lsh_recalls, lsh_qps, marker='o', color='#00d2ff', label='LSH')
    for i, txt in enumerate(lsh_labels):
        ax.annotate(str(txt), (lsh_recalls[i], lsh_qps[i]), textcoords="offset points", xytext=(0,5), ha='center', color='white')
        
    ax.axhline(exact_qps, color='white', linestyle='--', label='Exact Baseline')
    
    ax.set_yscale('log')
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('Recall@10', color='white')
    ax.set_ylabel('Queries Per Second (QPS)', color='white')
    ax.set_title('LSH Speed-Accuracy Tradeoff', color='white')
    ax.tick_params(colors='white')
    ax.grid(alpha=0.2)
    legend = ax.legend()
    for text in legend.get_texts():
        text.set_color('black')
    
    plt.savefig('d:/itgeeks/benchmark/results/lsh_tradeoff.png', dpi=300, facecolor='#1a1a2e')
    plt.close()
    
    # 2. IVF Tradeoff
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')
    
    ivf_recalls = [r['recall'] for r in ivf_data]
    ivf_qps = [r['qps'] for r in ivf_data]
    ivf_labels = [r['nprobe'] for r in ivf_data]
    
    ax.plot(ivf_recalls, ivf_qps, marker='s', color='#ff6b35', label='IVF')
    for i, txt in enumerate(ivf_labels):
        ax.annotate(str(txt), (ivf_recalls[i], ivf_qps[i]), textcoords="offset points", xytext=(0,5), ha='center', color='white')
        
    ax.axhline(exact_qps, color='white', linestyle='--', label='Exact Baseline')
    
    ax.set_yscale('log')
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('Recall@10', color='white')
    ax.set_ylabel('Queries Per Second (QPS)', color='white')
    ax.set_title('IVF Speed-Accuracy Tradeoff', color='white')
    ax.tick_params(colors='white')
    ax.grid(alpha=0.2)
    legend = ax.legend()
    for text in legend.get_texts():
        text.set_color('black')
    
    plt.savefig('d:/itgeeks/benchmark/results/ivf_tradeoff.png', dpi=300, facecolor='#1a1a2e')
    plt.close()
    
    # 3. Combined Tradeoff
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')
    
    ax.plot(lsh_recalls, lsh_qps, marker='o', color='#00d2ff', label='LSH (tables)')
    ax.plot(ivf_recalls, ivf_qps, marker='s', color='#ff6b35', label='IVF (nprobe)')
    ax.axhline(exact_qps, color='white', linestyle='--', label='Exact Baseline')
    
    ax.set_yscale('log')
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('Recall@10', color='white')
    ax.set_ylabel('Queries Per Second (QPS)', color='white')
    ax.set_title('VecForge: Speed vs Accuracy Tradeoff (50,000 vectors, d=384)', color='white')
    ax.tick_params(colors='white')
    ax.grid(alpha=0.2)
    legend = ax.legend()
    for text in legend.get_texts():
        text.set_color('black')
    
    plt.savefig('d:/itgeeks/benchmark/results/combined_tradeoff.png', dpi=300, facecolor='#1a1a2e')
    plt.close()

if __name__ == "__main__":
    main()
