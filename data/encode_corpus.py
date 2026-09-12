import os
import json
import numpy as np
from rich.console import Console
from sklearn.datasets import fetch_20newsgroups
import sys

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    pass

console = Console()

def main():
    console.print("[bold green]Encoding Corpus with Sentence-Transformers[/bold green]")
    
    categories = ['sci.space', 'comp.graphics', 'rec.sport.baseball', 
                  'sci.med', 'talk.politics.misc', 'rec.autos', 'misc.forsale']
    
    console.print("Downloading/Loading 20 Newsgroups dataset...")
    dataset = fetch_20newsgroups(subset='all', categories=categories,
                                 remove=('headers', 'footers', 'quotes'))
    
    console.print("Cleaning texts...")
    cleaned_texts = []
    labels = []
    
    for text, target in zip(dataset.data, dataset.target):
        text = " ".join(text.split())
        if 80 <= len(text) <= 600:
            cleaned_texts.append(text)
            labels.append(dataset.target_names[target])
            if len(cleaned_texts) == 5000:
                break
                
    console.print(f"Retained {len(cleaned_texts)} texts.")
    
    model_dir = 'd:/itgeeks/models'
    os.makedirs(model_dir, exist_ok=True)
    
    console.print("Loading model 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=model_dir)
    
    console.print("Encoding texts...")
    embeddings = model.encode(cleaned_texts, normalize_embeddings=True, show_progress_bar=True)
    
    query_texts = []
    query_labels = []
    for text, target in zip(dataset.data, dataset.target):
        text = " ".join(text.split())
        if 80 <= len(text) <= 600:
            if text not in cleaned_texts:
                query_texts.append(text)
                query_labels.append(dataset.target_names[target])
                if len(query_texts) == 500:
                    break
    
    query_embeddings = model.encode(query_texts, normalize_embeddings=True, show_progress_bar=True)
    
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from vecforge import ExactIndex
    
    exact_index = ExactIndex(384)
    db_ids = [f"doc_{i}" for i in range(len(embeddings))]
    exact_index.insert_batch(db_ids, embeddings)
    
    ground_truth = {}
    console.print("Computing ground truth for corpus queries...")
    for i in range(len(query_embeddings)):
        results = exact_index.search(query_embeddings[i], k=10)
        ground_truth[f"query_{i}"] = [res[0] for res in results]
        
    doc_data = [{"id": db_ids[i], "text": cleaned_texts[i], "category": labels[i]} for i in range(len(cleaned_texts))]
    
    os.makedirs('d:/itgeeks/data', exist_ok=True)
    np.save('d:/itgeeks/data/embeddings.npy', embeddings)
    np.save('d:/itgeeks/data/query_embeddings.npy', query_embeddings)
    with open('d:/itgeeks/data/texts.json', 'w') as f:
        json.dump(doc_data, f)
    with open('d:/itgeeks/data/corpus_ground_truth.json', 'w') as f:
        json.dump(ground_truth, f)
        
    console.print("[bold green]Corpus encoding complete![/bold green]")

if __name__ == "__main__":
    main()
