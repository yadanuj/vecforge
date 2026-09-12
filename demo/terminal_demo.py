import os
import sys
import json
import time
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.text import Text
from rich.align import Align
from sentence_transformers import SentenceTransformer

try:
    from vecforge import ExactIndex, LSHIndex, IVFIndex
except ImportError:
    # Fallback to local import if run outside module
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from vecforge import ExactIndex, LSHIndex, IVFIndex

console = Console()

def print_banner():
    banner = """
[bold cyan]
██╗   ██╗███████╗██████╗ ███████╗██████╗ ██████╗  ██████╗ ███████╗
██║   ██║██╔════╝██╔════╝ ██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝
██║   ██║█████╗  ██║      █████╗  ██║   ██║██████╔╝██║  ███╗█████╗  
╚██╗ ██╔╝██╔══╝  ██║      ██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝  
 ╚████╔╝ ███████╗╚██████╗ ██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗
  ╚═══╝  ╚══════╝ ╚═════╝ ╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝
[/bold cyan]
    """
    console.print(Align.center(banner))
    console.print(Align.center("[bold white]A Vector Database Built from Scratch[/bold white]"))
    console.print(Align.center("[dim]Version 1.0.0 | No FAISS, No Pinecone, Just Numpy[/dim]\n"))

def format_score(score):
    if score >= 0.6:
        return f"[bold green]{score:.4f}[/bold green]"
    elif score >= 0.4:
        return f"[bold yellow]{score:.4f}[/bold yellow]"
    else:
        return f"[bold red]{score:.4f}[/bold red]"

def main():
    print_banner()

    # Data Loading
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    embeddings_path = os.path.join(data_dir, "embeddings.npy")
    texts_path = os.path.join(data_dir, "texts.json")

    if not os.path.exists(embeddings_path) or not os.path.exists(texts_path):
        console.print(Panel("[bold red]Data files not found![/bold red]\nPlease run [yellow]python -m data.encode_corpus[/yellow] first to generate embeddings.", title="Error"))
        sys.exit(1)

    with console.status("[bold cyan]Loading corpus and embeddings...[/bold cyan]") as status:
        embeddings = np.load(embeddings_path)
        with open(texts_path, "r", encoding="utf-8") as f:
            texts_data = json.load(f)
            
        texts = [item.get("text", "") for item in texts_data]
        categories = [item.get("category", "N/A") for item in texts_data]
        doc_ids = [item.get("id", str(i)) for i, item in enumerate(texts_data)]
        
        dim = embeddings.shape[1]
        console.print(f"[green]Loaded {len(embeddings)} vectors of dimension {dim}.[/green]")

    with console.status("[bold cyan]Loading SentenceTransformer model...[/bold cyan]") as status:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        console.print("[green]Model loaded.[/green]\n")

    # Build Indexes
    exact_index = ExactIndex(dim=dim)
    lsh_index = LSHIndex(dim=dim, num_tables=12, num_bits=16)
    ivf_index = IVFIndex(dim=dim, n_clusters=64)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task1 = progress.add_task("[cyan]Building ExactIndex...", total=1)
        exact_index.insert_batch(doc_ids, embeddings)
        progress.update(task1, advance=1)

        task2 = progress.add_task("[cyan]Building LSHIndex...", total=1)
        lsh_index.insert_batch(doc_ids, embeddings)
        progress.update(task2, advance=1)

        task3 = progress.add_task("[cyan]Training IVFIndex...", total=1)
        ivf_index.train(embeddings)
        progress.update(task3, advance=1)
        
        task4 = progress.add_task("[cyan]Populating IVFIndex...", total=1)
        ivf_index.insert_batch(doc_ids, embeddings)
        progress.update(task4, advance=1)
        
    console.print("\n[bold green]✓ All indexes built successfully![/bold green]\n")

    # Map ID to metadata
    doc_map = {doc_id: {"text": t, "category": c} for doc_id, t, c in zip(doc_ids, texts, categories)}

    while True:
        console.print(Panel(
            "[1] Search   [2] Compare Indexes   [3] Insert   [4] Delete   [5] Stats   [6] Quit",
            title="[bold cyan]Main Menu[/bold cyan]",
            expand=False
        ))
        choice = Prompt.ask("Select an option", choices=["1", "2", "3", "4", "5", "6"])

        if choice == "1":
            query = Prompt.ask("\n[bold cyan]Enter search query[/bold cyan]")
            query_vec = model.encode([query])[0]
            
            t0 = time.time()
            results = lsh_index.search(query_vec, k=5)
            t1 = time.time()
            
            table = Table(title=f"Search Results (LSH Index) - {((t1-t0)*1000):.2f} ms")
            table.add_column("Rank", justify="center", style="cyan")
            table.add_column("Score", justify="center")
            table.add_column("Category", style="magenta")
            table.add_column("Text Snippet")
            
            for rank, (doc_id, score) in enumerate(results, 1):
                meta = doc_map.get(doc_id, {"text": "Unknown", "category": "Unknown"})
                snippet = meta["text"][:120] + "..." if len(meta["text"]) > 120 else meta["text"]
                table.add_row(str(rank), format_score(score), meta["category"], snippet)
                
            console.print(table)
            print()

        elif choice == "2":
            query = Prompt.ask("\n[bold cyan]Enter search query[/bold cyan]")
            query_vec = model.encode([query])[0]
            
            # Exact
            t0 = time.time()
            exact_res = exact_index.search(query_vec, k=5)
            t_exact = (time.time() - t0) * 1000
            
            # LSH
            t0 = time.time()
            lsh_res = lsh_index.search(query_vec, k=5)
            t_lsh = (time.time() - t0) * 1000
            
            # IVF
            t0 = time.time()
            ivf_res = ivf_index.search(query_vec, k=5, nprobe=4)
            t_ivf = (time.time() - t0) * 1000
            
            exact_ids = set(doc_id for doc_id, _ in exact_res)
            lsh_ids = set(doc_id for doc_id, _ in lsh_res)
            ivf_ids = set(doc_id for doc_id, _ in ivf_res)
            
            lsh_recall = len(exact_ids.intersection(lsh_ids)) / len(exact_ids) if exact_ids else 0
            ivf_recall = len(exact_ids.intersection(ivf_ids)) / len(exact_ids) if exact_ids else 0
            
            console.print(f"\n[bold white]Comparison for query:[/bold white] '{query}'\n")
            
            # Print side by side? Hard to do simple tables side by side, let's print sequentially beautifully
            console.print(f"[bold cyan]Exact Index[/bold cyan] ({t_exact:.2f} ms) [dim](Ground Truth)[/dim]")
            for doc_id, score in exact_res:
                console.print(f"  {format_score(score)} - {doc_map.get(doc_id, {}).get('text', '')[:60]}...")
            
            console.print(f"\n[bold cyan]LSH Index[/bold cyan] ({t_lsh:.2f} ms) [dim]Recall: {lsh_recall*100:.0f}%[/dim]")
            for doc_id, score in lsh_res:
                console.print(f"  {format_score(score)} - {doc_map.get(doc_id, {}).get('text', '')[:60]}...")

            console.print(f"\n[bold cyan]IVF Index[/bold cyan] ({t_ivf:.2f} ms, nprobe=4) [dim]Recall: {ivf_recall*100:.0f}%[/dim]")
            for doc_id, score in ivf_res:
                console.print(f"  {format_score(score)} - {doc_map.get(doc_id, {}).get('text', '')[:60]}...")
            print()

        elif choice == "3":
            text = Prompt.ask("\n[bold cyan]Enter document text[/bold cyan]")
            doc_id = f"custom_{int(time.time())}"
            vec = model.encode([text])[0]
            
            exact_index.insert(doc_id, vec)
            lsh_index.insert(doc_id, vec)
            ivf_index.insert(doc_id, vec)
            doc_map[doc_id] = {"text": text, "category": "Custom"}
            
            console.print(f"[bold green]Document '{doc_id}' inserted into all indexes.[/bold green]\n")

        elif choice == "4":
            doc_id = Prompt.ask("\n[bold cyan]Enter document ID to delete[/bold cyan]")
            
            e_del = exact_index.delete(doc_id)
            l_del = lsh_index.delete(doc_id)
            i_del = ivf_index.delete(doc_id)
            
            if e_del or l_del or i_del:
                if doc_id in doc_map:
                    del doc_map[doc_id]
                console.print(f"[bold green]Document '{doc_id}' deleted from indexes.[/bold green]\n")
            else:
                console.print(f"[bold red]Document '{doc_id}' not found.[/bold red]\n")

        elif choice == "5":
            stats = f"""
[bold]Exact Index:[/bold] {len(exact_index)} vectors, Dim: {dim}
[bold]LSH Index:[/bold]   {len(lsh_index)} vectors, Dim: {dim}, Tables: {lsh_index.num_tables}, Bits: {lsh_index.num_bits}
[bold]IVF Index:[/bold]   {len(ivf_index)} vectors, Dim: {dim}, Clusters: {ivf_index.n_clusters}
            """
            console.print(Panel(stats, title="[bold cyan]Index Statistics[/bold cyan]"))
            print()

        elif choice == "6":
            console.print("[bold cyan]Goodbye![/bold cyan]")
            break

if __name__ == "__main__":
    main()
