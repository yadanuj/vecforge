import os
import sys
import json
import time
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from sentence_transformers import SentenceTransformer

try:
    from vecforge import ExactIndex, LSHIndex, IVFIndex
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from vecforge import ExactIndex, LSHIndex, IVFIndex

app = Flask(__name__)

# Globals for data and models
model = None
exact_index = None
lsh_index = None
ivf_index = None
doc_map = {}

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VecForge</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-start: #0f0f1a;
            --bg-end: #1a1a2e;
            --glass: rgba(255, 255, 255, 0.05);
            --glass-border: rgba(255, 255, 255, 0.1);
            --text: #e2e8f0;
            --cyan: #00d2ff;
            --purple: #7b2ff7;
            --orange: #ff6b35;
        }

        body {
            margin: 0;
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, var(--bg-start), var(--bg-end));
            color: var(--text);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            overflow-x: hidden;
        }

        header {
            margin-top: 3rem;
            text-align: center;
        }

        h1 {
            font-size: 4rem;
            margin: 0;
            background: linear-gradient(90deg, var(--cyan), var(--purple), var(--orange));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradient-shift 5s infinite linear;
        }

        @keyframes gradient-shift {
            0% { filter: hue-rotate(0deg); }
            100% { filter: hue-rotate(360deg); }
        }

        .subtitle {
            font-size: 1.2rem;
            opacity: 0.8;
            margin-top: 0.5rem;
        }

        .search-container {
            width: 100%;
            max-width: 800px;
            margin: 2rem 0;
            position: relative;
        }

        .search-bar {
            width: 100%;
            padding: 1.5rem 2rem;
            font-size: 1.2rem;
            border-radius: 50px;
            border: 1px solid var(--glass-border);
            background: var(--glass);
            color: #fff;
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
            box-sizing: border-box;
        }

        .search-bar:focus {
            outline: none;
            border-color: var(--cyan);
            box-shadow: 0 0 20px rgba(0, 210, 255, 0.3);
            background: rgba(255, 255, 255, 0.1);
        }

        .controls {
            display: flex;
            gap: 1rem;
            justify-content: center;
            margin-bottom: 2rem;
        }

        button {
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            border: 1px solid var(--glass-border);
            background: var(--glass);
            color: #fff;
            cursor: pointer;
            backdrop-filter: blur(5px);
            transition: all 0.2s ease;
            font-weight: 600;
        }

        button:hover {
            background: rgba(255, 255, 255, 0.15);
            transform: translateY(-2px);
        }

        button.active {
            border-color: var(--cyan);
            color: var(--cyan);
        }

        .results-container {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 2rem;
            width: 90%;
            max-width: 1400px;
            margin-bottom: 3rem;
        }

        .column {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .column h2 {
            text-align: center;
            font-size: 1.2rem;
            margin-bottom: 0.5rem;
            display: flex;
            flex-direction: column;
        }

        .column h2 span.time {
            font-size: 0.9rem;
            font-weight: 400;
            color: var(--cyan);
            margin-top: 0.3rem;
        }

        .card {
            background: var(--glass);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            padding: 1.5rem;
            backdrop-filter: blur(10px);
            transition: transform 0.2s ease;
            position: relative;
        }

        .card:hover {
            transform: translateY(-5px);
            border-color: rgba(255, 255, 255, 0.3);
        }

        .score-badge {
            position: absolute;
            top: -10px;
            right: -10px;
            background: #222;
            color: #fff;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            border: 1px solid var(--glass-border);
        }

        .score-high { border-color: #00e676; color: #00e676; }
        .score-med { border-color: #ffca28; color: #ffca28; }
        .score-low { border-color: #ff5252; color: #ff5252; }

        .category {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--purple);
            margin-bottom: 0.5rem;
            font-weight: 800;
        }

        .text-snippet {
            font-size: 0.95rem;
            line-height: 1.5;
            color: #ccc;
        }

        .spinner {
            display: none;
            width: 50px;
            height: 50px;
            border: 3px solid rgba(255,255,255,0.1);
            border-radius: 50%;
            border-top-color: var(--cyan);
            animation: spin 1s ease-in-out infinite;
            margin: 2rem auto;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        @media (max-width: 900px) {
            .results-container {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <header>
        <h1>VecForge</h1>
        <div class="subtitle">Vector Database Built from Scratch</div>
    </header>

    <div class="search-container">
        <input type="text" id="searchInput" class="search-bar" placeholder="Enter search query..." autocomplete="off">
    </div>

    <div class="controls">
        <button id="modeSingle" class="active" onclick="setMode('single')">LSH Search</button>
        <button id="modeCompare" onclick="setMode('compare')">Compare Indexes</button>
    </div>

    <div id="spinner" class="spinner"></div>

    <div class="results-container" id="resultsContainer">
        <!-- Results injected here -->
    </div>

    <script>
        let mode = 'single';

        function setMode(newMode) {
            mode = newMode;
            document.getElementById('modeSingle').classList.toggle('active', mode === 'single');
            document.getElementById('modeCompare').classList.toggle('active', mode === 'compare');
            if (document.getElementById('searchInput').value.trim() !== '') {
                performSearch();
            }
        }

        function getScoreClass(score) {
            if (score >= 0.6) return 'score-high';
            if (score >= 0.4) return 'score-med';
            return 'score-low';
        }

        function buildCard(result) {
            const cls = getScoreClass(result.score);
            return `
                <div class="card">
                    <div class="score-badge ${cls}">${result.score.toFixed(4)}</div>
                    <div class="category">${result.category || 'N/A'}</div>
                    <div class="text-snippet">${result.text}</div>
                </div>
            `;
        }

        function renderColumn(title, timeMs, results, recall = null) {
            let html = `<div class="column">
                <h2>${title} <span class="time">${timeMs.toFixed(2)} ms${recall !== null ? ` | Recall: ${(recall*100).toFixed(0)}%` : ''}</span></h2>`;
            results.forEach(r => {
                html += buildCard(r);
            });
            html += `</div>`;
            return html;
        }

        async function performSearch() {
            const query = document.getElementById('searchInput').value.trim();
            if (!query) return;

            document.getElementById('spinner').style.display = 'block';
            document.getElementById('resultsContainer').innerHTML = '';

            try {
                if (mode === 'single') {
                    const res = await fetch('/api/search', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({query: query, index: 'lsh', k: 5})
                    });
                    const data = await res.json();
                    document.getElementById('resultsContainer').innerHTML = renderColumn('LSH Index', data.time_ms, data.results);
                    document.getElementById('resultsContainer').style.gridTemplateColumns = '1fr';
                    document.getElementById('resultsContainer').style.maxWidth = '600px';
                } else {
                    const res = await fetch('/api/compare', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({query: query})
                    });
                    const data = await res.json();
                    let html = '';
                    html += renderColumn('Exact Index', data.exact.time_ms, data.exact.results);
                    html += renderColumn('LSH Index', data.lsh.time_ms, data.lsh.results, data.lsh.recall);
                    html += renderColumn('IVF Index', data.ivf.time_ms, data.ivf.results, data.ivf.recall);
                    document.getElementById('resultsContainer').innerHTML = html;
                    document.getElementById('resultsContainer').style.gridTemplateColumns = 'repeat(3, 1fr)';
                    document.getElementById('resultsContainer').style.maxWidth = '1400px';
                }
            } catch (err) {
                console.error(err);
                document.getElementById('resultsContainer').innerHTML = '<div style="text-align:center;width:100%;color:#ff5252;">Error performing search.</div>';
            } finally {
                document.getElementById('spinner').style.display = 'none';
            }
        }

        document.getElementById('searchInput').addEventListener('keypress', function (e) {
            if (e.key === 'Enter') {
                performSearch();
            }
        });
    </script>
</body>
</html>
"""

def init_app():
    global model, exact_index, lsh_index, ivf_index, doc_map
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    embeddings_path = os.path.join(data_dir, "embeddings.npy")
    texts_path = os.path.join(data_dir, "texts.json")
    
    if os.path.exists(embeddings_path) and os.path.exists(texts_path):
        print("Loading data...")
        embeddings = np.load(embeddings_path)
        with open(texts_path, "r", encoding="utf-8") as f:
            texts_data = json.load(f)
            
        doc_ids = [item.get("id", str(i)) for i, item in enumerate(texts_data)]
        dim = embeddings.shape[1]
        
        for i, item in enumerate(texts_data):
            doc_map[doc_ids[i]] = item
            
        print("Loading Model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("Building Indexes...")
        exact_index = ExactIndex(dim=dim)
        exact_index.insert_batch(doc_ids, embeddings)
        
        lsh_index = LSHIndex(dim=dim, num_tables=12, num_bits=16)
        lsh_index.insert_batch(doc_ids, embeddings)
        
        ivf_index = IVFIndex(dim=dim, n_clusters=64)
        ivf_index.train(embeddings)
        ivf_index.insert_batch(doc_ids, embeddings)
        
        print("App initialized successfully.")
    else:
        print("Data files not found. Generate data before running the app.")

@app.route("/")
def index():
    return render_template_string(HTML_CONTENT)

@app.route("/api/search", methods=["POST"])
def search():
    data = request.json
    query = data.get("query", "")
    idx_type = data.get("index", "lsh")
    k = data.get("k", 5)
    
    query_vec = model.encode([query])[0]
    
    t0 = time.time()
    if idx_type == "exact":
        res = exact_index.search(query_vec, k=k)
    elif idx_type == "ivf":
        res = ivf_index.search(query_vec, k=k, nprobe=4)
    else:
        res = lsh_index.search(query_vec, k=k)
    t1 = time.time()
    
    results = []
    for doc_id, score in res:
        meta = doc_map.get(doc_id, {})
        results.append({
            "id": doc_id,
            "score": float(score),
            "text": meta.get("text", "")[:200] + "..." if len(meta.get("text", "")) > 200 else meta.get("text", ""),
            "category": meta.get("category", "N/A")
        })
        
    return jsonify({
        "time_ms": (t1-t0)*1000,
        "results": results
    })

@app.route("/api/compare", methods=["POST"])
def compare():
    data = request.json
    query = data.get("query", "")
    k = 5
    
    query_vec = model.encode([query])[0]
    
    # Exact
    t0 = time.time()
    exact_res = exact_index.search(query_vec, k=k)
    exact_time = (time.time() - t0) * 1000
    
    # LSH
    t0 = time.time()
    lsh_res = lsh_index.search(query_vec, k=k)
    lsh_time = (time.time() - t0) * 1000
    
    # IVF
    t0 = time.time()
    ivf_res = ivf_index.search(query_vec, k=k, nprobe=4)
    ivf_time = (time.time() - t0) * 1000
    
    exact_ids = set(r[0] for r in exact_res)
    lsh_ids = set(r[0] for r in lsh_res)
    ivf_ids = set(r[0] for r in ivf_res)
    
    lsh_recall = len(exact_ids.intersection(lsh_ids)) / len(exact_ids) if exact_ids else 0
    ivf_recall = len(exact_ids.intersection(ivf_ids)) / len(exact_ids) if exact_ids else 0
    
    def format_res(res):
        out = []
        for doc_id, score in res:
            meta = doc_map.get(doc_id, {})
            out.append({
                "id": doc_id,
                "score": float(score),
                "text": meta.get("text", "")[:150] + "..." if len(meta.get("text", "")) > 150 else meta.get("text", ""),
                "category": meta.get("category", "N/A")
            })
        return out
        
    return jsonify({
        "exact": {"time_ms": exact_time, "results": format_res(exact_res)},
        "lsh": {"time_ms": lsh_time, "results": format_res(lsh_res), "recall": lsh_recall},
        "ivf": {"time_ms": ivf_time, "results": format_res(ivf_res), "recall": ivf_recall},
    })

if __name__ == "__main__":
    init_app()
    app.run(host="0.0.0.0", port=5000, debug=False)
