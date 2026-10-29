"""
FastAPI Backend Application
Integrated Image Compression and Feature Analysis System
Deployable on Render (https://render.com)
"""

import os
import json
from typing import Optional
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Ensure proper Matplotlib non-interactive backend
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")

from src.pipeline import IntegratedSystemPipeline
from src.dataset_generator import generate_synthetic_dataset

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(CURRENT_DIR, "dataset")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "output")

app = FastAPI(
    title="Image Compression & Feature Analysis Backend",
    description="Linear Algebra & Vector Calculus API (SVD, PCA, Vector Space Isomorphism, Gradient Field)",
    version="1.0.0"
)

# Enable CORS for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure dataset and output directories exist
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
if len(os.listdir(DATASET_DIR)) == 0:
    generate_synthetic_dataset(DATASET_DIR)

# Ensure baseline pipeline has executed once
metrics_file = os.path.join(OUTPUT_DIR, "metrics_summary.json")
if not os.path.exists(metrics_file):
    init_pipeline = IntegratedSystemPipeline(
        dataset_dir=DATASET_DIR,
        output_dir=OUTPUT_DIR,
        svd_rank=12,
        pca_components=6,
        edge_threshold=0.12
    )
    init_pipeline.run_pipeline()


@app.get("/health")
@app.get("/api/health")
def health_check():
    """Render health check endpoint."""
    return {
        "status": "healthy",
        "service": "Linear Algebra & Vector Calculus Image Analysis Backend",
        "modules": ["Module 1: Matrix & Rank", "Module 2: SVD", "Module 3: Vector Space & PCA", "Module 4: Gradient"]
    }


@app.get("/api/metrics")
def get_metrics():
    """Returns the latest computed mathematical metrics."""
    if not os.path.exists(metrics_file):
        raise HTTPException(status_code=404, detail="Metrics not yet computed.")
    with open(metrics_file, "r") as f:
        return json.load(f)


@app.post("/api/run")
def run_pipeline_api(
    svd_rank: int = Query(12, ge=1, le=64, description="SVD truncated rank k"),
    pca_components: int = Query(6, ge=1, le=15, description="Number of PCA components p"),
    edge_threshold: float = Query(0.12, ge=0.01, le=0.5, description="Gradient magnitude edge threshold")
):
    """Executes the complete sequential pipeline with custom parameters."""
    pipeline = IntegratedSystemPipeline(
        dataset_dir=DATASET_DIR,
        output_dir=OUTPUT_DIR,
        svd_rank=svd_rank,
        pca_components=pca_components,
        edge_threshold=edge_threshold
    )
    results = pipeline.run_pipeline()
    return {
        "message": "Pipeline execution successful",
        "summary": results["summary"]
    }


@app.get("/api/plots/{plot_name}")
def get_plot(plot_name: str):
    """Serves generated diagnostic figures."""
    allowed_plots = {
        "rank_analysis.png",
        "svd_compression_stages.png",
        "pca_eigenfeatures.png",
        "pca_projections_2d.png",
        "reconstruction_comparison.png",
        "gradient_edge_analysis.png"
    }
    if plot_name not in allowed_plots:
        raise HTTPException(status_code=404, detail=f"Plot {plot_name} not found. Allowed: {list(allowed_plots)}")
    
    file_path = os.path.join(OUTPUT_DIR, plot_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Plot file has not been generated yet.")
    return FileResponse(file_path, media_type="image/png")


@app.get("/", response_class=HTMLResponse)
def index_dashboard():
    """Serves an interactive Web UI dashboard on the root domain."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Image Compression & Feature Analysis System</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0f172a;
      --card-bg: #1e293b;
      --accent: #38bdf8;
      --accent-hover: #0284c7;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --border: #334155;
      --success: #34d399;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
      padding: 2rem 1rem;
    }
    .container { max-width: 1200px; margin: 0 auto; }
    header { text-align: center; margin-bottom: 2.5rem; }
    header h1 { font-size: 2.2rem; font-weight: 700; color: #fff; margin-bottom: 0.5rem; }
    header p { color: var(--text-muted); font-size: 1.05rem; }
    .badge {
      display: inline-block;
      background: rgba(56, 189, 248, 0.15);
      color: var(--accent);
      border: 1px solid var(--accent);
      padding: 0.25rem 0.75rem;
      border-radius: 9999px;
      font-size: 0.85rem;
      font-weight: 600;
      margin-top: 0.5rem;
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.5rem;
      margin-bottom: 2rem;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .controls { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1.25rem; align-items: end; }
    .control-group label { display: block; font-size: 0.875rem; font-weight: 500; margin-bottom: 0.5rem; color: var(--text-muted); }
    .control-group input[type="number"], .control-group input[type="range"] {
      width: 100%; padding: 0.6rem 0.8rem; background: #0f172a; border: 1px solid var(--border); border-radius: 6px; color: #fff; font-size: 0.95rem;
    }
    button.btn {
      background: var(--accent);
      color: #0f172a;
      border: none;
      padding: 0.75rem 1.5rem;
      border-radius: 6px;
      font-weight: 600;
      cursor: pointer;
      font-size: 1rem;
      transition: background 0.2s;
    }
    button.btn:hover { background: var(--accent-hover); }
    button.btn:disabled { opacity: 0.6; cursor: not-allowed; }
    .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-top: 1rem; }
    .metric-card {
      background: #0f172a; border: 1px solid var(--border); border-radius: 8px; padding: 1rem; text-align: center;
    }
    .metric-card .label { font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-card .val { font-size: 1.4rem; font-weight: 700; color: var(--accent); margin-top: 0.3rem; }
    .plots-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr)); gap: 1.5rem; }
    .plot-item {
      background: var(--card-bg); border: 1px solid var(--border); border-radius: 10px; overflow: hidden;
    }
    .plot-item h3 { font-size: 1rem; padding: 1rem; background: rgba(15, 23, 42, 0.6); border-bottom: 1px solid var(--border); }
    .plot-item img { width: 100%; display: block; background: #fff; }
    .formula-banner {
      background: rgba(15, 23, 42, 0.6); border-left: 4px solid var(--accent); padding: 1rem; border-radius: 4px; font-family: 'Fira Code', monospace; font-size: 0.9rem; margin-bottom: 1.5rem;
    }
    footer { text-align: center; margin-top: 3rem; color: var(--text-muted); font-size: 0.85rem; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>Integrated Image Compression & Feature Analysis</h1>
      <p>Linear Algebra & Vector Calculus Computational Engine (Project #56)</p>
      <span class="badge">Live Backend on Render</span>
    </header>

    <div class="formula-banner">
      Images &rarr; Matrix &rarr; Rank &rarr; SVD &rarr; Vectorization &rarr; PCA &rarr; Projection &rarr; Reconstruction &rarr; &nabla;I &rarr; Edge Map
    </div>

    <div class="card">
      <h2 style="font-size: 1.25rem; margin-bottom: 1rem;">Interactive Pipeline Parameters</h2>
      <div class="controls">
        <div class="control-group">
          <label for="svdRank">SVD Truncated Rank k (1 - 64):</label>
          <input type="number" id="svdRank" value="12" min="1" max="64">
        </div>
        <div class="control-group">
          <label for="pcaComp">PCA Components p (1 - 15):</label>
          <input type="number" id="pcaComp" value="6" min="1" max="15">
        </div>
        <div class="control-group">
          <label for="edgeThresh">Edge Threshold &tau; (0.01 - 0.5):</label>
          <input type="number" id="edgeThresh" value="0.12" min="0.01" max="0.5" step="0.01">
        </div>
        <div class="control-group">
          <button class="btn" id="runBtn" onclick="runPipeline()">Run Pipeline</button>
        </div>
      </div>
      <div id="statusText" style="margin-top: 0.8rem; font-size: 0.9rem; color: var(--accent);"></div>
    </div>

    <div class="card">
      <h2 style="font-size: 1.25rem; margin-bottom: 0.5rem;">Computed Mathematical Metrics</h2>
      <div class="metrics-grid" id="metricsGrid">
        <div class="metric-card"><div class="label">SVD Rank k</div><div class="val" id="mSvdRank">--</div></div>
        <div class="metric-card"><div class="label">PCA Components</div><div class="val" id="mPcaComp">--</div></div>
        <div class="metric-card"><div class="label">Avg SVD PSNR</div><div class="val" id="mSvdPsnr">--</div></div>
        <div class="metric-card"><div class="label">Total PSNR</div><div class="val" id="mTotPsnr">--</div></div>
        <div class="metric-card"><div class="label">PCA Explained Var</div><div class="val" id="mPcaVar">--</div></div>
        <div class="metric-card"><div class="label">Gradient Corr</div><div class="val" id="mGradCorr">--</div></div>
        <div class="metric-card"><div class="label">Edge Jaccard IoU</div><div class="val" id="mEdgeIou">--</div></div>
        <div class="metric-card"><div class="label">Eckart-Young Error</div><div class="val" id="mEyError">--</div></div>
      </div>
    </div>

    <h2 style="font-size: 1.35rem; margin-bottom: 1rem;">Mathematical & Diagnostic Visualizations</h2>
    <div class="plots-grid">
      <div class="plot-item">
        <h3>1. Singular Value Spectrum & Energy Concentration</h3>
        <img id="plotRank" src="/api/plots/rank_analysis.png" alt="Rank Analysis">
      </div>
      <div class="plot-item">
        <h3>2. SVD Low-Rank Approximation Progression</h3>
        <img id="plotSvd" src="/api/plots/svd_compression_stages.png" alt="SVD Stages">
      </div>
      <div class="plot-item">
        <h3>3. PCA Principal Features & Scree Plot</h3>
        <img id="plotPca" src="/api/plots/pca_eigenfeatures.png" alt="PCA Features">
      </div>
      <div class="plot-item">
        <h3>4. Orthogonal Projection onto Principal Subspace</h3>
        <img id="plotProj" src="/api/plots/pca_projections_2d.png" alt="PCA Projection">
      </div>
      <div class="plot-item">
        <h3>5. Multi-Stage Reconstruction Comparison</h3>
        <img id="plotRecon" src="/api/plots/reconstruction_comparison.png" alt="Reconstruction">
      </div>
      <div class="plot-item">
        <h3>6. Module 4: Vector Calculus Gradient & Edge Map</h3>
        <img id="plotGrad" src="/api/plots/gradient_edge_analysis.png" alt="Vector Calculus">
      </div>
    </div>

    <footer>
      Linear Algebra & Vector Calculus Mini Project &bull; Powered by FastAPI & Python &bull; Deployed on Render
    </footer>
  </div>

  <script>
    async function loadMetrics() {
      try {
        const res = await fetch('/api/metrics');
        if (res.ok) {
          const m = await res.json();
          document.getElementById('mSvdRank').innerText = m.svd_rank_k;
          document.getElementById('mPcaComp').innerText = m.pca_components_p;
          document.getElementById('mSvdPsnr').innerText = m.avg_svd_psnr_db.toFixed(1) + ' dB';
          document.getElementById('mTotPsnr').innerText = m.avg_total_psnr_db.toFixed(1) + ' dB';
          document.getElementById('mPcaVar').innerText = m.pca_explained_variance_pct.toFixed(1) + '%';
          document.getElementById('mGradCorr').innerText = m.avg_gradient_corr.toFixed(3);
          document.getElementById('mEdgeIou').innerText = m.avg_edge_jaccard_iou.toFixed(3);
          document.getElementById('mEyError').innerText = m.eckart_young_discrepancy.toExponential(1);
        }
      } catch (err) {
        console.error(err);
      }
    }

    async function runPipeline() {
      const btn = document.getElementById('runBtn');
      const status = document.getElementById('statusText');
      const svdRank = document.getElementById('svdRank').value;
      const pcaComp = document.getElementById('pcaComp').value;
      const edgeThresh = document.getElementById('edgeThresh').value;

      btn.disabled = true;
      status.innerText = 'Executing mathematical pipeline on server...';

      try {
        const res = await fetch(`/api/run?svd_rank=${svdRank}&pca_components=${pcaComp}&edge_threshold=${edgeThresh}`, {
          method: 'POST'
        });
        if (res.ok) {
          status.innerText = 'Pipeline completed! Refreshing visualizations...';
          await loadMetrics();
          const t = new Date().getTime();
          ['plotRank', 'plotSvd', 'plotPca', 'plotProj', 'plotRecon', 'plotGrad'].forEach(id => {
            const el = document.getElementById(id);
            const baseSrc = el.src.split('?')[0];
            el.src = baseSrc + '?t=' + t;
          });
          status.innerText = 'Finished successfully!';
        } else {
          status.innerText = 'Execution error: ' + res.statusText;
        }
      } catch (e) {
        status.innerText = 'Network request failed: ' + e.message;
      } finally {
        btn.disabled = false;
      }
    }

    window.onload = loadMetrics;
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting server on port {port}...")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
