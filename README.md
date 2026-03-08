# Memory-Optimized QRAG: Dual-Quantized Retrieval with On-Demand Reconstruction

> **NeurIPS 2025 Workshop Submission**
> Aditya Girish et al., 2025

QRAG is a memory-efficient retrieval method for Retrieval-Augmented Generation (RAG) pipelines. It compresses embedding corpora via **dual-level quantization** — 8-bit PCA projections (E1) and 4-bit residuals — and reconstructs embeddings on-demand at query time, achieving high recall at a fraction of the memory cost of full-precision storage.

---

## Key Results

Benchmarked on two BEIR datasets with 768-dim embeddings (`all-mpnet-base-v2`), compared against real FAISS baselines.

### BEIR/SciFact — 5,000 documents

| Method | Recall@10 | Memory | Compression |
|---|---|---|---|
| Flat (FP32) | 1.000 | 15.4 MB | 1.0× |
| HNSW | 0.999 | 16.6 MB | 0.9× |
| IVF-Flat | 0.987 | 16.2 MB | 1.0× |
| IVF-PQ | 0.807 | 1.3 MB | **11.4×** |
| **QRAG-64** | **0.959** | **3.5 MB** | **4.4×** |
| QRAG-128 | 0.963 | 5.1 MB | 3.0× |
| QRAG-256 | 0.971 | 8.3 MB | 1.8× |

### BEIR/FiQA — 57,000 documents

| Method | Recall@10 | Memory | Compression |
|---|---|---|---|
| Flat (FP32) | 1.000 | 175.1 MB | 1.0× |
| HNSW | 0.991 | 189.7 MB | 0.9× |
| IVF-Flat | 0.998 | 178.0 MB | 1.0× |
| IVF-PQ | 0.736 | 8.4 MB | **20.8×** |
| **QRAG-64** | **0.921** | **40.1 MB** | **4.4×** |
| QRAG-128 | 0.949 | 58.4 MB | 3.0× |
| QRAG-256 | 0.952 | 94.9 MB | 1.9× |

### The Core Claim

At a **4.4× compression ratio**, QRAG achieves **0.92–0.96 Recall@10** across both datasets. IVF-PQ — the strongest memory-efficient baseline — requires **20× compression** to reach a similar memory footprint and only achieves **0.74 recall**, a gap of **+18 points**.

Unlike IVF-PQ, which has a single fixed operating point, QRAG's `projection_dim` parameter provides a **smooth, continuous recall-vs-memory tradeoff** — practitioners can dial in exactly the operating point they need.

### PCA Variance vs Recall (SciFact)

| Projection Dim | PCA Variance | Recall@10 |
|---|---|---|
| 32 | 0.534 | 0.937 |
| 64 | 0.681 | 0.959 |
| 96 | 0.767 | 0.966 |
| 128 | 0.828 | 0.963 |
| 192 | 0.904 | 0.969 |
| 256 | 0.946 | 0.971 |
| 384 | 0.980 | 0.972 |

Notable finding: QRAG-64 preserves only 68% of PCA variance but achieves 96% of perfect recall — the recall-variance relationship is strongly sublinear, suggesting QRAG's residual correction effectively compensates for low-variance projections.

---

## Method

QRAG compresses retrieval corpora in two stages:

**Stage 1 — E1 Projection (8-bit)**
A PCA projection reduces each embedding from dimension `D` to `projection_dim`. The projected vectors are quantized to 8-bit and stored. A normalized copy is indexed with FAISS HNSW for fast coarse search.

**Stage 2 — Residual Compression (4-bit)**
The reconstruction error (`original − PCA_inverse(projection)`) is quantized to 4-bit with bit-packing (two values per byte).

**At query time:**
1. Project the query and search the HNSW index for `k1` candidates (~100–200)
2. Dequantize E1 projections for candidates → reconstruct base vectors
3. Dequantize 4-bit residuals → add to base vectors
4. Re-rank reconstructed candidates by cosine similarity → return top-k

This two-stage approach means the full corpus never needs to be loaded into memory simultaneously.

---

## Repository Structure

```
.
├── final-8bitM1-4bitE1.py     # Standalone QRAG demo with metric export
├── test-QRAG.py               # Benchmark vs simulated baselines (15 methods)
├── qrag_benchmark.py          # Real FAISS baseline benchmark (GloVe-100)
├── qrag_benchmark_768.py      # Real FAISS baseline benchmark (768-dim BEIR)
├── qrag_kaggle.py             # GPU-accelerated Kaggle notebook version
├── repro.sh                   # One-command reproduction script
├── requirements.txt
├── reports/
│   ├── figures/               # Comparison plots
│   └── tables/                # CSV summaries and ablation study
└── results/
    ├── qrag_demo/             # Demo metrics (JSON + TXT)
    ├── pareto_*.png           # Recall vs memory Pareto curves
    ├── compression_*.png      # Compression ratio vs recall
    ├── pca_variance_*.png     # PCA variance ablation
    ├── combined_pareto.png    # Main paper figure (both datasets)
    └── summary_*.csv          # Full results tables
```

---

## Reproduction

### Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Quick demo (synthetic data, no downloads)

```bash
python final-8bitM1-4bitE1.py \
  --n_docs 20000 \
  --embedding_dim 768 \
  --projection_dim 256 \
  --n_test_queries 30 \
  --save_dir results/qrag_demo
```

### Real BEIR benchmark (reproduces paper results)

```bash
# SciFact — 5k docs, encodes in ~3 min on CPU
python qrag_benchmark_768.py --dataset scifact --max_docs 5000 --n_queries 500

# FiQA — 57k docs, use --use_cached after first run
python qrag_benchmark_768.py --dataset fiqa --max_docs 57000 --n_queries 500

# Subsequent runs (skip re-encoding)
python qrag_benchmark_768.py --dataset fiqa --max_docs 57000 --use_cached
```

### Kaggle GPU (fastest — ~3 min encoding for 57k docs)

1. Create a new Kaggle notebook, set **Accelerator → GPU T4 x2**
2. In the first cell: `!pip install -q sentence-transformers datasets`
3. Paste `qrag_kaggle.py` into the next cell and run

Outputs save to `/kaggle/working/results/`.

### Full simulated benchmark (15 baselines)

```bash
python test-QRAG.py
```

Artifacts saved to `reports/figures/`, `reports/tables/`, `results/`.

---

## Baselines

All FAISS baselines are **real implementations**, not simulated:

| Baseline | Description |
|---|---|
| **Flat (FP32)** | Exact brute-force inner product — gold standard |
| **IVF-Flat** | Inverted file index, full-precision vectors |
| **IVF-PQ** | Inverted file + product quantization — strongest memory-efficient baseline |
| **HNSW** | Hierarchical navigable small world graph |

---

## Requirements

```
faiss-cpu>=1.7.4
scikit-learn>=1.3.0
numpy>=1.24.0
matplotlib>=3.7.0
sentence-transformers>=2.2.0
datasets>=2.14.0
tqdm
```

---

## Notes

- FAISS falls back to NumPy-based search automatically if unavailable
- On Apple Silicon: encoding runs on CPU to avoid MPS multiprocessing segfaults; set `TOKENIZERS_PARALLELISM=false`
- On Kaggle GPU: FAISS runs on CPU (T4 shared memory limit prevents IVF-PQ GPU transfer); GPU is used for encoding only
- All benchmarks use `all-mpnet-base-v2` (768-dim, normalized embeddings)

---

## Citation

```bibtex
@misc{girish2025qrag,
  title  = {Memory-Optimized QRAG: Dual-Quantized Retrieval with On-Demand Reconstruction},
  author = {Aditya Girish et al.},
  year   = {2025}
}
```
