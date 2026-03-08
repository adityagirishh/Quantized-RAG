"""
QRAG NeurIPS Workshop Benchmark
================================
Tests QRAG (4-bit residuals + 8-bit E1) against REAL FAISS baselines:
  - Exact (FlatIP)          — gold standard
  - IVF-Flat                — classic approximate
  - IVF-PQ                  — the strongest memory-efficient baseline
  - HNSW                    — graph-based ANN

Dataset: GloVe-100 (downloaded automatically ~150MB) + synthetic fallback.

Outputs:
  results/qrag_metrics.json          — all numbers
  results/pareto_curve.png           — recall vs memory Pareto plot
  results/baseline_comparison.png    — bar charts
  results/summary_table.csv          — CSV for the paper

Install deps first:
  pip install faiss-cpu scikit-learn matplotlib numpy requests tqdm
"""

import os, time, json, urllib.request, zipfile, shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv
from typing import Optional
from sklearn.decomposition import PCA

# ── Try importing FAISS ──────────────────────────────────────────────────────
try:
    import faiss
    FAISS_OK = True
    print("[OK] FAISS available")
except ImportError:
    FAISS_OK = False
    print("[WARN] FAISS not found. Install with: pip install faiss-cpu")
    print("       Baselines will be skipped; only QRAG (numpy) will run.")

os.makedirs("results", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_glove(max_vecs: int = 50_000):
    """Download GloVe-100 and return (embeddings float32, words list)."""
    glove_dir  = "data/glove"
    glove_txt  = os.path.join(glove_dir, "glove.6B.100d.txt")
    glove_zip  = os.path.join(glove_dir, "glove.6B.zip")
    glove_url  = "https://nlp.stanford.edu/data/glove.6B.zip"

    os.makedirs(glove_dir, exist_ok=True)

    if not os.path.exists(glove_txt):
        print(f"Downloading GloVe-100 (~850 MB zip) …")
        print("  (This only happens once. Use --synthetic to skip.)")
        urllib.request.urlretrieve(glove_url, glove_zip,
            reporthook=lambda b,bs,t: print(f"\r  {b*bs/1e6:.0f}/{t/1e6:.0f} MB", end=""))
        print()
        with zipfile.ZipFile(glove_zip) as z:
            z.extractall(glove_dir)
        os.remove(glove_zip)

    print(f"Loading GloVe-100 (first {max_vecs:,} vectors) …")
    words, vecs = [], []
    with open(glove_txt, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_vecs:
                break
            parts = line.split()
            words.append(parts[0])
            vecs.append(np.array(parts[1:], dtype="float32"))

    embeddings = np.stack(vecs)
    # L2-normalise
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings /= np.maximum(norms, 1e-12)
    print(f"  Loaded {len(embeddings):,} vectors, dim={embeddings.shape[1]}")
    return embeddings, words


def make_synthetic(n: int = 50_000, dim: int = 128, seed: int = 42):
    """Clustered synthetic embeddings as fallback."""
    print(f"Generating {n:,} synthetic embeddings (dim={dim}) …")
    rng = np.random.RandomState(seed)
    centers = rng.normal(scale=2.0, size=(50, dim)).astype("float32")
    X = rng.normal(size=(n, dim)).astype("float32")
    for i in range(n):
        X[i] += centers[i % 50] * 0.4
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X /= np.maximum(norms, 1e-12)
    return X, [str(i) for i in range(n)]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  QUANTISERS
# ─────────────────────────────────────────────────────────────────────────────

class FourBitQuantizer:
    levels = 15
    def quantize(self, data):
        mins = np.percentile(data, 1, axis=0)
        maxs = np.percentile(data, 99, axis=0)
        ranges = np.maximum(maxs - mins, 1e-8)
        scaled = np.clip((data - mins) / ranges, 0, 1)
        q = np.round(scaled * self.levels).astype(np.uint8)
        N, D = q.shape
        W = (D + 1) // 2
        packed = np.zeros((N, W), dtype=np.uint8)
        for d in range(D):
            b = d // 2
            if d % 2 == 0:
                packed[:, b] |= q[:, d]
            else:
                packed[:, b] |= (q[:, d] << 4)
        return packed, mins, maxs

    def dequantize(self, packed, mins, maxs, shape):
        N, D = shape
        out = np.zeros((N, D), dtype=np.uint8)
        for d in range(D):
            b = d // 2
            if b < packed.shape[1]:
                if d % 2 == 0:
                    out[:, d] = packed[:, b] & 0x0F
                else:
                    out[:, d] = (packed[:, b] >> 4) & 0x0F
        ranges = maxs - mins
        return mins + (out.astype("float32") / self.levels) * ranges


class EightBitQuantizer:
    levels = 255
    def quantize(self, data):
        mins = np.percentile(data, 1, axis=0)
        maxs = np.percentile(data, 99, axis=0)
        ranges = np.maximum(maxs - mins, 1e-8)
        q = np.round(np.clip((data - mins) / ranges, 0, 1) * self.levels).astype(np.uint8)
        return q, mins, maxs

    def dequantize(self, q, mins, maxs, shape):
        return mins + (q.astype("float32") / self.levels) * (maxs - mins)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  QRAG
# ─────────────────────────────────────────────────────────────────────────────

class QRAG:
    def __init__(self, projection_dim: int = 64):
        self.projection_dim = projection_dim
        self.fitted = False

    def fit(self, embeddings: np.ndarray):
        N, D = embeddings.shape
        self.N, self.D = N, D
        pd = min(self.projection_dim, D, N - 1)

        # PCA
        self.pca = PCA(n_components=pd, random_state=0)
        self.pca.fit(embeddings)
        var = np.sum(self.pca.explained_variance_ratio_)
        print(f"  PCA({pd}) explains {var:.3f} variance")

        e1_raw = self.pca.transform(embeddings).astype("float32")
        norms  = np.linalg.norm(e1_raw, axis=1, keepdims=True)
        e1_norm = e1_raw / np.maximum(norms, 1e-12)

        # FAISS index on normalised projections
        if FAISS_OK:
            self.index = faiss.IndexHNSWFlat(pd, 32)
            self.index.hnsw.efConstruction = 200
            self.index.add(e1_norm)
        else:
            self.index = e1_norm  # numpy fallback

        # Residuals
        base = self.pca.inverse_transform(e1_raw).astype("float32")
        residuals = embeddings - base

        self.q4 = FourBitQuantizer()
        self.q8 = EightBitQuantizer()

        self.packed_res, self.res_min, self.res_max = self.q4.quantize(residuals)
        self.packed_e1,  self.e1_min,  self.e1_max  = self.q8.quantize(e1_raw)

        self.fitted = True

    def search(self, query: np.ndarray, k1: int = 100, k_final: int = 10):
        q_raw  = self.pca.transform(query.reshape(1, -1)).astype("float32")
        q_norm = q_raw / np.maximum(np.linalg.norm(q_raw), 1e-12)

        if FAISS_OK:
            _, ids = self.index.search(q_norm, k1)
            cands = ids[0]
        else:
            scores = self.index @ q_norm.reshape(-1)
            cands  = np.argsort(-scores)[:k1]

        e1_raw_c = self.q8.dequantize(self.packed_e1[cands], self.e1_min, self.e1_max,
                                      (len(cands), self.pca.n_components_))
        base_c   = self.pca.inverse_transform(e1_raw_c).astype("float32")
        res_c    = self.q4.dequantize(self.packed_res[cands], self.res_min, self.res_max,
                                      (len(cands), self.D))
        recon    = base_c + res_c
        recon   /= np.maximum(np.linalg.norm(recon, axis=1, keepdims=True), 1e-12)

        qn = query / np.maximum(np.linalg.norm(query), 1e-12)
        sc = recon @ qn
        top = np.argsort(-sc)[:k_final]
        return cands[top].tolist(), sc[top]

    def memory_mb(self):
        mb = (self.packed_res.nbytes + self.packed_e1.nbytes +
              self.res_min.nbytes + self.res_max.nbytes +
              self.e1_min.nbytes  + self.e1_max.nbytes) / 1e6
        if FAISS_OK and hasattr(self.index, "ntotal"):
            mb += self.index.ntotal * self.pca.n_components_ * 4 / 1e6
        else:
            mb += self.index.nbytes / 1e6
        return mb


# ─────────────────────────────────────────────────────────────────────────────
# 4.  FAISS BASELINES
# ─────────────────────────────────────────────────────────────────────────────

def build_flat(embeddings):
    idx = faiss.IndexFlatIP(embeddings.shape[1])
    idx.add(embeddings)
    return idx

def build_ivfflat(embeddings, nlist=None):
    d    = embeddings.shape[1]
    nlist = nlist or max(4, int(4 * np.sqrt(len(embeddings))))
    q    = faiss.IndexFlatIP(d)
    idx  = faiss.IndexIVFFlat(q, d, nlist, faiss.METRIC_INNER_PRODUCT)
    idx.train(embeddings)
    idx.add(embeddings)
    idx.nprobe = max(1, nlist // 8)
    return idx

def build_ivfpq(embeddings, nlist=None, m=None):
    d     = embeddings.shape[1]
    nlist = nlist or max(4, int(4 * np.sqrt(len(embeddings))))
    m     = m or max(4, d // 8)
    # m must divide d
    while d % m != 0 and m > 1:
        m -= 1
    q   = faiss.IndexFlatIP(d)
    idx = faiss.IndexIVFPQ(q, d, nlist, m, 8)
    idx.train(embeddings)
    idx.add(embeddings)
    idx.nprobe = max(1, nlist // 8)
    return idx

def build_hnsw(embeddings, M=32):
    d   = embeddings.shape[1]
    idx = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
    idx.hnsw.efConstruction = 200
    idx.add(embeddings)
    return idx

def index_memory_mb(index, embeddings):
    """Rough memory estimate for a FAISS index in MB."""
    n, d = embeddings.shape
    name = type(index).__name__
    if "Flat" in name and "IVF" not in name and "HNSW" not in name:
        return n * d * 4 / 1e6
    elif "IVFPQ" in name:
        nlist = index.nlist
        # PQ codes + coarse centroids
        codes_bytes = n * index.pq.M  # 1 byte per sub-quantizer
        centroids_bytes = nlist * d * 4
        return (codes_bytes + centroids_bytes) / 1e6
    elif "IVFFlat" in name:
        nlist = index.nlist
        return (n * d * 4 + nlist * d * 4) / 1e6
    elif "HNSW" in name:
        # vectors + graph edges
        M = index.hnsw.nb_neighbors(0)
        return (n * d * 4 + n * M * 2 * 4) / 1e6
    return n * d * 4 / 1e6


def eval_index(index, embeddings, queries, ground_truth, k=10):
    recalls, times = [], []
    for i, q in enumerate(queries):
        t0 = time.time()
        if FAISS_OK:
            qf = q.reshape(1, -1).astype("float32")
            _, ids = index.search(qf, k)
            res = ids[0].tolist()
        else:
            sc  = embeddings @ q
            res = np.argsort(-sc)[:k].tolist()
        times.append(time.time() - t0)
        recalls.append(len(set(res) & set(ground_truth[i])) / k)
    return float(np.mean(recalls)), float(np.std(recalls)), float(np.mean(times) * 1000)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  SWEEP  (recall vs memory Pareto)
# ─────────────────────────────────────────────────────────────────────────────

def qrag_sweep(embeddings, queries, ground_truth, projection_dims, k1=100, k=10):
    """Run QRAG at multiple projection dims to trace the Pareto front."""
    results = []
    for pd in projection_dims:
        print(f"  QRAG projection_dim={pd} …", end=" ", flush=True)
        qrag = QRAG(projection_dim=pd)
        qrag.fit(embeddings)
        recalls, times = [], []
        for i, q in enumerate(queries):
            t0 = time.time()
            ids, _ = qrag.search(q, k1=k1, k_final=k)
            times.append(time.time() - t0)
            recalls.append(len(set(ids) & set(ground_truth[i])) / k)
        mem = qrag.memory_mb()
        r   = float(np.mean(recalls))
        print(f"recall={r:.3f}  mem={mem:.1f}MB")
        results.append({"label": f"QRAG-{pd}", "recall": r,
                        "memory_mb": mem, "latency_ms": float(np.mean(times)*1000)})
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6.  PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_pareto(qrag_points, baselines, baseline_memory, out_path):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#ffffff")

    # QRAG sweep curve
    qx = [p["memory_mb"]  for p in qrag_points]
    qy = [p["recall"]     for p in qrag_points]
    order = np.argsort(qx)
    ax.plot([qx[i] for i in order], [qy[i] for i in order],
            "o-", color="#2563eb", lw=2.5, ms=7, label="QRAG (this work)", zorder=5)
    for p in qrag_points:
        ax.annotate(p["label"], (p["memory_mb"], p["recall"]),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=7.5, color="#1e40af")

    # Baselines as scatter
    colours = {"Flat": "#dc2626", "IVF-Flat": "#d97706",
               "IVF-PQ": "#16a34a", "HNSW": "#7c3aed"}
    markers = {"Flat": "s", "IVF-Flat": "^", "IVF-PQ": "D", "HNSW": "P"}
    for name, rec in baselines.items():
        mem = baseline_memory[name]
        ax.scatter(mem, rec, s=120, color=colours.get(name, "grey"),
                   marker=markers.get(name, "o"), zorder=6,
                   label=f"{name} ({mem:.0f}MB, R={rec:.3f})")
        ax.annotate(name, (mem, rec), textcoords="offset points",
                    xytext=(6, -10), fontsize=8, color=colours.get(name, "grey"))

    baseline_full = baseline_memory.get("Flat", None)
    if baseline_full:
        ax.axvline(baseline_full, color="#dc2626", ls="--", lw=1, alpha=0.5,
                   label="Full FP32 memory")

    ax.set_xlabel("Memory (MB)", fontsize=12)
    ax.set_ylabel("Recall@10", fontsize=12)
    ax.set_title("Recall@10 vs Memory — QRAG vs FAISS Baselines\n(→ upper-left is better)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_bars(baselines, qrag_best, out_path):
    names   = list(baselines.keys()) + [f"QRAG-best"]
    recalls = [baselines[n] for n in baselines] + [qrag_best["recall"]]
    colours = ["#dc2626","#d97706","#16a34a","#7c3aed","#2563eb"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_facecolor("#f8f9fa")
    bars = ax.bar(names, recalls, color=colours[:len(names)], width=0.55, edgecolor="white")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Recall@10", fontsize=12)
    ax.set_title("Recall@10 Comparison: QRAG vs Real FAISS Baselines", fontsize=13,
                 fontweight="bold")
    ax.axhline(0.9, color="grey", ls="--", lw=1, alpha=0.5, label="0.9 target")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true",
                        help="Skip GloVe download; use synthetic data")
    parser.add_argument("--n_docs",    type=int, default=50_000)
    parser.add_argument("--n_queries", type=int, default=200)
    parser.add_argument("--k",         type=int, default=10)
    parser.add_argument("--k1",        type=int, default=100,
                        help="QRAG coarse candidate count")
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    if args.synthetic:
        embeddings, _ = make_synthetic(args.n_docs, dim=128)
    else:
        try:
            embeddings, _ = load_glove(args.n_docs)
        except Exception as e:
            print(f"[WARN] GloVe load failed ({e}). Falling back to synthetic.")
            embeddings, _ = make_synthetic(args.n_docs, dim=128)

    N, D = embeddings.shape
    baseline_memory_mb = N * D * 4 / 1e6
    print(f"\nDataset: {N:,} vectors × dim {D}  ({baseline_memory_mb:.1f} MB FP32)\n")

    # ── Queries + ground truth ────────────────────────────────────────────────
    rng = np.random.RandomState(0)
    q_idx = rng.choice(N, args.n_queries, replace=False)
    queries = []
    for i in q_idx:
        q = embeddings[i] + 0.02 * rng.normal(size=D).astype("float32")
        q /= np.maximum(np.linalg.norm(q), 1e-12)
        queries.append(q)

    print("Computing exact ground truth …")
    ground_truth = []
    for q in queries:
        sc = embeddings @ q
        ground_truth.append(np.argsort(-sc)[:args.k].tolist())

    # ── FAISS baselines ───────────────────────────────────────────────────────
    baseline_results  = {}
    baseline_mem_dict = {}

    if FAISS_OK:
        configs = [
            ("Flat",     lambda: build_flat(embeddings)),
            ("IVF-Flat", lambda: build_ivfflat(embeddings)),
            ("IVF-PQ",   lambda: build_ivfpq(embeddings)),
            ("HNSW",     lambda: build_hnsw(embeddings)),
        ]
        for name, builder in configs:
            print(f"Building {name} index …", end=" ", flush=True)
            t0  = time.time()
            idx = builder()
            bt  = time.time() - t0
            r, std, lat = eval_index(idx, embeddings, queries, ground_truth, args.k)
            mem = index_memory_mb(idx, embeddings)
            baseline_results[name]  = r
            baseline_mem_dict[name] = mem
            print(f"recall={r:.3f} ±{std:.3f}  mem={mem:.1f}MB  build={bt:.1f}s  lat={lat:.1f}ms")
    else:
        print("[SKIP] FAISS not available — skipping baselines")

    # ── QRAG sweep ────────────────────────────────────────────────────────────
    # Projection dims to sweep (capped at D)
    pds = [d for d in [16, 32, 64, 96, 128] if d < D]
    if D <= 128:
        pds = [d for d in [8, 16, 32, 48, 64] if d < D]

    print("\nRunning QRAG sweep …")
    qrag_points = qrag_sweep(embeddings, queries, ground_truth, pds, k1=args.k1, k=args.k)

    # Best QRAG = highest recall
    qrag_best = max(qrag_points, key=lambda x: x["recall"])
    print(f"\nBest QRAG: {qrag_best['label']}  recall={qrag_best['recall']:.3f}"
          f"  mem={qrag_best['memory_mb']:.1f}MB")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots …")
    if FAISS_OK:
        plot_pareto(qrag_points, baseline_results, baseline_mem_dict,
                    "results/pareto_curve.png")
        plot_bars(baseline_results, qrag_best,
                  "results/baseline_comparison.png")
    else:
        # Pareto with QRAG only
        fig, ax = plt.subplots(figsize=(8, 5))
        qx = [p["memory_mb"] for p in qrag_points]
        qy = [p["recall"]    for p in qrag_points]
        ax.plot(sorted(qx), [qy[i] for i in np.argsort(qx)], "o-", color="#2563eb")
        ax.set_xlabel("Memory (MB)"); ax.set_ylabel("Recall@10")
        ax.set_title("QRAG Recall vs Memory"); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig("results/pareto_curve.png", dpi=150); plt.close()

    # ── CSV summary ───────────────────────────────────────────────────────────
    rows = []
    for name, rec in baseline_results.items():
        rows.append({"method": name, "recall_at_10": f"{rec:.4f}",
                     "memory_mb": f"{baseline_mem_dict[name]:.1f}",
                     "compression_vs_flat": f"{baseline_memory_mb / baseline_mem_dict[name]:.2f}x"})
    for p in qrag_points:
        rows.append({"method": p["label"],
                     "recall_at_10": f"{p['recall']:.4f}",
                     "memory_mb": f"{p['memory_mb']:.1f}",
                     "compression_vs_flat": f"{baseline_memory_mb / p['memory_mb']:.2f}x"})

    csv_path = "results/summary_table.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method","recall_at_10","memory_mb","compression_vs_flat"])
        w.writeheader(); w.writerows(rows)
    print(f"  Saved: {csv_path}")

    # ── JSON metrics ──────────────────────────────────────────────────────────
    metrics = {
        "dataset": "glove-100" if not args.synthetic else "synthetic",
        "n_docs": N, "embedding_dim": D,
        "baseline_memory_fp32_mb": baseline_memory_mb,
        "baselines": {k: {"recall_at_10": v, "memory_mb": baseline_mem_dict.get(k)}
                      for k, v in baseline_results.items()},
        "qrag_sweep": qrag_points,
        "qrag_best": qrag_best,
    }
    with open("results/qrag_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("  Saved: results/qrag_metrics.json")

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n" + "="*65)
    print(f"{'Method':<18} {'Recall@10':>10} {'Memory(MB)':>12} {'Compression':>13}")
    print("-"*65)
    for r in rows:
        print(f"{r['method']:<18} {r['recall_at_10']:>10} {r['memory_mb']:>12} {r['compression_vs_flat']:>13}")
    print("="*65)
    print("\nDone. Check the results/ folder for plots and tables.")


if __name__ == "__main__":
    main()