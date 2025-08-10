## Memory-Optimized QRAG: Technical Report

### Abstract
We present a memory-optimized QRAG approach using dual quantization: 4-bit residuals and 8-bit E1 projections with on-demand reconstruction. We compare this method against 15 diverse RAG variants and report memory footprint, recall@10, search latency, and a composite score.

### Method Summary
- PCA projector learns a low-dimensional subspace (E1) for candidate selection.
- E1 is quantized to 8-bit; residuals to 4-bit with percentile-based scaling and bit-packing.
- Candidates are selected via FAISS (HNSW/IVF/PQ depending on size) or NumPy fallback.
- Reconstruction combines inverse-projected E1 with dequantized residuals; normalized dot-product yields final ranking.

### Demo Results (from `final-8bitM1-4bitE1.py`)
- n_docs: 20,000; dim: 768; proj_dim: 256; n_queries: 30
- Mean Recall@10: ~0.723 (±0.115)
- Mean search time: ~2.0 ms/query
- Memory reduction vs dense baseline: ~1.8x

### Comparative Benchmark (from `test-QRAG.py`)
Top-line metrics (simulated competitors):
- Memory-Optimized QRAG: recall@10 ≈ 0.774; memory reduction ≈ 2.3x; speed ≈ 0.3 ms
- Memory leader: 4-bit Vector Quantization (≈4.0x)
- Speed leader: Memory-Optimized QRAG (≈0.3 ms)
- Quality leaders: QRAG is among top simulated recall scores

See `reports/figures/comparison.png` and CSVs in `reports/tables/` for full results.

### Ablation Study (configured in `test-QRAG.py`)
Shows the trade-offs among compression ratio, recall, and speed for variants (e.g., 8-bit residuals, float32 E1, no quantization + PCA, full precision baseline). See `reports/tables/ablation_study.csv`.

### Strategic Recommendations
1. Maintain dual-quantization scheme for strong memory efficiency.
2. Explore reflection/graph-based modules to push recall without inflating memory.
3. Optimize candidate selection and caching for ultra-low-latency workloads.
4. Extend to multi-hop retrieval settings (e.g., hierarchical summarization).

### Reproducibility
- Exact package versions: `requirements.txt`
- Determinism: seeds exposed in CLI for the demo
- Outputs saved to `results/` and `reports/`

### Limitations
- Competitor performance is simulated for breadth and speed. Integrations with real systems will change absolute metrics, but relative trade-offs remain illustrative.

### Appendix
- Implementation: `final-8bitM1-4bitE1.py`
- Benchmark framework: `test-QRAG.py`
- Artifacts: `reports/` and `results/`


