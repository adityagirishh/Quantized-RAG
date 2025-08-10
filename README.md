## Production-QRAG: Memory-Optimized QRAG Benchmark Suite

This repository provides a research-grade evaluation and demonstration of a Memory-Optimized QRAG approach using 4-bit residuals and 8-bit E1 projections, alongside a comprehensive comparison against 15 representative RAG baselines.

### Key Components
- `final-8bitM1-4bitE1.py`: Standalone demo of the 4-bit residuals + 8-bit E1 QRAG pipeline with metric export.
- `test-QRAG.py`: Comprehensive benchmark simulator comparing QRAG vs 15 competitors; produces ranked tables, visualizations, and reports.

### Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Quick Start
- Run the QRAG demo (saves metrics to `results/qrag_demo/`):
```bash
python final-8bitM1-4bitE1.py --n_docs 20000 --embedding_dim 768 --projection_dim 256 --n_test_queries 30 --save_dir results/qrag_demo
```

- Run the comprehensive benchmark (saves figures/tables and reports):
```bash
python test-QRAG.py
```

Artifacts will be saved under:
- `reports/figures/` — comparison plots
- `reports/tables/` — CSV summaries and ablation study
- `results/` — recommendations and executive summary

### Outputs Overview
- Figures: `reports/figures/comparison.png`
- Tables: `reports/tables/comparison.csv`, `reports/tables/ranked_by_composite_score.csv`, `reports/tables/ablation_study.csv`
- Reports: `results/executive_summary.md`, `results/recommendations.txt`, `results/recommendations.json`
- Demo metrics: `results/qrag_demo/qrag_metrics.json`, `results/qrag_demo/qrag_metrics.txt`

### Notes
- FAISS is used when available; the demo falls back to a NumPy-based candidate search automatically.
- The comprehensive benchmark uses simulated behaviors for competitors to enable quick comparative analysis.

### Citation
If you use this repository for academic work, please cite as:

Aditya Girish et al., "Memory-Optimized QRAG: Dual-Quantized Retrieval with On-Demand Reconstruction," 2025.


