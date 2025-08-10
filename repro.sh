#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python final-8bitM1-4bitE1.py --n_docs 20000 --embedding_dim 768 --projection_dim 256 --n_test_queries 30 --save_dir results/qrag_demo
python test-QRAG.py

echo "Artifacts saved to reports/ and results/"


