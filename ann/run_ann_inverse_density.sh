#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/geng/Codes/sgl_cosmo/ann"
PYTHON_BIN="/home/geng/miniforge3/envs/sgl-ann/bin/python"

"${PYTHON_BIN}" "${ROOT_DIR}/reconstruct_pantheon_ann_inverse_density.py" \
  --n-realizations 400 \
  --cv-folds 5 \
  --max-iter 8000 \
  --n-jobs 20 \
  --blas-threads 1 \
  --derivative-weight 0.0 \
  --curvature-penalty-weight 0.0 \
  --distance-curvature-penalty-weight 0.0 \
  --derivative-smoothing 1.2 \
  --density-weight-strength 1.0 \
  --density-k 7 \
  --density-weight-power 9.0
