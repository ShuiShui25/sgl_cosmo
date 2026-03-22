#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/geng/Codes/sgl_cosmo/ann"
PYTHON_BIN="/home/geng/miniforge3/envs/sgl-ann/bin/python"

"${PYTHON_BIN}" "${ROOT_DIR}/reconstruct_pantheon_ann_variable_lengthscale.py" \
  --n-realizations 400 \
  --cv-folds 5 \
  --max-iter 4000 \
  --n-jobs 20 \
  --blas-threads 1 \
  --derivative-weight 0.5 \
  --curvature-penalty-weight 0.2 \
  --distance-curvature-penalty-weight 0.3 \
  --derivative-smoothing 1.2
