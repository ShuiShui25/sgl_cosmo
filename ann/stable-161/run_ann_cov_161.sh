#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/geng/Codes/sgl_cosmo/ann"
PYTHON_BIN="/home/geng/miniforge3/envs/sgl-ann/bin/python"
SCRIPT_NAME="reconstruct_ann_cov_binnedcv.py"
LENS_FITS="${ROOT_DIR}/data/SGLTable_ANN.fits"

"${PYTHON_BIN}" "${ROOT_DIR}/${SCRIPT_NAME}" \
  --lens-fits "${LENS_FITS}" \
  --n-realizations 400 \
  --cv-folds 5 \
  --cv-bin-width 0.5 \
  --max-iter 8000 \
  --n-jobs 20 \
  --blas-threads 1 \
  --derivative-weight 0.0 \
  --curvature-penalty-weight 0.0 \
  --distance-curvature-penalty-weight 0.3 \
  --derivative-smoothing 1.2 \
  --bin-weight-strength 0.5 \
  --bin-weight-power 1.0 \
  --output-dir "${ROOT_DIR}/output/mb_ann_cov_161_weighted_run"
