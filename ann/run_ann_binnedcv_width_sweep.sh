#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/geng/Codes/sgl_cosmo/ann"
PYTHON_BIN="/home/geng/miniforge3/envs/sgl-ann/bin/python"
SCRIPT_NAME="reconstruct_pantheon_ann_unweighted_binnedcv.py"

COMMON_ARGS=(
  --n-realizations 400
  --cv-folds 5
  --max-iter 8000
  --n-jobs 20
  --blas-threads 1
  --derivative-weight 0.0
  --curvature-penalty-weight 0.0
  --distance-curvature-penalty-weight 0.3
  --derivative-smoothing 1.2
  --bin-weight-strength 0.5
  --bin-weight-power 1.0
)

run_variant() {
  local width="$1"
  local outdir="$2"
  echo "===== Running ${SCRIPT_NAME} with cv-bin-width=${width} ====="
  "${PYTHON_BIN}" "${ROOT_DIR}/${SCRIPT_NAME}" \
    "${COMMON_ARGS[@]}" \
    --cv-bin-width "${width}" \
    --output-dir "${ROOT_DIR}/output/${outdir}"
  echo
}

run_variant "0.1" "mb_ann_binnedcv_bw01_run"
run_variant "0.2" "mb_ann_binnedcv_bw02_run"
run_variant "0.4" "mb_ann_binnedcv_bw04_run"
run_variant "0.5" "mb_ann_binnedcv_bw05_run"
