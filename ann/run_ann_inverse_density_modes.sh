#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/geng/Codes/sgl_cosmo/ann"
PYTHON_BIN="/home/geng/miniforge3/envs/sgl-ann/bin/python"

COMMON_ARGS=(
  --n-realizations 400
  --cv-folds 5
  --max-iter 8000
  --n-jobs 20
  --blas-threads 1
  --derivative-weight 0.0
  --derivative-smoothing 1.2
  --density-weight-strength 1.0
  --density-k 10
  --density-weight-power 5.0
)

run_variant() {
  local script_name="$1"
  echo
  echo "===== Running ${script_name} ====="
  "${PYTHON_BIN}" "${ROOT_DIR}/${script_name}" "${COMMON_ARGS[@]}"
}

# run_variant "reconstruct_pantheon_ann_inverse_density_standard.py"
# run_variant "reconstruct_pantheon_ann_inverse_density_rerror.py"
run_variant "reconstruct_pantheon_ann_inverse_density_rsmooth.py"
