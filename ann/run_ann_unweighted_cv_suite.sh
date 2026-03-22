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
  --curvature-penalty-weight 0.0
  --distance-curvature-penalty-weight 0.3
  --derivative-smoothing 1.2
)

run_variant() {
  local script_name="$1"
  shift
  echo
  echo "===== Running ${script_name} ====="
  "${PYTHON_BIN}" "${ROOT_DIR}/${script_name}" "${COMMON_ARGS[@]}" "$@"
}

# run_variant "reconstruct_pantheon_ann_unweighted_expandgrid.py"
# run_variant "reconstruct_pantheon_ann_unweighted_repeatedkfold.py" --cv-repeats 3
run_variant "reconstruct_pantheon_ann_unweighted_binnedcv.py" --cv-bin-width 0.2
# run_variant "reconstruct_pantheon_ann_unweighted_ronly.py"
