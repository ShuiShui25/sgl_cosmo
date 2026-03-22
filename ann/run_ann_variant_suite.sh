#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/geng/Codes/sgl_cosmo/ann"
PYTHON_BIN="/home/geng/miniforge3/envs/sgl-ann/bin/python"

COMMON_ARGS=(
  --n-realizations 400
  --cv-folds 5
  --max-iter 4000
  --n-jobs 20
  --blas-threads 1
  --derivative-weight 0.5
  --curvature-penalty-weight 0.2
  --distance-curvature-penalty-weight 0.3
  --derivative-smoothing 1.2
  --highz-score-strength 2.0
  --highz-score-power 8.0
)

SCRIPTS=(
  # "reconstruct_pantheon_ann_bootstrap.py"
  # "reconstruct_pantheon_ann_highalpha.py"
  # "reconstruct_pantheon_ann_highz_weight.py"
  "reconstruct_pantheon_ann_highz_score.py"
)

for script_name in "${SCRIPTS[@]}"; do
  script_path="${ROOT_DIR}/${script_name}"
  echo "[RUN] ${script_name}"
  "${PYTHON_BIN}" "${script_path}" "${COMMON_ARGS[@]}"
  echo "[DONE] ${script_name}"
done

echo "[ALL DONE] Completed bootstrap and ANN variant runs."
