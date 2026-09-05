#!/usr/bin/env bash
# Completed sgl146 baseline, using explicit user-supplied data and output paths.
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/run_cosmology.sh INPUT.fits OUTPUT_DIR [wcdm|w0wa|all] [extra inference options...]" >&2
  exit 2
fi
fits="$1"
outdir="$2"
model="${3:-all}"
if [[ $# -ge 3 ]]; then shift 3; else shift 2; fi
case "$model" in
  wcdm|w0wa) models=("$model") ;;
  all) models=(wcdm w0wa) ;;
  *) echo "Unknown model: $model" >&2; exit 2 ;;
esac
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for model in "${models[@]}"; do
  "${PYTHON:-python}" "$script_dir/infer_cosmology.py" \
    --fits "$fits" --outdir "$outdir/$model" \
    --cosmo_model "$model" --npz_label "$model" \
    --seed 42 --warmup 4000 --chunk 2000 --max_chunks 30 \
    --target_accept 0.96 --max_tree_depth 10 --min_ESS 1000 \
    --ckpt_every 1 --ann_weight 1.0 --ann_nu 4.0 \
    --zl_scale_mode none --use_normal_intercept_prior \
    --intercept_prior_sigma 0.08 "$@"
done
