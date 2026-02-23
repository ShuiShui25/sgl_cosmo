#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Quick Diagnostic Strategy
# Compare 143 vs 144 on full DES sample with short runs
# -----------------------------

PYTHON_BIN="${PYTHON_BIN:-/home/geng/miniforge3/envs/jax/bin/python}"
BASE="/home/geng/Codes/lensing_cosmo/jax01"
OUTDIR="${BASE}/output_des_sim"
REPORT="${OUTDIR}/run_report_des_quick_143_vs_144.md"

SCRIPT143="${BASE}/jax_check143.py"
SCRIPT144="${BASE}/jax_check144.py"

FITS_WCDM="${BASE}/data/sim_data/DESa_sim_l3_w_evoPL.fits"
FITS_W0WA="${BASE}/data/sim_data/DESa_sim_l3_w0wa_evoPL.fits"

# Quick settings (full sample, short run)
COMMON=(
  --outdir "$OUTDIR"
  --seed 42
  --warmup 1000
  --chunk 100
  --max_chunks 1
  --target_accept 0.95
  --min_ESS 0
  --accept_mean_min 0.0
  --div_rate_thresh 1.0
)

mkdir -p "$OUTDIR"

{
  echo "# DES quick compare: jax_check143 vs jax_check144"
  echo
  echo "- Date: $(date -Iseconds)"
  echo "- Common args: ${COMMON[*]}"
  echo
} > "$REPORT"

run_one () {
  local tag="$1"
  local script="$2"
  local fits="$3"
  local model="$4"
  local label="$5"
  local log_tmp
  log_tmp="$(mktemp)"

  {
    echo "## ${tag}"
    echo
    echo '```bash'
    echo "$PYTHON_BIN $script --fits $fits ${COMMON[*]} --cosmo_model $model --npz_label $label"
    echo '```'
    echo
    echo '```'
  } >> "$REPORT"

  "$PYTHON_BIN" "$script" --fits "$fits" "${COMMON[@]}" --cosmo_model "$model" --npz_label "$label" \
    | tee -a "$REPORT" > "$log_tmp"

  {
    echo '```'
    echo
  } >> "$REPORT"

  rm -f "$log_tmp"
}

# 4 runs: 143/144 x wcdm/w0wa
run_one "143 wCDM" "$SCRIPT143" "$FITS_WCDM" wcdm des_quick_143_wcdm
run_one "144 wCDM" "$SCRIPT144" "$FITS_WCDM" wcdm des_quick_144_wcdm
run_one "143 w0wa" "$SCRIPT143" "$FITS_W0WA" w0wa des_quick_143_w0wa
run_one "144 w0wa" "$SCRIPT144" "$FITS_W0WA" w0wa des_quick_144_w0wa

# NPZ boundary summary
"$PYTHON_BIN" - <<'PY' | tee -a "$REPORT"
import os
import numpy as np

out = "/home/geng/Codes/lensing_cosmo/jax01/output_des_sim"
labels = [
    ("143_wcdm", "des_quick_143_wcdm"),
    ("144_wcdm", "des_quick_144_wcdm"),
    ("143_w0wa", "des_quick_143_w0wa"),
    ("144_w0wa", "des_quick_144_w0wa"),
]

print("## Boundary Summary")
print()
print("| run | g0_edge(<1.81) | d0_edge(<1.81) | log_sig_g_top(>-0.2) | gamma0_med | delta0_med | log_sig_g_med |")
print("| --- | --- | --- | --- | --- | --- | --- |")
for name, lb in labels:
    p = os.path.join(out, f"posterior_minimal_chunks_{lb}.npz")
    d = np.load(p)
    g0 = d["gamma0"]
    d0 = d["delta0"]
    lsg = d["log_sig_g"]
    print(
        f"| {name} | {np.mean(g0<1.81):.3f} | {np.mean(d0<1.81):.3f} | {np.mean(lsg>-0.2):.3f} | "
        f"{np.median(g0):.4f} | {np.median(d0):.4f} | {np.median(lsg):.4f} |"
    )
PY

echo
echo "Done. Report: $REPORT"
