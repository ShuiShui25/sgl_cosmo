#!/usr/bin/env bash
set -euo pipefail

PY="/home/geng/miniforge3/envs/jax/bin/python"
RUN="/home/geng/Codes/lensing_cosmo/jax01/jax_check145.py"
OUTDIR="/home/geng/Codes/lensing_cosmo/jax01/output_des_sim"
FITS="/home/geng/Codes/lensing_cosmo/jax01/data/sim_data/DESa_sim_l3_w_evoPL.fits"
REPORT="${OUTDIR}/run_report_des_ann_sweep_145.md"

WEIGHTS=(0.00 0.10 0.20 0.30 0.50)

COMMON=(
  --fits "${FITS}"
  --outdir "${OUTDIR}"
  --seed 42
  --maxN 300
  --warmup 1000
  --chunk 100
  --max_chunks 1
  --target_accept 0.95
  --min_ESS 0
  --ckpt_every 0
  --cosmo_model wcdm
  --use_normal_intercept_prior
  --intercept_prior_sigma 0.08
)

mkdir -p "${OUTDIR}"

{
  echo "# DES ANN Weight Sweep (jax_check145)"
  echo
  echo "- Date: $(date -Iseconds)"
  echo "- Script: ${RUN}"
  echo "- Data: ${FITS}"
  echo "- Weights: ${WEIGHTS[*]}"
  echo "- Common args: ${COMMON[*]}"
  echo
} > "${REPORT}"

for w in "${WEIGHTS[@]}"; do
  wp="${w/./p}"
  label="des145_wcdm_ann${wp}"
  log="/tmp/${label}.log"

  cmd=("${PY}" "${RUN}" "${COMMON[@]}" --ann_weight "${w}" --npz_label "${label}")

  echo "[RUN] ann_weight=${w} label=${label}"
  {
    echo "## ann_weight=${w}"
    echo
    echo '```bash'
    printf '%q ' "${cmd[@]}"
    echo
    echo '```'
    echo
  } >> "${REPORT}"

  "${cmd[@]}" | tee "${log}"

  {
    echo '```text'
    cat "${log}"
    echo '```'
    echo
  } >> "${REPORT}"
done

"${PY}" - <<'PY' >> "${REPORT}"
import glob, os, numpy as np
outdir = "/home/geng/Codes/lensing_cosmo/jax01/output_des_sim"
paths = sorted(glob.glob(os.path.join(outdir, "posterior_minimal_chunks_des145_wcdm_ann*.npz")))

print("## Boundary Summary")
print()
print("| label | g0_edge(<1.81) | d0_edge(<1.81) | log_sig_g_top(>-0.2) | gamma0_med | delta0_med | log_sig_g_med |")
print("| --- | --- | --- | --- | --- | --- | --- |")

for p in paths:
    d = np.load(p)
    g = d["gamma0"]
    de = d["delta0"]
    lsg = d["log_sig_g"]
    label = os.path.basename(p).replace("posterior_minimal_chunks_", "").replace(".npz", "")
    print(f"| {label} | {np.mean(g<1.81):.3f} | {np.mean(de<1.81):.3f} | {np.mean(lsg>-0.2):.3f} | {np.median(g):.4f} | {np.median(de):.4f} | {np.median(lsg):.4f} |")
PY

echo "[DONE] Report: ${REPORT}"
