#!/usr/bin/env bash
set -euo pipefail

PY="/home/geng/miniforge3/envs/jax/bin/python"
RUN="/home/geng/Codes/lensing_cosmo/jax01/jax_check140.py"
OUTDIR="/home/geng/Codes/lensing_cosmo/jax01/output"
FITS="/home/geng/Codes/lensing_cosmo/jax01/data/SGLTable_ANN.fits"
REPORT="${OUTDIR}/run_report_sgl140.md"
LOG_W0WA="/tmp/sgl140_w0wa_main.log"

mkdir -p "${OUTDIR}"

COMMON=(
  --fits "${FITS}"
  --outdir "${OUTDIR}"
  --seed 42
  --warmup 4000
  --chunk 2000
  --max_chunks 30
  --target_accept 0.96
  --min_ESS 1600
  --ckpt_every 1
)

run_case() {
  local model="$1"
  local label="$2"
  local logf="$3"
  local cmd=(
    "${PY}" "${RUN}"
    "${COMMON[@]}"
    --cosmo_model "${model}"
    --npz_label "${label}"
  )

  {
    echo "## ${model}"
    echo
    echo '```bash'
    printf '%q ' "${cmd[@]}"
    echo
    echo '```'
    echo
  } >> "${REPORT}"

  "${cmd[@]}" | tee "${logf}"

  {
    echo "### Raw Log (${model})"
    echo
    echo '```text'
    cat "${logf}"
    echo '```'
    echo
  } >> "${REPORT}"
}

{
  echo "# SGL run (jax_check140)"
  echo
  echo "- Date: $(date -Iseconds)"
  echo "- Script: ${RUN}"
  echo "- Data: ${FITS}"
  echo "- Output dir: ${OUTDIR}"
  echo
  echo "Runs: w0wa"
  echo "## Commands"
  echo
} > "${REPORT}"

run_case "w0wa" "sgl140_w0wa_main" "${LOG_W0WA}"

"${PY}" - <<'PY' >> "${REPORT}"
import os
import numpy as np

outdir = "/home/geng/Codes/lensing_cosmo/jax01/output"
label = "sgl140_w0wa_main"
keys = ["Om", "w0", "wa", "gamma0", "gamma_s", "log_sig_g", "delta0", "delta_s", "log_sig_d", "beta0", "log_sig_b"]
npz = os.path.join(outdir, f"posterior_minimal_chunks_{label}.npz")

print("## NPZ Summary")
print()
if not os.path.exists(npz):
    print(f"- Missing npz: {npz}")
else:
    d = np.load(npz)
    print(f"- npz: {npz}")
    print(f"- samples: {len(d[keys[0]])}")
    print("| param | median | p16 | p84 |")
    print("| --- | --- | --- | --- |")
    for k in keys:
        x = d[k]
        print(f"| {k} | {np.median(x):.4f} | {np.quantile(x, 0.16):.4f} | {np.quantile(x, 0.84):.4f} |")
PY

echo "[DONE] Report: ${REPORT}"
echo "[DONE] NPZ(w0wa): ${OUTDIR}/posterior_minimal_chunks_sgl140_w0wa_main.npz"
