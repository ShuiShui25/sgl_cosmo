#!/usr/bin/env bash
set -euo pipefail

PY="/home/geng/miniforge3/envs/jax/bin/python"
RUN="/home/geng/Codes/lensing_cosmo/jax01/jax_check146.py"
OUTDIR="/home/geng/Codes/lensing_cosmo/jax01/output"
FITS="/home/geng/Codes/lensing_cosmo/jax01/data/SGLTable_ANN.fits"
REPORT="${OUTDIR}/run_report_sgl146.md"
LOG_WCDM="/tmp/sgl146_wcdm_main.log"
LOG_W0WA="/tmp/sgl146_w0wa_main.log"

mkdir -p "${OUTDIR}"

COMMON=(
  --fits "${FITS}"
  --outdir "${OUTDIR}"
  --seed 42
  --warmup 4000
  --chunk 2000
  --max_chunks 30
  --target_accept 0.96
  --min_ESS 1000
  --ckpt_every 1
  --ann_weight 1.00
  --ann_nu 4.0
  --use_normal_intercept_prior
  --intercept_prior_sigma 0.08
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
  echo "# SGL run (jax_check146.py)"
  echo
  echo "- Date: $(date -Iseconds)"
  echo "- Script: ${RUN}"
  echo "- Data: ${FITS}"
  echo "- Output dir: ${OUTDIR}"
  echo
  echo "Runs: wcdm + w0wa"
  echo "## Commands"
  echo
} > "${REPORT}"

run_case "wcdm" "sgl146_wcdm_main" "${LOG_WCDM}"
run_case "w0wa" "sgl146_w0wa_main" "${LOG_W0WA}"

"${PY}" - <<'PY' >> "${REPORT}"
import os, numpy as np
outdir = "/home/geng/Codes/lensing_cosmo/jax01/output"
cases = [
    ("wcdm", "sgl146_wcdm_main", ["Om","w","gamma0","gamma_s","log_sig_g","delta0","delta_s","log_sig_d","beta0","log_sig_b"]),
    ("w0wa", "sgl146_w0wa_main", ["Om","w0","wa","gamma0","gamma_s","log_sig_g","delta0","delta_s","log_sig_d","beta0","log_sig_b"]),
]

print("## NPZ Summary")
print()
for model, label, keys in cases:
    npz = os.path.join(outdir, f"posterior_minimal_chunks_{label}.npz")
    print(f"### {model}")
    if not os.path.exists(npz):
        print(f"- Missing npz: {npz}")
        print()
        continue
    d = np.load(npz)
    print(f"- npz: {npz}")
    print(f"- samples: {len(d[keys[0]])}")
    print("| param | median | p16 | p84 |")
    print("| --- | --- | --- | --- |")
    for k in keys:
        x = d[k]
        print(f"| {k} | {np.median(x):.4f} | {np.quantile(x,0.16):.4f} | {np.quantile(x,0.84):.4f} |")
    print(f"- g0_edge(<1.81): {np.mean(d['gamma0']<1.81):.3f}")
    print(f"- d0_edge(<1.81): {np.mean(d['delta0']<1.81):.3f}")
    print(f"- log_sig_g_top(>-0.2): {np.mean(d['log_sig_g']>-0.2):.3f}")
    print()
PY

echo "[DONE] Report: ${REPORT}"
echo "[DONE] NPZ(wcdm): ${OUTDIR}/posterior_minimal_chunks_sgl146_wcdm_main.npz"
echo "[DONE] NPZ(w0wa): ${OUTDIR}/posterior_minimal_chunks_sgl146_w0wa_main.npz"
