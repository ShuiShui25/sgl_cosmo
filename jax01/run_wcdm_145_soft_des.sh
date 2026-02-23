#!/usr/bin/env bash
set -euo pipefail

PY="/home/geng/miniforge3/envs/jax/bin/python"
RUN="/home/geng/Codes/lensing_cosmo/jax01/jax_check145.py"
OUTDIR="/home/geng/Codes/lensing_cosmo/jax01/output_des_sim"
FITS="/home/geng/Codes/lensing_cosmo/jax01/data/sim_data/DESa_sim_l3_w_evoPL.fits"
LABEL="des145_wcdm_soft_main"
REPORT="${OUTDIR}/run_report_des_wcdm_145_soft.md"
LOG="/tmp/${LABEL}.log"

CMD=(
  "${PY}" "${RUN}"
  --fits "${FITS}"
  --outdir "${OUTDIR}"
  --seed 42
  --warmup 2500
  --chunk 1200
  --max_chunks 12
  --target_accept 0.97
  --min_ESS 1000
  --ckpt_every 1
  --cosmo_model wcdm
  --ann_weight 0.10
  --ann_nu 4.0
  --use_normal_intercept_prior
  --intercept_prior_sigma 0.08
  --npz_label "${LABEL}"
)

mkdir -p "${OUTDIR}"

{
  echo "# DES wCDM run (jax_check145 soft constraints)"
  echo
  echo "- Date: $(date -Iseconds)"
  echo "- Script: ${RUN}"
  echo "- Data: ${FITS}"
  echo "- Model: wCDM only"
  echo "- Notes: linear soft constraints + robust ANN (Student-t)"
  echo
  echo "## Command"
  echo
  echo '```bash'
  printf '%q ' "${CMD[@]}"
  echo
  echo '```'
  echo
} > "${REPORT}"

# Keep progress bar in CLI; full text is still captured into markdown.
"${CMD[@]}" | tee "${LOG}"

{
  echo "## Raw Log"
  echo
  echo '```text'
  cat "${LOG}"
  echo '```'
  echo
} >> "${REPORT}"

"${PY}" - <<'PY' >> "${REPORT}"
import os, numpy as np
outdir = "/home/geng/Codes/lensing_cosmo/jax01/output_des_sim"
label = "des145_wcdm_soft_main"
npz = os.path.join(outdir, f"posterior_minimal_chunks_{label}.npz")
print("## NPZ Summary")
print()
if not os.path.exists(npz):
    print(f"Missing npz: {npz}")
else:
    d = np.load(npz)
    g, de, lsg = d["gamma0"], d["delta0"], d["log_sig_g"]
    w = d["w"]
    print(f"- npz: {npz}")
    print(f"- samples: {len(g)}")
    print(f"- g0_edge(<1.81): {np.mean(g<1.81):.3f}")
    print(f"- d0_edge(<1.81): {np.mean(de<1.81):.3f}")
    print(f"- log_sig_g_top(>-0.2): {np.mean(lsg>-0.2):.3f}")
    print(f"- gamma0 median: {np.median(g):.4f}")
    print(f"- delta0 median: {np.median(de):.4f}")
    print(f"- log_sig_g median: {np.median(lsg):.4f}")
    print(f"- w median: {np.median(w):.4f} (p16={np.quantile(w,0.16):.4f}, p84={np.quantile(w,0.84):.4f})")
PY

echo "[DONE] Report: ${REPORT}"
echo "[DONE] NPZ: ${OUTDIR}/posterior_minimal_chunks_${LABEL}.npz"
