#!/usr/bin/env bash
set -euo pipefail

PY="/home/geng/miniforge3/envs/jax/bin/python"
RUN="/home/geng/Codes/lensing_cosmo/jax01/jax_check145.py"
OUTDIR="/home/geng/Codes/lensing_cosmo/jax01/output_des_sim"
FITS="/home/geng/Codes/lensing_cosmo/jax01/data/sim_data/DESa_sim_l3_w_evoPL.fits"
REPORT="${OUTDIR}/run_report_des_intercept_sigma_sweep_145.md"

SIGMAS=(0.08 0.10 0.12)

COMMON=(
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
)

mkdir -p "${OUTDIR}"

{
  echo "# DES intercept_prior_sigma sweep (jax_check145, wCDM)"
  echo
  echo "- Date: $(date -Iseconds)"
  echo "- Script: ${RUN}"
  echo "- Data: ${FITS}"
  echo "- Sweep: intercept_prior_sigma = ${SIGMAS[*]}"
  echo "- Truth target: delta0=2.26, delta_s=-0.16"
  echo "- Common args: ${COMMON[*]}"
  echo
} > "${REPORT}"

for s in "${SIGMAS[@]}"; do
  sp="${s/./p}"
  label="des145_wcdm_soft_sig${sp}"
  log="/tmp/${label}.log"

  cmd=("${PY}" "${RUN}" "${COMMON[@]}" --intercept_prior_sigma "${s}" --npz_label "${label}")

  echo "[RUN] intercept_prior_sigma=${s} label=${label}"
  {
    echo "## intercept_prior_sigma=${s}"
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
paths = sorted(glob.glob(os.path.join(outdir, "posterior_minimal_chunks_des145_wcdm_soft_sig*.npz")))

print("## Delta Summary")
print()
print("| label | delta0_med | delta0_p16 | delta0_p84 | delta_s_med | delta_s_p16 | delta_s_p84 |")
print("| --- | --- | --- | --- | --- | --- | --- |")
for p in paths:
    d = np.load(p)
    de0 = d["delta0"]
    des = d["delta_s"]
    label = os.path.basename(p).replace("posterior_minimal_chunks_", "").replace(".npz", "")
    print(
        f"| {label} | {np.median(de0):.4f} | {np.quantile(de0,0.16):.4f} | {np.quantile(de0,0.84):.4f} "
        f"| {np.median(des):.4f} | {np.quantile(des,0.16):.4f} | {np.quantile(des,0.84):.4f} |"
    )
PY

echo "[DONE] Report: ${REPORT}"
