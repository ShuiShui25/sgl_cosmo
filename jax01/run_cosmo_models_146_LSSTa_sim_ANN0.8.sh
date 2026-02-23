#!/usr/bin/env bash
set -euo pipefail

PY="/home/geng/miniforge3/envs/jax/bin/python"
RUN="/home/geng/Codes/lensing_cosmo/jax01/jax_check146.py"
OUTDIR="/home/geng/Codes/lensing_cosmo/jax01/output_lsst_sim"
FITS_WCDM="/home/geng/Codes/lensing_cosmo/jax01/data/sim_data/LSSTa_sim_l3_w_evoPL.fits"
FITS_W0WA="/home/geng/Codes/lensing_cosmo/jax01/data/sim_data/LSSTa_sim_l3_w0wa_evoPL.fits"
REPORT="${OUTDIR}/run_report_lsst_sim146_ann0.80.md"

COMMON=(
  --outdir "${OUTDIR}"
  --seed 42
  --warmup 4000
  --chunk 2000
  --max_chunks 30
  --target_accept 0.95
  --min_ESS 2000
  --ckpt_every 1
  --ann_weight 0.80
  --ann_nu 2.1
  --use_normal_intercept_prior
  --intercept_prior_sigma 0.08
)

mkdir -p "${OUTDIR}"

{
  echo "# LSSTa sim run (jax_check146)"
  echo
  echo "- Date: $(date -Iseconds)"
  echo "- Script: ${RUN}"
  echo "- Settings: soft constraints + robust ANN Student-t"
  echo "- Fixed: intercept_prior_sigma=0.08"
  echo "- Common args: ${COMMON[*]}"
  echo
} > "${REPORT}"

run_case () {
  local model="$1"
  local fits="$2"
  local label="$3"
  local log="/tmp/${label}.log"

  local cmd=("${PY}" "${RUN}" "${COMMON[@]}" --fits "${fits}" --cosmo_model "${model}" --npz_label "${label}")

  echo "[RUN] model=${model} label=${label}"

  {
    echo "## ${model}"
    echo
    echo "- fits: ${fits}"
    echo "- label: ${label}"
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
}

run_case "wcdm" "${FITS_WCDM}" "lsst146_wcdm_ann080"
run_case "w0wa" "${FITS_W0WA}" "lsst146_w0wa_ann080"

"${PY}" - <<'PY' >> "${REPORT}"
import os, numpy as np
outdir = "/home/geng/Codes/lensing_cosmo/jax01/output_lsst_sim"
cases = [
    ("wcdm", "lsst146_wcdm_ann080", ["Om", "w", "gamma0", "gamma_s", "delta0", "delta_s"]),
    ("w0wa", "lsst146_w0wa_ann080", ["Om", "w0", "wa", "gamma0", "gamma_s", "delta0", "delta_s"]),
]

print("## Posterior Summary")
print()
for name, label, keys in cases:
    npz = os.path.join(outdir, f"posterior_minimal_chunks_{label}.npz")
    print(f"### {name}")
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
    if "gamma0" in d and "delta0" in d and "log_sig_g" in d:
        g = d["gamma0"]
        de = d["delta0"]
        lsg = d["log_sig_g"]
        print(f"- g0_edge(<1.81): {np.mean(g<1.81):.3f}")
        print(f"- d0_edge(<1.81): {np.mean(de<1.81):.3f}")
        print(f"- log_sig_g_top(>-0.2): {np.mean(lsg>-0.2):.3f}")
    print()
PY

echo "[DONE] Report: ${REPORT}"
