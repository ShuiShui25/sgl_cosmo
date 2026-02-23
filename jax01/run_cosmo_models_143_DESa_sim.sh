#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT="/home/geng/Codes/lensing_cosmo/jax01/jax_check143.py"
OUTDIR="/home/geng/Codes/lensing_cosmo/jax01/output_des_sim"
REPORT="${OUTDIR}/run_report_des_sim143.md"
SUMMARY_ROWS="${OUTDIR}/summary_rows_des_sim143.txt"

FITS_WCDM="/home/geng/Codes/lensing_cosmo/jax01/data/sim_data/DESa_sim_l3_w_evoPL.fits"
FITS_W0WA="/home/geng/Codes/lensing_cosmo/jax01/data/sim_data/DESa_sim_l3_w0wa_evoPL.fits"

COMMON=(
  --outdir "$OUTDIR"
  --seed 42
  --warmup 4000
  --chunk 2000
  --max_chunks 30
  --target_accept 0.95
  --min_ESS 1000
)

mkdir -p "$OUTDIR"
: > "$SUMMARY_ROWS"

{
  echo "# Cosmology Model Run Report (jax_check143 DESa sim)"
  echo ""
  echo "- Script: $SCRIPT"
  echo "- Outdir: $OUTDIR"
  echo "- Date: $(date -Iseconds)"
  echo ""
} > "$REPORT"

run_model() {
  local label="$1"
  local model="$2"
  local fits="$3"
  local log_tmp
  log_tmp="$(mktemp)"

  {
    echo "## Run: ${label}"
    echo ""
    echo "Command:"
    echo "\`\`\`bash"
    echo "$PYTHON_BIN $SCRIPT --fits $fits ${COMMON[*]} --cosmo_model $model --npz_label $label"
    echo "\`\`\`"
    echo ""
    echo "Output:"
    echo "\`\`\`"
  } >> "$REPORT"

  "$PYTHON_BIN" "$SCRIPT" --fits "$fits" "${COMMON[@]}" --cosmo_model "$model" --npz_label "$label" \
    | tee -a "$REPORT" > "$log_tmp"

  {
    echo "\`\`\`"
    echo ""
  } >> "$REPORT"

  local step_size divs min_ess
  step_size=$(grep -E "Learned step_size=" "$log_tmp" | tail -n 1 | sed 's/\r//g')
  divs=$(grep -E "divergences \(this chunk\) = [0-9]+" "$log_tmp" | tail -n 1 | sed 's/[^0-9]//g')
  min_ess=$(grep -E "min ESS over [0-9]+ params = [0-9\.]+" "$log_tmp" | tail -n 1 | sed 's/.*= //')

  step_size=${step_size:-"(n/a)"}
  divs=${divs:-"(n/a)"}
  min_ess=${min_ess:-"(n/a)"}

  echo "| ${label} | ${step_size} | ${divs} | ${min_ess} |" >> "$SUMMARY_ROWS"

  rm -f "$log_tmp"
}

{
  echo "## WCDM"
  echo "- Fits: $FITS_WCDM"
  echo ""
} >> "$REPORT"
run_model "des_sim_wcdm" "wcdm" "$FITS_WCDM"

{
  echo "## W0WA"
  echo "- Fits: $FITS_W0WA"
  echo ""
} >> "$REPORT"
run_model "des_sim_w0wa" "w0wa" "$FITS_W0WA"

{
  echo "## Final Comparison"
  echo ""
  echo "| Run | Step Size (last) | Divergences (last) | Min ESS (last) |"
  echo "| --- | --- | --- | --- |"
  cat "$SUMMARY_ROWS"
} >> "$REPORT"
