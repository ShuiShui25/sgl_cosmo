#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT="/home/geng/Codes/lensing_cosmo/jax01/jax_check140.py"
FITS="/home/geng/Codes/lensing_cosmo/jax01/data/SGLTable_ANN_40.fits"
OUTDIR="/home/geng/Codes/lensing_cosmo/jax01/output_test40"
LABEL_BASE="test40a"
REPORT="${OUTDIR}/run_report_test40a.md"
SUMMARY_ROWS="${OUTDIR}/summary_rows_${LABEL_BASE}.txt"

COMMON_ARGS=(
  --fits "$FITS"
  --outdir "$OUTDIR"
  --seed 42
  --warmup 2000
  --chunk 1000
  --max_chunks 10
  --target_accept 0.9
  --min_ESS 500
)

mkdir -p "$OUTDIR"
: > "$SUMMARY_ROWS"

# Start a fresh report for this session.
{
  echo "# Cosmology Model Run Report"
  echo ""
  echo "- Script: $SCRIPT"
  echo "- Fits: $FITS"
  echo "- Outdir: $OUTDIR"
  echo "- Label base: $LABEL_BASE"
  echo "- Date: $(date -Iseconds)"
  echo ""
} > "$REPORT"

run_model() {
  local model="$1"
  local label="${LABEL_BASE}_${model}"
  local log_tmp
  log_tmp="$(mktemp)"

  {
    echo "## Model: ${model}"
    echo ""
    echo "Command:"
    echo "\`\`\`bash"
    echo "$PYTHON_BIN $SCRIPT ${COMMON_ARGS[*]} --cosmo_model $model --npz_label $label"
    echo "\`\`\`"
    echo ""
    echo "Output:"
    echo "\`\`\`"
  } >> "$REPORT"

  # Stream all output (including warmup and every chunk) into report and temp log.
  "$PYTHON_BIN" "$SCRIPT" "${COMMON_ARGS[@]}" --cosmo_model "$model" --npz_label "$label" \
    | tee -a "$REPORT" > "$log_tmp"

  {
    echo "\`\`\`"
    echo ""
  } >> "$REPORT"

  # Extract key metrics (last occurrence).
  local step_size divs min_ess
  step_size=$(grep -E "Learned step_size=" "$log_tmp" | tail -n 1 | sed 's/\r//g')
  divs=$(grep -E "divergences \(this chunk\) = [0-9]+" "$log_tmp" | tail -n 1 | sed 's/[^0-9]//g')
  min_ess=$(grep -E "min ESS over [0-9]+ params = [0-9\.]+" "$log_tmp" | tail -n 1 | sed 's/.*= //')

  step_size=${step_size:-"(n/a)"}
  divs=${divs:-"(n/a)"}
  min_ess=${min_ess:-"(n/a)"}

  echo "| ${model} | ${step_size} | ${divs} | ${min_ess} |" >> "$SUMMARY_ROWS"

  rm -f "$log_tmp"
}

# Same session, same seed/chunk/warmup/max_chunks via COMMON_ARGS.
run_model wcdm
run_model w0wa
run_model wphi

{
  echo "## Final Comparison"
  echo ""
  echo "| Model | Step Size (last) | Divergences (last) | Min ESS (last) |"
  echo "| --- | --- | --- | --- |"
  cat "$SUMMARY_ROWS"
  echo ""
  echo "## Outputs"
  echo "- wcdm: ${OUTDIR}/posterior_minimal_chunks_${LABEL_BASE}_wcdm.npz"
  echo "- w0wa: ${OUTDIR}/posterior_minimal_chunks_${LABEL_BASE}_w0wa.npz"
  echo "- wphi: ${OUTDIR}/posterior_minimal_chunks_${LABEL_BASE}_wphi.npz"
} >> "$REPORT"
