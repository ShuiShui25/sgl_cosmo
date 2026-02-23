#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT="/home/geng/Codes/lensing_cosmo/jax01/jax_check140.py"
OUTDIR="/home/geng/Codes/lensing_cosmo/jax01/output_test40"
REPORT="${OUTDIR}/run_report_test140.md"
SUMMARY_ROWS="${OUTDIR}/summary_rows_test140.txt"

SMALL_FITS="/home/geng/Codes/lensing_cosmo/jax01/data/SGLTable_ANN_40.fits"
FULL_FITS="/home/geng/Codes/lensing_cosmo/jax01/data/SGLTable_ANN.fits"

SMALL_LABEL="test140_small"
FULL_LABEL="test140_full"

COMMON_SMALL=(
  --fits "$SMALL_FITS"
  --outdir "$OUTDIR"
  --seed 42
  --warmup 2000
  --chunk 1000
  --max_chunks 8
  --target_accept 0.9
  --min_ESS 500
)

COMMON_FULL=(
  --fits "$FULL_FITS"
  --outdir "$OUTDIR"
  --seed 42
  --warmup 4000
  --chunk 2000
  --max_chunks 30
  --target_accept 0.95
  --min_ESS 800
)

mkdir -p "$OUTDIR"
: > "$SUMMARY_ROWS"

# Start a fresh report for this run.
{
  echo "# Cosmology Model Run Report (jax_check140)"
  echo ""
  echo "- Script: $SCRIPT"
  echo "- Outdir: $OUTDIR"
  echo "- Date: $(date -Iseconds)"
  echo ""
} > "$REPORT"

run_model() {
  local label_base="$1"
  local model="$2"
  shift 2
  local -a common=("$@")
  local label="${label_base}_${model}"
  local log_tmp
  log_tmp="$(mktemp)"

  {
    echo "## Run: ${label}"
    echo ""
    echo "Command:"
    echo "\`\`\`bash"
    echo "$PYTHON_BIN $SCRIPT ${common[*]} --cosmo_model $model --npz_label $label"
    echo "\`\`\`"
    echo ""
    echo "Output:"
    echo "\`\`\`"
  } >> "$REPORT"

  # Stream output into report and temp log for summary parsing.
  "$PYTHON_BIN" "$SCRIPT" "${common[@]}" --cosmo_model "$model" --npz_label "$label" \
    | tee -a "$REPORT" > "$log_tmp"

  {
    echo "\`\`\`"
    echo ""
  } >> "$REPORT"

  # Summary row (last occurrence)
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

run_group() {
  local label_base="$1"
  shift
  local -a common=("$@")

  run_model "$label_base" wcdm "${common[@]}"
  run_model "$label_base" w0wa "${common[@]}"
  run_model "$label_base" wphi "${common[@]}"
}

{
  echo "## Small Sample"
  echo "- Fits: $SMALL_FITS"
  echo ""
} >> "$REPORT"
run_group "$SMALL_LABEL" "${COMMON_SMALL[@]}"

{
  echo "## Full Sample"
  echo "- Fits: $FULL_FITS"
  echo ""
} >> "$REPORT"
run_group "$FULL_LABEL" "${COMMON_FULL[@]}"

{
  echo "## Final Comparison"
  echo ""
  echo "| Run | Step Size (last) | Divergences (last) | Min ESS (last) |"
  echo "| --- | --- | --- | --- |"
  cat "$SUMMARY_ROWS"
} >> "$REPORT"
