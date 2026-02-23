#!/usr/bin/env bash
set -euo pipefail

PY=/home/geng/miniforge3/envs/jax/bin/python
ROOT=/home/geng/Codes/lensing_cosmo/jax01
OUT=${ROOT}/output_compare_145_146
FITS=${ROOT}/data/SGLTable_ANN_40.fits

mkdir -p "${OUT}"

COMMON=(
  --fits "${FITS}"
  --outdir "${OUT}"
  --seed 42
  --maxN 40
  --warmup 120
  --chunk 80
  --max_chunks 3
  --target_accept 0.90
  --min_ESS 0
  --cosmo_model w0wa
  --ann_weight 0.10
  --zl_scale_mode none
)

echo "[RUN] 145 baseline"
"${PY}" "${ROOT}/jax_check145.py" "${COMMON[@]}" --npz_label cmp145_annshort 2>&1 | tee "${OUT}/run_cmp145_annshort.log"

echo "[RUN] 146 ann-lens-prior"
"${PY}" "${ROOT}/jax_check146.py" "${COMMON[@]}" --npz_label cmp146_annshort 2>&1 | tee "${OUT}/run_cmp146_annshort.log"

NPZ145="${OUT}/posterior_minimal_chunks_cmp145_annshort.npz"
NPZ146="${OUT}/posterior_minimal_chunks_cmp146_annshort.npz"
MD="${OUT}/run_report_compare_145_146_ann_short.md"

"${PY}" - << 'PY2' "${NPZ145}" "${NPZ146}" "${MD}"
import numpy as np
import sys
from pathlib import Path

npz145, npz146, md = map(Path, sys.argv[1:4])

bounds = {
    'Om': (0.02, 0.8),
    'w': (-4.0, 2.0),
    'w0': (-4.0, 2.0),
    'wa': (-6.0, 6.0),
    'gamma0': (1.8, 2.5),
    'gamma_s': (-1.5, 1.5),
    'delta0': (1.8, 3.0),
    'delta_s': (-1.5, 1.5),
    'log_sig_g': (-10.0, -0.1),
    'log_sig_d': (-10.0, -0.1),
}

def load_stats(p):
    d = np.load(p)
    out = {}
    for k in d.files:
        v = np.asarray(d[k]).reshape(-1)
        if v.size == 0 or not np.issubdtype(v.dtype, np.number):
            continue
        out[k] = {
            'n': int(v.size),
            'mean': float(np.mean(v)),
            'med': float(np.median(v)),
            'q16': float(np.quantile(v, 0.16)),
            'q84': float(np.quantile(v, 0.84)),
        }
        if k in bounds:
            lo, hi = bounds[k]
            span = hi - lo
            eps = 0.02 * span
            out[k]['edge_frac'] = float(np.mean((v <= lo + eps) | (v >= hi - eps)))
    return out

s145 = load_stats(npz145)
s146 = load_stats(npz146)

cosmo = [k for k in ('Om','w','w0','wa','alpha') if k in s145 or k in s146]
lens = [k for k in ('gamma0','gamma_s','delta0','delta_s','log_sig_g','log_sig_d','beta0','log_sig_b') if k in s145 or k in s146]

lines = []
lines.append('# 145 vs 146 ANN short comparison\n')
lines.append(f'- NPZ145: `{npz145}`')
lines.append(f'- NPZ146: `{npz146}`\n')

lines.append('## Cosmology drift')
for k in cosmo:
    a, b = s145.get(k), s146.get(k)
    if not a or not b:
        continue
    dmed = b['med'] - a['med']
    line = f"- `{k}`: med 145={a['med']:.5f} -> 146={b['med']:.5f} (Δ={dmed:+.5f})"
    if 'edge_frac' in a and 'edge_frac' in b:
        line += f", edge_frac 145={a['edge_frac']:.3f}, 146={b['edge_frac']:.3f}"
    lines.append(line)

lines.append('\n## Lens-pop drift / edge')
for k in lens:
    a, b = s145.get(k), s146.get(k)
    if not a or not b:
        continue
    dmed = b['med'] - a['med']
    line = f"- `{k}`: med 145={a['med']:.5f} -> 146={b['med']:.5f} (Δ={dmed:+.5f})"
    if 'edge_frac' in a and 'edge_frac' in b:
        line += f", edge_frac 145={a['edge_frac']:.3f}, 146={b['edge_frac']:.3f}"
    lines.append(line)

md.write_text('\n'.join(lines) + '\n')
print(f'[OK] wrote {md}')
PY2

echo "[DONE] report: ${MD}"
