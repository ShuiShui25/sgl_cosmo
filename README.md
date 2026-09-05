# sgl_cosmo

Minimal Python methods for hierarchical cosmology inference from galaxy-scale strong-lensing distance ratios, associated with Geng et al., **Hierarchical Cosmological Constraints through a Strong-lensing Distance Ratio**, ApJS **285**, 4 (2026), [doi:10.3847/1538-4365/ae6782](https://doi.org/10.3847/1538-4365/ae6782).

## What this release contains

- `scripts/infer_cosmology.py`: JAX/NumPyro hierarchical lens-population and cosmology inference.
- `scripts/run_cosmology.sh`: portable settings from the completed `sgl146` wCDM/w0wa runs.
- `scripts/summarize_posterior.py`: marginal KDE modes, medians and 16th/84th percentiles.
- `scripts/reconstruct_ann.py`: optional Pantheon+ ANN ensemble reconstruction, derived from the later `ann/ANN_cov_r` implementation.

Only methods, this README and dependencies are distributed. Supply your own input catalogs; posterior chains, FITS files, plots, notebooks and experiment reports are not included in the current source tree.

## Install

Use Python **3.11** on Linux:

```bash
git clone https://github.com/ShuiShui25/sgl_cosmo.git
cd sgl_cosmo
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The requirements install CPU JAX. For an accelerator, follow the [official JAX installation instructions](https://docs.jax.dev/en/latest/installation.html), keeping JAX and jaxlib compatible with the pinned versions. `JAX_PLATFORMS=cpu` explicitly selects CPU. A full 161-lens NUTS run can take hours; the ANN ensemble is also computationally expensive.

## Prepare a lens catalog

The shortest workflow starts with an existing FITS table containing:

| Column | Meaning / unit |
| --- | --- |
| `zl`, `zs` | Lens and source redshift; `0 < zl < zs` |
| `theta_E` | Einstein radius, arcsec |
| `theta_ap` | Equivalent circular spectroscopic aperture radius, arcsec |
| `sigma_ap`, `sigma_ap_err` | Aperture velocity dispersion and its 1-sigma uncertainty, km/s |
| `dd_ANN`, `dd_error_ANN` | ANN distance ratio D_ls/D_s and its positive uncertainty, dimensionless |

All columns must be finite; radii, dispersions, errors and ANN ratios must be positive. The script validates the input and converts arcseconds to radians. Prepare the scientific sample selection yourself. The paper uses the 161-system compilation of Chen et al. (2019), described in its Section 3. For the historical run, use the original `SGLTable_ANN.fits`; this release does not distribute it. Obtain that prepared catalog from the authors if it is not already available to you.

```bash
mkdir -p data
# Place your prepared lens catalog at data/lenses.fits.
bash scripts/run_cosmology.sh data/lenses.fits output/baseline all
# Or run one model:
bash scripts/run_cosmology.sh data/lenses.fits output/wcdm_only wcdm
```

The wrapper uses the active `python`; override it with `PYTHON=/path/to/python`. Both input and output paths are supplied by the user. Additional inference options follow the model argument. For a short installation check (not a scientific result):

```bash
JAX_PLATFORMS=cpu bash scripts/run_cosmology.sh data/lenses.fits output/smoke wcdm \
  --maxN 3 --warmup 50 --chunk 20 --max_chunks 1 --max_tree_depth 4 --min_ESS 0
```

A short run may fail the sampler quality checks; inspect its diagnostics and increase warmup rather than interpreting it as a constraint.

## Output and continuation

For each model, the wrapper creates a separate directory with:

- `posterior_minimal_chunks_MODEL.npz`: accepted samples and divergence flags; warmup excluded.
- `checkpoint_MODEL/checkpoint.pkl`: sampler state and accumulated posterior history.
- `checkpoint_MODEL/run_config.json`: settings plus input/script SHA-256 hashes.

```bash
python scripts/summarize_posterior.py output/baseline/wcdm/posterior_minimal_chunks_wcdm.npz
bash scripts/run_cosmology.sh data/lenses.fits output/baseline wcdm --resume
```

Resume uses the same command and adds `--resume`; `--max_chunks` then specifies additional chunks. Input data and sampling/model settings must match. Reusing a run directory without `--resume` is rejected. Resume only checkpoints you created locally, since the checkpoint format is Python pickle. Legacy checkpoints without sample history are not supported.

Sampling is single-chain, with chunk diagnostics for acceptance, divergences, tree depth, E-BFMI and effective sample size (ESS). Reaching the maximum chunk count is not proof of convergence. Run independent seeds in separate output directories and assess mixing and convergence before scientific use; a single-chain ESS threshold cannot establish between-chain convergence.

## Baseline and published results

The release preserves the likelihood and priors of `jax01/jax_check146.py`, whose completed chains are used by the repository's final comparison plots. The default wrapper settings are seed 42, 4000 warmup steps, 2000 samples per chunk, at most 30 chunks, target acceptance 0.96, minimum monitored ESS 1000, ANN weight 1 and Student-t degrees of freedom 4. The Einstein-radius fractional uncertainty is 0.05. Redshift is centered at the sample median without rescaling; intercept priors are truncated normals centered at gamma0=2.1 and delta0=2.2 with width 0.08.

The paper's Table 1 reports the following SGL-only constraints:

| Model | Omega_m | w / w0 | wa |
| --- | --- | --- | --- |
| wCDM | 0.32 (+0.10, -0.11) | -1.00 (+0.57, -0.97) | — |
| w0waCDM | 0.348 (+0.099, -0.099) | -1.22 (+0.79, -1.21) | -1.5 (+3.4, -3.1) |

These are published reference values, not new measurements made by this release. The local wCDM plot marks marginal peaks near Omega_m=0.3248 and w=-1.0079; the full saved chain has medians near 0.3278 and -1.3332. Modes and medians differ for skewed posteriors. The summary script reports both separately and does not shift chains or attach median-centered uncertainties to a mode.

There are known differences between available code and the published description:

- The historical code uses uniform w/w0 in [-4, 2] and wa in [-6, 6], whereas the paper states w0 in [-3, 1] and wa in [-5, 5]. These historical bounds are preserved explicitly; this is an executable historical baseline, not a claim of exact reproduction of every published assumption.
- The inference evaluates a joint lens/cosmology likelihood with a weighted Student-t ANN constraint on the lens-side distance ratio. It does not implement the separately described two-stage population-prior analysis.
- It consumes per-lens `dd_error_ANN`, not an inter-lens covariance matrix. Later files named `cov` do not change that inference assumption.
- April 14–15 covariance experiments contain interrupted runs and altered priors/input ratios, so they are not the default stable inference configuration.

This minimal release covers SGL-only inference. The paper's Planck/BAO combinations and LSST forecasts require additional data and analysis, and are not reproduced by the commands above.

## Optional: reconstruct ANN ratios from Pantheon+

Use the public [Pantheon+ data release](https://github.com/PantheonPlusSH0ES/DataRelease/tree/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR), downloading `Pantheon+SH0ES.dat` and `Pantheon+SH0ES_STAT+SYS.cov`. The table needs `zHD` and `m_b_corr`; rows with `IS_CALIBRATOR != 0` are excluded when that column is present. The covariance must correspond to the full input row order; selection is applied internally to both axes.

The following reproduces the later ANN wrapper's settings, with one worker for portability (`--n-jobs` can be increased):

```bash
python scripts/reconstruct_ann.py \
  --data data/Pantheon+SH0ES.dat \
  --cov data/Pantheon+SH0ES_STAT+SYS.cov \
  --lens-fits data/lenses.fits \
  --output-dir output/ann \
  --n-realizations 400 --cv-folds 5 --cv-bin-width 0.5 \
  --max-iter 8000 --n-jobs 1 --blas-threads 1 \
  --derivative-weight 0 --curvature-penalty-weight 0 \
  --distance-curvature-penalty-weight 0.3 --derivative-smoothing 1.2 \
  --bin-weight-strength 0.5 --bin-weight-power 1.0
bash scripts/run_cosmology.sh output/ann/SGLTable_ANN_cov.fits output/reconstructed all
```

The reconstruction draws magnitude realizations from the supplied SN covariance, fits an MLP ensemble with redshift-binned cross-validation, and computes each lens ratio from paired predictions at zl and zs. It writes a new `SGLTable_ANN_cov.fits` under `--output-dir` (or the explicit `--output-fits` path), plus reconstruction tables and covariance products. It refuses to overwrite an existing output FITS. It preserves the supplied lens columns and adds/replaces only the ANN ratio columns in the new table.

This later ANN ensemble produces a new input realization and is not verified to recreate the original publication's ANN catalog. Keep its cosmology results in a separate output directory.

For all available arguments:

```bash
python scripts/infer_cosmology.py --help
python scripts/reconstruct_ann.py --help
```

## Validation of this release

Dependencies were installed in a clean Python 3.11 CPU environment. Both cosmology models passed short sampling runs; continuation retained earlier samples. Fixed-state log densities match the original `sgl146` model. ANN input/output handling was exercised with a small synthetic catalog and a reduced CV grid. These checks establish installation and execution, not convergence of a new full 161-lens scientific run.

## License

This project is licensed under the GNU General Public License v3.0 only (`GPL-3.0-only`). See [LICENSE](LICENSE) for the full text.
