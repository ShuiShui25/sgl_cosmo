#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fully-JAX hierarchical inference demo (single file), with:
- One-time warmup to learn step_size / inverse_mass_matrix
- Chunked sampling that reuses learned step_size / inv_mass (no further adaptation)
- Host device count configurable via CLI --host_devices
- Checkpoint/resume (stores only z, step_size, inv_mass, rng_key)
- Per-chunk diagnostics: divergences + ESS
- Early stop if either:
    (a) reached max_chunks
    (b) convergence by ESS: min(ESS over monitored params) >= --min_ESS

Data (FITS) columns expected (edit COL mapping if needed):
  zl, zs
  theta_E [arcsec]
  theta_ap [arcsec]
  sigma_ap [km/s]
  sigma_ap_err [km/s]
  dd_ANN, dd_error_ANN
"""

import os
import argparse
import pickle
import numpy as np

from astropy.table import Table
import astropy.units as u

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import effective_sample_size


# -----------------------------
# constants
# -----------------------------
C_KM_S = 299792.458  # km/s


# -----------------------------
# JAX helpers
# -----------------------------
def _safe_log(x, tiny=1e-30):
    return jnp.log(jnp.maximum(x, tiny))


def _linear_interp_1d(x, xp, fp):
    """
    JAX-friendly 1D linear interpolation.
    x: (...,)
    xp: (M,) increasing
    fp: (M,)
    returns: (...,)
    """
    x = jnp.asarray(x)
    xp = jnp.asarray(xp)
    fp = jnp.asarray(fp)

    x_clipped = jnp.clip(x, xp[0], xp[-1])
    idx = jnp.searchsorted(xp, x_clipped, side="right") - 1
    idx = jnp.clip(idx, 0, xp.shape[0] - 2)

    x0 = xp[idx]
    x1 = xp[idx + 1]
    y0 = fp[idx]
    y1 = fp[idx + 1]

    t = (x_clipped - x0) / jnp.maximum(x1 - x0, 1e-30)
    return y0 + t * (y1 - y0)


def E_z_flat_wcdm(z, Om, w):
    """
    E(z)=H(z)/H0 for flat wCDM constant w.
    """
    zp1 = 1.0 + z
    Ode = 1.0 - Om
    return jnp.sqrt(Om * zp1**3 + Ode * zp1**(3.0 * (1.0 + w)))


def build_I_of_z_grid(Om, w, z_grid):
    """
    I(z)=∫0^z dz'/E(z') on a fixed z_grid via cumulative trapezoid.
    """
    Ez = E_z_flat_wcdm(z_grid, Om, w)
    invE = 1.0 / jnp.maximum(Ez, 1e-30)
    dz = z_grid[1:] - z_grid[:-1]
    trap = 0.5 * (invE[1:] + invE[:-1]) * dz
    I = jnp.concatenate([jnp.array([0.0], dtype=z_grid.dtype), jnp.cumsum(trap)])
    return I


def dr_th_from_I(zl, zs, z_grid, I_grid):
    """
    dr_th = (I(zs)-I(zl)) / I(zs)
    zl,zs: (N,)
    """
    I_s = _linear_interp_1d(zs, z_grid, I_grid)
    I_l = _linear_interp_1d(zl, z_grid, I_grid)
    denom = jnp.maximum(I_s, 1e-30)
    return (I_s - I_l) / denom


# -----------------------------
# lensing+dynamics mapping in JAX
# -----------------------------
def f_prime_jax(g, d, beta):
    """
    Vectorized JAX version of f_prime using gammaln for stability.
    """
    g = jnp.asarray(g)
    d = jnp.asarray(d)
    beta = jnp.asarray(beta)

    pref = 1.0 / (2.0 * jnp.sqrt(jnp.pi))
    term1 = (g + d - 5.0) * (g + d - 2.0 - 2.0 * beta) / (d - 3.0)

    a = (g + d - 2.0) / 2.0
    b = (g + d) / 2.0
    ln_num = gammaln(a) + gammaln(b)

    c = (g + d - 3.0) / 2.0
    e = (g + d - 1.0) / 2.0
    A = jnp.exp(gammaln(b) + gammaln(c))
    B = beta * jnp.exp(gammaln(a) + gammaln(e))
    den = A - B

    ln_term3 = (
        gammaln((d - 1.0) / 2.0) + gammaln((g - 1.0) / 2.0)
        - gammaln(d / 2.0) - gammaln(g / 2.0)
    )
    term3 = jnp.exp(ln_term3)

    fp = pref * term1 * jnp.exp(ln_num) * term3 / jnp.maximum(den, 1e-30)
    return fp


def ln_dd_obs_jax(g, d, beta, thetaE, theta_ap, sigma_ap):
    fp = f_prime_jax(g, d, beta)
    ratio = thetaE / jnp.maximum(theta_ap, 1e-30)
    dd = (C_KM_S**2.0) / (4.0 * jnp.pi) * (thetaE / jnp.maximum(sigma_ap**2.0, 1e-30)) * (ratio ** (g - 2.0)) / jnp.maximum(fp, 1e-30)
    return _safe_log(dd)


def ln_triangular_pdf(x, left, mode, right):
    x = jnp.asarray(x)
    in_support = (x >= left) & (x <= right)

    num1 = 2.0 * (x - left)
    den1 = (right - left) * (mode - left)
    pdf1 = num1 / jnp.maximum(den1, 1e-30)

    num2 = 2.0 * (right - x)
    den2 = (right - left) * (right - mode)
    pdf2 = num2 / jnp.maximum(den2, 1e-30)

    pdf = jnp.where(x <= mode, pdf1, pdf2)
    pdf = jnp.where(in_support, pdf, 0.0)
    return _safe_log(pdf)


# -----------------------------
# Data loader (host side)
# -----------------------------
def load_fits_minimal(fits_path: str, maxN=None):
    tab = Table.read(fits_path)

    COL = dict(
        zl="zl",
        zs="zs",
        theta_E="theta_E",
        theta_ap="theta_ap",
        sigma_ap="sigma_ap",
        sigma_ap_err="sigma_ap_err",
        dr_ann="dd_ANN",
        dr_ann_err="dd_error_ANN",
    )

    if maxN is not None:
        tab = tab[: int(maxN)]

    zl = np.asarray(tab[COL["zl"]], dtype=float)
    zs = np.asarray(tab[COL["zs"]], dtype=float)

    thetaE_arcsec = np.asarray(tab[COL["theta_E"]], dtype=float)
    theta_ap_arcsec = np.asarray(tab[COL["theta_ap"]], dtype=float)

    thetaE = thetaE_arcsec * u.arcsec.to("radian")
    theta_ap = theta_ap_arcsec * u.arcsec.to("radian")

    sigma_ap = np.asarray(tab[COL["sigma_ap"]], dtype=float)
    sigma_ap_err = np.asarray(tab[COL["sigma_ap_err"]], dtype=float)

    dr_ann = np.asarray(tab[COL["dr_ann"]], dtype=float)
    dr_ann_err = np.asarray(tab[COL["dr_ann_err"]], dtype=float)

    return zl, zs, thetaE, theta_ap, sigma_ap, sigma_ap_err, dr_ann, dr_ann_err


# -----------------------------
# NumPyro model
# -----------------------------
def fullhier_model(
    zl, zs, thetaE, theta_ap, sigma_ap, sigma_ap_err, dr_ann, dr_ann_err,
    *,
    z_grid,
    d_thetaE=0.05,
    min_rel_sn=1e-4,
):
    N = zl.shape[0]

    # ---- Priors ----
    Om = numpyro.sample("Om", dist.Uniform(0.02, 0.8))
    w  = numpyro.sample("w",  dist.Uniform(-4.0, 2.0))

    gamma0 = numpyro.sample("gamma0", dist.Uniform(1.5, 2.5))
    gamma_s = numpyro.sample("gamma_s", dist.Uniform(-3.0, 3.0))
    log_sig_g = numpyro.sample("log_sig_g", dist.Uniform(-4.0, -0.1))

    delta0 = numpyro.sample("delta0", dist.Uniform(1.7, 3.0))
    delta_s = numpyro.sample("delta_s", dist.Uniform(-3.0, 3.0))
    log_sig_d = numpyro.sample("log_sig_d", dist.Uniform(-4.0, -0.1))

    beta0 = numpyro.sample("beta0", dist.Uniform(-0.8, 1.0))
    log_sig_b = numpyro.sample("log_sig_b", dist.Uniform(-10.0, -0.1))

    sig_g = jnp.exp(log_sig_g)
    sig_d = jnp.exp(log_sig_d)
    sig_b = jnp.exp(log_sig_b)

    u_g = numpyro.sample("u_g", dist.Normal(0.0, 1.0).expand([N]))
    u_d = numpyro.sample("u_d", dist.Normal(0.0, 1.0).expand([N]))
    u_b = numpyro.sample("u_b", dist.Normal(0.0, 1.0).expand([N]))

    gamma_i = gamma0 + gamma_s * zl + sig_g * u_g
    delta_i = delta0 + delta_s * zl + sig_d * u_d
    beta_i  = beta0 +            sig_b * u_b

    # ---- Hard bounds via penalty ----
    good = (
        jnp.isfinite(gamma_i) & jnp.isfinite(delta_i) & jnp.isfinite(beta_i) &
        (gamma_i > 1.01) & (gamma_i < 2.99) &
        (delta_i > 0.51) & (delta_i < 2.99) &
        (beta_i  > -0.99) & (beta_i  < 0.99) &
        jnp.isfinite(zl) & jnp.isfinite(zs) & (zs > zl) &
        (thetaE > 0.0) & (theta_ap > 0.0) &
        (sigma_ap > 0.0) & (sigma_ap_err > 0.0)
    )
    numpyro.factor("hard_bounds", jnp.sum(jnp.where(good, 0.0, -1e20)))

    # ---- dr_th via grid integral + interp ----
    I_grid = build_I_of_z_grid(Om, w, z_grid)
    dr_th = dr_th_from_I(zl, zs, z_grid, I_grid)
    ln_dr_th = _safe_log(dr_th)

    # ---- ln(dr_obs) from lensing+dynamics ----
    ln_dr_obs = ln_dd_obs_jax(gamma_i, delta_i, beta_i, thetaE, theta_ap, sigma_ap)

    # ---- Main likelihood (ln-space Gaussian) ----
    var_ln = ((gamma_i - 1.0) ** 2) * (d_thetaE ** 2) + 4.0 * (sigma_ap_err / jnp.maximum(sigma_ap, 1e-30)) ** 2
    var_ln = jnp.maximum(var_ln, 1e-30)
    main_loglike = (
        -0.5 * (ln_dr_th - ln_dr_obs) ** 2 / var_ln
        -0.5 * jnp.log(2.0 * jnp.pi * var_ln)
    )

    # ---- ANN prior in logspace ----
    rel = dr_ann_err / jnp.maximum(dr_ann, 1e-30)
    sig_ln = jnp.maximum(rel, min_rel_sn)
    ann_var = jnp.maximum(sig_ln**2, 1e-30)
    ann_loglike = (
        -0.5 * (ln_dr_th - _safe_log(dr_ann)) ** 2 / ann_var
        -0.5 * jnp.log(2.0 * jnp.pi * ann_var)
    )

    # ---- beta triangular prior per lens ----
    left, mode, right = -0.5, 0.102, 0.656
    beta_lp = ln_triangular_pdf(beta_i, left, mode, right)

    total = jnp.sum(main_loglike) + jnp.sum(ann_loglike) + jnp.sum(beta_lp)
    numpyro.factor("loglik", total)


# -----------------------------
# checkpoint I/O
# -----------------------------
def _to_host(x):
    return jax.device_get(x)


def save_checkpoint(path, *, rng_key, z, step_size, inv_mass):
    payload = {
        "rng_key": np.array(_to_host(rng_key)),
        "z": _to_host(z),
        "step_size": float(_to_host(step_size)),
        "inv_mass": _to_host(inv_mass),
    }
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def load_checkpoint(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)

    rng_key = jnp.asarray(payload["rng_key"])
    z = jax.tree_util.tree_map(lambda a: jnp.asarray(a), payload["z"])
    step_size = jnp.asarray(payload["step_size"])
    inv_mass = jax.tree_util.tree_map(lambda a: jnp.asarray(a), payload["inv_mass"])
    return rng_key, z, step_size, inv_mass


def _extract_adapt_params(state):
    """
    Robustly extract step_size and inverse_mass_matrix from NumPyro state.
    """
    adapt_state = getattr(state, "adapt_state", None)
    if adapt_state is None:
        raise RuntimeError("Could not find adapt_state in mcmc.last_state; cannot extract step_size/inv_mass.")

    step_size = getattr(adapt_state, "step_size", None)
    if step_size is None and isinstance(adapt_state, dict):
        step_size = adapt_state.get("step_size", None)

    inv_mass = getattr(adapt_state, "inverse_mass_matrix", None)
    if inv_mass is None and isinstance(adapt_state, dict):
        inv_mass = adapt_state.get("inverse_mass_matrix", None)

    if step_size is None or inv_mass is None:
        raise RuntimeError("Failed to extract step_size and/or inverse_mass_matrix from adapt_state.")

    return step_size, inv_mass


# -----------------------------
# diagnostics
# -----------------------------
def compute_ess_dict(acc_samples: dict, keys):
    """
    Compute ESS for each key using accumulated samples (single chain).
    numpyro.diagnostics.effective_sample_size requires x.ndim >= 2 with shape
    (num_chains, num_draws, ...). For 1 chain, we wrap (draws,) -> (1, draws).

    Returns: ess_per_key (dict), min_ess (float)
    """
    ess_per = {}
    min_ess = np.inf

    for k in keys:
        x = acc_samples.get(k, None)
        if x is None or len(x) < 20:
            ess = np.nan
        else:
            # ensure 1D float array on host
            x = np.asarray(x, dtype=np.float64).reshape(-1)
            # wrap to (chains=1, draws)
            x2 = jnp.asarray(x)[None, :]
            ess_val = effective_sample_size(x2)  # returns scalar array for scalar params
            ess = float(np.asarray(jax.device_get(ess_val)).reshape(-1)[0])

        ess_per[k] = ess
        if np.isfinite(ess):
            min_ess = min(min_ess, ess)

    if not np.isfinite(min_ess):
        min_ess = np.nan

    return ess_per, min_ess

def get_chunk_divergences(mcmc):
    """
    Return number of divergences in this chunk if available, else None.
    """
    try:
        extra = mcmc.get_extra_fields(group_by_chain=False)
        if "diverging" in extra:
            div = np.array(_to_host(extra["diverging"])).astype(bool)
            return int(div.sum())
    except Exception:
        pass
    return None


# -----------------------------
# driver
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", required=True, help="Input FITS table.")
    ap.add_argument("--outdir", default="out_jax_numpyro", help="Output directory.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--maxN", type=int, default=0, help="Use only first maxN lenses (0 => all).")

    # sampling control (1 chain only)
    ap.add_argument("--warmup", type=int, default=2000, help="NUTS warmup steps (only used if not resuming).")
    ap.add_argument("--chunk", type=int, default=1000, help="Samples per chunk.")
    ap.add_argument("--max_chunks", type=int, default=10, help="Max number of chunks to run.")
    ap.add_argument("--target_accept", type=float, default=0.8, help="NUTS target_accept_prob.")

    # convergence by ESS
    ap.add_argument("--min_ESS", type=float, default=0.0,
                    help="Convergence threshold: stop early if min ESS over monitored params >= min_ESS. "
                         "Set 0 to disable early stop by ESS.")

    # integration grid
    ap.add_argument("--dz", type=float, default=1e-3, help="z-grid spacing for integral I(z).")
    ap.add_argument("--zmax_pad", type=float, default=0.05, help="Pad added to max(zs) for grid upper bound.")

    # host devices
    ap.add_argument("--host_devices", type=int, default=1, help="numpyro.set_host_device_count(k)")

    # checkpoint
    ap.add_argument("--resume", action="store_true", help="Resume from checkpoint in outdir if present.")
    ap.add_argument("--ckpt_name", default="checkpoint.pkl", help="Checkpoint filename inside outdir.")
    ap.add_argument("--ckpt_every", type=int, default=1, help="Save checkpoint every N chunks.")
    ap.add_argument("--npz_label", default="wCDM", help="Checkpoint filename inside outdir.")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # MUST set host device count very early
    numpyro.set_host_device_count(int(args.host_devices))

    ckpt_path = os.path.join(args.outdir, args.ckpt_name)

    print("[INFO] JAX devices:", jax.devices())
    print("[INFO] Default backend:", jax.default_backend())
    print(f"[INFO] host_devices={args.host_devices}  (num local devices: {jax.local_device_count()})")

    maxN = None if args.maxN <= 0 else int(args.maxN)
    zl, zs, thetaE, theta_ap, sigma_ap, sigma_ap_err, dr_ann, dr_ann_err = load_fits_minimal(args.fits, maxN=maxN)

    # move data to JAX arrays
    zl = jnp.asarray(zl, dtype=jnp.float32)
    zs = jnp.asarray(zs, dtype=jnp.float32)
    thetaE = jnp.asarray(thetaE, dtype=jnp.float32)
    theta_ap = jnp.asarray(theta_ap, dtype=jnp.float32)
    sigma_ap = jnp.asarray(sigma_ap, dtype=jnp.float32)
    sigma_ap_err = jnp.asarray(sigma_ap_err, dtype=jnp.float32)
    dr_ann = jnp.asarray(dr_ann, dtype=jnp.float32)
    dr_ann_err = jnp.asarray(dr_ann_err, dtype=jnp.float32)

    N = int(zl.shape[0])
    print(f"[INFO] N_lens={N}")

    # z_grid
    zmax = float(jnp.max(zs)) + float(args.zmax_pad)
    dz = float(args.dz)
    M = int(np.ceil(zmax / dz)) + 1
    z_grid = jnp.linspace(0.0, zmax, M, dtype=jnp.float32)
    print(f"[INFO] z_grid: M={M}, zmax≈{zmax:.4f}, dz={dz:g}")

    # model closure
    def model():
        return fullhier_model(
            zl, zs, thetaE, theta_ap, sigma_ap, sigma_ap_err, dr_ann, dr_ann_err,
            z_grid=z_grid,
        )

    # run control
    rng_key = jax.random.PRNGKey(int(args.seed))
    z0 = None
    step_size = None
    inv_mass = None

    # resume if requested
    if args.resume and os.path.exists(ckpt_path):
        rng_key, z0, step_size, inv_mass = load_checkpoint(ckpt_path)
        print(f"[INFO] Resumed from checkpoint: {ckpt_path}")
        print(f"[INFO] step_size={float(_to_host(step_size)):.3e}")
        do_warmup = False
    else:
        do_warmup = True

    # monitored parameters (first 10)
    keep_keys = [
        "Om", "w",
        "gamma0", "gamma_s", "log_sig_g",
        "delta0", "delta_s", "log_sig_d",
        "beta0", "log_sig_b",
    ]

    # accumulate samples as 1D arrays for ESS
    acc = {k: np.empty((0,), dtype=np.float64) for k in keep_keys}

    stopped_by = None

    for ci in range(int(args.max_chunks)):
        print(f"\n[RUN] chunk {ci + 1}/{int(args.max_chunks)}")
        rng_key, subkey = jax.random.split(rng_key)

        if do_warmup:
            # ---------- Warmup chunk: adapt step size / mass matrix ----------
            kernel = NUTS(model, target_accept_prob=float(args.target_accept))
            mcmc = MCMC(
                kernel,
                num_warmup=int(args.warmup),
                num_samples=int(args.chunk),
                num_chains=1,
                chain_method="sequential",
                progress_bar=True,
            )
            mcmc.run(subkey)

            state = mcmc.last_state
            step_size, inv_mass = _extract_adapt_params(state)
            z0 = state.z
            do_warmup = False

            print(f"[INFO] Learned step_size={float(_to_host(step_size)):.3e} from warmup; will reuse inv_mass thereafter.")

        else:
            # ---------- Subsequent chunks: reuse warmup params; NO adaptation ----------
            kernel = NUTS(
                model,
                target_accept_prob=float(args.target_accept),
                step_size=step_size,
                inverse_mass_matrix=inv_mass,
                adapt_step_size=False,
                adapt_mass_matrix=False,
            )
            mcmc = MCMC(
                kernel,
                num_warmup=0,
                num_samples=int(args.chunk),
                num_chains=1,
                chain_method="sequential",
                progress_bar=True,
            )
            mcmc.run(subkey, init_params=z0)
            z0 = mcmc.last_state.z

        # ---- diagnostics: divergences (this chunk) ----
        n_div = get_chunk_divergences(mcmc)
        if n_div is None:
            print("[DIAG] divergences: (not available)")
        else:
            print(f"[DIAG] divergences (this chunk) = {n_div}")

        # ---- collect samples ----
        s = mcmc.get_samples(group_by_chain=False)
        for k in keep_keys:
            if k in s:
                new = np.array(_to_host(s[k]), dtype=np.float64).reshape(-1)
                acc[k] = np.concatenate([acc[k], new], axis=0)

        # ---- diagnostics: ESS (accumulated) ----
        ess_per, min_ess = compute_ess_dict(acc, keep_keys)
        # concise print
        print(f"[DIAG] ESS (accumulated): min ESS over 10 params = {min_ess:.1f}")
        # optional: print each
        # for k in keep_keys:
        #     print(f"       ESS[{k}] = {ess_per[k]:.1f}")

        # ---- checkpoint ----
        if int(args.ckpt_every) > 0 and ((ci + 1) % int(args.ckpt_every) == 0):
            save_checkpoint(
                ckpt_path,
                rng_key=rng_key,
                z=z0,
                step_size=step_size,
                inv_mass=inv_mass,
            )
            print(f"[CKPT] Saved: {ckpt_path}")

        # ---- stopping condition (either) ----
        if float(args.min_ESS) > 0.0 and np.isfinite(min_ess) and (min_ess >= float(args.min_ESS)):
            stopped_by = f"convergence: min_ess={min_ess:.1f} >= min_ESS={float(args.min_ESS):.1f}"
            print(f"[STOP] {stopped_by}")
            break

        # else: continue until max_chunks naturally

    if stopped_by is None:
        stopped_by = f"max_chunks reached ({int(args.max_chunks)})"
        print(f"\n[STOP] {stopped_by}")

    # ---- save final posterior npz (concatenated) ----
    out = {k: acc[k] for k in keep_keys}
    out_path = os.path.join(args.outdir, f"posterior_minimal_chunks_{args.npz_label}.npz")
    np.savez(out_path, **out)

    print(f"\n[DONE] Saved: {out_path}")
    print(f"[DONE] Checkpoint: {ckpt_path}")
    print(f"[DONE] Stopped by: {stopped_by}")


if __name__ == "__main__":
    main()
