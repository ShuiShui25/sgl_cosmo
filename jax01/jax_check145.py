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

Cosmology models:
  --cosmo_model wcdm   : flat wCDM with constant w (params: Om, w)
  --cosmo_model w0wa   : flat w0waCDM (CPL)         (params: Om, w0, wa)
  --cosmo_model wphi   : flat w_ϕCDM (thawing)      (params: Om, w0, alpha)

In E(z) we optionally include a fixed radiation density Ωr via CLI --Omega_r (default 0).

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
from collections import deque

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


# -----------------------------
# Cosmology: E(z) for flat models
# -----------------------------
def E_z_flat_wcdm(z, Om, w, Or=0.0):
    """
    E(z)=H(z)/H0 for (nearly) flat wCDM constant w, with optional radiation Ωr:

      E^2(z)=Ωr(1+z)^4 + Ωm(1+z)^3 + Ωde(1+z)^{3(1+w)}

    where Ωde=1-Ωm-Ωr.
    """
    zp1 = 1.0 + z
    Or = jnp.asarray(Or)
    Ode = 1.0 - Om - Or
    E2 = Or * zp1**4 + Om * zp1**3 + Ode * zp1**(3.0 * (1.0 + w))
    return jnp.sqrt(jnp.maximum(E2, 1e-30))


def E_z_flat_w0wa(z, Om, w0, wa, Or=0.0):
    """
    E(z)=H(z)/H0 for flat CPL w0waCDM (with optional radiation Ωr):

      w(z)=w0 + wa*z/(1+z)
      rho_de(z)/rho_de0 = (1+z)^{3(1+w0+wa)} * exp[-3 wa z/(1+z)]

    Thus
      E^2(z)=Ωr(1+z)^4 + Ωm(1+z)^3 + Ωde * rho_de(z)/rho_de0

    where Ωde=1-Ωm-Ωr.
    """
    zp1 = 1.0 + z
    Or = jnp.asarray(Or)
    Ode = 1.0 - Om - Or
    f_de = zp1 ** (3.0 * (1.0 + w0 + wa)) * jnp.exp(-3.0 * wa * z / jnp.maximum(zp1, 1e-30))
    E2 = Or * zp1**4 + Om * zp1**3 + Ode * f_de
    return jnp.sqrt(jnp.maximum(E2, 1e-30))


def E_z_flat_wphi(z, Om, w0, alpha, Or=0.0):
    """
    E(z)=H(z)/H0 for flat w_ϕCDM (thawing), matching your figure:

      w_ϕ(z) = -1 + (1+w0) * (1/(1+z))^alpha = -1 + (1+w0)(1+z)^(-alpha)

      E^2(z) = Ωr(1+z)^4 + Ωm(1+z)^3
               + Ωϕ * exp[ 3(1+w0)/alpha * ( 1 - (1+z)^(-alpha) ) ] ,

    where Ωϕ = 1 - Ωm - Ωr.

    We also implement the alpha -> 0 limit to avoid division by ~0:
      alpha -> 0  =>  w(z) -> w0 (constant)  =>  rho_de(z) ∝ (1+z)^{3(1+w0)}.
    """
    zp1 = 1.0 + z
    Or = jnp.asarray(Or)
    Ode = 1.0 - Om - Or

    alpha = jnp.asarray(alpha)

    f_de_lim = zp1 ** (3.0 * (1.0 + w0))
    f_de_gen = jnp.exp(3.0 * (1.0 + w0) / jnp.maximum(alpha, 1e-12) * (1.0 - zp1 ** (-alpha)))
    f_de = jnp.where(jnp.abs(alpha) < 1e-6, f_de_lim, f_de_gen)

    E2 = Or * zp1**4 + Om * zp1**3 + Ode * f_de
    return jnp.sqrt(jnp.maximum(E2, 1e-30))


def build_I_of_z_grid(cosmo_model, cosmo_params, z_grid):
    """
    I(z)=∫0^z dz'/E(z') on a fixed z_grid via cumulative trapezoid.

    cosmo_model: "wcdm" / "w0wa" / "wphi"
    cosmo_params:
      - wcdm: (Om, w, Or)
      - w0wa: (Om, w0, wa, Or)
      - wphi: (Om, w0, alpha, Or)
    """
    if cosmo_model == "wcdm":
        Om, w, Or = cosmo_params
        Ez = E_z_flat_wcdm(z_grid, Om, w, Or=Or)
    elif cosmo_model == "w0wa":
        Om, w0, wa, Or = cosmo_params
        Ez = E_z_flat_w0wa(z_grid, Om, w0, wa, Or=Or)
    elif cosmo_model == "wphi":
        Om, w0, alpha, Or = cosmo_params
        Ez = E_z_flat_wphi(z_grid, Om, w0, alpha, Or=Or)
    else:
        raise ValueError(f"Unknown cosmo_model={cosmo_model}")

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
    cosmo_model="wcdm",
    Omega_r=0.0,
    d_thetaE=0.05,
    min_rel_sn=1e-4,
    ann_weight=0.10,
    ann_nu=4.0,
    zl_scale_mode="none",
    use_normal_slope_prior=False,
    slope_prior_sigma=0.5,
    use_normal_intercept_prior=True,
    intercept_prior_sigma=0.08,
    log_sig_g_max=-0.1,
    log_sig_d_max=-0.1,
):
    N = zl.shape[0]

    # ---- Cosmology priors ----
    Om = numpyro.sample("Om", dist.Uniform(0.02, 0.8))

    # Fixed radiation density (default 0). Keep it in cosmo_params so E(z) matches the figure when desired.
    Or = jnp.asarray(Omega_r, dtype=zl.dtype)

    if cosmo_model == "wcdm":
        w = numpyro.sample("w", dist.Uniform(-4.0, 2.0))
        cosmo_params = (Om, w, Or)
    elif cosmo_model == "w0wa":
        w0 = numpyro.sample("w0", dist.Uniform(-4.0, 2.0))
        wa = numpyro.sample("wa", dist.Uniform(-6.0, 6.0))
        cosmo_params = (Om, w0, wa, Or)
    elif cosmo_model == "wphi":
        # w_phiCDM: params (Om, w0, alpha)
        w0 = numpyro.sample("w0", dist.Uniform(-4.0, 2.0))
        # alpha>0 is typical for this parameterization; keep it away from exactly 0 for numerical stability.
        alpha = numpyro.sample("alpha", dist.Uniform(0.35, 0.55))
        cosmo_params = (Om, w0, alpha, Or)
    else:
        raise ValueError(f"Unknown cosmo_model={cosmo_model}. Use 'wcdm', 'w0wa', or 'wphi'.")

    # ---- Lens population priors ----
    if use_normal_intercept_prior:
        gamma0 = numpyro.sample("gamma0", dist.TruncatedNormal(loc=2.1, scale=intercept_prior_sigma, low=1.8, high=2.5))
        delta0 = numpyro.sample("delta0", dist.TruncatedNormal(loc=2.2, scale=intercept_prior_sigma, low=1.8, high=3.0))
    else:
        gamma0 = numpyro.sample("gamma0", dist.Uniform(1.8, 2.5))
        delta0 = numpyro.sample("delta0", dist.Uniform(1.8, 3.0))

    if use_normal_slope_prior:
        gamma_s = numpyro.sample("gamma_s", dist.Normal(0.0, slope_prior_sigma))
        delta_s = numpyro.sample("delta_s", dist.Normal(0.0, slope_prior_sigma))
    else:
        gamma_s = numpyro.sample("gamma_s", dist.Uniform(-1.5, 1.5))
        delta_s = numpyro.sample("delta_s", dist.Uniform(-1.5, 1.5))

    log_sig_g = numpyro.sample("log_sig_g", dist.Uniform(-10.0, log_sig_g_max))
    log_sig_d = numpyro.sample("log_sig_d", dist.Uniform(-10.0, log_sig_d_max))

    beta0 = numpyro.sample("beta0", dist.Uniform(-0.8, 1.0))
    log_sig_b = numpyro.sample("log_sig_b", dist.Uniform(-10.0, -0.1))

    sig_g = jnp.exp(log_sig_g)
    sig_d = jnp.exp(log_sig_d)
    sig_b = jnp.exp(log_sig_b)

    u_g = numpyro.sample("u_g", dist.Normal(0.0, 1.0).expand([N]))
    u_d = numpyro.sample("u_d", dist.Normal(0.0, 1.0).expand([N]))
    u_b = numpyro.sample("u_b", dist.Normal(0.0, 1.0).expand([N]))

    # Only difference vs jax_check141:
    # use differentiable bounded maps for gamma/delta to avoid hard-bound cliffs.
    # Center (and optionally scale) zl to reduce intercept-slope degeneracy.
    zl_center = jnp.median(zl)
    zlc = zl - zl_center
    if zl_scale_mode == "std":
        zl_scale = jnp.maximum(jnp.std(zl), 1e-6)
        zlc = zlc / zl_scale
    elif zl_scale_mode == "mad":
        zl_scale = jnp.maximum(1.4826 * jnp.median(jnp.abs(zlc)), 1e-6)
        zlc = zlc / zl_scale
    elif zl_scale_mode == "none":
        pass
    else:
        raise ValueError(f"Unknown zl_scale_mode={zl_scale_mode}")

    gamma_i = gamma0 + gamma_s * zlc + sig_g * u_g
    delta_i = delta0 + delta_s * zlc + sig_d * u_d
    beta_i  = beta0 +            sig_b * u_b

    # ---- Soft constraints to avoid singular regions while keeping differentiability ----
    soft_eps = 0.02
    g_min = 1.02
    d_min = 1.02
    gd_min = 3.02
    d3_margin = 0.05
    beta_max = 0.99

    pen_g = jax.nn.softplus((g_min - gamma_i) / soft_eps)
    pen_d = jax.nn.softplus((d_min - delta_i) / soft_eps)
    pen_gd = jax.nn.softplus((gd_min - (gamma_i + delta_i)) / soft_eps)
    pen_d3 = jax.nn.softplus((d3_margin - jnp.abs(delta_i - 3.0)) / soft_eps)
    pen_beta = jax.nn.softplus((jnp.abs(beta_i) - beta_max) / soft_eps)

    soft_penalty = jnp.sum(pen_g + pen_d + pen_gd + pen_d3 + pen_beta)
    numpyro.factor("soft_constraints", -25.0 * soft_penalty)

    # Keep hard checks only for invalid data/non-finite states
    good = (
        jnp.isfinite(gamma_i) & jnp.isfinite(delta_i) & jnp.isfinite(beta_i) &
        jnp.isfinite(zl) & jnp.isfinite(zs) & (zs > zl) &
        (thetaE > 0.0) & (theta_ap > 0.0) &
        (sigma_ap > 0.0) & (sigma_ap_err > 0.0)
    )
    numpyro.factor("hard_checks", jnp.sum(jnp.where(good, 0.0, -1e20)))

    # ---- dr_th via grid integral + interp ----
    I_grid = build_I_of_z_grid(cosmo_model, cosmo_params, z_grid)
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

    # ---- ANN prior in logspace (robust Student-t) ----
    rel = dr_ann_err / jnp.maximum(dr_ann, 1e-30)
    sig_ln = jnp.maximum(rel, min_rel_sn)
    ann_var = jnp.maximum(sig_ln**2, 1e-30)
    ann_std = jnp.sqrt(ann_var)
    ann_resid = (ln_dr_th - _safe_log(dr_ann)) / jnp.maximum(ann_std, 1e-30)
    ann_nu_eff = jnp.maximum(jnp.asarray(ann_nu, dtype=zl.dtype), 2.1)
    ann_loglike = dist.StudentT(df=ann_nu_eff, loc=0.0, scale=1.0).log_prob(ann_resid) - jnp.log(jnp.maximum(ann_std, 1e-30))

    # ---- beta triangular prior per lens ----
    left, mode, right = -0.5, 0.102, 0.656
    beta_lp = ln_triangular_pdf(beta_i, left, mode, right)

    main_loglike_sum = jnp.sum(main_loglike)
    ann_loglike_sum = jnp.sum(ann_loglike)
    numpyro.deterministic("main_loglike_sum", main_loglike_sum)
    numpyro.deterministic("ann_loglike_sum", ann_loglike_sum)

    total = main_loglike_sum + ann_weight * ann_loglike_sum + jnp.sum(beta_lp)
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

    Returns:
      ess_per_key (dict),
      min_ess (float),
      min3 (list of tuples): (param_index, param_name, ess)
    """
    ess_per = {}
    min_ess = np.inf

    for k in keys:
        x = acc_samples.get(k, None)
        if x is None or len(x) < 20:
            ess = np.nan
        else:
            x = np.asarray(x, dtype=np.float64).reshape(-1)
            x2 = jnp.asarray(x)[None, :]
            ess_val = effective_sample_size(x2)
            ess = float(np.asarray(jax.device_get(ess_val)).reshape(-1)[0])

        ess_per[k] = ess
        if np.isfinite(ess):
            min_ess = min(min_ess, ess)

    if not np.isfinite(min_ess):
        min_ess = np.nan

    finite = [
        (i, k, ess_per[k])
        for i, k in enumerate(keys)
        if np.isfinite(ess_per[k])
    ]
    finite.sort(key=lambda t: t[2])
    min3 = finite[:3]

    return ess_per, min_ess, min3


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


def get_chunk_extra_diagnostics(mcmc, max_tree_depth=10):
    """
    Return per-chunk extra diagnostics if available:
      - accept_mean
      - tree_depth_mean
      - ebfmi
      - max_tree_depth_hit (fraction in [0,1])
    """
    accept_mean = None
    tree_depth_mean = None
    ebfmi = None
    max_tree_depth_hit = None

    try:
        extra = mcmc.get_extra_fields(group_by_chain=False)
    except Exception:
        return accept_mean, tree_depth_mean, ebfmi, max_tree_depth_hit

    if "accept_prob" in extra:
        ap = np.array(_to_host(extra["accept_prob"]), dtype=np.float64).reshape(-1)
        if ap.size > 0:
            accept_mean = float(np.mean(ap))
    elif "acceptance_rate" in extra:
        ap = np.array(_to_host(extra["acceptance_rate"]), dtype=np.float64).reshape(-1)
        if ap.size > 0:
            accept_mean = float(np.mean(ap))

    if "tree_depth" in extra:
        td = np.array(_to_host(extra["tree_depth"]), dtype=np.float64).reshape(-1)
        if td.size > 0:
            tree_depth_mean = float(np.mean(td))
            max_tree_depth_hit = float(np.mean(td >= float(max_tree_depth)))
    elif "num_steps" in extra:
        ns = np.array(_to_host(extra["num_steps"]), dtype=np.float64).reshape(-1)
        ns = ns[ns > 0]
        if ns.size > 0:
            # Approximate tree depth from leapfrog step count.
            tree_depth_mean = float(np.mean(np.log2(ns) + 1.0))
            max_steps = float((2 ** int(max_tree_depth)) - 1)
            max_tree_depth_hit = float(np.mean(ns >= max_steps))

    if "energy" in extra:
        e = np.array(_to_host(extra["energy"]), dtype=np.float64).reshape(-1)
        if e.size >= 2:
            var_e = float(np.var(e))
            if var_e > 0.0:
                de = np.diff(e)
                ebfmi = float(np.var(de) / var_e)

    return accept_mean, tree_depth_mean, ebfmi, max_tree_depth_hit


# -----------------------------
# driver
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", required=True, help="Input FITS table.")
    ap.add_argument("--outdir", default="out_jax_numpyro", help="Output directory.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--maxN", type=int, default=0, help="Use only first maxN lenses (0 => all).")

    # cosmology switch
    ap.add_argument("--cosmo_model", default="wcdm", choices=["wcdm", "w0wa", "wphi"],
                    help="Cosmology model: wcdm (Om,w) / w0wa (Om,w0,wa) / wphi (Om,w0,alpha).")
    ap.add_argument("--Omega_r", type=float, default=0.0,
                    help="Fixed radiation density Ωr used in E(z). Default 0.0 (ignored).")

    # sampling control (1 chain only)
    ap.add_argument("--warmup", type=int, default=2000, help="NUTS warmup steps (only used if not resuming).")
    ap.add_argument("--chunk", type=int, default=1000, help="Samples per chunk.")
    ap.add_argument("--max_chunks", type=int, default=10, help="Max number of chunks to run.")
    ap.add_argument("--target_accept", type=float, default=0.8, help="NUTS target_accept_prob.")
    ap.add_argument("--max_tree_depth", type=int, default=10, help="NUTS max_tree_depth.")
    ap.add_argument("--div_rate_thresh", type=float, default=0.1, help="Drop chunk if divergence rate exceeds this threshold.")
    ap.add_argument("--accept_mean_min", type=float, default=0.8, help="Drop chunk if acceptance mean is below this threshold.")
    ap.add_argument("--bad_chunk_warmup", type=int, default=500, help="Warmup steps for re-adaptation after a bad chunk.")
    ap.add_argument("--recent_chunks", type=int, default=3, help="Number of recent accepted chunks used for ESS_recent.")
    ap.add_argument("--ann_weight", type=float, default=0.10, help="Weight multiplier for ANN log-likelihood term.")
    ap.add_argument("--ann_nu", type=float, default=4.0, help="Degrees of freedom for robust Student-t ANN log-likelihood (>=2.1).")
    ap.add_argument("--zl_scale_mode", choices=["none", "std", "mad"], default="none",
                    help="Optional scaling for centered zl used in slope parameterization.")
    ap.add_argument("--use_normal_slope_prior", action="store_true",
                    help="Use Normal(0, slope_prior_sigma) for gamma_s and delta_s.")
    ap.add_argument("--slope_prior_sigma", type=float, default=0.5,
                    help="Std for Normal slope priors when --use_normal_slope_prior is enabled.")
    ap.add_argument("--use_normal_intercept_prior", action=argparse.BooleanOptionalAction, default=True,
                    help="Use truncated normal priors for gamma0 and delta0.")
    ap.add_argument("--intercept_prior_sigma", type=float, default=0.08,
                    help="Std for truncated normal intercept priors when enabled.")
    ap.add_argument("--log_sig_g_max", type=float, default=-0.1,
                    help="Upper bound for log_sig_g prior Uniform(-10, log_sig_g_max).")
    ap.add_argument("--log_sig_d_max", type=float, default=-0.1,
                    help="Upper bound for log_sig_d prior Uniform(-10, log_sig_d_max).")

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
    ap.add_argument("--npz_label", default="", help="Label suffix for posterior npz filename (optional).")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    EXTRA_FIELDS = ("accept_prob", "energy", "num_steps", "diverging")

    def _make_mcmc(kernel, num_warmup, num_samples):
        kwargs = dict(
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=1,
            chain_method="sequential",
            progress_bar=True,
        )
        return MCMC(kernel, **kwargs)

    # MUST set host device count very early
    numpyro.set_host_device_count(int(args.host_devices))

    ckpt_dir = os.path.join(args.outdir, f"checkpoint_{args.npz_label}")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, args.ckpt_name)

    print("[INFO] JAX devices:", jax.devices())
    print("[INFO] Default backend:", jax.default_backend())
    print(f"[INFO] host_devices={args.host_devices}  (num local devices: {jax.local_device_count()})")
    print(f"[INFO] cosmo_model={args.cosmo_model}")
    print(f"[INFO] Omega_r={float(args.Omega_r):.3e}")
    print(f"[INFO] ann_weight={float(args.ann_weight):.3f}, ann_nu={float(args.ann_nu):.2f}, zl_scale_mode={args.zl_scale_mode}")
    print(f"[INFO] priors: normal_slope={args.use_normal_slope_prior}, normal_intercept={args.use_normal_intercept_prior}, "
          f"log_sig_g_max={float(args.log_sig_g_max):.2f}, log_sig_d_max={float(args.log_sig_d_max):.2f}")

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
            cosmo_model=args.cosmo_model,
            Omega_r=float(args.Omega_r),
            ann_weight=float(args.ann_weight),
            ann_nu=float(args.ann_nu),
            zl_scale_mode=args.zl_scale_mode,
            use_normal_slope_prior=bool(args.use_normal_slope_prior),
            slope_prior_sigma=float(args.slope_prior_sigma),
            use_normal_intercept_prior=bool(args.use_normal_intercept_prior),
            intercept_prior_sigma=float(args.intercept_prior_sigma),
            log_sig_g_max=float(args.log_sig_g_max),
            log_sig_d_max=float(args.log_sig_d_max),
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

    # monitored parameters (dynamic by cosmology model)
    if args.cosmo_model == "wcdm":
        cosmo_keys = ["Om", "w"]
        label_default = "wCDM"
    elif args.cosmo_model == "w0wa":
        cosmo_keys = ["Om", "w0", "wa"]
        label_default = "w0waCDM"
    else:  # wphi
        cosmo_keys = ["Om", "w0", "alpha"]
        label_default = "wphiCDM"

    keep_keys = cosmo_keys + [
        "gamma0", "gamma_s", "log_sig_g",
        "delta0", "delta_s", "log_sig_d",
        "beta0", "log_sig_b",
    ]

    # accumulate samples as 1D arrays for ESS
    acc = {k: np.empty((0,), dtype=np.float64) for k in keep_keys}
    acc_div = np.empty((0,), dtype=bool)

    stopped_by = None
    force_readapt_next = False
    recent_queue = deque(maxlen=max(1, int(args.recent_chunks)))

    for ci in range(int(args.max_chunks)):
        print(f"\n[RUN] chunk {ci + 1}/{int(args.max_chunks)}")
        rng_key, subkey = jax.random.split(rng_key)

        if do_warmup or force_readapt_next:
            # ---------- Warmup chunk: adapt step size / mass matrix ----------
            warm_steps = int(args.warmup) if do_warmup else int(args.bad_chunk_warmup)
            if force_readapt_next:
                print(f"[INFO] Re-adapting after bad chunk: warmup={warm_steps}")
            kernel = NUTS(
                model,
                target_accept_prob=float(args.target_accept),
                max_tree_depth=int(args.max_tree_depth),
            )
            mcmc = _make_mcmc(kernel, warm_steps, int(args.chunk))
            mcmc.run(subkey, extra_fields=EXTRA_FIELDS)

            state = mcmc.last_state
            step_size, inv_mass = _extract_adapt_params(state)
            z0 = state.z
            do_warmup = False
            force_readapt_next = False

            print(f"[INFO] Learned step_size={float(_to_host(step_size)):.3e} from warmup; will reuse inv_mass thereafter.")

        else:
            # ---------- Subsequent chunks: reuse warmup params; NO adaptation ----------
            kernel = NUTS(
                model,
                target_accept_prob=float(args.target_accept),
                max_tree_depth=int(args.max_tree_depth),
                step_size=step_size,
                inverse_mass_matrix=inv_mass,
                adapt_step_size=False,
                adapt_mass_matrix=False,
            )
            mcmc = _make_mcmc(kernel, 0, int(args.chunk))
            try:
                mcmc.run(subkey, extra_fields=EXTRA_FIELDS, init_params=z0)
                z0 = mcmc.last_state.z
            except RuntimeError as err:
                # Continue robustly if previous unconstrained state is invalid.
                if "Cannot find valid initial parameters" not in str(err):
                    raise
                print("[WARN] Invalid init_params from previous chunk; retrying with fresh warmup.")
                rng_key, retry_key = jax.random.split(rng_key)
                kernel = NUTS(
                    model,
                    target_accept_prob=float(args.target_accept),
                    max_tree_depth=int(args.max_tree_depth),
                )
                mcmc = _make_mcmc(kernel, int(args.warmup), int(args.chunk))
                mcmc.run(retry_key, extra_fields=EXTRA_FIELDS)
                state = mcmc.last_state
                step_size, inv_mass = _extract_adapt_params(state)
                z0 = state.z
                print(f"[INFO] Relearned step_size={float(_to_host(step_size)):.3e} after fallback warmup.")

        # ---- diagnostics: divergences (this chunk) ----
        n_div = get_chunk_divergences(mcmc)
        if n_div is None:
            print("[DIAG] divergences: (not available)")
        else:
            print(f"[DIAG] divergences (this chunk) = {n_div}")
        accept_mean, tree_depth_mean, ebfmi, max_td_hit = get_chunk_extra_diagnostics(
            mcmc, max_tree_depth=int(args.max_tree_depth)
        )
        if accept_mean is None:
            print("[DIAG] acceptance mean: (not available)")
        else:
            print(f"[DIAG] acceptance mean: {accept_mean:.4f}")
        if tree_depth_mean is None:
            print("[DIAG] tree depth mean: (not available)")
        else:
            print(f"[DIAG] tree depth mean: {tree_depth_mean:.3f}")
        if ebfmi is None:
            print("[DIAG] E-BFMI (this chunk): (not available)")
        else:
            print(f"[DIAG] E-BFMI (this chunk): {ebfmi:.4f}")
        if max_td_hit is None:
            print("[DIAG] max_tree_depth hit rate: (not available)")
        else:
            print(f"[DIAG] max_tree_depth hit rate: {max_td_hit:.3f}")

        # ---- chunk quality gating ----
        is_bad_chunk = False
        bad_reasons = []
        if n_div is not None:
            div_rate = float(n_div) / float(int(args.chunk))
            if div_rate > float(args.div_rate_thresh):
                is_bad_chunk = True
                bad_reasons.append(f"div_rate={div_rate:.3f}>{float(args.div_rate_thresh):.3f}")
        if accept_mean is not None and accept_mean < float(args.accept_mean_min):
            is_bad_chunk = True
            bad_reasons.append(f"accept_mean={accept_mean:.3f}<{float(args.accept_mean_min):.3f}")

        s = mcmc.get_samples(group_by_chain=False)
        extra = mcmc.get_extra_fields(group_by_chain=False)
        div = None
        if "diverging" in extra:
            div = np.array(_to_host(extra["diverging"])).astype(bool).reshape(-1)

        if "main_loglike_sum" in s:
            main_sum = np.asarray(_to_host(s["main_loglike_sum"]), dtype=np.float64).reshape(-1)
            print(f"[DIAG] sum(main_loglike): mean={main_sum.mean():.3f}, last={main_sum[-1]:.3f}")
        else:
            print("[DIAG] sum(main_loglike): (not available)")

        if "ann_loglike_sum" in s:
            ann_sum = np.asarray(_to_host(s["ann_loglike_sum"]), dtype=np.float64).reshape(-1)
            print(f"[DIAG] sum(ann_loglike): mean={ann_sum.mean():.3f}, last={ann_sum[-1]:.3f}")
        else:
            print("[DIAG] sum(ann_loglike): (not available)")

        if is_bad_chunk:
            print(f"[GATE] dropped bad chunk: {'; '.join(bad_reasons)}")
            force_readapt_next = True
            continue

        # ---- collect accepted chunk samples ----
        if div is not None:
            acc_div = np.concatenate([acc_div, div], axis=0)

        chunk_payload = {}
        for k in keep_keys:
            if k in s:
                new = np.array(_to_host(s[k]), dtype=np.float64).reshape(-1)
                acc[k] = np.concatenate([acc[k], new], axis=0)
                chunk_payload[k] = new
        recent_queue.append(chunk_payload)

        # ---- diagnostics: ESS dual-track ----
        ess_per_all, min_ess_all, min3_all = compute_ess_dict(acc, keep_keys)
        print(f"[DIAG] ESS_all: min ESS over {len(keep_keys)} params = {min_ess_all:.1f}")
        if min3_all:
            msg = ", ".join([f"({idx},{name},{ess:.1f})" for idx, name, ess in min3_all])
            print(f"[DIAG] ESS_all min-3 (index,param,ESS) = {msg}")
        else:
            print("[DIAG] ESS_all min-3: (not available)")

        recent_acc = {k: np.empty((0,), dtype=np.float64) for k in keep_keys}
        for payload in recent_queue:
            for k in keep_keys:
                if k in payload and payload[k].size > 0:
                    recent_acc[k] = np.concatenate([recent_acc[k], payload[k]], axis=0)
        ess_per_recent, min_ess_recent, min3_recent = compute_ess_dict(recent_acc, keep_keys)
        print(f"[DIAG] ESS_recent({len(recent_queue)} chunks): min ESS over {len(keep_keys)} params = {min_ess_recent:.1f}")
        if min3_recent:
            msg = ", ".join([f"({idx},{name},{ess:.1f})" for idx, name, ess in min3_recent])
            print(f"[DIAG] ESS_recent min-3 (index,param,ESS) = {msg}")
        else:
            print("[DIAG] ESS_recent min-3: (not available)")

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
        if float(args.min_ESS) > 0.0 and np.isfinite(min_ess_all) and (min_ess_all >= float(args.min_ESS)):
            stopped_by = f"convergence: min_ess_all={min_ess_all:.1f} >= min_ESS={float(args.min_ESS):.1f}"
            print(f"[STOP] {stopped_by}")
            break

    if stopped_by is None:
        stopped_by = f"max_chunks reached ({int(args.max_chunks)})"
        print(f"\n[STOP] {stopped_by}")

    # ---- save final posterior npz (concatenated) ----
    label = args.npz_label.strip() if args.npz_label.strip() else label_default
    out = {k: acc[k] for k in keep_keys}
    out["diverging"] = acc_div
    out_path = os.path.join(args.outdir, f"posterior_minimal_chunks_{label}.npz")
    np.savez(out_path, **out)

    print(f"\n[DONE] Saved: {out_path}")
    print(f"[DONE] Checkpoint: {ckpt_path}")
    print(f"[DONE] Stopped by: {stopped_by}")


if __name__ == "__main__":
    main()
