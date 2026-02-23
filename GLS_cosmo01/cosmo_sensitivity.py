#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 12:29:15 2025

@author: gengs
"""

import numpy as np
import math
from scipy.integrate import quad
from scipy.special import gamma as Gamma, exp1  # E1(x)
import astropy.units as u

# -------------------------
# constants
# -------------------------
c_kms = 299792.458  # km/s

# -------------------------
# Koopmans (2005) f^{-1} — vectorized
# -------------------------
def f_inv_vec(gamma3d, delta, beta):
    """
    Vectorized 1/f(gamma, delta, beta) of Koopmans (2005).
    Accepts scalars or arrays; returns array broadcast to common shape.
    """
    g  = np.asarray(gamma3d, dtype=float)
    d  = np.asarray(delta,    dtype=float)
    b  = np.asarray(beta,     dtype=float)
    # broadcast
    g, d, b = np.broadcast_arrays(g, d, b)

    xi   = g + d - 2.0
    pref = 1.0 / (2.0 * np.sqrt(np.pi))
    term1 = (xi - 3.0) * (xi - 2.0*b) / (d - 3.0)

    num = Gamma(xi/2.0) * Gamma((xi+2.0)/2.0)
    den = Gamma((xi+2.0)/2.0) * Gamma((xi-1.0)/2.0) - b * Gamma(xi/2.0) * Gamma((xi+1.0)/2.0)
    term2 = num / den

    term3 = Gamma((d-1.0)/2.0) * Gamma((g-1.0)/2.0) / (Gamma(d/2.0) * Gamma(g/2.0))

    out = pref * term1 * term2 * term3

    # domain masks: avoid singularities and non-finite results
    bad = (np.isclose(d, 3.0) |
           np.isclose(xi, 3.0) |
           np.isclose(xi, 2.0*b) |
           ~np.isfinite(out))
    return np.where(bad, np.nan, out)

# -------------------------
# w_phi distances
# w_phi(z) = -1 + (1+w0) exp(-alpha z)
# F(z) = exp{ 3 e^{alpha} (1+w0) [E1(alpha) - E1(alpha(1+z))] }
# H(z)^2/H0^2 = Om(1+z)^3 + Ok(1+z)^2 + (1-Om-Ok) F(z)
# -------------------------
def _da_wphi_scalar(z1, z2, w0, alpha, Ok, H0, Om, c=c_kms):
    """Scalar D_A(z1->z2) [Mpc] for the w_phi model (used internally)."""
    if np.isclose(z1, z2) or (z2 < z1):
        return 0.0

    Ode = 1.0 - Om - Ok
    A = 3.0 * np.exp(alpha) * (1.0 + w0)
    E1_alpha = exp1(alpha)

    def F_of_z(z):
        return np.exp(A * (E1_alpha - exp1(alpha * (1.0 + z))))

    def Ez_inv(z):
        Ez2 = Om*(1.0+z)**3 + Ok*(1.0+z)**2 + Ode*F_of_z(z)
        return 1.0/np.sqrt(Ez2)

    def sinn(ok, x):
        if ok < 0.0:
            k = np.sqrt(-ok);  return np.sin(k*x)/k
        if ok > 0.0:
            k = np.sqrt(ok);   return np.sinh(k*x)/k
        return x

    chi, _ = quad(Ez_inv, z1, z2, epsabs=0, epsrel=1e-8, limit=200)
    return (c / H0) * sinn(Ok, chi) / (1.0 + z2)

def da_wphi_vec(z1, z2, w0, alpha, Ok, H0, Om, c=c_kms):
    """
    Vectorized angular-diameter distance D_A(z1->z2) [Mpc] for w_phi cosmology.
    All inputs can be scalars or arrays; output broadcasts to common shape.
    """
    z1 = np.asarray(z1, dtype=float)
    z2 = np.asarray(z2, dtype=float)
    w0 = float(w0); alpha = float(alpha); Ok = float(Ok); H0 = float(H0); Om = float(Om); c = float(c)

    # broadcast z1 and z2 onto a common shape
    z1b, z2b = np.broadcast_arrays(z1, z2)
    out = np.empty_like(z1b, dtype=float)

    # flatten, compute per-pair, then reshape
    it = np.nditer([z1b, z2b, out], op_flags=[['readonly'], ['readonly'], ['writeonly']])
    for zi, zj, o in it:
        o[...] = _da_wphi_scalar(float(zi), float(zj), w0, alpha, Ok, H0, Om, c)
    return out

# -------------------------
# σ_los for spherical EPL (vectorized, w_phi cosmology)
# -------------------------
def sigma_EPL_wphi_vec(zl, zs, theta_E_arcsec, theta_ap_arcsec,
                       gamma3d, delta, beta,
                       w0, alpha, Ok, H0, Om, c=c_kms):
    """
    Luminosity-weighted LOS dispersion within circular aperture for spherical power-law mass,
    using w_phi cosmology distances. Accepts scalars or arrays; all inputs broadcast.
    """
    zl = np.asarray(zl, dtype=float)
    zs = np.asarray(zs, dtype=float)
    thE = np.asarray(theta_E_arcsec, dtype=float) * u.arcsec.to('rad')
    thA = np.asarray(theta_ap_arcsec, dtype=float) * u.arcsec.to('rad')

    # Broadcast all lens/source dependent arrays to a common shape
    zl, zs, thE, thA, gamma3d, delta, beta = np.broadcast_arrays(zl, zs, thE, thA, gamma3d, delta, beta)

    # Distances [Mpc]
    Ds  = da_wphi_vec(0.0, zs, w0, alpha, Ok, H0, Om, c)
    Dl  = da_wphi_vec(0.0, zl, w0, alpha, Ok, H0, Om, c)
    Dls = da_wphi_vec(zl,  zs, w0, alpha, Ok, H0, Om, c)

    # f^{-1}(γ,δ,β)
    fi = f_inv_vec(gamma3d, delta, beta)

    # Prefactor and final σ (km/s). All terms already broadcast.
    fac = (c**2) / (4.0 * np.pi) * (Ds / Dls) * thE**(gamma3d - 1.0) * thA**(2.0 - gamma3d)
    sigma = np.sqrt(fac / fi)

    # If zs <= zl, D_A(zl->zs)=0 → fac→∞；我们返回 NaN 以显式标记无效对
    bad_pair = (zs <= zl) | ~np.isfinite(Dls) | (Dls <= 0)
    return np.where(bad_pair, np.nan, sigma)



from dataclasses import dataclass
from typing import Optional, Literal
from astropy import units as u
from astropy import constants as const
import numpy as np
try:
    from scipy.integrate import quad
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

C_KMS = const.c.to(u.km/u.s).value

ModelT = Literal['constw', 'cpl', 'wphi']

@dataclass
class CosmoParams:
    H0: float = 70.0            # [km/s/Mpc]
    Om0: float = 0.3            # Ω_m
    Ok0: float = 0.0            # Ω_k
    Or0: float = 0.0            # Ω_r (通常可忽略)
    # ---- choose dark-energy model via `model` ----
    model: ModelT = 'cpl'       # 'constw' | 'cpl' | 'wphi'
    # const-w:
    w: Optional[float] = None
    # CPL: w(a)=w0+wa(1-a)
    w0: float = -1.0
    wa: float = 0.0
    # w_phi: w(z) = -1 + (1+w0) exp(-alpha z)
    alpha: Optional[float] = None

    def Ode0(self) -> float:
        return 1.0 - self.Om0 - self.Ok0 - self.Or0

def _rho_de_factor(z: float, cp: CosmoParams) -> float:
    """
    ρ_DE(z)/ρ_DE(0) for the chosen dark-energy model.
    - constw: (1+z)^{3(1+w)}
    - cpl:    (1+z)^{3(1+w0+wa)} * exp[-3 wa z/(1+z)]
    - wphi:   F(z) per your definition using exp1 (E1).
              F(z) = exp{ 3 e^{alpha} (1+w0) [E1(alpha) - E1(alpha(1+z))] }
    """
    m = cp.model
    if m == 'constw':
        w = cp.w if cp.w is not None else -1.0
        return (1.0 + z)**(3.0 * (1.0 + w))

    if m == 'cpl':
        return (1.0 + z)**(3.0 * (1.0 + cp.w0 + cp.wa)) * np.exp(-3.0 * cp.wa * z / (1.0 + z))

    if m == 'wphi':
        if cp.alpha is None:
            raise ValueError("For model='wphi', please set CosmoParams.alpha (and w0).")
        A = 3.0 * np.exp(cp.alpha) * (1.0 + cp.w0)
        return np.exp(A * (exp1(cp.alpha) - exp1(cp.alpha * (1.0 + z))))

    raise ValueError(f"Unknown DE model: {m}")

def Ez(z: float, cp: CosmoParams) -> float:
    """E(z) = H(z)/H0 with chosen dark-energy model."""
    de = cp.Ode0() * _rho_de_factor(z, cp)
    return np.sqrt(cp.Om0*(1.0+z)**3 + cp.Or0*(1.0+z)**4 + cp.Ok0*(1.0+z)**2 + de)

def _chi_quad(z: float, cp: CosmoParams) -> float:
    """无量纲共动径向距离 χ(z) = ∫_0^z dz'/E(z')."""
    if _HAS_SCIPY:
        val, _ = quad(lambda zp: 1.0 / Ez(zp, cp), 0.0, float(z), epsrel=1e-8, epsabs=0.0, limit=256)
        return val
    zgrid = np.linspace(0.0, float(z), 4097)
    return np.trapz(1.0 / np.vectorize(Ez)(zgrid, cp), zgrid)

def _Sk(x: float, Ok0: float) -> float:
    """曲率函数 S_k(x)，x = √|Ω_k| χ。"""
    if abs(Ok0) < 1e-12: return x
    return np.sinh(x) if Ok0 > 0.0 else np.sin(x)

def D_M(z: float, cp: CosmoParams) -> float:
    """横向共动距离 D_M(z) [Mpc]."""
    chi = _chi_quad(z, cp)
    if abs(cp.Ok0) < 1e-12:
        return (C_KMS / cp.H0) * chi
    sqrtOk = np.sqrt(abs(cp.Ok0))
    return (C_KMS / cp.H0) * _Sk(sqrtOk * chi, cp.Ok0) / sqrtOk

def D_A(z: float, cp: CosmoParams) -> float:
    """角径距 D_A(0→z) [Mpc]."""
    return D_M(z, cp) / (1.0 + z)

def D_A12(z1: float, z2: float, cp: CosmoParams) -> float:
    """两红移间角径距 D_A(z1→z2) [Mpc], 需 z2>z1。"""
    if not (z2 > z1):
        raise ValueError("Require z2 > z1 for D_A12.")
    chi1 = _chi_quad(z1, cp); chi2 = _chi_quad(z2, cp)
    dchi = chi2 - chi1
    if abs(cp.Ok0) < 1e-12:
        DM12 = (C_KMS / cp.H0) * dchi
    else:
        sqrtOk = np.sqrt(abs(cp.Ok0))
        DM12 = (C_KMS / cp.H0) * _Sk(sqrtOk * dchi, cp.Ok0) / sqrtOk
    return DM12 / (1.0 + z2)

# ---- wrappers (unchanged signatures + new args: model, alpha) ----
def Dl(zl: float, cp: CosmoParams) -> float: return D_A(zl, cp)
def Ds(zs: float, cp: CosmoParams) -> float: return D_A(zs, cp)
def Dls(zl: float, zs: float, cp: CosmoParams) -> float: return D_A12(zl, zs, cp)

# =========================
# 1) Galaxy–galaxy lens: R = Ds / Dls
# =========================
def galaxy_lens_distance_ratio(zl: float, zs: float, *,
                               H0: float, omega_m: float,
                               omega_k: float = 0.0, omega_r: float = 0.0,
                               w: Optional[float] = None,
                               w0: float = -1.0, wa: float = 0.0,
                               model: ModelT = 'cpl', alpha: Optional[float] = None) -> float:
    cp = CosmoParams(H0=H0, Om0=omega_m, Ok0=omega_k, Or0=omega_r,
                     model=model, w=w, w0=w0, wa=wa, alpha=alpha)
    return Ds(zs, cp) / Dls(zl, zs, cp)

# =========================
# 2) Time-delay: D_dt = (1+zl) Dl Ds / Dls
# =========================
def time_delay_distance(zl: float, zs: float, *,
                        H0: float, omega_m: float,
                        omega_k: float = 0.0, omega_r: float = 0.0,
                        w: Optional[float] = None,
                        w0: float = -1.0, wa: float = 0.0,
                        model: ModelT = 'cpl', alpha: Optional[float] = None) -> float:
    cp = CosmoParams(H0=H0, Om0=omega_m, Ok0=omega_k, Or0=omega_r,
                     model=model, w=w, w0=w0, wa=wa, alpha=alpha)
    return (1.0 + zl) * Dl(zl, cp) * Ds(zs, cp) / Dls(zl, zs, cp)

# =========================
# 3) DSP: β = (D_ls2/D_s2) / (D_ls1/D_s1)
# =========================
def double_source_beta(zl: float, zs1: float, zs2: float, *,
                       H0: float, omega_m: float,
                       omega_k: float = 0.0, omega_r: float = 0.0,
                       w: Optional[float] = None,
                       w0: float = -1.0, wa: float = 0.0,
                       model: ModelT = 'cpl', alpha: Optional[float] = None) -> float:
    if not (zl < zs1 < zs2):
        raise ValueError("Require zl < zs1 < zs2 for double-source lensing.")
    cp = CosmoParams(H0=H0, Om0=omega_m, Ok0=omega_k, Or0=omega_r,
                     model=model, w=w, w0=w0, wa=wa, alpha=alpha)
    num = Dls(zl, zs2, cp) / Ds(zs2, cp)
    den = Dls(zl, zs1, cp) / Ds(zs1, cp)
    return num / den

# -------- vectorized shells: add (model, alpha) pass-through --------
def galaxy_lens_distance_ratio_vec(
    zl, zs, *, H0, omega_m, omega_k=0.0, omega_r=0.0,
    w=None, w0=-1.0, wa=0.0,
    model: ModelT = 'cpl', alpha: Optional[float] = None
):
    cp = CosmoParams(H0=H0, Om0=omega_m, Ok0=omega_k, Or0=omega_r,
                     model=model, w=w, w0=w0, wa=wa, alpha=alpha)
    zlA, zsA = np.broadcast_arrays(np.asarray(zl, float), np.asarray(zs, float))
    out = np.empty_like(zlA, dtype=float)
    it = np.nditer([zlA, zsA, out], flags=['multi_index'],
                   op_flags=[['readonly'], ['readonly'], ['writeonly']])
    for zl_i, zs_i, o in it:
        zl_f, zs_f = float(zl_i), float(zs_i)
        o[...] = np.nan if zs_f <= zl_f else Ds(zs_f, cp) / Dls(zl_f, zs_f, cp)
    return out

def time_delay_distance_vec(
    zl, zs, *, H0, omega_m, omega_k=0.0, omega_r=0.0,
    w=None, w0=-1.0, wa=0.0,
    model: ModelT = 'cpl', alpha: Optional[float] = None
):
    cp = CosmoParams(H0=H0, Om0=omega_m, Ok0=omega_k, Or0=omega_r,
                     model=model, w=w, w0=w0, wa=wa, alpha=alpha)
    zlA, zsA = np.broadcast_arrays(np.asarray(zl, float), np.asarray(zs, float))
    out = np.empty_like(zlA, dtype=float)
    it = np.nditer([zlA, zsA, out], flags=['multi_index'],
                   op_flags=[['readonly'], ['readonly'], ['writeonly']])
    for zl_i, zs_i, o in it:
        zl_f, zs_f = float(zl_i), float(zs_i)
        o[...] = np.nan if zs_f <= zl_f else (1.0 + zl_f) * Dl(zl_f, cp) * Ds(zs_f, cp) / Dls(zl_f, zs_f, cp)
    return out

def double_source_beta_vec(
    zl, zs1, zs2, *, H0, omega_m, omega_k=0.0, omega_r=0.0,
    w=None, w0=-1.0, wa=0.0,
    model: ModelT = 'cpl', alpha: Optional[float] = None
):
    cp = CosmoParams(H0=H0, Om0=omega_m, Ok0=omega_k, Or0=omega_r,
                     model=model, w=w, w0=w0, wa=wa, alpha=alpha)
    zlA, z1A, z2A = np.broadcast_arrays(np.asarray(zl, float),
                                        np.asarray(zs1, float),
                                        np.asarray(zs2, float))
    out = np.empty_like(zlA, dtype=float)
    it = np.nditer([zlA, z1A, z2A, out], flags=['multi_index'],
                   op_flags=[['readonly'], ['readonly'], ['readonly'], ['writeonly']])
    for zl_i, z1_i, z2_i, o in it:
        zl_f, z1_f, z2_f = float(zl_i), float(z1_i), float(z2_i)
        if not (zl_f < z1_f < z2_f):
            o[...] = np.nan
        else:
            num = Dls(zl_f, z2_f, cp) / Ds(z2_f, cp)
            den = Dls(zl_f, z1_f, cp) / Ds(z1_f, cp)
            o[...] = num / den
    return out



#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.ticker import LogFormatterMathtext
from typing import Dict, Tuple

# ========= YOU MUST PROVIDE / IMPORT THESE =========
# from your_module import galaxy_lens_distance_ratio, time_delay_distance, double_source_beta

def R_scalar(zl: float, zs: float, cosmo: Dict) -> float:
    return galaxy_lens_distance_ratio(float(zl), float(zs), **cosmo)

def Ddt_scalar(zl: float, zs: float, cosmo: Dict) -> float:
    return time_delay_distance(float(zl), float(zs), **cosmo)

def beta_scalar(zl: float, zs1: float, zs2: float, cosmo: Dict) -> float:
    return double_source_beta(float(zl), float(zs1), float(zs2), **cosmo)

# ----------- Cosmology helpers -----------
def _prepare_cosmo(model: str, H0: float, Om0: float, Ok0: float,
                   w: float = None, w0: float = None, wa: float = None, alpha: float = None) -> Dict:
    c = dict(H0=H0, omega_m=Om0, omega_k=Ok0, model=model)
    if model == 'constw':
        c['w'] = float(w)
    elif model == 'cpl':
        c['w0'] = float(w0); c['wa'] = float(wa)
    elif model == 'wphi':
        c['w0'] = float(w0); c['alpha'] = float(alpha)
    else:
        raise ValueError("model must be one of {'constw','cpl','wphi'}")
    return c

PRANGE = {
    'constw': {'H0':(40.0,100.0),'Om0':(0.0,1.0),'Ok0':(-0.5,0.5),'w':(-2.0,0.0)},
    'cpl'   : {'H0':(40.0,100.0),'Om0':(0.0,1.0),'Ok0':(-0.5,0.5),'w0':(-2.0,0.0),'wa':(-4.0,1.0)},
    'wphi'  : {'H0':(40.0,100.0),'Om0':(0.0,1.0),'Ok0':(-0.5,0.5),'w0':(-2.0,0.0),'alpha':(1.0,2.0)},
}

def _fd_step(model: str, p: str, val: float,
             eps_rel: float = 1e-5, eps_abs: float = 1e-5) -> Tuple[float, float]:
    lo, hi = PRANGE[model].get(p, (None, None))
    if p in ('Ok0','w','w0','wa','Om0'):
        dp = max(eps_abs, 1e-4)
    elif p == 'alpha':
        base = max(val, 1e-6)
        dp = max(eps_abs, base*eps_rel)
    elif p == 'H0':
        dp = max(eps_abs, 1e-3*max(1.0, val))
    else:
        dp = max(eps_abs, abs(val)*eps_rel)
    p_minus, p_plus = val - dp, val + dp
    if lo is not None and hi is not None:
        p_minus = np.clip(p_minus, lo, hi)
        p_plus  = np.clip(p_plus,  lo, hi)
        if np.isclose(p_minus, p_plus):
            p_minus = max(lo, val - 2*dp)
            p_plus  = min(hi, val + 2*dp)
    return float(p_minus), float(p_plus)

def _dln_obs_dp(obs_fn, zl, zs_or_tuple, cosmo_in: Dict, p: str,
                eps_rel=1e-5, eps_abs=1e-5) -> float:
    # base value
    if obs_fn is beta_scalar:
        zl_f = float(zl); zs1_f, zs2_f = float(zs_or_tuple[0]), float(zs_or_tuple[1])
        base = obs_fn(zl_f, zs1_f, zs2_f, cosmo_in)
    else:
        zl_f = float(zl); zs_f = float(zs_or_tuple)
        base = obs_fn(zl_f, zs_f, cosmo_in)
    if not np.isfinite(base) or base <= 0:
        return np.nan

    model = cosmo_in.get('model', 'cpl')

    def _apply_param(d: Dict, key: str, val: float):
        dd = dict(d)
        if key == 'H0':   dd['H0'] = val
        elif key == 'Om0': dd['omega_m'] = val
        elif key == 'Ok0': dd['omega_k'] = val
        else:              dd[key] = val
        return dd

    cur = (cosmo_in['H0'] if p=='H0' else
           cosmo_in['omega_m'] if p=='Om0' else
           cosmo_in['omega_k'] if p=='Ok0' else
           cosmo_in.get(p, None))
    if cur is None:
        return np.nan

    p_minus, p_plus = _fd_step(model, p, float(cur), eps_rel=eps_rel, eps_abs=eps_abs)
    c_m = _apply_param(cosmo_in, p, p_minus)
    c_p = _apply_param(cosmo_in, p, p_plus)

    if obs_fn is beta_scalar:
        bm = obs_fn(zl_f, zs1_f, zs2_f, c_m)
        bp = obs_fn(zl_f, zs1_f, zs2_f, c_p)
    else:
        bm = obs_fn(zl_f, zs_f, c_m)
        bp = obs_fn(zl_f, zs_f, c_p)

    if (bm <= 0) or (bp <= 0) or (not np.isfinite(bm)) or (not np.isfinite(bp)):
        return np.nan

    return float((np.log(bp) - np.log(bm)) / (p_plus - p_minus))

def dlnR_dp(zl, zs, cosmo, p, **kw):        return _dln_obs_dp(R_scalar,   zl, zs,       cosmo, p, **kw)
def dlnDdt_dp(zl, zs, cosmo, p, **kw):      return _dln_obs_dp(Ddt_scalar, zl, zs,       cosmo, p, **kw)
def dlnbeta_dp(zl, zs1, zs2, cosmo, p, **kw): return _dln_obs_dp(beta_scalar, zl, (zs1, zs2), cosmo, p, **kw)

# ----------- Tk GUI -----------
class SensitivityGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sensitivity Map — single distance combination")
        self.geometry("1050x700")

        # top controls
        top = ttk.Frame(self); top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        # Model dropdown
        ttk.Label(top, text="Model:").grid(row=0, column=0, sticky="e")
        self.model_var = tk.StringVar(value='wphi')
        self.model_cb = ttk.Combobox(top, textvariable=self.model_var,
                                     values=['constw','cpl','wphi'], width=10, state='readonly')
        self.model_cb.grid(row=0, column=1, sticky="w", padx=4)
        self.model_cb.bind("<<ComboboxSelected>>", self._on_model_change)

        # Parameter to differentiate
        ttk.Label(top, text="d/d(param):").grid(row=0, column=2, sticky="e")
        self.param_var = tk.StringVar(value='alpha')
        self.param_cb = ttk.Combobox(top, textvariable=self.param_var,
                                     values=['w0','alpha'], width=10, state='readonly')
        self.param_cb.grid(row=0, column=3, sticky="w", padx=4)
        self.param_cb.bind("<<ComboboxSelected>>", self._auto_plot)

        # Observable dropdown
        ttk.Label(top, text="Observable:").grid(row=0, column=4, sticky="e")
        self.obs_var = tk.StringVar(value='R')
        self.obs_cb = ttk.Combobox(top, textvariable=self.obs_var,
                                   values=['R','D_dt','beta_DSP'], width=10, state='readonly')
        self.obs_cb.grid(row=0, column=5, sticky="w", padx=4)
        self.obs_cb.bind("<<ComboboxSelected>>", self._auto_plot)

        # ---- sliders for cosmological parameters (dynamic per model) ----
        sliders = ttk.LabelFrame(self, text="Cosmology (sliders)")
        sliders.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        self.slider_defs = {}
        def add_slider(parent, key, text, lo, hi, val, col):
            lab = ttk.Label(parent, text=text)
            lab.grid(row=0, column=2*col, sticky="e")
            scale = ttk.Scale(parent, from_=lo, to=hi, orient=tk.HORIZONTAL)
            scale.set(val)
            scale.grid(row=0, column=2*col+1, sticky="we", padx=6)
            val_label = ttk.Label(parent, text=f"{val:.3f}")
            val_label.grid(row=1, column=2*col+1)
            parent.columnconfigure(2*col+1, weight=1)
            self.slider_defs[key] = (scale, val_label, lo, hi)
            scale.bind("<B1-Motion>", self._on_slider_move)
            scale.bind("<ButtonRelease-1>", self._auto_plot)
            return scale

        # default model wphi sliders visible; others created but toggled by model
        self.s_H0   = add_slider(sliders, 'H0',   "H0 [km/s/Mpc]", PRANGE['wphi']['H0'][0],   PRANGE['wphi']['H0'][1],   70.0, 0)
        self.s_Om0  = add_slider(sliders, 'Om0',  "Ωm",            PRANGE['wphi']['Om0'][0],  PRANGE['wphi']['Om0'][1],  0.3,  1)
        self.s_Ok0  = add_slider(sliders, 'Ok0',  "Ωk",            PRANGE['wphi']['Ok0'][0],  PRANGE['wphi']['Ok0'][1],  0.0,  2)
        self.s_w    = add_slider(sliders, 'w',    "w (const-w)",   PRANGE['constw']['w'][0],  PRANGE['constw']['w'][1], -1.0,  3)
        self.s_w0   = add_slider(sliders, 'w0',   "w0",            -2.0, 0.0, -1.0,  4)
        self.s_wa   = add_slider(sliders, 'wa',   "wa",            -4.0, 1.0,  0.0,  5)
        self.s_alp  = add_slider(sliders, 'alpha',"alpha",          1.0, 2.0,  1.2,  6)

        # ---- grid & sigma (text boxes) ----
        gridf = ttk.LabelFrame(self, text="Grid & β_DSP settings (type values)")
        gridf.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        self.e = {}
        def add_entry(lbl, default, col):
            ttk.Label(gridf, text=lbl).grid(row=0, column=2*col, sticky="e")
            var = tk.StringVar(value=str(default))
            ent = ttk.Entry(gridf, textvariable=var, width=10)
            ent.grid(row=0, column=2*col+1, sticky="w", padx=4)
            ent.bind("<Return>", self._auto_plot)
            self.e[lbl] = var

        add_entry("zl_min", 0.05, 0); add_entry("zl_max", 1.5, 1); add_entry("zl_step", 0.05, 2)
        add_entry("zs_min", 0.10, 3); add_entry("zs_max", 3.0, 4); add_entry("zs_step", 0.05, 5)
        add_entry("zs2_fixed", 2.0, 6); add_entry("zs1_step", 0.05, 7); add_entry("sigma_frac", 0.1, 8)

        # ---- plot area ----
        plotf = ttk.Frame(self)
        plotf.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=6)

        self.fig, self.ax = plt.subplots(figsize=(7.8, 5.8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plotf)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, plotf)
        self.toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # init visibility and first plot
        self._on_model_change()
        self._plot()

    # ---- UI helpers ----
    def _on_slider_move(self, event=None):
        # update numeric labels next to sliders
        for key, (scale, label, lo, hi) in self.slider_defs.items():
            label.config(text=f"{scale.get():.3f}")

    def _on_model_change(self, event=None):
        m = self.model_var.get()
        # update parameter choices
        if m == 'constw':
            params = ['H0','Om0','Ok0','w']; default = 'w'
        elif m == 'cpl':
            params = ['H0','Om0','Ok0','w0','wa']; default = 'w0'
        else:
            params = ['H0','Om0','Ok0','w0','alpha']; default = 'alpha'
        self.param_cb['values'] = params
        if self.param_var.get() not in params:
            self.param_var.set(default)

        # show/hide sliders
        for key, (scale, _, lo, hi) in self.slider_defs.items():
            scale.master.grid()  # ensure parent frame is visible
        # hide irrelevant EoS sliders
        self._set_slider_visible('w',   m == 'constw')
        self._set_slider_visible('w0',  m in ('cpl','wphi'))
        self._set_slider_visible('wa',  m == 'cpl')
        self._set_slider_visible('alpha', m == 'wphi')

        self._auto_plot()

    def _set_slider_visible(self, key, visible: bool):
        scale, label, lo, hi = self.slider_defs[key]
        state = 'normal' if visible else 'disabled'
        scale.configure(state=state)
        label.configure(state=state)

    def _get_grid_float(self, name: str) -> float:
        try:
            return float(self.e[name].get())
        except Exception:
            raise ValueError(f"Invalid value for {name}: {self.e[name].get()}")

    def _auto_plot(self, event=None):
        # debounce small slider jitters by after() – but keep it simple:
        self.after(50, self._plot)

    # ---- core plot ----
    def _plot(self):
        try:
            model = self.model_var.get()
            param = self.param_var.get()
            obs   = self.obs_var.get()

            # cosmology from sliders
            H0  = float(self.slider_defs['H0'][0].get())
            Om0 = float(self.slider_defs['Om0'][0].get())
            Ok0 = float(self.slider_defs['Ok0'][0].get())
            w   = float(self.slider_defs['w'][0].get())
            w0  = float(self.slider_defs['w0'][0].get())
            wa  = float(self.slider_defs['wa'][0].get())
            alp = float(self.slider_defs['alpha'][0].get())

            # grid & sigma from entries
            zl_min  = self._get_grid_float("zl_min")
            zl_max  = self._get_grid_float("zl_max")
            zl_step = self._get_grid_float("zl_step")
            zs_min  = self._get_grid_float("zs_min")
            zs_max  = self._get_grid_float("zs_max")
            zs_step = self._get_grid_float("zs_step")
            zs2_fix = self._get_grid_float("zs2_fixed")
            zs1_step= self._get_grid_float("zs1_step")
            sigma   = self._get_grid_float("sigma_frac")

            # build cosmology dict
            if model == 'constw':
                cosmo = _prepare_cosmo(model, H0, Om0, Ok0, w=w)
            elif model == 'cpl':
                cosmo = _prepare_cosmo(model, H0, Om0, Ok0, w0=w0, wa=wa)
            else:
                cosmo = _prepare_cosmo(model, H0, Om0, Ok0, w0=w0, alpha=alp)

            # grids
            zl_grid = np.arange(zl_min, zl_max + 1e-12, zl_step)
            zs_grid = np.arange(zs_min, zs_max + 1e-12, zs_step)
            ZL, ZS  = np.meshgrid(zl_grid, zs_grid, indexing="xy")

            zs1_grid = np.arange(zs_min, min(zs2_fix - 1e-6, zs_max) + 1e-12, zs1_step)
            ZL_b, ZS1 = np.meshgrid(zl_grid, zs1_grid, indexing="xy")

            # colormap and levels
            vmin, vmax = 1e-4, 1e1
            levels01 = np.logspace(-4, 0, 16)
            levels02 = np.linspace(1, 10, 10)[1:]
            levels = np.r_[levels01, levels02]
            norm = LogNorm(vmin=vmin, vmax=vmax)

            # choose derivative function
            if obs == 'R':
                gridX, gridY = zl_grid, zs_grid
                data = np.full_like(ZL, np.nan, dtype=float)
                for i in range(ZL.shape[0]):
                    for j in range(ZL.shape[1]):
                        zl, zs = ZL[i, j], ZS[i, j]
                        if not (zs > zl > 0.0):
                            continue
                        val = dlnR_dp(zl, zs, cosmo, param)
                        data[i, j] = abs(val) / max(sigma, 1e-30) if np.isfinite(val) else np.nan
                arr = np.ma.masked_invalid(data)
                title_tag = r"\mathcal{R}_{DR}"
                ylab = r"$z_s$"
            elif obs == 'D_dt':
                gridX, gridY = zl_grid, zs_grid
                data = np.full_like(ZL, np.nan, dtype=float)
                for i in range(ZL.shape[0]):
                    for j in range(ZL.shape[1]):
                        zl, zs = ZL[i, j], ZS[i, j]
                        if not (zs > zl > 0.0):
                            continue
                        val = dlnDdt_dp(zl, zs, cosmo, param)
                        data[i, j] = abs(val) / max(sigma, 1e-30) if np.isfinite(val) else np.nan
                arr = np.ma.masked_invalid(data)
                title_tag = r"\mathcal{R}_{\Delta t}"
                ylab = r"$z_s$"
            else:  # beta_DSP
                gridX, gridY = zl_grid, zs1_grid
                data = np.full_like(ZL_b, np.nan, dtype=float)
                eps = 1e-6
                for i in range(ZL_b.shape[0]):
                    for j in range(ZL_b.shape[1]):
                        zl, zs1 = ZL_b[i, j], ZS1[i, j]
                        if not (zl + eps < zs1 - eps < zs2_fix - eps):
                            continue
                        val = dlnbeta_dp(zl, zs1, zs2_fix, cosmo, param)
                        data[i, j] = abs(val) / max(sigma, 1e-30) if np.isfinite(val) else np.nan
                arr = np.ma.masked_invalid(data)
                title_tag = r"\mathcal{R}_{DSPDR}"
                ylab = fr"$z_{{s1}}$ (fixed $z_{{s2}}={zs2_fix:.2f}$)"

            ptex_map = dict(H0=r"H_0", Om0=r"\Omega_m", Ok0=r"\Omega_k",
                            w=r"w", w0=r"w_0", wa=r"w_a", alpha=r"\alpha")
            ptex = ptex_map.get(param, param)

            # draw
            self.ax.clear()
            cf = self.ax.contourf(gridX, gridY, arr, levels=levels, norm=norm, cmap="viridis")
            self.ax.set_xlabel(r"$z_l$")
            self.ax.set_ylabel(ylab)
            self.ax.grid(alpha=0.2)
            self.ax.set_title(fr"$|\partial \ln {title_tag}/\partial {ptex}|/\sigma$")

            # colorbar inside the same axes’ figure
            # remove old colorbars by clearing the figure’s existing ones
            for cax in self.fig.axes[1:]:
                self.fig.delaxes(cax)
            cbar = self.fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"),
                                     ax=self.ax, ticks=levels, format=LogFormatterMathtext())
            cbar.set_label("Sensitivity")

            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Error", str(e))

# ---------- main ----------
if __name__ == "__main__":
    app = SensitivityGUI()
    app.mainloop()
