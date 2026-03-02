# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 01:12:12 2025

@author: poilo
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
try:
    from scipy.integrate import quad
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False
from astropy import units as u
from astropy import constants as const

C_KMS = const.c.to(u.km/u.s).value

@dataclass
class CosmoParams:
    H0: float = 70.0            # [km/s/Mpc]
    Om0: float = 0.3            # Ω_m
    Ok0: float = 0.0            # Ω_k
    Or0: float = 0.0            # Ω_r (可忽略时设 0)
    # 暗能量状态方程：二选一
    w: Optional[float] = None      # 常数 w
    w0: float = -1.0            # CPL: w(a)=w0+wa(1-a)
    wa: float = 0.0

    def Ode0(self) -> float:
        return 1.0 - self.Om0 - self.Ok0 - self.Or0

def _rho_de_factor(z: float, cp: CosmoParams) -> float:
    """ρ_DE(z)/ρ_DE(0) 的无量纲演化因子。"""
    if cp.w is not None:  # 常数 w
        return (1.0 + z)**(3.0 * (1.0 + cp.w))
    # CPL: w(a)=w0+wa(1-a) => ρ_DE ∝ (1+z)^{3(1+w0+wa)} exp[-3 wa z/(1+z)]
    return (1.0 + z)**(3.0 * (1.0 + cp.w0 + cp.wa)) * np.exp(-3.0 * cp.wa * z / (1.0 + z))

def Ez(z: float, cp: CosmoParams) -> float:
    """E(z) = H(z)/H0."""
    de = cp.Ode0() * _rho_de_factor(z, cp)
    return np.sqrt(cp.Om0 * (1.0 + z)**3 + cp.Or0 * (1.0 + z)**4 + cp.Ok0 * (1.0 + z)**2 + de)

def _chi_quad(z: float, cp: CosmoParams) -> float:
    """无量纲共动径向距离 χ(z) = ∫_0^z dz'/E(z')."""
    if _HAS_SCIPY:
        val, _ = quad(lambda zp: 1.0 / Ez(zp, cp), 0.0, float(z), epsrel=1e-8, epsabs=0.0, limit=256)
        return val
    # 简单梯形积分后备方案
    zgrid = np.linspace(0.0, float(z), 4097)
    return np.trapz(1.0 / np.vectorize(Ez)(zgrid, cp), zgrid)

def _Sk(x: float, Ok0: float) -> float:
    """曲率函数 S_k(x)。注意这里的 x 是 √|Ω_k| χ。"""
    if abs(Ok0) < 1e-12:
        return x
    if Ok0 > 0.0:  # open, sinh
        return np.sinh(x)
    else:          # closed, sin
        return np.sin(x)

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
    chi1 = _chi_quad(z1, cp)
    chi2 = _chi_quad(z2, cp)
    dchi = chi2 - chi1
    if abs(cp.Ok0) < 1e-12:
        DM12 = (C_KMS / cp.H0) * dchi
    else:
        sqrtOk = np.sqrt(abs(cp.Ok0))
        DM12 = (C_KMS / cp.H0) * _Sk(sqrtOk * dchi, cp.Ok0) / sqrtOk
    return DM12 / (1.0 + z2)

# ---- 封装常用距离 ----
def Dl(zl: float, cp: CosmoParams) -> float: return D_A(zl, cp)
def Ds(zs: float, cp: CosmoParams) -> float: return D_A(zs, cp)
def Dls(zl: float, zs: float, cp: CosmoParams) -> float: return D_A12(zl, zs, cp)

# =========================
# 1) 星系–星系透镜：R = Ds / Dls
# =========================
def galaxy_lens_distance_ratio(zl: float, zs: float, *,
                               H0: float, omega_m: float,
                               omega_k: float = 0.0, omega_r: float = 0.0,
                               w: Optional[float] = None, w0: float = -1.0, wa: float = 0.0) -> float:
    """
    返回 R = D_s / D_ls （无量纲），仅含宇宙学部分。
    注：真实建模还需乘以结构因子 F(γ, β_ani, θ_ap/θ_E) 才能与(θ_E, σ_ap)组合。
    """
    cp = CosmoParams(H0=H0, Om0=omega_m, Ok0=omega_k, Or0=omega_r, w=w, w0=w0, wa=wa)
    return Ds(zs, cp) / Dls(zl, zs, cp)

# =========================
# 2) 时延 AGN：D_dt = (1+zl) Dl Ds / Dls
# =========================
def time_delay_distance(zl: float, zs: float, *,
                        H0: float, omega_m: float,
                        omega_k: float = 0.0, omega_r: float = 0.0,
                        w: Optional[float] = None, w0: float = -1.0, wa: float = 0.0) -> float:
    """
    返回时延距离 D_Δt [Mpc]： (1+zl) * Dl * Ds / Dls
    注意：观测上 D_Δt 还会被外会聚 κ_ext 等效缩放，建模需单独处理。
    """
    cp = CosmoParams(H0=H0, Om0=omega_m, Ok0=omega_k, Or0=omega_r, w=w, w0=w0, wa=wa)
    return (1.0 + zl) * Dl(zl, cp) * Ds(zs, cp) / Dls(zl, zs, cp)

# =========================
# 3) 双源透镜：β_DSP = (D_ls2/D_s2) / (D_ls1/D_s1)
# =========================
def double_source_beta(zl: float, zs1: float, zs2: float, *,
                       H0: float, omega_m: float,
                       omega_k: float = 0.0, omega_r: float = 0.0,
                       w: Optional[float] = None, w0: float = -1.0, wa: float = 0.0) -> float:
    """
    返回双源透镜的宇宙学量 β_DSP （无量纲）。
    要求 zl < zs1 < zs2。
    """
    if not (zl < zs1 < zs2):
        raise ValueError("Require zl < zs1 < zs2 for double-source lensing.")
    cp = CosmoParams(H0=H0, Om0=omega_m, Ok0=omega_k, Or0=omega_r, w=w, w0=w0, wa=wa)
    num = Dls(zl, zs2, cp) / Ds(zs2, cp)
    den = Dls(zl, zs1, cp) / Ds(zs1, cp)
    return num / den

import numpy as np

def _as_float(x):
    # 确保传给积分器的是标量 float
    return float(np.asarray(x))

def galaxy_lens_distance_ratio_vec(
    zl, zs, *, H0, omega_m, omega_k=0.0, omega_r=0.0,
    w=None, w0=-1.0, wa=0.0
):
    """
    向量化版本：返回 R = Ds/Dls（与 zl、zs 广播后的形状一致）。
    不满足 zs>zl 的元素返回 np.nan。
    """
    cp = CosmoParams(H0=H0, Om0=omega_m, Ok0=omega_k, Or0=omega_r, w=w, w0=w0, wa=wa)
    zlA, zsA = np.broadcast_arrays(np.asarray(zl, float), np.asarray(zs, float))
    out = np.empty_like(zlA, dtype=float)
    it = np.nditer([zlA, zsA, out], flags=['multi_index'],
                   op_flags=[['readonly'], ['readonly'], ['writeonly']])
    for zl_i, zs_i, o in it:
        zl_f, zs_f = _as_float(zl_i), _as_float(zs_i)
        if zs_f <= zl_f:
            o[...] = np.nan
        else:
            # o[...] = Ds(zs_f, cp) / Dls(zl_f, zs_f, cp)
            o[...] = Dls(zl_f, zs_f, cp) / Ds(zs_f, cp)
    return out

def time_delay_distance_vec(
    zl, zs, *, H0, omega_m, omega_k=0.0, omega_r=0.0,
    w=None, w0=-1.0, wa=0.0
):
    """
    向量化版本：返回 D_dt [Mpc] = (1+zl) * Dl * Ds / Dls。
    不满足 zs>zl 的元素返回 np.nan。
    """
    cp = CosmoParams(H0=H0, Om0=omega_m, Ok0=omega_k, Or0=omega_r, w=w, w0=w0, wa=wa)
    zlA, zsA = np.broadcast_arrays(np.asarray(zl, float), np.asarray(zs, float))
    out = np.empty_like(zlA, dtype=float)
    it = np.nditer([zlA, zsA, out], flags=['multi_index'],
                   op_flags=[['readonly'], ['readonly'], ['writeonly']])
    for zl_i, zs_i, o in it:
        zl_f, zs_f = _as_float(zl_i), _as_float(zs_i)
        if zs_f <= zl_f:
            o[...] = np.nan
        else:
            o[...] = (1.0 + zl_f) * Dl(zl_f, cp) * Ds(zs_f, cp) / Dls(zl_f, zs_f, cp)
    return out

def double_source_beta_vec(
    zl, zs1, zs2, *, H0, omega_m, omega_k=0.0, omega_r=0.0,
    w=None, w0=-1.0, wa=0.0
):
    """
    向量化版本：返回 β_DSP = (D_ls2/D_s2)/(D_ls1/D_s1)。
    仅当 zl < zs1 < zs2 时有效，否则该元素 np.nan。
    """
    cp = CosmoParams(H0=H0, Om0=omega_m, Ok0=omega_k, Or0=omega_r, w=w, w0=w0, wa=wa)
    zlA, z1A, z2A = np.broadcast_arrays(np.asarray(zl, float),
                                        np.asarray(zs1, float),
                                        np.asarray(zs2, float))
    out = np.empty_like(zlA, dtype=float)
    it = np.nditer([zlA, z1A, z2A, out], flags=['multi_index'],
                   op_flags=[['readonly'], ['readonly'], ['readonly'], ['writeonly']])
    for zl_i, z1_i, z2_i, o in it:
        zl_f, z1_f, z2_f = _as_float(zl_i), _as_float(z1_i), _as_float(z2_i)
        if not (zl_f < z1_f < z2_f):
            o[...] = np.nan
        else:
            num = Dls(zl_f, z2_f, cp) / Ds(z2_f, cp)
            den = Dls(zl_f, z1_f, cp) / Ds(z1_f, cp)
            o[...] = num / den
    return out

import os
import numpy as np
import pandas as pd
import scipy.integrate as sci
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy import units as u
from astropy import constants as const
from scipy.special import gamma as fgamma

# ───────────────────────────────────────────────────────────────
# 0) Load data
# ───────────────────────────────────────────────────────────────
tab = Table.read(r"/home/astrodust/SG/sgl_cosmo/GLS_cosmo01/Data/SGLTable03.fits")

# Make plain numpy arrays (drop units if present)
zl        = np.asarray(tab['zl'], dtype=float)
zs        = np.asarray(tab['zs'], dtype=float)
theta_ap_arcsec = np.asarray(tab['theta_ap'], dtype=float)
sigma_ap  = np.asarray(tab['sigma_ap'], dtype=float)

# Aperture‐correction parameters (defined but not used below)
eta, eta_err = -0.066, 0.035
def calibrate(sigma_ap_nocal, theta_eff, theta_ap, eta):
    return sigma_ap_nocal * (theta_eff / (2*theta_ap))**eta
def sigma_uncertainty(sigma_ap_nocal, sigma_ap_err_nocal, theta_eff, theta_ap, eta, eta_err):
    fac = (theta_eff / (2*theta_ap))
    err_stat2 = sigma_ap_err_nocal**2
    err_AC2   = (sigma_ap_err_nocal**2) * fac**(2*eta) \
              + (sigma_ap_nocal**2) * fac**(2*eta) * (np.log(fac)**2) * (eta_err**2)
    err_sys2  = (0.03 * sigma_ap_nocal)**2
    return np.sqrt(err_stat2 + err_AC2 + err_sys2)

# ───────────────────────────────────────────────────────────────
# σ–θ_E relation for a spherical power-law (γ)
# ───────────────────────────────────────────────────────────────
pi = np.pi

def f_gamma(g):
    a = -1.0/np.sqrt(pi)
    b = ((5.0 - 2.0*g)*(1.0 - g)) / (3.0 - g)
    c = fgamma(g - 1.0) / fgamma(g - 1.5)
    d = fgamma((g - 1.0)/2.0) / fgamma(g/2.0)
    return a * b * c * d**2

def f_prime(g, d, beta):
    pref = 1.0 / (2.0 * np.sqrt(np.pi))
    term1 = (g + d - 5.0) * (g + d - 2.0 - 2.0*beta) / (d - 3.0)
    num = fgamma((g + d - 2.0)/2.0) * fgamma((g + d)/2.0)
    den = fgamma((g + d)/2.0) * fgamma((g + d - 3.0)/2.0) \
        - beta * fgamma((g + d - 2.0)/2.0) * fgamma((g + d - 1.0)/2.0)
    term3 = fgamma((d - 1.0)/2.0) * fgamma((g - 1.0)/2.0) / (fgamma(d/2.0) * fgamma(g/2.0))
    return pref * term1 * (num/den) * term3

cosmo = dict(H0=70.0, omega_m=0.3, omega_k=0.0, w=-1.0)

R = galaxy_lens_distance_ratio_vec(zl, zs, **cosmo)
Ddt = time_delay_distance_vec(zl, zs, **cosmo)

import numpy as np
import emcee
from multiprocessing import Pool
import os
from astropy.constants import G, c, M_sun

import math

# Load values - GLOBAL VALUES
c_km = c.to('km/s').value
pi = np.pi

def E2_wCDM(z, Om, ok, w, Or=0.0):

    Ode = 1.0 - Om - ok - Or
    if Ode <= 0.0:
        return np.nan

    return Om*(1.0+z)**3 + Or*(1.0+z)**4 + ok*(1.0+z)**2 + Ode*(1.0+z)**(3.0*(1.0+w))

def E2_positive_on_grid(Om, ok, w, z1, z2, n=8):
    if z2 <= z1:
        return True  
    zz = np.linspace(z1, z2, n)
    Ez2 = E2_wCDM(zz, Om, ok, w)
    return np.all(np.isfinite(Ez2)) and np.all(Ez2 > 0.0)

def Sk_of_chi(chi, ok):
    # comoving transverse distance projector with curvature
    if abs(ok) < 1e-10:
        return chi
    s = np.sqrt(abs(ok))
    y = s * chi

    if abs(y) < 1e-6:

        return chi + (np.sign(ok) * ok) * chi**3 / 6.0
    return np.sinh(y)/s if ok > 0 else np.sin(y)/s


def integrate_da_good_kw(z1, z2, w, ok, H0, Om, c_km):
    if not (np.isfinite(z1) and np.isfinite(z2)):
        return np.nan
    if z2 <= z1:
        return 0.0

    if not E2_positive_on_grid(Om, ok, w, z1, z2, n=8):
        return np.nan

    def inv_E(z):
        Ez2 = E2_wCDM(z, Om, ok, w)

        if not np.isfinite(Ez2) or Ez2 <= 0.0:
            return np.inf
        return 1.0 / np.sqrt(Ez2)

    chi, err = sci.quad(inv_E, z1, z2, epsabs=1e-8, epsrel=1e-8, limit=200)
    if not np.isfinite(chi):
        return np.nan


    Dc = (c_km / H0) * chi
    Dm = Sk_of_chi(Dc * H0 / c_km, ok) * (c_km / H0) 
    DA = Dm / (1.0 + z2)
    return DA

# -------------------------------------------------
# 0) 理论量：dd_th = D_ls / D_s  （保持你的实现不变）
# -------------------------------------------------
def dd_th(Om, w, zl, zs, H0):
    D_s  = integrate_da_good_kw(0.,  zs, w, 0., H0, Om, c_km)
    D_ls = integrate_da_good_kw(zl, zs, w, 0., H0, Om, c_km)
    return D_ls / D_s

# =================================================
# 1) 直接观测量与 log-likelihood
# =================================================
def lnlike_dd_direct(Om, w, zl_arr, zs_arr, dd_obs_arr, dd_obs_err_arr, H0,
                     sigma_is_fractional=False, sigma_floor=1e-6):
    """
    观测给定为直接的 dd_obs (= D_ls/D_s)，误差是绝对误差或分数误差。
    在 log 空间做高斯似然： ln dd_th vs ln dd_obs
    """
    zl_arr = np.asarray(zl_arr, float)
    zs_arr = np.asarray(zs_arr, float)
    dd_obs_arr = np.asarray(dd_obs_arr, float)
    dd_obs_err_arr = np.asarray(dd_obs_err_arr, float)

    # 基本掩码：物理 & 有限
    m = (zs_arr > zl_arr) & np.isfinite(dd_obs_arr) & np.isfinite(dd_obs_err_arr) \
        & (dd_obs_arr > 0) & (dd_obs_err_arr > 0)
    if not np.any(m):
        return -np.inf

    zl = zl_arr[m]; zs = zs_arr[m]
    dd_obs = dd_obs_arr[m]; dd_err = dd_obs_err_arr[m]

    # 理论值（逐个算，兼容你现有的 dd_th 标量函数）
    dd_th_arr = np.array([dd_th(Om, w, zli, zsi, H0) for zli, zsi in zip(zl, zs)])
    if np.any(~np.isfinite(dd_th_arr)) or np.any(dd_th_arr <= 0):
        return -np.inf

    # log 空间误差
    if sigma_is_fractional:
        sigma_ln = np.clip(dd_err, sigma_floor, None)     # dd_err 已是分数误差
    else:
        sigma_ln = np.clip(dd_err / dd_obs, sigma_floor, None)

    delta = np.log(dd_th_arr) - np.log(dd_obs)
    var   = sigma_ln**2

    # 带常数项的高斯对数似然（不带常数也可以，结果只差一个常数）
    lnL = -0.5 * np.sum(delta**2 / var + np.log(2*np.pi*var))
    return float(lnL)

# 统一的先验
def lnprior_2D(theta):
    Om, w = theta
    if 0.02 < Om < 1.0 and -4.0 < w < 2.0:
        return 0.0
    return -np.inf

def lnprob_cos2D_direct(theta, zl, zs, dd_obs, dd_obs_err, H0,
                        sigma_is_fractional=False):
    lp = lnprior_2D(theta)
    if not np.isfinite(lp):
        return -np.inf
    Om, w = theta
    ll = lnlike_dd_direct(Om, w, zl, zs, dd_obs, dd_obs_err, H0,
                          sigma_is_fractional=sigma_is_fractional)
    return lp + ll

# =================================================
# 2) 运行 emcee（多进程 + 每 2000 步 checkpoint）
# =================================================
import numpy as np
import emcee
import os, glob
from multiprocessing import Pool

def run_mcmc_dd(zl, zs, dd_obs, dd_obs_err, H0,
                init=(0.3, -1.0), nwalkers=32, nsteps=10000,
                threads=8, out_prefix="mcmc_dd_direct",
                sigma_is_fractional=False, random_seed=42,
                checkpoint_size = 1000,
                burnin_to_save=1000, thin_to_save=1):
    """
    以 dd_obs 直接约束 (Om, w)。
    - 每 2000 步写 checkpoint: {out_prefix}_chk{chunk:03d}.npz
    - 结束时仅保存丢弃 burn-in、thin 后且保持三维结构的链：
        {out_prefix}_chain_kept.npy  (shape = (nsteps_kept, nwalkers, ndim))
    - 随后删除所有 checkpoint 文件。
    """
    rng = np.random.default_rng(random_seed)
    ndim = 2
    Om0, w0 = init

    p0 = np.array([Om0, w0])
    spread = np.array([0.02, 0.05])
    p0_walkers = p0 + spread * rng.normal(size=(nwalkers, ndim))

    # 保证初始化落在先验范围
    for k in range(nwalkers):
        tries = 0
        while not np.isfinite(lnprior_2D(p0_walkers[k])) and tries < 1000:
            p0_walkers[k] = p0 + spread * rng.normal(size=ndim)
            tries += 1
        if not np.isfinite(lnprior_2D(p0_walkers[k])):
            raise RuntimeError("Could not initialize walkers inside prior bounds.")

    log_prob_args = (zl, zs, dd_obs, dd_obs_err, H0, sigma_is_fractional)

    with Pool(processes=threads) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, lnprob_cos2D_direct, args=log_prob_args, pool=pool
        )

        state = p0_walkers
        chunk = checkpoint_size
        n_chunks = int(np.ceil(nsteps / chunk))
        for i in range(n_chunks):
            n_this = chunk if (i < n_chunks - 1) else (nsteps - chunk * (n_chunks - 1))
            state = sampler.run_mcmc(state, n_this, progress=True)

            # --- checkpoint ---
            chk_path = f"{out_prefix}_chk{i+1:03d}.npz"
            np.savez(
                chk_path,
                chain=sampler.get_chain(),
                log_prob=sampler.get_log_prob(),
                acceptance_fraction=sampler.acceptance_fraction,
                last_state_pos=state.coords,
                last_state_log_prob=state.log_prob
            )
            print(f"[Checkpoint] saved to {chk_path}")

        # -------- 保存丢弃 burn-in + 抽样后的三维链 --------
        # emcee 的 get_chain 支持 discard/thin 并保持 (nsteps_kept, nwalkers, ndim) 形状
        chain_kept = sampler.get_chain(discard=burnin_to_save, thin=thin_to_save, flat=False)
        kept_path = f"{out_prefix}_chain_kept.npy"
        np.save(kept_path, chain_kept)
        print(f"[Saved] kept chain -> {kept_path}  shape={chain_kept.shape}")

        # -------- 删除所有 checkpoint 文件 --------
        for fp in glob.glob(f"{out_prefix}_chk*.npz"):
            try:
                os.remove(fp)
                print(f"[Cleanup] removed {fp}")
            except Exception as e:
                print(f"[Cleanup] failed to remove {fp}: {e}")

        return sampler, state



out_path = r"/home/astrodust/SG/sgl_cosmo/codes/Output"
out_name = r"-0.1_bias"


sampler, state = run_mcmc_dd(
    zl, zs, R*0.9, R*0.01, H0=70.0,
    init=(0.3, -1.0), nwalkers=200, nsteps=10000,
    threads=12, out_prefix=f"{out_path}/{out_name}",
    sigma_is_fractional=False,
    checkpoint_size = 1000
)
chain = sampler.get_chain(discard=2000, thin=10, flat=True)
