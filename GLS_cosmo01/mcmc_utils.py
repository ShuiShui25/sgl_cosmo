#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 16:13:31 2024

@author: gengs
"""
from astropy.table import Table
import numpy as np
import astropy.units as u
import scipy.integrate as sci
import math
from scipy.special import gamma
from derivative import df_delta, df_gamma
from astropy.constants import G, c, M_sun
import os
from scipy.stats import median_abs_deviation
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import emcee
from multiprocessing import Pool
import pandas as pd
from scipy.special import gamma
from scipy.integrate import quad
from scipy import integrate
import time
from astropy import constants as const
import warnings
# warnings.filterwarnings("ignore")
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


def integrate_da_good_evow(z1, z2, w0, wa, Ok, H0, Om, c=c):
    """
    Angular-diameter distance D_A(z1->z2) in Mpc for CPL w(z)=w0+wa*z/(1+z).
    """
    z1 = np.atleast_1d(z1)
    z2 = np.atleast_1d(z2)

    if len(z1) > 1 and len(z2) > 1 and len(z1) != len(z2):
        raise ValueError("z1 and z2 must be the same length, or one must be scalar.")
    if len(z1) == 1 and len(z2) > 1:
        z1 = np.full_like(z2, z1)
    elif len(z2) == 1 and len(z1) > 1:
        z2 = np.full_like(z1, z2)

    Ode = 1.0 - Om - Ok

    def integrand(z):
        # CPL: ρ_de(z)/ρ_de0 = (1+z)^{3(1+w0+wa)} * exp[-3*wa*z/(1+z)]
        de_factor = (1.0+z)**(3.0*(1.0+w0+wa)) * np.exp(-3.0*wa*z/(1.0+z))
        Ez2 = Om*(1.0+z)**3 + Ok*(1.0+z)**2 + Ode*de_factor
        return 1.0/np.sqrt(Ez2)

    def sinn(Ok, x):
        if Ok < 0.0:
            k = np.sqrt(abs(Ok))
            return np.sin(k*x)/k
        elif Ok > 0.0:
            k = np.sqrt(abs(Ok))
            return np.sinh(k*x)/k
        else:
            return x

    out = []
    for zs, ze in np.column_stack((z1, z2)):
        chi, _ = sci.quad(integrand, zs, ze)
        DA = (c_km/H0) * sinn(Ok, chi) / (1.0 + ze)
        out.append(DA)

    out = np.array(out)
    return out if out.size > 1 else out[0]
    

from scipy.special import exp1  # E1(x) = ∫_x^∞ (e^{-t}/t) dt

def integrate_da_wphi(z1, z2, w0, alpha, Ok, H0, Om, c=c_km):
    """
    Angular-diameter distance D_A(z1 -> z2) in Mpc for the w_phi thawing-fit model:
        w_phi(z) = -1 + (1 + w0) * exp(-alpha * z)
        F(z; w_phi) = exp( 3 * exp(alpha) * (1+w0) * [ E1(alpha) - E1(alpha*(1+z)) ] )

    H(z)^2 / H0^2 = Om*(1+z)^3 + Ok*(1+z)^2 + Ode*F(z),  where Ode = 1-Om-Ok.

    Parameters
    ----------
    z1, z2 : float or array
        Start and end redshifts (allow broadcasting as in the original function).
    w0     : float
        Present-day w_phi(0) value.
    alpha  : float
        Exponential fall-off parameter in the thawing-fit.
    Ok     : float
        Curvature density parameter Ω_k.
    H0     : float
        Hubble constant in km s^-1 Mpc^-1.
    Om     : float
        Matter density parameter Ω_m.
    c      : float
        Speed of light in km s^-1 (default 299792.458).

    Returns
    -------
    D_A : float or ndarray
        Angular-diameter distance in Mpc from z1 to z2 (same shape/broadcasting as inputs).
    """
    z1 = np.atleast_1d(z1)
    z2 = np.atleast_1d(z2)

    if len(z1) > 1 and len(z2) > 1 and len(z1) != len(z2):
        raise ValueError("z1 and z2 must be the same length, or one must be scalar.")
    if len(z1) == 1 and len(z2) > 1:
        z1 = np.full_like(z2, z1)
    elif len(z2) == 1 and len(z1) > 1:
        z2 = np.full_like(z1, z2)

    Ode = 1.0 - Om - Ok

    # Precompute constants appearing in F(z)
    A = 3.0 * np.exp(alpha) * (1.0 + w0)
    E1_alpha = exp1(alpha)

    def F_of_z(z):
        # F(z) = exp( A * (E1(alpha) - E1(alpha*(1+z))) )
        return np.exp(A * (E1_alpha - exp1(alpha * (1.0 + z))))

    def Ez_inv(z):
        Ez2 = Om * (1.0 + z) ** 3 + Ok * (1.0 + z) ** 2 + Ode * F_of_z(z)
        return 1.0 / np.sqrt(Ez2)

    def sinn(Ok, x):
        if Ok < 0.0:
            k = np.sqrt(-Ok)
            return np.sin(k * x) / k
        elif Ok > 0.0:
            k = np.sqrt(Ok)
            return np.sinh(k * x) / k
        else:
            return x

    out = []
    for zs, ze in np.column_stack((z1, z2)):
        # Handle ze == zs quickly
        if np.isclose(ze, zs):
            out.append(0.0)
            continue
        # Line-of-sight comoving distance χ = ∫ dz / E(z)
        chi, _ = sci.quad(Ez_inv, zs, ze, epsabs=0, epsrel=1e-8, limit=200)
        DA = (c / H0) * sinn(Ok, chi) / (1.0 + ze)
        out.append(DA)

    out = np.array(out)
    return out if out.size > 1 else out[0]




def f_gama(x):
    a = -1 / np.sqrt(np.pi)
    b = ((5 - 2 * x) * (1 - x)) / (3 - x)
    c = gamma(x - 1) / gamma(x - (3 / 2))
    d = gamma((x - 1) / 2) / gamma(x / 2)

    return a * b * c * np.square(d)

#def f_prime(gamma_val, delta, beta):
#    prefactor = 1. / (2. * np.sqrt(np.pi))
#    term1 = (gamma_val + delta - 5.) * (gamma_val + delta - 2. - 2. * beta) / (delta - 3.)
#    term2_numerator = gamma((gamma_val + delta - 2.) / 2.) * gamma((gamma_val + delta) / 2.)
#    term2_denominator = gamma((gamma_val + delta) / 2.) * gamma((gamma_val + delta - 3.) / 2.) - beta * gamma((gamma_val + delta - 2.) / 2.) * gamma((gamma_val + delta - 1.) / 2.)
#    term3 = gamma((delta - 1.) / 2.) * gamma((gamma_val - 1.) / 2.) / (gamma(delta / 2.) * gamma(gamma_val / 2.))
#    
#    return prefactor * term1 * (term2_numerator / term2_denominator) * term3
    
def f_prime(g, d, beta):
    pref = 1.0 / (2.0 * np.sqrt(np.pi))
    term1 = (g + d - 5.0) * (g + d - 2.0 - 2.0*beta) / (d - 3.0)
    num = gamma((g + d - 2.0)/2.0) * gamma((g + d)/2.0)
    den = gamma((g + d)/2.0) * gamma((g + d - 3.0)/2.0) \
        - beta * gamma((g + d - 2.0)/2.0) * gamma((g + d - 1.0)/2.0)
    term3 = gamma((d - 1.0)/2.0) * gamma((g - 1.0)/2.0) / (gamma(d/2.0) * gamma(g/2.0))
    return pref * term1 * (num/den) * term3

def dd_th(Om, w, zl, zs, H0):
    D_s = integrate_da_good_kw(0., zs, w, 0., H0, Om, c_km)
    D_ls = integrate_da_good_kw(zl, zs, w, 0., H0, Om, c_km)
    return D_ls / D_s

def dd_th_evow(Om, w0, wa, zl, zs, H0):
    D_s  = integrate_da_good_evow(0.0, zs, w0, wa, 0.0, H0, Om, c_km)
    D_ls = integrate_da_good_evow(zl,  zs, w0, wa, 0.0, H0, Om, c_km)
    return D_ls / D_s
    
def dd_th_wphi(Om, w0, alpha, zl, zs, H0):
    D_s  = integrate_da_good_evow(0.0, zs, w0, alpha, 0.0, H0, Om, c_km)
    D_ls = integrate_da_good_evow(zl,  zs, w0, alpha, 0.0, H0, Om, c_km)
    return D_ls / D_s
    
def dd_th_kw(Om, ok, w, zl, zs, H0):
    if (not np.isfinite(zl)) or (not np.isfinite(zs)) or zs <= zl:
        return np.nan 
    Ds  = integrate_da_good_kw(0.0, zs, w, ok, H0, Om, c_km)
    Dls = integrate_da_good_kw(zl,  zs, w, ok, H0, Om, c_km)
    if (not np.isfinite(Ds)) or (not np.isfinite(Dls)) or Ds <= 0.0 or Dls <= 0.0:
        return np.nan
    return Dls / Ds
    
def dd_th_kevow(Om, ok, w0, wa, zl, zs, H0):
    """
    D_ls/D_s for w0–wa with curvature.
    Assumes integrate_da_good_evow signature: (z1, z2, w0, wa, ok, H0, Om, c_km)
    """
    D_s  = integrate_da_good_evow(0.0, zs, w0, wa, ok, H0, Om, c_km)
    D_ls = integrate_da_good_evow(zl,  zs, w0, wa, ok, H0, Om, c_km)
    return D_ls / D_s

def dd_obs(gamma_var, delta, beta, thetaE, theta_ap, sigma_ap):
    dd = c_km**2. / (4. * np.pi) * thetaE / sigma_ap**2.0 * (thetaE / theta_ap)**(gamma_var - 2.0) / f_prime(gamma_var, delta, beta)
    return dd
    
def ln_f_sph(g, d, beta):
    return np.log(f_prime(g, d, beta))
    
def dlnf_dx(g, d, beta, var='g', h=1e-4):
    if var=='g':
        return (ln_f_sph(g+h, d, beta) - ln_f_sph(g-h, d, beta)) / (2*h)
    if var=='d':
        return (ln_f_sph(g, d+h, beta) - ln_f_sph(g, d-h, beta)) / (2*h)
    if var=='b':
        return (ln_f_sph(g, d, beta+h) - ln_f_sph(g, d, beta-h)) / (2*h)
    raise ValueError("var must be 'g','d', or 'b'")
    
def ln_dd_obs(g, d, beta, thetaE, theta_ap, sigma_ap):
    return np.log(dd_obs(g, d, beta, thetaE, theta_ap, sigma_ap))
    
### 2D Omega_m+w
    
def lnchi2(Om, w, g, d, beta, zl, zs, H0,
           thetaE, sigma_ap, sig_sigma, theta_ap,
           d_thetaE=0.05, sig_g=0.0, sig_d=0.0, sig_b=0.0):
    dr_th = dd_th(Om, w, zl, zs, H0)
    ln_dr_th = np.log(dr_th)

    ln_dr_obs = ln_dd_obs(g, d, beta, thetaE, theta_ap, sigma_ap)

    dlnf_dg = dlnf_dx(g, d, beta, 'g')
    dlnf_dd = dlnf_dx(g, d, beta, 'd')
    dlnf_db = dlnf_dx(g, d, beta, 'b')

    var_ln = (
        (g - 1.0)**2 * (d_thetaE)**2
        + 4.0 * (sig_sigma / sigma_ap)**2
        + (np.log(thetaE / theta_ap) - dlnf_dg)**2 * sig_g**2
        + (dlnf_dd)**2 * sig_d**2
        + (dlnf_db)**2 * sig_b**2
    )

    var_ln = np.maximum(var_ln, 1e-16)
    return ((ln_dr_th - ln_dr_obs)**2) / var_ln


def lnlike(Om, w, g, d, beta, zl, zs, H0,
           thetaE, sigma_ap, sig_sigma, theta_ap,
           d_thetaE=0.05, sig_g=0.0, sig_d=0.0, sig_b=0.0):
    lnchi2_val = lnchi2(Om, w, g, d, beta, zl, zs, H0,
                        thetaE, sigma_ap, sig_sigma, theta_ap,
                        d_thetaE=d_thetaE, sig_g=sig_g, sig_d=sig_d, sig_b=sig_b)
    return -0.5 * lnchi2_val


def lnprob_cos2D(x,
                 beta0, beta_err,
                 gamma_var, delta_var, gamma_err, delta_err,
                 zl, zs, thetaE, theta_ap, H0, sigma_ap, sigma_ap_err):

    Om, w = x
    beta = beta0

    if not (0.02 < Om < 1.0) or not (-4.0 < w < 2.0):
        return -np.inf

    total_lnL = 0.0

    for i in range(len(zl)):
        f_g = f_prime(gamma_var[i], delta_var[i], beta)
        if (not np.isfinite(f_g)) or (f_g <= 0.0):
            return -np.inf

        lnL_i = lnlike(Om, w,
                       gamma_var[i], delta_var[i], beta,
                       zl[i], zs[i], H0,
                       thetaE[i], sigma_ap[i], sigma_ap_err[i], theta_ap[i],
                       d_thetaE=0.05,
                       sig_g=gamma_err[i],
                       sig_d=delta_err[i],
                       sig_b=beta_err)
        if not np.isfinite(lnL_i):
            return -np.inf

        total_lnL += lnL_i

    ln_prior_beta = -0.5 * ((beta - beta0) / beta_err)**2 - np.log(np.sqrt(2*np.pi) * beta_err)
    return total_lnL + ln_prior_beta


### # 4D
### # gamma, omega_m, w, beta

def lnchi2_4D_core(Om, w, g0, beta,
                   d0, ds, zl_i, zl_med, zs_i, H0,
                   thetaE_i, sigma_ap_i, sig_sigma_i, theta_ap_i,
                   d_thetaE=0.05, sig_b=0.0):
    """
    Core per-system chi2 using:
      gamma_i = g0 (constant across lenses)
      delta_i = d0 + ds * (zl_i - zl_med)   # <-- from lumi_params
    Error model keeps theta_E and sigma_ap terms and beta prior width.
    """
    # evolution
    gamma_i = g0
    delta_i = d0 + ds * (zl_i - zl_med)

    # sanity for model
    f_g = f_prime(gamma_i, delta_i, beta)
    if (not np.isfinite(f_g)) or (f_g <= 0.0) or (not (1.5 < gamma_i < 2.5)):
        return np.inf  # invalid → kills likelihood upstream

    # theory/obs ln distance-ratio
    dr_th    = dd_th(Om, w, zl_i, zs_i, H0)
    ln_th    = np.log(dr_th)
    ln_obs   = ln_dd_obs(gamma_i, delta_i, beta, thetaE_i, theta_ap_i, sigma_ap_i)

    # derivatives – we only keep beta's contribution in var, as requested
    dlnf_db  = dlnf_dx(gamma_i, delta_i, beta, 'b')

    # variance: keep thetaE & sigma_ap terms + beta prior width
    var_ln = (
        (gamma_i - 1.0)**2 * (d_thetaE)**2
        + 4.0 * (sig_sigma_i / max(sigma_ap_i, 1e-16))**2
        + (dlnf_db**2) * (sig_b**2)
    )
    var_ln = max(var_ln, 1e-16)

    return ((ln_th - ln_obs)**2) / var_ln


def lnprob_cos4D(x, d0, ds, beta0, beta_err,
                 zl, zs, thetaE, theta_ap, H0, sigma_ap, sigma_ap_err):
    """
    Sample only [Om, w, g0, beta].
    Build delta_i from lumi_params = (d0, ds).
    Apply a Gaussian prior on beta around (beta0, beta_err).
    """
    Om, w, g0, beta = x

    # simple box priors
    if not (0.02 < Om < 1.0):
        return -np.inf
    if not (-4.0 < w < 2.0):
        return -np.inf

    # median lens-z for centered evolution
    zl_med = np.median(zl)

    # accumulate lnL over systems
    total_lnL = 0.0
    for i in range(len(zl)):
        chi2_i = lnchi2_4D_core(
            Om, w, g0, beta,
            d0, ds, zl[i], zl_med, zs[i], H0,
            thetaE[i], sigma_ap[i], sigma_ap_err[i], theta_ap[i],
            d_thetaE=0.05, sig_b=beta_err
        )
        if not np.isfinite(chi2_i):
            return -np.inf
        total_lnL += (-0.5 * chi2_i)

    # Gaussian prior on beta
    ln_prior_beta = -0.5 * ((beta - beta0) / beta_err)**2 - np.log(np.sqrt(2*np.pi) * beta_err)

    return total_lnL + ln_prior_beta



### # 5D_gamma
### # gamma_0, gamma_s, omega_m, w, beta
def lnchi2_5D_gamma(Om, w, g0, gs, beta,
                   d0, ds, zl_i, zl_med, zs_i, H0,
                   thetaE_i, sigma_ap_i, sig_sigma_i, theta_ap_i,
                   d_thetaE=0.05, sig_b=0.0):
    """
    Core per-system chi2 using:
      gamma_i = g0 (constant across lenses)
      delta_i = d0 + ds * (zl_i - zl_med)   # <-- from lumi_params
    Error model keeps theta_E and sigma_ap terms and beta prior width.
    """
    # evolution
    gamma_i = g0 + gs * (zl_i - zl_med)
    delta_i = d0 + ds * (zl_i - zl_med)

    # sanity for model
    f_g = f_prime(gamma_i, delta_i, beta)
    if (not np.isfinite(f_g)) or (f_g <= 0.0) or (not (1.5 < gamma_i < 2.5)):
        return np.inf  # invalid → kills likelihood upstream

    # theory/obs ln distance-ratio
    dr_th    = dd_th(Om, w, zl_i, zs_i, H0)
    ln_th    = np.log(dr_th)
    ln_obs   = ln_dd_obs(gamma_i, delta_i, beta, thetaE_i, theta_ap_i, sigma_ap_i)

    # derivatives – we only keep beta's contribution in var, as requested
    dlnf_db  = dlnf_dx(gamma_i, delta_i, beta, 'b')

    # variance: keep thetaE & sigma_ap terms + beta prior width
    var_ln = (
        (gamma_i - 1.0)**2 * (d_thetaE)**2
        + 4.0 * (sig_sigma_i / max(sigma_ap_i, 1e-16))**2
        + (dlnf_db**2) * (sig_b**2)
    )
    var_ln = max(var_ln, 1e-16)

    return ((ln_th - ln_obs)**2) / var_ln


def lnprob_cos5D_gamma(x, d0, ds, beta0, beta_err,
                 zl, zs, thetaE, theta_ap, H0, sigma_ap, sigma_ap_err):
    """
    Sample only [Om, w, g0, beta].
    Build delta_i from lumi_params = (d0, ds).
    Apply a Gaussian prior on beta around (beta0, beta_err).
    """
    Om, w, g0, gs, beta = x

    # simple box priors
    if not (0.02 < Om < 1.0):
        return -np.inf
    if not (-4.0 < w < 2.0):
        return -np.inf

    # median lens-z for centered evolution
    zl_med = np.median(zl)

    # accumulate lnL over systems
    total_lnL = 0.0
    for i in range(len(zl)):
        chi2_i = lnchi2_5D_gamma(
            Om, w, g0, gs, beta,
            d0, ds, zl[i], zl_med, zs[i], H0,
            thetaE[i], sigma_ap[i], sigma_ap_err[i], theta_ap[i],
            d_thetaE=0.05, sig_b=beta_err
        )
        if not np.isfinite(chi2_i):
            return -np.inf
        total_lnL += (-0.5 * chi2_i)

    # Gaussian prior on beta
    ln_prior_beta = -0.5 * ((beta - beta0) / beta_err)**2 - np.log(np.sqrt(2*np.pi) * beta_err)

    return total_lnL + ln_prior_beta


### # 6D_w0wa_gamma
### # gamma_0, gamma_s, omega_m, w0, wa, beta

def lnchi2_evow_core(Om, w0, wa, g0, gs, beta,
                     d0, ds, zl_i, zl_med, zs_i, H0,
                     thetaE_i, sigma_ap_i, sig_sigma_i, theta_ap_i,
                     d_thetaE=0.05, sig_b=0.0):
    """
    Per-system χ² for the w0–wa cosmology with linear gamma & delta evolution:
      gamma_i = g0 + gs * (zl_i - zl_med)
      delta_i = d0 + ds * (zl_i - zl_med)
    Error model keeps thetaE & sigma_ap terms + beta prior width only.
    """
    # evolved structure params
    g_i = g0 + gs * (zl_i - zl_med)
    d_i = d0 + ds * (zl_i - zl_med)

    # physical sanity
    f_g = f_prime(g_i, d_i, beta)
    if (not np.isfinite(f_g)) or (f_g <= 0.0) or (not (1.2 < g_i < 2.8)):
        return np.inf

    # theory/obs ln(distance ratio)
    dr_th   = dd_th_evow(Om, w0, wa, zl_i, zs_i, H0)
    ln_th   = np.log(dr_th)
    ln_obs  = ln_dd_obs(g_i, d_i, beta, thetaE_i, theta_ap_i, sigma_ap_i)

    # derivative wrt beta for its prior propagation
    dlnf_db = dlnf_dx(g_i, d_i, beta, 'b')

    # variance: keep thetaE & sigma_ap + beta prior width
    var_ln = (
        (g_i - 1.0)**2 * (d_thetaE)**2
        + 4.0 * (sig_sigma_i / max(sigma_ap_i, 1e-16))**2
        + (dlnf_db**2) * (sig_b**2)
    )
    var_ln = max(var_ln, 1e-16)

    return ((ln_th - ln_obs)**2) / var_ln


def lnprob_cos6D_w0wa_gamma(x, d0, ds, beta0, beta_err,
                            zl, zs, thetaE, theta_ap, H0, sigma_ap, sigma_ap_err):
    """
    Sample only [Om, w0, wa, g0, gs, beta].
    Build gamma_i and delta_i from (g0, gs) and (d0, ds) centered at median(zl).
    Apply a Gaussian prior on beta around (beta0, beta_err).
    """
    Om, w0, wa, g0, gs, beta = x

    # box priors (adjust as you like)
    if not (0.02 < Om < 1.0):       return -np.inf
    if not (-1.9 < w0 < 0.6):       return -np.inf
    if not (-4.0 < wa < 1.0):       return -np.inf
    if not (1.0  < g0 < 3.0):       return -np.inf
    if not (-2.0 < gs < 2.0):       return -np.inf

    zl_med = np.median(zl)

    total_lnL = 0.0
    for i in range(len(zl)):
        chi2_i = lnchi2_evow_core(
            Om, w0, wa, g0, gs, beta,
            d0, ds, zl[i], zl_med, zs[i], H0,
            thetaE[i], sigma_ap[i], sigma_ap_err[i], theta_ap[i],
            d_thetaE=0.05, sig_b=beta_err
        )
        if not np.isfinite(chi2_i):
            return -np.inf
        total_lnL += (-0.5 * chi2_i)

    # Gaussian prior on beta
    ln_prior_beta = -0.5 * ((beta - beta0) / beta_err)**2 - np.log(np.sqrt(2*np.pi) * beta_err)

    return total_lnL + ln_prior_beta


### # 6D_wphi_gamma
### # gamma_0, gamma_s, omega_m, w0, alpha, beta

def lnchi2_wphi_core(Om, w0, alpha, g0, gs, beta,
                     d0, ds, zl_i, zl_med, zs_i, H0,
                     thetaE_i, sigma_ap_i, sig_sigma_i, theta_ap_i,
                     d_thetaE=0.05, sig_b=0.0):
    """
    Per-system χ² for the w0–wa cosmology with linear gamma & delta evolution:
      gamma_i = g0 + gs * (zl_i - zl_med)
      delta_i = d0 + ds * (zl_i - zl_med)
    Error model keeps thetaE & sigma_ap terms + beta prior width only.
    """
    # evolved structure params
    g_i = g0 + gs * (zl_i - zl_med)
    d_i = d0 + ds * (zl_i - zl_med)

    # physical sanity
    f_g = f_prime(g_i, d_i, beta)
    if (not np.isfinite(f_g)) or (f_g <= 0.0) or (not (1.2 < g_i < 2.8)):
        return np.inf

    # theory/obs ln(distance ratio)
    dr_th   = dd_th_wphi(Om, w0, alpha, zl_i, zs_i, H0)
    ln_th   = np.log(dr_th)
    ln_obs  = ln_dd_obs(g_i, d_i, beta, thetaE_i, theta_ap_i, sigma_ap_i)

    # derivative wrt beta for its prior propagation
    dlnf_db = dlnf_dx(g_i, d_i, beta, 'b')

    # variance: keep thetaE & sigma_ap + beta prior width
    var_ln = (
        (g_i - 1.0)**2 * (d_thetaE)**2
        + 4.0 * (sig_sigma_i / max(sigma_ap_i, 1e-16))**2
        + (dlnf_db**2) * (sig_b**2)
    )
    var_ln = max(var_ln, 1e-16)

    return ((ln_th - ln_obs)**2) / var_ln


def lnprob_cos6D_wphi_gamma(x, d0, ds, beta0, beta_err,
                            zl, zs, thetaE, theta_ap, H0, sigma_ap, sigma_ap_err):
    """
    Sample only [Om, w0, wa, g0, gs, beta].
    Build gamma_i and delta_i from (g0, gs) and (d0, ds) centered at median(zl).
    Apply a Gaussian prior on beta around (beta0, beta_err).
    """
    Om, w0, alpha, g0, gs, beta = x

    # box priors (adjust as you like)
    if not (0.02 < Om < 1.0):       return -np.inf
    if not (-1.6 < w0 < -0.4):       return -np.inf
    if not (1.35 < alpha < 1.55):       return -np.inf
    if not (1.0  < g0 < 3.0):       return -np.inf
    if not (-2.0 < gs < 2.0):       return -np.inf

    zl_med = np.median(zl)

    total_lnL = 0.0
    for i in range(len(zl)):
        chi2_i = lnchi2_wphi_core(
            Om, w0, alpha, g0, gs, beta,
            d0, ds, zl[i], zl_med, zs[i], H0,
            thetaE[i], sigma_ap[i], sigma_ap_err[i], theta_ap[i],
            d_thetaE=0.05, sig_b=beta_err
        )
        if not np.isfinite(chi2_i):
            return -np.inf
        total_lnL += (-0.5 * chi2_i)

    # Gaussian prior on beta
    ln_prior_beta = -0.5 * ((beta - beta0) / beta_err)**2 - np.log(np.sqrt(2*np.pi) * beta_err)

    return total_lnL + ln_prior_beta

### Omega_m+w0+wa

def lnchi2_evow(Om, w0, wa, g, d, beta, zl, zs, H0,
                thetaE, sigma_ap, sig_sigma, theta_ap,
                d_thetaE=0.05, sig_g=0.0, sig_d=0.0, sig_b=0.0):
    dr_th = dd_th_evow(Om, w0, wa, zl, zs, H0)
    ln_dr_th = np.log(dr_th)

    ln_dr_obs = ln_dd_obs(g, d, beta, thetaE, theta_ap, sigma_ap)

    dlnf_dg = dlnf_dx(g, d, beta, 'g')
    dlnf_dd = dlnf_dx(g, d, beta, 'd')
    dlnf_db = dlnf_dx(g, d, beta, 'b')

    var_ln = (
        (g - 1.0)**2 * (d_thetaE)**2
        + 4.0 * (sig_sigma / sigma_ap)**2
        + (np.log(thetaE / theta_ap) - dlnf_dg)**2 * sig_g**2
        + (dlnf_dd)**2 * sig_d**2
        + (dlnf_db)**2 * sig_b**2
    )
    var_ln = np.maximum(var_ln, 1e-16)
    return ((ln_dr_th - ln_dr_obs)**2) / var_ln

def lnlike_evow(Om, w0, wa, g, d, beta, zl, zs, H0,
                thetaE, sigma_ap, sig_sigma, theta_ap,
                d_thetaE=0.05, sig_g=0.0, sig_d=0.0, sig_b=0.0):
    lnchi2_val = lnchi2_evow(Om, w0, wa, g, d, beta, zl, zs, H0,
                             thetaE, sigma_ap, sig_sigma, theta_ap,
                             d_thetaE=d_thetaE, sig_g=sig_g, sig_d=sig_d, sig_b=sig_b)
    return -0.5 * lnchi2_val

def lnprob_cos3D_evow(x,
                      beta0, beta_err,
                      gamma_var, delta_var, gamma_err, delta_err,
                      zl, zs, thetaE, theta_ap, H0, sigma_ap, sigma_ap_err):
    Om, w0, wa = x
    beta = beta0

    if not (0.02 < Om < 1.0) or not (-1.9 < w0 < 0.6) or not (-4.0 < wa < 1.0):
        return -np.inf

    total_lnL = 0.0
    for i in range(len(zl)):
        f_g = f_prime(gamma_var[i], delta_var[i], beta)
        if (not np.isfinite(f_g)) or (f_g <= 0.0):
            return -np.inf

        lnL_i = lnlike_evow(Om, w0, wa,
                            gamma_var[i], delta_var[i], beta,
                            zl[i], zs[i], H0,
                            thetaE[i], sigma_ap[i], sigma_ap_err[i], theta_ap[i],
                            d_thetaE=0.05,
                            sig_g=gamma_err[i],
                            sig_d=delta_err[i],
                            sig_b=beta_err)
        if not np.isfinite(lnL_i):
            return -np.inf
        total_lnL += lnL_i

    ln_prior_beta = -0.5 * ((beta - beta0)/beta_err)**2 - np.log(np.sqrt(2*np.pi) * beta_err)
    return total_lnL + ln_prior_beta
    
    
### Omega_m+w0+alpha

#def lnchi2_wphi(Om, w0, alpha, g, d, beta, zl, zs, H0,
#                thetaE, sigma_ap, sig_sigma, theta_ap,
#                d_thetaE=0.05, sig_g=0.0, sig_d=0.0, sig_b=0.0):
#    dr_th = dd_th_wphi(Om, w0, alpha, zl, zs, H0)
#    ln_dr_th = np.log(dr_th)
#
#    ln_dr_obs = ln_dd_obs(g, d, beta, thetaE, theta_ap, sigma_ap)
#
#    dlnf_dg = dlnf_dx(g, d, beta, 'g')
#    dlnf_dd = dlnf_dx(g, d, beta, 'd')
#    dlnf_db = dlnf_dx(g, d, beta, 'b')
#
#    var_ln = (
#        (g - 1.0)**2 * (d_thetaE)**2
#        + 4.0 * (sig_sigma / sigma_ap)**2
#        + (np.log(thetaE / theta_ap) - dlnf_dg)**2 * sig_g**2
#        + (dlnf_dd)**2 * sig_d**2
#        + (dlnf_db)**2 * sig_b**2
#    )
#    var_ln = np.maximum(var_ln, 1e-16)
#    return ((ln_dr_th - ln_dr_obs)**2) / var_ln
#
#def lnlike_wphi(Om, w0, alpha, g, d, beta, zl, zs, H0,
#                thetaE, sigma_ap, sig_sigma, theta_ap,
#                d_thetaE=0.05, sig_g=0.0, sig_d=0.0, sig_b=0.0):
#    lnchi2_val = lnchi2_wphi(Om, w0, alpha, g, d, beta, zl, zs, H0,
#                             thetaE, sigma_ap, sig_sigma, theta_ap,
#                             d_thetaE=d_thetaE, sig_g=sig_g, sig_d=sig_d, sig_b=sig_b)
#    return -0.5 * lnchi2_val
    
def lnchi2_wphi(Om, w0, alpha, g, d, beta, zl, zs, H0,
                thetaE, sigma_ap, sig_sigma, theta_ap,
                d_thetaE=0.05, sig_g=0.0, sig_d=0.0, sig_b=0.0):

    # ---- theory distance ratio (guard) ----
    dr_th = dd_th_wphi(Om, w0, alpha, zl, zs, H0)
    if not np.isfinite(dr_th) or dr_th <= 0.0:
        return np.inf
    ln_dr_th = np.log(dr_th)

    # ---- observed ln(dd) (assumes ln_dd_obs already stable wrt inputs) ----
    if (thetaE <= 0.0) or (theta_ap <= 0.0) or (sigma_ap <= 0.0):
        return np.inf
    ln_dr_obs = ln_dd_obs(g, d, beta, thetaE, theta_ap, sigma_ap)

    # ---- derivatives ----
    dlnf_dg = dlnf_dx(g, d, beta, 'g')
    dlnf_dd = dlnf_dx(g, d, beta, 'd')
    dlnf_db = dlnf_dx(g, d, beta, 'b')

    # ---- variance (with clamps) ----
    # clamp denominators and logs to avoid NaNs
    _sigma_ap = max(float(sigma_ap), 1e-16)
    _thetaE   = max(float(thetaE),   1e-16)
    _theta_ap = max(float(theta_ap), 1e-16)

    log_ratio = np.log(_thetaE / _theta_ap)

    var_ln = (
        (g - 1.0)**2 * (d_thetaE)**2
        + 4.0 * (float(sig_sigma) / _sigma_ap)**2
        + (log_ratio - dlnf_dg)**2 * (float(sig_g)**2)
        + (dlnf_dd)**2 * (float(sig_d)**2)
        + (dlnf_db)**2 * (float(sig_b)**2)
    )
    if not np.isfinite(var_ln):
        return np.inf

    var_ln = max(float(var_ln), 1e-16)

    chi2 = ((ln_dr_th - ln_dr_obs)**2) / var_ln
    if not np.isfinite(chi2):
        return np.inf
    return chi2


def lnlike_wphi(Om, w0, alpha, g, d, beta, zl, zs, H0,
                thetaE, sigma_ap, sig_sigma, theta_ap,
                d_thetaE=0.05, sig_g=0.0, sig_d=0.0, sig_b=0.0):
    chi2 = lnchi2_wphi(Om, w0, alpha, g, d, beta, zl, zs, H0,
                       thetaE, sigma_ap, sig_sigma, theta_ap,
                       d_thetaE=d_thetaE, sig_g=sig_g, sig_d=sig_d, sig_b=sig_b)
    if not np.isfinite(chi2):
        return -np.inf
    return -0.5 * chi2


def lnprob_cos3D_wphi(x,
                      beta0, beta_err,
                      gamma_var, delta_var, gamma_err, delta_err,
                      zl, zs, thetaE, theta_ap, H0, sigma_ap, sigma_ap_err):
    Om, w0, alpha = x
    beta = beta0

    if not (0.02 < Om < 1.0) or not (-1.6 < w0 < -0.4) or not (1.35 < alpha < 1.55):
        return -np.inf

    total_lnL = 0.0
    for i in range(len(zl)):
        f_g = f_prime(gamma_var[i], delta_var[i], beta)
        if (not np.isfinite(f_g)) or (f_g <= 0.0):
            return -np.inf

        lnL_i = lnlike_wphi(Om, w0, alpha,
                            gamma_var[i], delta_var[i], beta,
                            zl[i], zs[i], H0,
                            thetaE[i], sigma_ap[i], sigma_ap_err[i], theta_ap[i],
                            d_thetaE=0.05,
                            sig_g=gamma_err[i],
                            sig_d=delta_err[i],
                            sig_b=beta_err)
        if not np.isfinite(lnL_i):
            return -np.inf
        total_lnL += lnL_i

    ln_prior_beta = -0.5 * ((beta - beta0)/beta_err)**2 - np.log(np.sqrt(2*np.pi) * beta_err)
    return total_lnL + ln_prior_beta

### Omega_k+w0+wa

def lnchi2_kevow(Om,ok, w0, wa, g, d, beta, zl, zs, H0,
                thetaE, sigma_ap, sig_sigma, theta_ap,
                d_thetaE=0.05, sig_g=0.0, sig_d=0.0, sig_b=0.0):
    dr_th = dd_th_kevow(Om,ok, w0, wa, zl, zs, H0)
    ln_dr_th = np.log(dr_th)

    ln_dr_obs = ln_dd_obs(g, d, beta, thetaE, theta_ap, sigma_ap)

    dlnf_dg = dlnf_dx(g, d, beta, 'g')
    dlnf_dd = dlnf_dx(g, d, beta, 'd')
    dlnf_db = dlnf_dx(g, d, beta, 'b')

    var_ln = (
        (g - 1.0)**2 * (d_thetaE)**2
        + 4.0 * (sig_sigma / sigma_ap)**2
        + (np.log(thetaE / theta_ap) - dlnf_dg)**2 * sig_g**2
        + (dlnf_dd)**2 * sig_d**2
        + (dlnf_db)**2 * sig_b**2
    )
    var_ln = np.maximum(var_ln, 1e-16)
    return ((ln_dr_th - ln_dr_obs)**2) / var_ln

def lnlike_kevow(Om, ok, w0, wa, g, d, beta, zl, zs, H0,
                thetaE, sigma_ap, sig_sigma, theta_ap,
                d_thetaE=0.05, sig_g=0.0, sig_d=0.0, sig_b=0.0):
    lnchi2_val = lnchi2_kevow(Om,ok, w0, wa, g, d, beta, zl, zs, H0,
                             thetaE, sigma_ap, sig_sigma, theta_ap,
                             d_thetaE=d_thetaE, sig_g=sig_g, sig_d=sig_d, sig_b=sig_b)
    return -0.5 * lnchi2_val

def lnprob_cos3D_kevow(x,
                      beta0, beta_err,
                      gamma_var, delta_var, gamma_err, delta_err,
                      zl, zs, thetaE, theta_ap, H0, sigma_ap, sigma_ap_err):
    ok, w0, wa = x
    Om = 0.3
    beta = beta0

    if not (-0.5 < ok < 0.5) or not (-1.9 < w0 < 0.6) or not (-4.0 < wa < 1.0):
        return -np.inf

    total_lnL = 0.0
    for i in range(len(zl)):
        f_g = f_prime(gamma_var[i], delta_var[i], beta)
        if (not np.isfinite(f_g)) or (f_g <= 0.0):
            return -np.inf

        lnL_i = lnlike_kevow(Om,ok, w0, wa,
                            gamma_var[i], delta_var[i], beta,
                            zl[i], zs[i], H0,
                            thetaE[i], sigma_ap[i], sigma_ap_err[i], theta_ap[i],
                            d_thetaE=0.05,
                            sig_g=gamma_err[i],
                            sig_d=delta_err[i],
                            sig_b=beta_err)
        if not np.isfinite(lnL_i):
            return -np.inf
        total_lnL += lnL_i

    ln_prior_beta = -0.5 * ((beta - beta0)/beta_err)**2 - np.log(np.sqrt(2*np.pi) * beta_err)
    return total_lnL + ln_prior_beta

### Omega_m+Omega_k+w

def lnchi2_mkw(Om, ok, w, g, d, beta, zl, zs, H0,
              thetaE, sigma_ap, sig_sigma, theta_ap,
              d_thetaE=0.05, sig_g=0.0, sig_d=0.0, sig_b=0.0):
    dr_th = dd_th_kw(Om, ok, w, zl, zs, H0)
    if (not np.isfinite(dr_th)) or dr_th <= 0.0:
        return np.inf

    ln_dr_th  = np.log(dr_th)
    ln_dr_obs = ln_dd_obs(g, d, beta, thetaE, theta_ap, sigma_ap)
    if (not np.isfinite(ln_dr_obs)):
        return np.inf

    dlnf_dg = dlnf_dx(g, d, beta, 'g')
    dlnf_dd = dlnf_dx(g, d, beta, 'd')
    dlnf_db = dlnf_dx(g, d, beta, 'b')

    var_ln = (
        (g - 1.0)**2 * (d_thetaE)**2
        + 4.0 * (sig_sigma / max(sigma_ap, 1e-12))**2
        + (np.log(thetaE / theta_ap) - dlnf_dg)**2 * sig_g**2
        + (dlnf_dd)**2 * sig_d**2
        + (dlnf_db)**2 * sig_b**2
    )
    var_ln = np.maximum(var_ln, 1e-16)
    return ((ln_dr_th - ln_dr_obs)**2) / var_ln




def lnlike_mkw(Om, ok, w, g, d, beta, zl, zs, H0,
              thetaE, sigma_ap, sig_sigma, theta_ap,
              d_thetaE=0.05, sig_g=0.0, sig_d=0.0, sig_b=0.0):
    return -0.5 * lnchi2_mkw(Om, ok, w, g, d, beta, zl, zs, H0,
                             thetaE, sigma_ap, sig_sigma, theta_ap,
                             d_thetaE=d_thetaE, sig_g=sig_g, sig_d=sig_d, sig_b=sig_b)


def lnprob_mkw(x,
              beta0, beta_err,
              gamma_var, delta_var, gamma_err, delta_err,
              zl, zs, thetaE, theta_ap, H0, sigma_ap, sigma_ap_err):
    Om, ok, w = x
    beta = beta0


    if not (0.02 < Om < 1.0): return -np.inf
    if not (-0.5 < ok < 0.5): return -np.inf
    if not (-4.0 < w  < 2.0): return -np.inf
    if (1.0 - Om - ok) <= 0.0: return -np.inf

    total_lnL = 0.0
    for i in range(len(zl)):
        if not (np.isfinite(zl[i]) and np.isfinite(zs[i])) or (zs[i] <= zl[i]):
            return -np.inf

        f_g = f_prime(gamma_var[i], delta_var[i], beta)
        if (not np.isfinite(f_g)) or (f_g <= 0.0): return -np.inf

        lnL_i = lnlike_mkw(Om, ok, w,
                          gamma_var[i], delta_var[i], beta,
                          zl[i], zs[i], H0,
                          thetaE[i], sigma_ap[i], sigma_ap_err[i], theta_ap[i],
                          d_thetaE=0.05,
                          sig_g=gamma_err[i],
                          sig_d=delta_err[i],
                          sig_b=beta_err)
        if not np.isfinite(lnL_i): return -np.inf
        total_lnL += lnL_i

    ln_prior_beta = -0.5*((beta - beta0)/beta_err)**2 - np.log(np.sqrt(2*np.pi)*beta_err)
    return total_lnL + ln_prior_beta

def lnprob_kw(x,
              beta0, beta_err,
              gamma_var, delta_var, gamma_err, delta_err,
              zl, zs, thetaE, theta_ap, H0, sigma_ap, sigma_ap_err):
    ok, w = x
    
    Om = 0.3
    beta = beta0


    if not (0.02 < Om < 1.0): return -np.inf
    if not (-0.5 < ok < 0.5): return -np.inf
    if not (-4.0 < w  < 2.0): return -np.inf
    if (1.0 - Om - ok) <= 0.0: return -np.inf

    total_lnL = 0.0
    for i in range(len(zl)):
        if not (np.isfinite(zl[i]) and np.isfinite(zs[i])) or (zs[i] <= zl[i]):
            return -np.inf

        f_g = f_prime(gamma_var[i], delta_var[i], beta)
        if (not np.isfinite(f_g)) or (f_g <= 0.0): return -np.inf

        lnL_i = lnlike_mkw(Om, ok, w,
                          gamma_var[i], delta_var[i], beta,
                          zl[i], zs[i], H0,
                          thetaE[i], sigma_ap[i], sigma_ap_err[i], theta_ap[i],
                          d_thetaE=0.05,
                          sig_g=gamma_err[i],
                          sig_d=delta_err[i],
                          sig_b=beta_err)
        if not np.isfinite(lnL_i): return -np.inf
        total_lnL += lnL_i

    ln_prior_beta = -0.5*((beta - beta0)/beta_err)**2 - np.log(np.sqrt(2*np.pi)*beta_err)
    return total_lnL + ln_prior_beta

def lnprob_mk(x,
              beta0, beta_err,
              gamma_var, delta_var, gamma_err, delta_err,
              zl, zs, thetaE, theta_ap, H0, sigma_ap, sigma_ap_err):
    Om, ok = x
    
    w = -1.
    beta = beta0


    if not (0.02 < Om < 1.0): return -np.inf
    if not (-0.5 < ok < 0.5): return -np.inf
    if not (-4.0 < w  < 2.0): return -np.inf
    if (1.0 - Om - ok) <= 0.0: return -np.inf

    total_lnL = 0.0
    for i in range(len(zl)):
        if not (np.isfinite(zl[i]) and np.isfinite(zs[i])) or (zs[i] <= zl[i]):
            return -np.inf

        f_g = f_prime(gamma_var[i], delta_var[i], beta)
        if (not np.isfinite(f_g)) or (f_g <= 0.0): return -np.inf

        lnL_i = lnlike_mkw(Om, ok, w,
                          gamma_var[i], delta_var[i], beta,
                          zl[i], zs[i], H0,
                          thetaE[i], sigma_ap[i], sigma_ap_err[i], theta_ap[i],
                          d_thetaE=0.05,
                          sig_g=gamma_err[i],
                          sig_d=delta_err[i],
                          sig_b=beta_err)
        if not np.isfinite(lnL_i): return -np.inf
        total_lnL += lnL_i

    ln_prior_beta = -0.5*((beta - beta0)/beta_err)**2 - np.log(np.sqrt(2*np.pi)*beta_err)
    return total_lnL + ln_prior_beta



    
#def chi_2(Om, w, gamma_var, delta_var, gamma_err, delta_err, beta, zl, zs, thetaE, theta_ap, H0, sigma_ap, sigma_ap_err):
#    dr_obs = dd_obs(gamma_var, delta_var, beta, zl, zs, thetaE, theta_ap, sigma_ap)
#    dr_th = dd_th(Om, w, zl, zs, H0)
#    num = (dr_th - dr_obs)**2.0
#    d_thetaE = (gamma_var - 1.)**2.0 * 0.05**2.0
#    d_sigma = 4. * sigma_ap_err**2.0 / sigma_ap**2.0
#    denom_obs2 = d_thetaE + d_sigma
#    return num / denom_obs2
#    
#def lnlike(Om, w, gamma_var, delta_var, gamma_err, delta_err, beta, zl, zs, thetaE, theta_ap, H0, sigma_ap, sigma_ap_err):
#    return np.exp(-chi_2(Om, w, gamma_var, delta_var, gamma_err, delta_err, beta, zl, zs, thetaE, theta_ap, H0, sigma_ap, sigma_ap_err) * 0.5)
    

    
def chi_2_gamma(Om, w, gamma_var, delta_var, beta, zl, zs, thetaE, theta_ap, H0, sigma_ap, sigma_ap_err):
    dr_obs = dd_obs(gamma_var, delta_var, beta, zl, zs, thetaE, theta_ap, sigma_ap)
    dr_th = dd_th(Om, w, zl, zs, H0)
    num = (dr_th - dr_obs)**2.0
    d_thetaE = (gamma_var - 1.)**2.0 * 0.05**2.0
    d_sigma = 4. * sigma_ap_err**2.0 / sigma_ap**2.0
    denom_obs2 = d_thetaE + d_sigma
    return num / denom_obs2
    
def lnlike_gamma(Om, w, gamma_var, delta_var, beta, zl, zs, thetaE, theta_ap, H0, sigma_ap, sigma_ap_err):
    return np.exp(-chi_2_gamma(Om, w, gamma_var, delta_var, beta, zl, zs, thetaE, theta_ap, H0, sigma_ap, sigma_ap_err) * 0.5)    

#def chi_2_evo(Om, w0, wa, gamma_var, delta_var, gamma_err, delta_err, beta, zl, zs, thetaE, theta_ap, H0, sigma_ap, sigma_ap_err):
#    dr_obs = dd_obs(gamma_var, delta_var, beta, zl, zs, thetaE, theta_ap, sigma_ap)
#    dr_th = dd_th_evow(Om, w0,wa, zl, zs, H0)
#    num = (dr_th - dr_obs)**2.0
#    d_thetaE = (gamma_var - 1.)**2.0 * 0.05**2.0
#    d_sigma = 4. * sigma_ap_err**2.0 / sigma_ap**2.0
#    d_gamma = (np.log(thetaE / theta_ap) - df_gamma(gamma_var, delta_var, beta) / f_prime(gamma_var, delta_var, beta))**2.0 * gamma_err**2.0
#    d_delta = (df_delta(gamma_var, delta_var, beta) / f_prime(gamma_var, delta_var, beta))**2.0 * delta_err**2.0
#    denom_obs2 = d_thetaE + d_sigma + d_gamma + d_delta
#    return num / denom_obs2
    
def chi_2_evo(Om, w0, wa, gamma_var, delta_var, gamma_err, delta_err, beta, zl, zs, thetaE, theta_ap, H0, sigma_ap, sigma_ap_err):
    dr_obs = dd_obs(gamma_var, delta_var, beta, zl, zs, thetaE, theta_ap, sigma_ap)
    dr_th = dd_th_evow(Om, w0,wa, zl, zs, H0)
    num = (dr_th - dr_obs)**2.0
    d_thetaE = (gamma_var - 1.)**2.0 * 0.05**2.0
    d_sigma = 4. * sigma_ap_err**2.0 / sigma_ap**2.0
#    d_gamma = (np.log(thetaE / theta_ap) - df_gamma(gamma_var, delta_var, beta) / f_prime(gamma_var, delta_var, beta))**2.0 * gamma_err**2.0
#    d_delta = (df_delta(gamma_var, delta_var, beta) / f_prime(gamma_var, delta_var, beta))**2.0 * delta_err**2.0
    denom_obs2 = d_thetaE + d_sigma
    return num / denom_obs2

def lnlike_evo(Om, w0, wa, gamma_var, delta_var, gamma_err, delta_err, beta, zl, zs, thetaE, theta_ap, H0, sigma_ap, sigma_ap_err):
    return np.exp(-chi_2_evo(Om, w0, wa, gamma_var, delta_var, gamma_err, delta_err, beta, zl, zs, thetaE, theta_ap, H0, sigma_ap, sigma_ap_err) * 0.5)

def lnprob_cos3D(x,beta0,beta_err, gamma_var, delta_var, gamma_err, delta_err, zl, zs, thetaE, theta_ap, H0, sigma_ap, sigma_ap_err):
    Om, w, beta = x

    total_lnlike = 0
    mu, sigma_beta = beta0,beta_err
    ln_prior_beta = -0.5 * np.log(2 * np.pi * sigma_beta**2) - ((beta - mu)**2 / (2 * sigma_beta**2))

    for i in range(len(zl)):
        f_g = f_prime(gamma_var[i], delta_var[i], beta)

        if not (0.0 < Om < 1.0 and f_g > 0. and -4 < w < 2.):
            return -np.inf

        f_prime_result = f_g
        if np.isnan(f_prime_result):
            return -np.inf

        lglnlike = np.log(lnlike(Om, w, gamma_var[i], delta_var[i], gamma_err[i],
                                     delta_err[i], beta, zl[i], zs[i], thetaE[i], theta_ap[i], H0,
                                     sigma_ap[i], sigma_ap_err[i]))
        if np.isnan(lglnlike):
            return -np.inf
        total_lnlike += lglnlike

    return total_lnlike + ln_prior_beta

# def lnprob_cos4D(x,beta0,beta_err, gamma_var, delta_var, gamma_err, delta_err, zl, zs, thetaE, theta_ap, H0, sigma_ap, sigma_ap_err):
#     Om, w0, wa, beta = x

#     total_lnlike = 0
#     mu, sigma_beta = beta0,beta_err
#     ln_prior_beta = -0.5 * np.log(2 * np.pi * sigma_beta**2) - ((beta - mu)**2 / (2 * sigma_beta**2))

#     for i in range(len(zl)):
#         f_g = f_prime(gamma_var[i], delta_var[i], beta)
#         w = w0+wa*zs[i]/(1.+zs[i])
#         # if not (0.0 < Om < 1.0 and f_g > 0. and -1.5 < w < -0.5):
#         if not (0.0 < Om < 1.0 and f_g > 0. and -1.0 < w0 < -0.5 and -2.0 < wa < 0.0 ):
#             return -np.inf

#         f_prime_result = f_g
#         if np.isnan(f_prime_result):
#             return -np.inf

#         lglnlike = np.log(lnlike_evo(Om, w0, wa, gamma_var[i], delta_var[i], gamma_err[i],
#                                      delta_err[i], beta, zl[i], zs[i], thetaE[i], theta_ap[i], H0,
#                                      sigma_ap[i], sigma_ap_err[i]))
#         if np.isnan(lglnlike):
#             return -np.inf
#         total_lnlike += lglnlike

#     return total_lnlike + ln_prior_beta
    
# def lnprob_cos4D_beta(x,
#                       beta0, beta_err,
#                       gamma_var, delta_var, gamma_err, delta_err,
#                       zl, zs, thetaE, theta_ap,
#                       H0,
#                       sigma_ap, sigma_ap_err):
# #    """
# #    Log-posterior for 4D cosmology + intrinsic beta scatter.
# #
# #    x             : array-like of length 4 ¡ú [Om, w, beta, sigma_b]
# #    beta0         : prior mean for beta
# #    beta_err      : measurement error on beta
# #    gamma_var¡­    : arrays of your data, same as in lnprob_cos3D
# #    """
#     Om, w, beta, sigma_b = x

#     # ----- hard priors -----
#     if not (0.0 < Om < 1.0 and -4.0 < w < 2.0 and -0.6 < beta <1. and 0.0 < sigma_b < 0.6):
#         return -np.inf

#     # ----- prior on beta with intrinsic scatter -----
#     mu = beta0
#     # total beta variance = measurement^2 + intrinsic^2
#     var_beta = beta_err**2 + sigma_b**2
#     ln_prior_beta = -0.5*(np.log(2*np.pi*var_beta) 
#                           + (beta - mu)**2/var_beta)

#     total_lnpost = ln_prior_beta

#     # ----- likelihood over your dataset -----
#     for i in range(len(zl)):
#         # compute f_prime(¡­) as before
#         f_g = f_prime(gamma_var[i],
#                       delta_var[i],
#                       beta)

#         # require physicality
#         if not (f_g > 0.0) or np.isnan(f_g):
#             return -np.inf

#         # your existing per-point likelihood function returns P>0
#         Li = lnlike(Om, w,
#                     gamma_var[i], delta_var[i],
#                     gamma_err[i], delta_err[i],
#                     beta,
#                     zl[i], zs[i],
#                     thetaE[i], theta_ap[i],
#                     H0,
#                     sigma_ap[i], sigma_ap_err[i])
#         if Li <= 0 or np.isnan(Li):
#             return -np.inf

#         total_lnpost += np.log(Li)

#     return total_lnpost

# def lnprob_cos4D_gamma(x,
#                       beta0, beta_err,
#                       gamma_var, delta_var, gamma_err, delta_err,
#                       zl, zs, thetaE, theta_ap,
#                       H0,
#                       sigma_ap, sigma_ap_err):
# #    """
# #    Log-posterior for 4D cosmology + intrinsic beta scatter.
# #
# #    x             : array-like of length 4 ¡ú [Om, w, beta, sigma_b]
# #    beta0         : prior mean for beta
# #    beta_err      : measurement error on beta
# #    gamma_var¡­    : arrays of your data, same as in lnprob_cos3D
# #    """
#     Om, w, gamma_var,beta, sigma_b = x

#     # ----- hard priors -----
#     if not (0.0 < Om < 1.0 and -4.0 < w < 2.0 and -0.6 < beta <1. and 0.0 < sigma_b < 0.6 and 1.8 < gamma_var < 2.3):
#         return -np.inf

#     # ----- prior on beta with intrinsic scatter -----
#     mu = beta0
#     # total beta variance = measurement^2 + intrinsic^2
#     var_beta = beta_err**2 + sigma_b**2
#     ln_prior_beta = -0.5*(np.log(2*np.pi*var_beta) 
#                           + (beta - mu)**2/var_beta)

#     total_lnpost = ln_prior_beta

#     # ----- likelihood over your dataset -----
#     for i in range(len(zl)):
#         # compute f_prime(¡­) as before
#         f_g = f_prime(gamma_var,
#                       delta_var[i],
#                       beta)

#         # require physicality
#         if not (f_g > 0.0) or np.isnan(f_g):
#             return -np.inf

#         # your existing per-point likelihood function returns P>0
#         Li = lnlike_gamma(Om, w,
#                     gamma_var,
#                     delta_var[i],
#                     beta,
#                     zl[i], zs[i],
#                     thetaE[i], theta_ap[i],
#                     H0,
#                     sigma_ap[i], sigma_ap_err[i])
#         if Li <= 0 or np.isnan(Li):
#             return -np.inf

#         total_lnpost += np.log(Li)

#     return total_lnpost

 
def lnprob_cos6D_gamma(
    x, delta, delta_err,
    zl, zs, thetaE, theta_ap,
    H0,
    sigma_ap, sigma_ap_err
):
    # """
    # Log-posterior for a 6D model:
    #   x = [Om, w, gamma_mean, sigma_gamma, beta, sigma_beta]

    # beta0, beta_err   : prior mean & measurement error for beta
    # gamma0, gamma_err : prior mean & measurement error for gamma
    # gamma_var_data, delta_var, ... : your data arrays
    # """
    Om, w, gamma_mean, sigma_gamma, beta, sigma_beta = x

    # 1) Hard parameter bounds
    if not (
        0.0 < Om < 1.0 and
        -4.0 < w < 2.0 and
        1.8 < gamma_mean < 2.3 and
        0.0 < sigma_gamma < 1.0 and
        -0.6 < beta < 1.0 and
        0.0 < sigma_beta  < 1.0
    ):
        return -np.inf

    # 2) Prior on gamma with intrinsic scatter σ_γ
    var_g = sigma_gamma**2
    ln_prior_gamma = -0.5 * (
        np.log(2*np.pi*var_g)
        + (gamma_mean - 2.0)**2 / var_g
    )

    # 3) Prior on beta with intrinsic scatter σ_β
    var_b = sigma_beta**2
    ln_prior_beta = -0.5 * (
        np.log(2*np.pi*var_b)
        + (beta - 0.0)**2 / var_b
    )

    # accumulate log‐posterior
    total_lnpost = ln_prior_gamma + ln_prior_beta

    # 4) Data likelihood loop
    for i in range(len(zl)):
        # lens‐model piece
        f_g = f_prime(gamma_mean,
                      2.173,
                      beta)
        if not (f_g > 0.0) or np.isnan(f_g):
            return -np.inf

        # your existing per-point likelihood
        L = lnlike_gamma(
            Om, w,
            gamma_mean, 2.173,
            beta,
            zl[i], zs[i],
            thetaE[i], theta_ap[i],
            H0,
            sigma_ap[i], sigma_ap_err[i]
        )
        if L <= 0 or np.isnan(L):
            return -np.inf

        total_lnpost += np.log(L)

    return total_lnpost


def lnprob_cos3D_evow_b(x,beta0,beta_err, gamma_var, delta_var, gamma_err, delta_err, zl, zs, thetaE, theta_ap, H0, sigma_ap, sigma_ap_err):
    Om, w0, wa = x
    beta = beta0

    total_lnlike = 0
#    mu, sigma_beta = beta0,beta_err
#    ln_prior_beta = -0.5 * np.log(2 * np.pi * sigma_beta**2) - ((beta - mu)**2 / (2 * sigma_beta**2))

    for i in range(len(zl)):
        f_g = f_prime(gamma_var[i], delta_var[i], beta)
        w = w0+wa*zs[i]/(1.+zs[i])
        # if not (0.0 < Om < 1.0 and f_g > 0. and -1.5 < w < -0.5):
        if not (0.0 < Om < 1.0 and f_g > 0. and -4.0 < w0 < 3.0 and -6.0 < wa < 6.0 ):
            return -np.inf

        f_prime_result = f_g
        if np.isnan(f_prime_result):
            return -np.inf

        lglnlike = np.log(lnlike_evo(Om, w0, wa, gamma_var[i], delta_var[i], gamma_err[i],
                                     delta_err[i], beta, zl[i], zs[i], thetaE[i], theta_ap[i], H0,
                                     sigma_ap[i], sigma_ap_err[i]))
        if np.isnan(lglnlike):
            return -np.inf
        total_lnlike += lglnlike

    return total_lnlike

def lnprob_cos2D_Om(x,beta0,beta_err, gamma_var, delta_var, gamma_err, delta_err, zl, zs, thetaE, theta_ap, H0, sigma_ap, sigma_ap_err):
    Om, beta = x
    w = -1.0
    total_lnlike = 0
    mu, sigma_beta = beta0,beta_err
    ln_prior_beta = -0.5 * np.log(2 * np.pi * sigma_beta**2) - ((beta - mu)**2 / (2 * sigma_beta**2))

    for i in range(len(zl)):
        f_g = f_prime(gamma_var[i], delta_var[i], beta)

        if not (0.0 < Om < 1.0 and f_g > 0. and -1.5 < w < -0.5):
            return -np.inf

        f_prime_result = f_g
        if np.isnan(f_prime_result):
            return -np.inf

        lglnlike = np.log(lnlike(Om, w, gamma_var[i], delta_var[i], gamma_err[i],
                                     delta_err[i], beta, zl[i], zs[i], thetaE[i], theta_ap[i], H0,
                                     sigma_ap[i], sigma_ap_err[i]))
        if np.isnan(lglnlike):
            return -np.inf
        total_lnlike += lglnlike

    return total_lnlike + ln_prior_beta

def lnprob_cos2D_w(x,beta0,beta_err, gamma_var, delta_var, gamma_err, delta_err, zl, zs, thetaE, theta_ap, H0, sigma_ap, sigma_ap_err):
    w, beta = x
    Om = 0.3
    total_lnlike = 0
    mu, sigma_beta = beta0,beta_err
    ln_prior_beta = -0.5 * np.log(2 * np.pi * sigma_beta**2) - ((beta - mu)**2 / (2 * sigma_beta**2))

    for i in range(len(zl)):
        f_g = f_prime(gamma_var[i], delta_var[i], beta)

        if not (0.0 < Om < 1.0 and f_g > 0. and -1.5 < w < -0.5):
            return -np.inf

        f_prime_result = f_g
        if np.isnan(f_prime_result):
            return -np.inf

        lglnlike = np.log(lnlike(Om, w, gamma_var[i], delta_var[i], gamma_err[i],
                                     delta_err[i], beta, zl[i], zs[i], thetaE[i], theta_ap[i], H0,
                                     sigma_ap[i], sigma_ap_err[i]))
        if np.isnan(lglnlike):
            return -np.inf
        total_lnlike += lglnlike

    return total_lnlike + ln_prior_beta