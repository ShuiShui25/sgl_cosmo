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
from mcmc_utils import lnprob_kw,lnprob_mk,lnprob_mkw
from mcmc_utils import lnprob_cos2D,lnprob_cos3D_evow,lnprob_cos3D
from mcmc_utils import lnprob_cos3D_evow_b,lnprob_cos3D_kevow, lnprob_cos3D_wphi
from mcmc_utils import lnprob_cos4D
from mcmc_utils import lnprob_cos5D_gamma
from mcmc_utils import lnprob_cos6D_gamma, lnprob_cos6D_w0wa_gamma, lnprob_cos6D_wphi_gamma


# Load values - GLOBAL VALUES
c = c.to('km/s').value
pi = np.pi
eta = -0.066
eta_err = 0.035

def calibrate(sigma_ap_nocal, theta_eff, theta_ap, eta):
    return sigma_ap_nocal * (theta_eff / (2 * theta_ap)) ** eta

def sigma_uncertainty(sigma_ap_nocal, sigma_ap_err_nocal, theta_eff, theta_ap, eta, eta_err):
    err_stat2 = sigma_ap_err_nocal**2.0
    err_AC2 = sigma_ap_err_nocal**2.0 * (theta_eff / (2.0 * theta_ap))**(2.0 * eta) + \
              sigma_ap_nocal**2.0 * (theta_eff / (2.0 * theta_ap))**(2.0 * eta) * np.log(theta_eff / (2.0 * theta_ap))**2.0 * eta_err**2.0
    err_sys2 = (0.03 * sigma_ap_nocal)**2.0
    return np.sqrt(err_stat2 + err_AC2 + err_sys2)*0.8

class LensCosmo:
    def __init__(self, params, param_errors, 
                 free_params, free_params_guess,
                 evo, prior,
                 lnprob_function,
                 filename, save_path,
                 nsteps=8000, nwalkers=200, ndim=3, ncpu=12,
                 checkpoint=False, nsteps_per_checkpoint=1000,
                 ):
        if len(params) < 3:
            raise ValueError("params must have at least 3 elements: [d0, ds, beta0].")
        if len(param_errors) < 3:
            raise ValueError("param_errors must have at least 3 elements: [σ_d0, σ_ds, σ_beta].")
        if not isinstance(free_params, list) or not all(isinstance(p, str) for p in free_params):
            raise TypeError("free_params must be a list of strings")
        if evo not in ['linear', 'CPL', 'gamma','linear-Etherington']:
            raise ValueError("evo must be linear, CPL, gamma,linear-Etherington")
        if not isinstance(prior, str):
            raise TypeError("prior must be a string")
        
        self.params = np.array(params)
        self.param_errors = np.array(param_errors)
        self.free_params = free_params
        self.free_params_guess = free_params_guess if free_params_guess is not None else [0.3, -1.0]
        self.nsteps = nsteps
        self.nwalkers = nwalkers
        self.ndim = len(free_params)
        self.ncpu = ncpu
        self.checkpoint = checkpoint
        self.nsteps_per_checkpoint = nsteps_per_checkpoint
        self.evo = evo
        self.prior = prior
        if lnprob_function == 'lnprob_cos2D':
            self.lnprob_function = lnprob_cos2D
            
        elif lnprob_function == 'lnprob_cos3D_evow':
            self.lnprob_function = lnprob_cos3D_evow
            
        elif lnprob_function == 'lnprob_cos3D_wphi':
            self.lnprob_function = lnprob_cos3D_wphi
            
        elif lnprob_function == 'lnprob_cos3D_kevow':
            self.lnprob_function = lnprob_cos3D_kevow
            
        elif lnprob_function == 'lnprob_kw':
            self.lnprob_function = lnprob_kw
            
        elif lnprob_function == 'lnprob_mk':
            self.lnprob_function = lnprob_mk
            
        elif lnprob_function == 'lnprob_mkw':
            self.lnprob_function = lnprob_mkw
            
        elif lnprob_function == 'lnprob_cos3D':
            self.lnprob_function = lnprob_cos3D
        
        elif lnprob_function == 'lnprob_cos3D_evow_b':
            self.lnprob_function = lnprob_cos3D_evow_b
        
        elif lnprob_function == 'lnprob_cos4D':
            self.lnprob_function = lnprob_cos4D
            
        elif lnprob_function == 'lnprob_cos4D_beta':
            self.lnprob_function = lnprob_cos4D_beta
            
        elif lnprob_function == 'lnprob_cos5D_gamma':
            self.lnprob_function = lnprob_cos5D_gamma
            
        elif lnprob_function == 'lnprob_cos6D_gamma':
            self.lnprob_function = lnprob_cos6D_gamma
            
        elif lnprob_function == 'lnprob_cos6D_w0wa_gamma':
            self.lnprob_function = lnprob_cos6D_w0wa_gamma
            
        elif lnprob_function == 'lnprob_cos6D_wphi_gamma':
            self.lnprob_function = lnprob_cos6D_wphi_gamma
        
        else:
            raise ValueError(
                f"mode {lnprob_function!r} not available; available ones are: "
                "lnprob_cos2D, lnprob_cos3D, lnprob_cos3D_evow_b, lnprob_cos4D,lnprob_cos4D_beta,"
                "lnprob_cos3D_evow, lnprob_cos3D_wphi,lnprob_cos3D_kevow,"
                "lnprob_cos5D_gamma, lnprob_cos6D_gamma, lnprob_cos6D_w0wa_gamma, lnprob_cos6D_wphi_gamma"
            )

        self.filename = filename
        self.save_path = save_path
        self.data_folder = os.path.join(self.save_path, f"{evo}-{prior}")
        os.makedirs(self.data_folder, exist_ok=True)

    def load_fits_data(self, evo, filename, eta, eta_err):
        """
        Load data from a FITS file.
        """
        lenstable  = Table.read(filename)
        theta_E_r  = lenstable['theta_E']  * u.arcsec.to('radian')
        theta_eff  = lenstable['theta_eff']* u.arcsec.to('radian')
        theta_ap_r = lenstable['theta_ap'] * u.arcsec.to('radian')
    
        # If you later want calibration, re-enable your calibrate() calls
        sigma_ap           = lenstable['sigma_ap']
        abs_delta_sigma_ap = lenstable['sigma_ap_err']
    
        zl = np.array(lenstable['zl'])
        zs = np.array(lenstable['zs'])
    
        lnprob_name = self.lnprob_function.__name__
    
        # Defaults (kept for API compatibility; unused by our new lnprobs)
        gamma_list = np.zeros_like(zl, dtype=float)
        delta_list = np.zeros_like(zl, dtype=float)
        gamma_err  = np.zeros_like(zl, dtype=float)
        delta_err  = np.zeros_like(zl, dtype=float)
    
        # Build gamma/delta only when an lnprob needs them precomputed
        if evo == 'linear':
            if lnprob_name in ('lnprob_cos4D','lnprob_cos5D_gamma', 'lnprob_cos6D_w0wa_gamma', 'lnprob_cos6D_wphi_gamma'):
                # New modes compute gamma/delta internally; nothing to do here.
                pass
            else:
                # Legacy: gamma = a0 + a1*zl ; delta = a2 + a3*zl
                gamma_list = self.params[0] + zl * self.params[1]
                delta_list = self.params[2] + zl * self.params[3]
                gamma_err  = np.sqrt(self.param_errors[0]**2.0 + (zl**2.0) * self.param_errors[1]**2.0)
                delta_err  = np.sqrt(self.param_errors[2]**2.0 + (zl**2.0) * self.param_errors[3]**2.0)
    
        elif evo == 'linear-Etherington':
            gamma_list = self.params[0] + (zl - 0.319) * self.params[1]
            delta_list = self.params[2] + (zl - 0.319) * self.params[3]
            gamma_err  = np.sqrt(self.param_errors[0]**2.0 + (zl**2.0) * self.param_errors[1]**2.0)
            delta_err  = np.sqrt(self.param_errors[2]**2.0 + (zl**2.0) * self.param_errors[3]**2.0)
    
        elif evo == 'CPL':
            zfac       = zl / (1.0 + zl)
            gamma_list = self.params[0] + zfac * self.params[1]
            delta_list = self.params[2] + zfac * self.params[3]
            gamma_err  = np.sqrt(self.param_errors[0]**2.0 + (zfac**2.0) * self.param_errors[1]**2.0)
            delta_err  = np.sqrt(self.param_errors[2]**2.0 + (zfac**2.0) * self.param_errors[3]**2.0)
    
        elif evo == 'gamma':
            gamma_list = np.zeros_like(zl, dtype=float)      # unused
            delta_list = np.full_like(zl, 2.173, dtype=float)
            gamma_err  = np.zeros_like(zl, dtype=float)
            delta_err  = np.full_like(zl, 0.085, dtype=float)
    
        data_dict = {
            'zl': zl,
            'zs': zs,
            'theta_E_r': theta_E_r,
            'theta_eff': theta_eff,
            'theta_ap_r': theta_ap_r,
            'sigma_ap': sigma_ap,
            'abs_delta_sigma_ap': abs_delta_sigma_ap,
            'gamma_list': gamma_list,
            'delta_list': delta_list,
            'gamma_err': gamma_err,
            'delta_err': delta_err,
        }
        return data_dict

    
    
    def mcmc(self, evo, data, free_params, free_params_guess):
        output_file = os.path.join(self.data_folder, f"mcmc_cosmop{self.ndim}D-{self.evo}-{self.prior}.npy")
        if os.path.exists(output_file):
            print(f"MCMC results already exist at {output_file}. Skipping MCMC run.")
            return None, None
    
        print(f"Running MCMC for mcmc_cosmop{self.ndim}D-{self.evo}-{self.prior}")
    
        H0 = 70.0
        p0 = [np.random.normal(loc=free_params_guess, scale=1e-4, size=self.ndim) for _ in range(self.nwalkers)]
        filename = os.path.join(self.data_folder, f"mcmc_cosmop{self.ndim}D-{self.evo}-{self.prior}.h5")
        backend  = emcee.backends.HDFBackend(filename)
        backend.reset(self.nwalkers, self.ndim)
    
        all_samples  = []
        all_ln_probs = []
    
        lnprob_name = self.lnprob_function.__name__
    
        with Pool(self.ncpu) as pool:
            if evo == 'linear':
                if lnprob_name in ('lnprob_cos4D', 'lnprob_cos5D_gamma',
                                   'lnprob_cos6D_w0wa_gamma', 'lnprob_cos6D_wphi_gamma'):
                    d0, ds   = self.params[0], self.params[1]
                    beta0    = self.params[2]
                    beta_err = self.param_errors[2]
                    sampler = emcee.EnsembleSampler(
                        self.nwalkers, self.ndim, self.lnprob_function,
                        args=(d0, ds, beta0, beta_err,
                              data['zl'], data['zs'], data['theta_E_r'], data['theta_ap_r'],
                              H0, data['sigma_ap'], data['abs_delta_sigma_ap']),
                        pool=pool, backend=backend
                    )
                else:
                    # Legacy linear-evolution modes that expect precomputed gamma/delta
                    sampler = emcee.EnsembleSampler(
                        self.nwalkers, self.ndim, self.lnprob_function,
                        args=(self.params[4], self.param_errors[4],
                              data['gamma_list'], data['delta_list'], data['gamma_err'], data['delta_err'],
                              data['zl'], data['zs'], data['theta_E_r'], data['theta_ap_r'],
                              H0, data['sigma_ap'], data['abs_delta_sigma_ap']),
                        pool=pool, backend=backend
                    )
    
            elif evo == 'linear-Etherington':
                sampler = emcee.EnsembleSampler(
                    self.nwalkers, self.ndim, self.lnprob_function,
                    args=(self.params[4], self.param_errors[4],
                          data['gamma_list'], data['delta_list'], data['gamma_err'], data['delta_err'],
                          data['zl'], data['zs'], data['theta_E_r'], data['theta_ap_r'],
                          H0, data['sigma_ap'], data['abs_delta_sigma_ap']),
                    pool=pool, backend=backend
                )
    
            elif evo == 'CPL':
                sampler = emcee.EnsembleSampler(
                    self.nwalkers, self.ndim, self.lnprob_function,
                    args=(self.params[4], self.param_errors[4],
                          data['gamma_list'], data['delta_list'], data['gamma_err'], data['delta_err'],
                          data['zl'], data['zs'], data['theta_E_r'], data['theta_ap_r'],
                          H0, data['sigma_ap'], data['abs_delta_sigma_ap']),
                    pool=pool, backend=backend
                )
    
            elif evo == 'gamma':
                sampler = emcee.EnsembleSampler(
                    self.nwalkers, self.ndim, self.lnprob_function,
                    args=(data['delta_list'], data['delta_err'],
                          data['zl'], data['zs'], data['theta_E_r'], data['theta_ap_r'],
                          H0, data['sigma_ap'], data['abs_delta_sigma_ap']),
                    pool=pool, backend=backend
                )
    
            # ---------- Run with robust autocorr handling ----------
            def _safe_tau_to_bt(sampler, *, tol=0):
                """
                Try to compute tau; return (burnin, thin, ok_flag).
                Falls back to (nsteps/4, 1, False) if tau isn't usable.
                """
                try:
                    tau = sampler.get_autocorr_time(tol=tol)
                    if (not np.all(np.isfinite(tau))) or np.any(tau <= 0):
                        raise emcee.autocorr.AutocorrError("non-finite or non-positive tau")
                    burnin = int(max(2 * np.max(tau), 10))
                    thin   = int(max(0.5 * np.min(tau), 1))
                    return burnin, thin, True
                except Exception as e:
                    # Will be overridden by caller with sensible defaults
                    return None, None, False
    
            if self.checkpoint:
                print(f'Running with checkpoints every {self.nsteps_per_checkpoint} steps')
                # conservative defaults in case tau never stabilizes during checkpoints
                default_burnin = max(self.nsteps // 4, 10)
                default_thin   = 1
    
                for _ in range(0, self.nsteps, self.nsteps_per_checkpoint):
                    sampler.run_mcmc(p0, self.nsteps_per_checkpoint, store=True, progress=True)
    
                    burnin, thin, ok = _safe_tau_to_bt(sampler, tol=0)
                    if ok:
                        print(f"Checkpoint: tau OK → burnin={burnin}, thin={thin}")
                        # Optional early-stop if we’ve well exceeded autocorr length
                        try:
                            tau_now = sampler.get_autocorr_time(tol=0)
                            if sampler.iteration / np.max(tau_now) > 50:
                                print("Sufficient chain length achieved. Stopping run.")
                                break
                        except Exception:
                            pass
                    else:
                        print("Autocorr not reliable yet. Continuing ...")
    
                    p0 = sampler.get_last_sample()
    
                # After checkpoint loop, finalize burnin/thin robustly
                burnin, thin, ok = _safe_tau_to_bt(sampler, tol=0)
                if not ok:
                    burnin, thin = default_burnin, default_thin
    
            else:
                sampler.run_mcmc(p0, self.nsteps, progress=True)
                burnin, thin, ok = _safe_tau_to_bt(sampler, tol=0)
                if not ok:
                    burnin = max(self.nsteps // 4, 10)
                    thin   = 1
    
            # Clip burnin so we don't discard more than available
            total_len = len(sampler.get_chain())
            burnin = int(min(max(burnin, 0), max(total_len - 1, 0)))
            thin   = int(max(thin, 1))
    
            all_samples  = sampler.get_chain(discard=burnin, thin=thin)
            all_ln_probs = sampler.get_log_prob(discard=burnin, thin=thin)
    
            chain_info = {
                'nsteps': total_len,
                'nsteps_without_burnin': len(sampler.get_chain(discard=burnin, thin=thin)),
                'burnin': burnin,
                'thin': thin
            }
            pd.DataFrame([chain_info]).to_csv(
                os.path.join(self.data_folder, f'mcmc_chain_logfile{self.ndim}D-{self.evo}-{self.prior}.csv')
            )
    
            np.save(output_file, all_samples)
            # np.save(os.path.join(self.data_folder, f'all_ln_probs_cosmop{self.ndim}D-{self.evo}-{self.prior}.npy'), all_ln_probs)
    
        print(f"MCMC run completed and results saved at {self.data_folder}.")




    