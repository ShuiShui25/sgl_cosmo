# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 18:10:30 2024

@author: poilo
"""

import emcee
import numpy as np
import os
from multiprocessing import Pool
from astropy.table import Table
from GLS_cosmo import LensCosmo, eta, eta_err
from astropy.table import Table
from astropy import units as u
import math
from astropy.constants import G, c, M_sun
import scipy.integrate as sci



def main_2D_linear_tri():
    # Base directory = the folder containing this script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define data and output directories relative to that
    data_dir   = os.path.join(base_dir, 'Data')
    output_dir = os.path.join(base_dir, 'Output')

    # Make sure Output exists
    os.makedirs(output_dir, exist_ok=True)

    # Define parameters for the LensCosmo class
    params           = [2.065, -0.20 ,  2.14, -0.09, 0.0]
    param_errors     = [0.046,  0.12,   0.16,  0.19, 0.087 ]
    free_params      = ['Om', 'w']
    free_params_guess= [0.3,  -1.0]
    evo              = 'linear'
    prior            = 'Tri'
    lnprob_func      = 'lnprob_cos2D'

    # Now both of these are relative
    save_path = output_dir
    filename  = os.path.join(data_dir, 'SGLTable03.fits')

    # Load data from a FITS file
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")

    # Initialize LensCosmo
    lens_cosmo = LensCosmo(
        params=params,
        param_errors=param_errors,
        free_params=free_params,
        free_params_guess=free_params_guess,
        evo=evo,
        prior=prior,
        lnprob_function=lnprob_func,
        filename=filename,
        save_path=save_path,
        nsteps=10000,
        nwalkers=200,
        ndim=len(free_params),
        ncpu=12,
        checkpoint=True,
        nsteps_per_checkpoint=2000
    )

    # Load and run
    data = lens_cosmo.load_fits_data(evo, filename, eta, eta_err)
    lens_cosmo.mcmc(evo, data, free_params, free_params_guess)
    
def main_2D_linear_22():
    # Base directory = the folder containing this script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define data and output directories relative to that
    data_dir   = os.path.join(base_dir, 'Data')
    output_dir = os.path.join(base_dir, 'Output')

    # Make sure Output exists
    os.makedirs(output_dir, exist_ok=True)

    # Define parameters for the LensCosmo class
    params           = [2.054, -0.19 ,  2.26, -0.16, 0.0]
    param_errors     = [0.041,  0.11,   0.13,  0.18, 0.085 ]
    free_params      = ['Om', 'w']
    free_params_guess= [0.3,  -1.0]
    evo              = 'linear'
    prior            = '0.22-beta0'
    lnprob_func      = 'lnprob_cos2D'

    # Now both of these are relative
    save_path = output_dir
    filename  = os.path.join(data_dir, 'SGLTable03.fits')

    # Load data from a FITS file
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")

    # Initialize LensCosmo
    lens_cosmo = LensCosmo(
        params=params,
        param_errors=param_errors,
        free_params=free_params,
        free_params_guess=free_params_guess,
        evo=evo,
        prior=prior,
        lnprob_function=lnprob_func,
        filename=filename,
        save_path=save_path,
        nsteps=10000,
        nwalkers=200,
        ndim=len(free_params),
        ncpu=12,
        checkpoint=True,
        nsteps_per_checkpoint=2000
    )

    # Load and run
    data = lens_cosmo.load_fits_data(evo, filename, eta, eta_err)
    lens_cosmo.mcmc(evo, data, free_params, free_params_guess)

def main_2D_linear_DESa_noevo():
    # Base directory = the folder containing this script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define data and output directories relative to that
    data_dir   = os.path.join(base_dir, 'Data')
    output_dir = os.path.join(base_dir, 'Output')

    # Make sure Output exists
    os.makedirs(output_dir, exist_ok=True)

    # Define parameters for the LensCosmo class
    params           = [2.078, 0.0 ,  2.173, 0., 0.22]
    param_errors     = [0.041,  0.11,   0.13,  0.18, 0.085 ]
    free_params      = ['Om', 'w']
    free_params_guess= [0.3,  -1.0]
    evo              = 'linear'
    prior            = 'DESa_noevo'
    lnprob_func      = 'lnprob_cos2D'

    # Now both of these are relative
    save_path = output_dir
    filename  = os.path.join(data_dir, 'DESa_sim_l3_w_noevoPL.fits')

    # Load data from a FITS file
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")

    # Initialize LensCosmo
    lens_cosmo = LensCosmo(
        params=params,
        param_errors=param_errors,
        free_params=free_params,
        free_params_guess=free_params_guess,
        evo=evo,
        prior=prior,
        lnprob_function=lnprob_func,
        filename=filename,
        save_path=save_path,
        nsteps=10000,
        nwalkers=200,
        ndim=len(free_params),
        ncpu=12,
        checkpoint=True,
        nsteps_per_checkpoint=2000
    )

    # Load and run
    data = lens_cosmo.load_fits_data(evo, filename, eta, eta_err)
    lens_cosmo.mcmc(evo, data, free_params, free_params_guess)

def main_2D_linear_11_DESa_evo():
    # Base directory = the folder containing this script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define data and output directories relative to that
    data_dir   = os.path.join(base_dir, 'Data')
    output_dir = os.path.join(base_dir, 'Output')

    # Make sure Output exists
    os.makedirs(output_dir, exist_ok=True)

    # Define parameters for the LensCosmo class
    params           = [2.054, -0.19 ,  2.26, -0.16, 0.11]
    param_errors     = [0.041,  0.11,   0.13,  0.18, 0.085 ]
    free_params      = ['Om', 'w']
    free_params_guess= [0.3,  -1.0]
    evo              = 'linear'
    prior            = 'evodata+evofit'
    lnprob_func      = 'lnprob_cos2D'

    # Now both of these are relative
    save_path = output_dir
    filename  = os.path.join(data_dir, 'DESa_sim_l3_w_evoPL.fits')

    # Load data from a FITS file
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")

    # Initialize LensCosmo
    lens_cosmo = LensCosmo(
        params=params,
        param_errors=param_errors,
        free_params=free_params,
        free_params_guess=free_params_guess,
        evo=evo,
        prior=prior,
        lnprob_function=lnprob_func,
        filename=filename,
        save_path=save_path,
        nsteps=10000,
        nwalkers=200,
        ndim=len(free_params),
        ncpu=12,
        checkpoint=True,
        nsteps_per_checkpoint=2000
    )

    # Load and run
    data = lens_cosmo.load_fits_data(evo, filename, eta, eta_err)
    lens_cosmo.mcmc(evo, data, free_params, free_params_guess)

def main_4D_linear_sim_noevo():
    base_dir   = os.path.dirname(os.path.abspath(__file__))
    data_dir   = os.path.join(base_dir, 'Data')
    output_dir = os.path.join(base_dir, 'Output')
    os.makedirs(output_dir, exist_ok=True)

    # -------- luminosity evolution + beta prior --------
    # params: [d0, ds, beta0]
    params           = [2.26, -0.16, 0.11]      # lumi_params + beta0
    param_errors     = [0.13,  0.18,  0.03]     

    free_params       = ['Om', 'w', 'g0', 'beta']
    free_params_guess = [0.3,  -1.0, 2.0, 0.11]

    evo         = 'linear'
    prior       = 'noevog-evod-LSSTa'
    lnprob_func = 'lnprob_cos4D'

    save_path = output_dir
    filename  = os.path.join(data_dir, 'LSSTa_sim_l3_w_evoPL.fits')

    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")

    lens_cosmo = LensCosmo(
        params=params,
        param_errors=param_errors,
        free_params=free_params,
        free_params_guess=free_params_guess,
        evo=evo,
        prior=prior,
        lnprob_function=lnprob_func,
        filename=filename,
        save_path=save_path,
        nsteps=10000,
        nwalkers=200,
        ndim=len(free_params),
        ncpu=12,
        checkpoint=True,
        nsteps_per_checkpoint=2000
    )

    data = lens_cosmo.load_fits_data(evo, filename, eta, eta_err)
    lens_cosmo.mcmc(evo, data, free_params, free_params_guess)
    
def main_5D_linear_sim_gamma_evo():
    base_dir   = os.path.dirname(os.path.abspath(__file__))
    data_dir   = os.path.join(base_dir, 'Data')
    output_dir = os.path.join(base_dir, 'Output')
    os.makedirs(output_dir, exist_ok=True)

    # -------- luminosity evolution + beta prior --------
    # params: [d0, ds, beta0]
    params           = [2.26, -0.16, 0.11]      # lumi_params + beta0
    param_errors     = [0.13,  0.18,  0.03]     

    free_params       = ['Om', 'w', 'g0','gs', 'beta']
    free_params_guess = [0.3,  -1.0, 2.05, -0.1, 0.11]

    evo         = 'linear'
    prior       = 'evog-evod-LSSTa'
    lnprob_func = 'lnprob_cos5D_gamma'

    save_path = output_dir
    filename  = os.path.join(data_dir, 'LSSTa_sim_l3_w_evoPL.fits')

    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")

    lens_cosmo = LensCosmo(
        params=params,
        param_errors=param_errors,
        free_params=free_params,
        free_params_guess=free_params_guess,
        evo=evo,
        prior=prior,
        lnprob_function=lnprob_func,
        filename=filename,
        save_path=save_path,
        nsteps=10000,
        nwalkers=200,
        ndim=len(free_params),
        ncpu=12,
        checkpoint=True,
        nsteps_per_checkpoint=2000
    )

    data = lens_cosmo.load_fits_data(evo, filename, eta, eta_err)
    lens_cosmo.mcmc(evo, data, free_params, free_params_guess)

def main_6D_w0wa_linear_DESa_gamma_evo():
    base_dir   = os.path.dirname(os.path.abspath(__file__))
    data_dir   = os.path.join(base_dir, 'Data')
    output_dir = os.path.join(base_dir, 'Output')
    os.makedirs(output_dir, exist_ok=True)

    # -------- luminosity evolution + beta prior --------
    # params: [d0, ds, beta0]
    params           = [2.26, -0.16, 0.11]      # lumi_params + beta0
    param_errors     = [0.13,  0.18,  0.03]     

    free_params = ['Om', 'w0', 'wa', 'g0','gs', 'beta'] 
    free_params_guess = [0.3, -0.75, -0.86, 2.05, -0.1, 0.11]

    evo         = 'linear'
    prior       = 'evog-evod-DESa-w0wa'
    lnprob_func = 'lnprob_cos6D_w0wa_gamma'

    save_path = output_dir
    filename  = os.path.join(data_dir, 'DESa_sim_l3_w0wa_evoPL.fits')

    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")

    lens_cosmo = LensCosmo(
        params=params,
        param_errors=param_errors,
        free_params=free_params,
        free_params_guess=free_params_guess,
        evo=evo,
        prior=prior,
        lnprob_function=lnprob_func,
        filename=filename,
        save_path=save_path,
        nsteps=10000,
        nwalkers=200,
        ndim=len(free_params),
        ncpu=12,
        checkpoint=True,
        nsteps_per_checkpoint=2000
    )

    data = lens_cosmo.load_fits_data(evo, filename, eta, eta_err)
    lens_cosmo.mcmc(evo, data, free_params, free_params_guess)

def main_6D_wphi_linear_DESa_gamma_evo():
    base_dir   = os.path.dirname(os.path.abspath(__file__))
    data_dir   = os.path.join(base_dir, 'Data')
    output_dir = os.path.join(base_dir, 'Output')
    os.makedirs(output_dir, exist_ok=True)

    # -------- luminosity evolution + beta prior --------
    # params: [d0, ds, beta0]
    params           = [2.26, -0.16, 0.11]      # lumi_params + beta0
    param_errors     = [0.13,  0.18,  0.03]     

    free_params = ['Om', 'w0', 'alpha', 'g0','gs', 'beta'] 
    free_params_guess = [0.3, -1.0, 1.45, 2.05, -0.1, 0.11]

    evo         = 'linear'
    prior       = 'evog-evod-DESa-wphi'
    lnprob_func = 'lnprob_cos6D_wphi_gamma'

    save_path = output_dir
    filename  = os.path.join(data_dir, 'DESa_sim_l3_wphi_evoPL.fits')

    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")

    lens_cosmo = LensCosmo(
        params=params,
        param_errors=param_errors,
        free_params=free_params,
        free_params_guess=free_params_guess,
        evo=evo,
        prior=prior,
        lnprob_function=lnprob_func,
        filename=filename,
        save_path=save_path,
        nsteps=10000,
        nwalkers=200,
        ndim=len(free_params),
        ncpu=12,
        checkpoint=True,
        nsteps_per_checkpoint=2000
    )

    data = lens_cosmo.load_fits_data(evo, filename, eta, eta_err)
    lens_cosmo.mcmc(evo, data, free_params, free_params_guess)
    
def main_2D_linear_18():
    # Base directory = the folder containing this script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define data and output directories relative to that
    data_dir   = os.path.join(base_dir, 'Data')
    output_dir = os.path.join(base_dir, 'Output')

    # Make sure Output exists
    os.makedirs(output_dir, exist_ok=True)

    # Define parameters for the LensCosmo class
    params           = [2.052, -0.19 ,  2.270, -0.18, 0.0]
    param_errors     = [0.041,  0.11,   0.098,  0.18,  0.1]
    free_params      = ['Om', 'w']
    free_params_guess= [0.3,  -1.0]
    evo              = 'linear'
    prior            = '0.18-beta0'
    lnprob_func      = 'lnprob_cos2D'

    # Now both of these are relative
    save_path = output_dir
    filename  = os.path.join(data_dir, 'SGLTable03.fits')

    # Load data from a FITS file
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")

    # Initialize LensCosmo
    lens_cosmo = LensCosmo(
        params=params,
        param_errors=param_errors,
        free_params=free_params,
        free_params_guess=free_params_guess,
        evo=evo,
        prior=prior,
        lnprob_function=lnprob_func,
        filename=filename,
        save_path=save_path,
        nsteps=10000,
        nwalkers=200,
        ndim=len(free_params),
        ncpu=12,
        checkpoint=True,
        nsteps_per_checkpoint=2000
    )

    # Load and run
    data = lens_cosmo.load_fits_data(evo, filename, eta, eta_err)
    lens_cosmo.mcmc(evo, data, free_params, free_params_guess)
    
def main_2D_CPL_tri():
    # Base directory = the folder containing this script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define data and output directories relative to that
    data_dir   = os.path.join(base_dir, 'Data')
    output_dir = os.path.join(base_dir, 'Output')

    # Make sure Output exists
    os.makedirs(output_dir, exist_ok=True)

    # Define parameters for the LensCosmo class
    params           = [2.079, -0.35 ,  2.17, -0.25, 0.0]
    param_errors     = [0.056,  0.23,   0.17,  0.4,   0.1 ]
    free_params      = ['Om', 'w']
    free_params_guess= [0.3,  -1.0]
    evo              = 'CPL'
    prior            = 'Tri-beta0'
    lnprob_func      = 'lnprob_cos2D'

    # Now both of these are relative
    save_path = output_dir
    filename  = os.path.join(data_dir, 'SGLTable03.fits')

    # Load data from a FITS file
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")

    # Initialize LensCosmo
    lens_cosmo = LensCosmo(
        params=params,
        param_errors=param_errors,
        free_params=free_params,
        free_params_guess=free_params_guess,
        evo=evo,
        prior=prior,
        lnprob_function=lnprob_func,
        filename=filename,
        save_path=save_path,
        nsteps=10000,
        nwalkers=200,
        ndim=len(free_params),
        ncpu=12,
        checkpoint=True,
        nsteps_per_checkpoint=2000
    )

    # Load and run
    data = lens_cosmo.load_fits_data(evo, filename, eta, eta_err)
    lens_cosmo.mcmc(evo, data, free_params, free_params_guess)
    
def main_2D_CPL_22():
    # Base directory = the folder containing this script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define data and output directories relative to that
    data_dir   = os.path.join(base_dir, 'Data')
    output_dir = os.path.join(base_dir, 'Output')

    # Make sure Output exists
    os.makedirs(output_dir, exist_ok=True)

    # Define parameters for the LensCosmo class
    params           = [2.069, -0.341 ,  2.29, -0.36, 0.0]
    param_errors     = [0.051,  0.23,   0.14,  0.39,  0.087]
    free_params      = ['Om', 'w']
    free_params_guess= [0.3,  -1.0]
    evo              = 'CPL'
    prior            = '0.22-beta0'
    lnprob_func      = 'lnprob_cos2D'

    # Now both of these are relative
    save_path = output_dir
    filename  = os.path.join(data_dir, 'SGLTable03.fits')

    # Load data from a FITS file
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")

    # Initialize LensCosmo
    lens_cosmo = LensCosmo(
        params=params,
        param_errors=param_errors,
        free_params=free_params,
        free_params_guess=free_params_guess,
        evo=evo,
        prior=prior,
        lnprob_function=lnprob_func,
        filename=filename,
        save_path=save_path,
        nsteps=10000,
        nwalkers=200,
        ndim=len(free_params),
        ncpu=12,
        checkpoint=True,
        nsteps_per_checkpoint=2000
    )

    # Load and run
    data = lens_cosmo.load_fits_data(evo, filename, eta, eta_err)
    lens_cosmo.mcmc(evo, data, free_params, free_params_guess)
    
def main_2D_CPL_18():
    # Base directory = the folder containing this script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define data and output directories relative to that
    data_dir   = os.path.join(base_dir, 'Data')
    output_dir = os.path.join(base_dir, 'Output')

    # Make sure Output exists
    os.makedirs(output_dir, exist_ok=True)

    # Define parameters for the LensCosmo class
    params           = [2.068, -0.34 ,  2.3, -0.34, 0.0]
    param_errors     = [0.051,  0.23,   0.11, 0.39, 0.087]
    free_params      = ['Om', 'w']
    free_params_guess= [0.3,  -1.0]
    evo              = 'CPL'
    prior            = '0.18-beta0'
    lnprob_func      = 'lnprob_cos2D'

    # Now both of these are relative
    save_path = output_dir
    filename  = os.path.join(data_dir, 'SGLTable03.fits')

    # Load data from a FITS file
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")

    # Initialize LensCosmo
    lens_cosmo = LensCosmo(
        params=params,
        param_errors=param_errors,
        free_params=free_params,
        free_params_guess=free_params_guess,
        evo=evo,
        prior=prior,
        lnprob_function=lnprob_func,
        filename=filename,
        save_path=save_path,
        nsteps=10000,
        nwalkers=200,
        ndim=len(free_params),
        ncpu=12,
        checkpoint=True,
        nsteps_per_checkpoint=2000
    )

    # Load and run
    data = lens_cosmo.load_fits_data(evo, filename, eta, eta_err)
    lens_cosmo.mcmc(evo, data, free_params, free_params_guess)
    
def main_2D_linear_LD():
    # Base directory = the folder containing this script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define data and output directories relative to that
    data_dir   = os.path.join(base_dir, 'Data')
    output_dir = os.path.join(base_dir, 'Output')

    # Make sure Output exists
    os.makedirs(output_dir, exist_ok=True)

    # Define parameters for the LensCosmo class
    params           = [2.058, 0.043 ,  2.058, 0.043, 0.0]
    param_errors     = [0.031,  0.215,  0.031, 0.215,  0.1]
    free_params      = ['Om', 'w']
    free_params_guess= [0.3,  -1.0]
    evo              = 'linear-Etherington'
    prior            = 'LD-g'
    lnprob_func      = 'lnprob_cos2D'

    # Now both of these are relative
    save_path = output_dir
    filename  = os.path.join(data_dir, 'SGLTable03.fits')

    # Load data from a FITS file
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")

    # Initialize LensCosmo
    lens_cosmo = LensCosmo(
        params=params,
        param_errors=param_errors,
        free_params=free_params,
        free_params_guess=free_params_guess,
        evo=evo,
        prior=prior,
        lnprob_function=lnprob_func,
        filename=filename,
        save_path=save_path,
        nsteps=10000,
        nwalkers=200,
        ndim=len(free_params),
        ncpu=12,
        checkpoint=True,
        nsteps_per_checkpoint=2000
    )

    # Load and run
    data = lens_cosmo.load_fits_data(evo, filename, eta, eta_err)
    lens_cosmo.mcmc(evo, data, free_params, free_params_guess)
    
def main_2D_linear_OL():
    # Base directory = the folder containing this script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define data and output directories relative to that
    data_dir   = os.path.join(base_dir, 'Data')
    output_dir = os.path.join(base_dir, 'Output')

    # Make sure Output exists
    os.makedirs(output_dir, exist_ok=True)

    # Define parameters for the LensCosmo class
    params           = [2.077, 0.248 , 2.077, 0.248, 0.0]
    param_errors     = [0.028, 0.174,  0.028, 0.174, 0.1]
    free_params      = ['Om', 'w']
    free_params_guess= [0.3,  -1.0]
    evo              = 'linear-Etherington'
    prior            = 'OL-g'
    lnprob_func      = 'lnprob_cos2D'

    # Now both of these are relative
    save_path = output_dir
    filename  = os.path.join(data_dir, 'SGLTable03.fits')

    # Load data from a FITS file
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")

    # Initialize LensCosmo
    lens_cosmo = LensCosmo(
        params=params,
        param_errors=param_errors,
        free_params=free_params,
        free_params_guess=free_params_guess,
        evo=evo,
        prior=prior,
        lnprob_function=lnprob_func,
        filename=filename,
        save_path=save_path,
        nsteps=10000,
        nwalkers=200,
        ndim=len(free_params),
        ncpu=12,
        checkpoint=True,
        nsteps_per_checkpoint=2000
    )

    # Load and run
    data = lens_cosmo.load_fits_data(evo, filename, eta, eta_err)
    lens_cosmo.mcmc(evo, data, free_params, free_params_guess)
    
def main_3D_linear_tri():
    # Base directory = the folder containing this script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define data and output directories relative to that
    data_dir   = os.path.join(base_dir, 'Data')
    output_dir = os.path.join(base_dir, 'Output')

    # Make sure Output exists
    os.makedirs(output_dir, exist_ok=True)

    # Define parameters for the LensCosmo class
    params           = [2.065, -0.20 ,  2.14, -0.09, -0.05]
    param_errors     = [0.046,  0.12,   0.16,  0.19, 0.087]
    free_params      = ['Om', 'w0', 'alpha']
    free_params_guess= [0.3, -0.8,  1.45]
    evo              = 'linear'
    prior            = 'wphi-Tri'
    lnprob_func      = 'lnprob_cos3D_wphi'

    # Now both of these are relative
    save_path = output_dir
    filename  = os.path.join(data_dir, 'SGLTable03.fits')

    # Load data from a FITS file
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")

    # Initialize LensCosmo
    lens_cosmo = LensCosmo(
        params=params,
        param_errors=param_errors,
        free_params=free_params,
        free_params_guess=free_params_guess,
        evo=evo,
        prior=prior,
        lnprob_function=lnprob_func,
        filename=filename,
        save_path=save_path,
        nsteps=10000,
        nwalkers=200,
        ndim=len(free_params),
        ncpu=12,
        checkpoint=True,
        nsteps_per_checkpoint=2000
    )

    # Load and run
    data = lens_cosmo.load_fits_data(evo, filename, eta, eta_err)
    lens_cosmo.mcmc(evo, data, free_params, free_params_guess)
    
def main_3D_linear_22():
    # Base directory = the folder containing this script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define data and output directories relative to that
    data_dir   = os.path.join(base_dir, 'Data')
    output_dir = os.path.join(base_dir, 'Output')

    # Make sure Output exists
    os.makedirs(output_dir, exist_ok=True)

    # Define parameters for the LensCosmo class
    params           = [2.054, -0.19 ,  2.26, -0.16, 0.11]
    param_errors     = [0.041,  0.11,   0.13,  0.18, 0.085 ]
    free_params      = ['Om', 'w0', 'alpha']
    free_params_guess= [0.3, -1.0,  1.45]
    evo              = 'linear'
    prior            = 'wphi-0.22'
    lnprob_func      = 'lnprob_cos3D_wphi'

    # Now both of these are relative
    save_path = output_dir
    filename  = os.path.join(data_dir, 'SGLTable03.fits')

    # Load data from a FITS file
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")

    # Initialize LensCosmo
    lens_cosmo = LensCosmo(
        params=params,
        param_errors=param_errors,
        free_params=free_params,
        free_params_guess=free_params_guess,
        evo=evo,
        prior=prior,
        lnprob_function=lnprob_func,
        filename=filename,
        save_path=save_path,
        nsteps=10000,
        nwalkers=200,
        ndim=len(free_params),
        ncpu=12,
        checkpoint=True,
        nsteps_per_checkpoint=2000
    )

    # Load and run
    data = lens_cosmo.load_fits_data(evo, filename, eta, eta_err)
    lens_cosmo.mcmc(evo, data, free_params, free_params_guess)
    
def main_3D_linear_18():
    # Base directory = the folder containing this script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define data and output directories relative to that
    data_dir   = os.path.join(base_dir, 'Data')
    output_dir = os.path.join(base_dir, 'Output')

    # Make sure Output exists
    os.makedirs(output_dir, exist_ok=True)

    # Define parameters for the LensCosmo class
    params           = [2.052, -0.19 ,  2.270, -0.18, 0.133]
    param_errors     = [0.041,  0.11,   0.098,  0.18,  0.1]
    free_params      = ['Ok', 'w0', 'wa']
    free_params_guess= [0.0, -0.8,  -1.0]
    evo              = 'linear'
    prior            = 'kevow-0.18'
    lnprob_func      = 'lnprob_cos3D_kevow'

    # Now both of these are relative
    save_path = output_dir
    filename  = os.path.join(data_dir, 'SGLTable03.fits')

    # Load data from a FITS file
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")

    # Initialize LensCosmo
    lens_cosmo = LensCosmo(
        params=params,
        param_errors=param_errors,
        free_params=free_params,
        free_params_guess=free_params_guess,
        evo=evo,
        prior=prior,
        lnprob_function=lnprob_func,
        filename=filename,
        save_path=save_path,
        nsteps=10000,
        nwalkers=200,
        ndim=len(free_params),
        ncpu=12,
        checkpoint=True,
        nsteps_per_checkpoint=2000
    )

    # Load and run
    data = lens_cosmo.load_fits_data(evo, filename, eta, eta_err)
    lens_cosmo.mcmc(evo, data, free_params, free_params_guess)
    
def main_3D_CPL_tri():
    # Base directory = the folder containing this script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define data and output directories relative to that
    data_dir   = os.path.join(base_dir, 'Data')
    output_dir = os.path.join(base_dir, 'Output')

    # Make sure Output exists
    os.makedirs(output_dir, exist_ok=True)

    # Define parameters for the LensCosmo class
    params           = [2.079, -0.35 ,  2.17, -0.25, 0.02]
    param_errors     = [0.056,  0.23,   0.17,  0.4,   0.1 ]
    free_params      = ['Ok', 'w0', 'wa']
    free_params_guess= [0.0, -0.8,  -1.0]
    evo              = 'CPL'
    prior            = 'kevow-Tri'
    lnprob_func      = 'lnprob_cos3D_kevow'

    # Now both of these are relative
    save_path = output_dir
    filename  = os.path.join(data_dir, 'SGLTable03.fits')

    # Load data from a FITS file
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")

    # Initialize LensCosmo
    lens_cosmo = LensCosmo(
        params=params,
        param_errors=param_errors,
        free_params=free_params,
        free_params_guess=free_params_guess,
        evo=evo,
        prior=prior,
        lnprob_function=lnprob_func,
        filename=filename,
        save_path=save_path,
        nsteps=10000,
        nwalkers=200,
        ndim=len(free_params),
        ncpu=12,
        checkpoint=True,
        nsteps_per_checkpoint=2000
    )

    # Load and run
    data = lens_cosmo.load_fits_data(evo, filename, eta, eta_err)
    lens_cosmo.mcmc(evo, data, free_params, free_params_guess)
    
def main_3D_CPL_22():
    # Base directory = the folder containing this script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define data and output directories relative to that
    data_dir   = os.path.join(base_dir, 'Data')
    output_dir = os.path.join(base_dir, 'Output')

    # Make sure Output exists
    os.makedirs(output_dir, exist_ok=True)

    # Define parameters for the LensCosmo class
    params           = [2.069, -0.341 ,  2.29, -0.36, 0.12]
    param_errors     = [0.051,  0.23,   0.14,  0.39,  0.087]
    free_params      = ['Ok', 'w0', 'wa']
    free_params_guess= [0.0, -0.8,  -1.0]
    evo              = 'CPL'
    prior            = 'kevow-0.22'
    lnprob_func      = 'lnprob_cos3D_kevow'

    # Now both of these are relative
    save_path = output_dir
    filename  = os.path.join(data_dir, 'SGLTable03.fits')

    # Load data from a FITS file
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")

    # Initialize LensCosmo
    lens_cosmo = LensCosmo(
        params=params,
        param_errors=param_errors,
        free_params=free_params,
        free_params_guess=free_params_guess,
        evo=evo,
        prior=prior,
        lnprob_function=lnprob_func,
        filename=filename,
        save_path=save_path,
        nsteps=10000,
        nwalkers=200,
        ndim=len(free_params),
        ncpu=12,
        checkpoint=True,
        nsteps_per_checkpoint=2000
    )

    # Load and run
    data = lens_cosmo.load_fits_data(evo, filename, eta, eta_err)
    lens_cosmo.mcmc(evo, data, free_params, free_params_guess)
    
def main_3D_CPL_18():
    # Base directory = the folder containing this script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define data and output directories relative to that
    data_dir   = os.path.join(base_dir, 'Data')
    output_dir = os.path.join(base_dir, 'Output')

    # Make sure Output exists
    os.makedirs(output_dir, exist_ok=True)

    # Define parameters for the LensCosmo class
    params           = [2.068, -0.34 ,  2.3, -0.34, 0.136]
    param_errors     = [0.051,  0.23,   0.11, 0.39, 0.087]
    free_params      = ['Ok', 'w0', 'wa']
    free_params_guess= [0.0, -0.8,  -1.0]
    evo              = 'CPL'
    prior            = 'kevow-0.18'
    lnprob_func      = 'lnprob_cos3D_kevow'

    # Now both of these are relative
    save_path = output_dir
    filename  = os.path.join(data_dir, 'SGLTable03.fits')

    # Load data from a FITS file
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")

    # Initialize LensCosmo
    lens_cosmo = LensCosmo(
        params=params,
        param_errors=param_errors,
        free_params=free_params,
        free_params_guess=free_params_guess,
        evo=evo,
        prior=prior,
        lnprob_function=lnprob_func,
        filename=filename,
        save_path=save_path,
        nsteps=10000,
        nwalkers=200,
        ndim=len(free_params),
        ncpu=12,
        checkpoint=True,
        nsteps_per_checkpoint=2000
    )

    # Load and run
    data = lens_cosmo.load_fits_data(evo, filename, eta, eta_err)
    lens_cosmo.mcmc(evo, data, free_params, free_params_guess)
    
def main_3D_linear_LD():
    # Base directory = the folder containing this script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define data and output directories relative to that
    data_dir   = os.path.join(base_dir, 'Data')
    output_dir = os.path.join(base_dir, 'Output')

    # Make sure Output exists
    os.makedirs(output_dir, exist_ok=True)

    # Define parameters for the LensCosmo class
    params           = [2.058, 0.043 ,  2.173, 0.0, 0.0]
    param_errors     = [0.031,  0.215,  0.085,  0.1,  0.1]
    free_params      = ['Om', 'w0', 'wa']
    free_params_guess= [0.3, -0.8,  -1.0]
    evo              = 'linear-Etherington'
    prior            = 'LD'
    lnprob_func      = 'lnprob_cos3D_evow'

    # Now both of these are relative
    save_path = output_dir
    filename  = os.path.join(data_dir, 'SGLTable03.fits')

    # Load data from a FITS file
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")

    # Initialize LensCosmo
    lens_cosmo = LensCosmo(
        params=params,
        param_errors=param_errors,
        free_params=free_params,
        free_params_guess=free_params_guess,
        evo=evo,
        prior=prior,
        lnprob_function=lnprob_func,
        filename=filename,
        save_path=save_path,
        nsteps=10000,
        nwalkers=200,
        ndim=len(free_params),
        ncpu=12,
        checkpoint=True,
        nsteps_per_checkpoint=2000
    )

    # Load and run
    data = lens_cosmo.load_fits_data(evo, filename, eta, eta_err)
    lens_cosmo.mcmc(evo, data, free_params, free_params_guess)
    
def main_3D_linear_OL():
    # Base directory = the folder containing this script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define data and output directories relative to that
    data_dir   = os.path.join(base_dir, 'Data')
    output_dir = os.path.join(base_dir, 'Output')

    # Make sure Output exists
    os.makedirs(output_dir, exist_ok=True)

    # Define parameters for the LensCosmo class
    params           = [2.077, 0.248 ,  2.173, 0.0, 0.0]
    param_errors     = [0.028,  0.174,  0.085, 0.1, 0.1]
    free_params      = ['Om', 'w0', 'wa']
    free_params_guess= [0.3, -0.8,  -1.0]
    evo              = 'linear-Etherington'
    prior            = 'OL'
    lnprob_func      = 'lnprob_cos3D_evow'

    # Now both of these are relative
    save_path = output_dir
    filename  = os.path.join(data_dir, 'SGLTable03.fits')

    # Load data from a FITS file
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")

    # Initialize LensCosmo
    lens_cosmo = LensCosmo(
        params=params,
        param_errors=param_errors,
        free_params=free_params,
        free_params_guess=free_params_guess,
        evo=evo,
        prior=prior,
        lnprob_function=lnprob_func,
        filename=filename,
        save_path=save_path,
        nsteps=10000,
        nwalkers=200,
        ndim=len(free_params),
        ncpu=12,
        checkpoint=True,
        nsteps_per_checkpoint=2000
    )

    # Load and run
    data = lens_cosmo.load_fits_data(evo, filename, eta, eta_err)
    lens_cosmo.mcmc(evo, data, free_params, free_params_guess)
    
def main_kw_linear_tri():
    # Base directory = the folder containing this script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define data and output directories relative to that
    data_dir   = os.path.join(base_dir, 'Data')
    output_dir = os.path.join(base_dir, 'Output')

    # Make sure Output exists
    os.makedirs(output_dir, exist_ok=True)

    # Define parameters for the LensCosmo class
    params           = [2.065, -0.20 ,  2.14, -0.09, -0.03]
    param_errors     = [0.046,  0.12,   0.16,  0.19, 0.087 ]
    free_params      = ['Ok','w']
    free_params_guess= [0.0, -1.0]
    evo              = 'linear'
    prior            = 'kw-Tri'
    lnprob_func      = 'lnprob_kw'

    # Now both of these are relative
    save_path = output_dir
    filename  = os.path.join(data_dir, 'SGLTable03.fits')

    # Load data from a FITS file
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")

    # Initialize LensCosmo
    lens_cosmo = LensCosmo(
        params=params,
        param_errors=param_errors,
        free_params=free_params,
        free_params_guess=free_params_guess,
        evo=evo,
        prior=prior,
        lnprob_function=lnprob_func,
        filename=filename,
        save_path=save_path,
        nsteps=10000,
        nwalkers=200,
        ndim=len(free_params),
        ncpu=12,
        checkpoint=True,
        nsteps_per_checkpoint=2000
    )

    # Load and run
    data = lens_cosmo.load_fits_data(evo, filename, eta, eta_err)
    lens_cosmo.mcmc(evo, data, free_params, free_params_guess)
    
def main_kw_linear_22():
    # Base directory = the folder containing this script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define data and output directories relative to that
    data_dir   = os.path.join(base_dir, 'Data')
    output_dir = os.path.join(base_dir, 'Output')

    # Make sure Output exists
    os.makedirs(output_dir, exist_ok=True)

    # Define parameters for the LensCosmo class
    params           = [2.054, -0.19 ,  2.26, -0.16, 0.11]
    param_errors     = [0.041,  0.11,   0.13,  0.18, 0.085 ]
    free_params      = ['Ok','w']
    free_params_guess= [ 0.0, -1.0]
    evo              = 'linear'
    prior            = 'kw-0.22'
    lnprob_func      = 'lnprob_kw'

    # Now both of these are relative
    save_path = output_dir
    filename  = os.path.join(data_dir, 'SGLTable03.fits')

    # Load data from a FITS file
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")

    # Initialize LensCosmo
    lens_cosmo = LensCosmo(
        params=params,
        param_errors=param_errors,
        free_params=free_params,
        free_params_guess=free_params_guess,
        evo=evo,
        prior=prior,
        lnprob_function=lnprob_func,
        filename=filename,
        save_path=save_path,
        nsteps=10000,
        nwalkers=200,
        ndim=len(free_params),
        ncpu=12,
        checkpoint=True,
        nsteps_per_checkpoint=2000
    )

    # Load and run
    data = lens_cosmo.load_fits_data(evo, filename, eta, eta_err)
    lens_cosmo.mcmc(evo, data, free_params, free_params_guess)
    
def main_kw_linear_18():
    # Base directory = the folder containing this script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define data and output directories relative to that
    data_dir   = os.path.join(base_dir, 'Data')
    output_dir = os.path.join(base_dir, 'Output')

    # Make sure Output exists
    os.makedirs(output_dir, exist_ok=True)

    # Define parameters for the LensCosmo class
    params           = [2.052, -0.19 ,  2.270, -0.18, 0.133]
    param_errors     = [0.041,  0.11,   0.098,  0.18,  0.1]
    free_params      = ['Ok','w']
    free_params_guess= [0.0, -1.0]
    evo              = 'linear'
    prior            = 'kw-0.18'
    lnprob_func      = 'lnprob_kw'

    # Now both of these are relative
    save_path = output_dir
    filename  = os.path.join(data_dir, 'SGLTable03.fits')

    # Load data from a FITS file
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")

    # Initialize LensCosmo
    lens_cosmo = LensCosmo(
        params=params,
        param_errors=param_errors,
        free_params=free_params,
        free_params_guess=free_params_guess,
        evo=evo,
        prior=prior,
        lnprob_function=lnprob_func,
        filename=filename,
        save_path=save_path,
        nsteps=10000,
        nwalkers=200,
        ndim=len(free_params),
        ncpu=12,
        checkpoint=True,
        nsteps_per_checkpoint=2000
    )

    # Load and run
    data = lens_cosmo.load_fits_data(evo, filename, eta, eta_err)
    lens_cosmo.mcmc(evo, data, free_params, free_params_guess)
    
def main_kw_CPL_tri():
    # Base directory = the folder containing this script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define data and output directories relative to that
    data_dir   = os.path.join(base_dir, 'Data')
    output_dir = os.path.join(base_dir, 'Output')

    # Make sure Output exists
    os.makedirs(output_dir, exist_ok=True)

    # Define parameters for the LensCosmo class
    params           = [2.079, -0.35 ,  2.17, -0.25, 0.02]
    param_errors     = [0.056,  0.23,   0.17,  0.4,   0.1 ]
    free_params      = ['Ok','w']
    free_params_guess= [0.0, -1.0]
    evo              = 'CPL'
    prior            = 'kw-Tri'
    lnprob_func      = 'lnprob_kw'

    # Now both of these are relative
    save_path = output_dir
    filename  = os.path.join(data_dir, 'SGLTable03.fits')

    # Load data from a FITS file
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")

    # Initialize LensCosmo
    lens_cosmo = LensCosmo(
        params=params,
        param_errors=param_errors,
        free_params=free_params,
        free_params_guess=free_params_guess,
        evo=evo,
        prior=prior,
        lnprob_function=lnprob_func,
        filename=filename,
        save_path=save_path,
        nsteps=10000,
        nwalkers=200,
        ndim=len(free_params),
        ncpu=12,
        checkpoint=True,
        nsteps_per_checkpoint=2000
    )

    # Load and run
    data = lens_cosmo.load_fits_data(evo, filename, eta, eta_err)
    lens_cosmo.mcmc(evo, data, free_params, free_params_guess)
    
def main_kw_CPL_22():
    # Base directory = the folder containing this script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define data and output directories relative to that
    data_dir   = os.path.join(base_dir, 'Data')
    output_dir = os.path.join(base_dir, 'Output')

    # Make sure Output exists
    os.makedirs(output_dir, exist_ok=True)

    # Define parameters for the LensCosmo class
    params           = [2.069, -0.341 ,  2.29, -0.36, 0.12]
    param_errors     = [0.051,  0.23,   0.14,  0.39,  0.087]
    free_params      = [ 'Ok','w']
    free_params_guess= [ 0.0, -1.0]
    evo              = 'CPL'
    prior            = 'kw-0.22'
    lnprob_func      = 'lnprob_kw'

    # Now both of these are relative
    save_path = output_dir
    filename  = os.path.join(data_dir, 'SGLTable03.fits')

    # Load data from a FITS file
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")

    # Initialize LensCosmo
    lens_cosmo = LensCosmo(
        params=params,
        param_errors=param_errors,
        free_params=free_params,
        free_params_guess=free_params_guess,
        evo=evo,
        prior=prior,
        lnprob_function=lnprob_func,
        filename=filename,
        save_path=save_path,
        nsteps=10000,
        nwalkers=200,
        ndim=len(free_params),
        ncpu=12,
        checkpoint=True,
        nsteps_per_checkpoint=2000
    )

    # Load and run
    data = lens_cosmo.load_fits_data(evo, filename, eta, eta_err)
    lens_cosmo.mcmc(evo, data, free_params, free_params_guess)
    
def main_kw_CPL_18():
    # Base directory = the folder containing this script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define data and output directories relative to that
    data_dir   = os.path.join(base_dir, 'Data')
    output_dir = os.path.join(base_dir, 'Output')

    # Make sure Output exists
    os.makedirs(output_dir, exist_ok=True)

    # Define parameters for the LensCosmo class
    params           = [2.068, -0.34 ,  2.3, -0.34, 0.136]
    param_errors     = [0.051,  0.23,   0.11, 0.39, 0.087]
    free_params      = ['Ok','w']
    free_params_guess= [0.0, -1.0]
    evo              = 'CPL'
    prior            = 'kw-0.18'
    lnprob_func      = 'lnprob_kw'

    # Now both of these are relative
    save_path = output_dir
    filename  = os.path.join(data_dir, 'SGLTable03.fits')

    # Load data from a FITS file
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")

    # Initialize LensCosmo
    lens_cosmo = LensCosmo(
        params=params,
        param_errors=param_errors,
        free_params=free_params,
        free_params_guess=free_params_guess,
        evo=evo,
        prior=prior,
        lnprob_function=lnprob_func,
        filename=filename,
        save_path=save_path,
        nsteps=10000,
        nwalkers=200,
        ndim=len(free_params),
        ncpu=12,
        checkpoint=True,
        nsteps_per_checkpoint=2000
    )

    # Load and run
    data = lens_cosmo.load_fits_data(evo, filename, eta, eta_err)
    lens_cosmo.mcmc(evo, data, free_params, free_params_guess)
    
def main_mk_linear_tri():
    # Base directory = the folder containing this script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define data and output directories relative to that
    data_dir   = os.path.join(base_dir, 'Data')
    output_dir = os.path.join(base_dir, 'Output')

    # Make sure Output exists
    os.makedirs(output_dir, exist_ok=True)

    # Define parameters for the LensCosmo class
    params           = [2.065, -0.20 ,  2.14, -0.09, -0.03]
    param_errors     = [0.046,  0.12,   0.16,  0.19, 0.087 ]
    free_params      = ['Om','Ok']
    free_params_guess= [0.3, 0.0]
    evo              = 'linear'
    prior            = 'mk-Tri'
    lnprob_func      = 'lnprob_mk'

    # Now both of these are relative
    save_path = output_dir
    filename  = os.path.join(data_dir, 'SGLTable03.fits')

    # Load data from a FITS file
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")

    # Initialize LensCosmo
    lens_cosmo = LensCosmo(
        params=params,
        param_errors=param_errors,
        free_params=free_params,
        free_params_guess=free_params_guess,
        evo=evo,
        prior=prior,
        lnprob_function=lnprob_func,
        filename=filename,
        save_path=save_path,
        nsteps=10000,
        nwalkers=200,
        ndim=len(free_params),
        ncpu=12,
        checkpoint=True,
        nsteps_per_checkpoint=2000
    )

    # Load and run
    data = lens_cosmo.load_fits_data(evo, filename, eta, eta_err)
    lens_cosmo.mcmc(evo, data, free_params, free_params_guess)
    
def main_mk_linear_22():
    # Base directory = the folder containing this script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define data and output directories relative to that
    data_dir   = os.path.join(base_dir, 'Data')
    output_dir = os.path.join(base_dir, 'Output')

    # Make sure Output exists
    os.makedirs(output_dir, exist_ok=True)

    # Define parameters for the LensCosmo class
    params           = [2.054, -0.19 ,  2.26, -0.16, 0.11]
    param_errors     = [0.041,  0.11,   0.13,  0.18, 0.085 ]
    free_params      = ['Om','Ok']
    free_params_guess= [0.3, 0.0]
    evo              = 'linear'
    prior            = 'mk-0.22'
    lnprob_func      = 'lnprob_mk'

    # Now both of these are relative
    save_path = output_dir
    filename  = os.path.join(data_dir, 'SGLTable03.fits')

    # Load data from a FITS file
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")

    # Initialize LensCosmo
    lens_cosmo = LensCosmo(
        params=params,
        param_errors=param_errors,
        free_params=free_params,
        free_params_guess=free_params_guess,
        evo=evo,
        prior=prior,
        lnprob_function=lnprob_func,
        filename=filename,
        save_path=save_path,
        nsteps=10000,
        nwalkers=200,
        ndim=len(free_params),
        ncpu=12,
        checkpoint=True,
        nsteps_per_checkpoint=2000
    )

    # Load and run
    data = lens_cosmo.load_fits_data(evo, filename, eta, eta_err)
    lens_cosmo.mcmc(evo, data, free_params, free_params_guess)
    
def main_mk_linear_18():
    # Base directory = the folder containing this script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define data and output directories relative to that
    data_dir   = os.path.join(base_dir, 'Data')
    output_dir = os.path.join(base_dir, 'Output')

    # Make sure Output exists
    os.makedirs(output_dir, exist_ok=True)

    # Define parameters for the LensCosmo class
    params           = [2.052, -0.19 ,  2.270, -0.18, 0.133]
    param_errors     = [0.041,  0.11,   0.098,  0.18,  0.1]
    free_params      = ['Om','Ok']
    free_params_guess= [0.3, 0.0]
    evo              = 'linear'
    prior            = 'mk-0.18'
    lnprob_func      = 'lnprob_mk'

    # Now both of these are relative
    save_path = output_dir
    filename  = os.path.join(data_dir, 'SGLTable03.fits')

    # Load data from a FITS file
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")

    # Initialize LensCosmo
    lens_cosmo = LensCosmo(
        params=params,
        param_errors=param_errors,
        free_params=free_params,
        free_params_guess=free_params_guess,
        evo=evo,
        prior=prior,
        lnprob_function=lnprob_func,
        filename=filename,
        save_path=save_path,
        nsteps=10000,
        nwalkers=200,
        ndim=len(free_params),
        ncpu=12,
        checkpoint=True,
        nsteps_per_checkpoint=2000
    )

    # Load and run
    data = lens_cosmo.load_fits_data(evo, filename, eta, eta_err)
    lens_cosmo.mcmc(evo, data, free_params, free_params_guess)
    
def main_mk_CPL_tri():
    # Base directory = the folder containing this script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define data and output directories relative to that
    data_dir   = os.path.join(base_dir, 'Data')
    output_dir = os.path.join(base_dir, 'Output')

    # Make sure Output exists
    os.makedirs(output_dir, exist_ok=True)

    # Define parameters for the LensCosmo class
    params           = [2.079, -0.35 ,  2.17, -0.25, 0.02]
    param_errors     = [0.056,  0.23,   0.17,  0.4,   0.1 ]
    free_params      = ['Om','Ok']
    free_params_guess= [0.3, 0.0]
    evo              = 'CPL'
    prior            = 'mk-Tri'
    lnprob_func      = 'lnprob_mk'

    # Now both of these are relative
    save_path = output_dir
    filename  = os.path.join(data_dir, 'SGLTable03.fits')

    # Load data from a FITS file
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")

    # Initialize LensCosmo
    lens_cosmo = LensCosmo(
        params=params,
        param_errors=param_errors,
        free_params=free_params,
        free_params_guess=free_params_guess,
        evo=evo,
        prior=prior,
        lnprob_function=lnprob_func,
        filename=filename,
        save_path=save_path,
        nsteps=10000,
        nwalkers=200,
        ndim=len(free_params),
        ncpu=12,
        checkpoint=True,
        nsteps_per_checkpoint=2000
    )

    # Load and run
    data = lens_cosmo.load_fits_data(evo, filename, eta, eta_err)
    lens_cosmo.mcmc(evo, data, free_params, free_params_guess)
    
def main_mk_CPL_22():
    # Base directory = the folder containing this script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define data and output directories relative to that
    data_dir   = os.path.join(base_dir, 'Data')
    output_dir = os.path.join(base_dir, 'Output')

    # Make sure Output exists
    os.makedirs(output_dir, exist_ok=True)

    # Define parameters for the LensCosmo class
    params           = [2.069, -0.341 ,  2.29, -0.36, 0.12]
    param_errors     = [0.051,  0.23,   0.14,  0.39,  0.087]
    free_params      = ['Om','Ok']
    free_params_guess= [0.3, 0.0]
    evo              = 'CPL'
    prior            = 'mk-0.22'
    lnprob_func      = 'lnprob_mk'

    # Now both of these are relative
    save_path = output_dir
    filename  = os.path.join(data_dir, 'SGLTable03.fits')

    # Load data from a FITS file
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")

    # Initialize LensCosmo
    lens_cosmo = LensCosmo(
        params=params,
        param_errors=param_errors,
        free_params=free_params,
        free_params_guess=free_params_guess,
        evo=evo,
        prior=prior,
        lnprob_function=lnprob_func,
        filename=filename,
        save_path=save_path,
        nsteps=10000,
        nwalkers=200,
        ndim=len(free_params),
        ncpu=12,
        checkpoint=True,
        nsteps_per_checkpoint=2000
    )

    # Load and run
    data = lens_cosmo.load_fits_data(evo, filename, eta, eta_err)
    lens_cosmo.mcmc(evo, data, free_params, free_params_guess)
    
def main_mk_CPL_18():
    # Base directory = the folder containing this script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define data and output directories relative to that
    data_dir   = os.path.join(base_dir, 'Data')
    output_dir = os.path.join(base_dir, 'Output')

    # Make sure Output exists
    os.makedirs(output_dir, exist_ok=True)

    # Define parameters for the LensCosmo class
    params           = [2.068, -0.34 ,  2.3, -0.34, 0.136]
    param_errors     = [0.051,  0.23,   0.11, 0.39, 0.087]
    free_params      = ['Om','Ok']
    free_params_guess= [0.3, 0.0]
    evo              = 'CPL'
    prior            = 'mk-0.18'
    lnprob_func      = 'lnprob_mk'

    # Now both of these are relative
    save_path = output_dir
    filename  = os.path.join(data_dir, 'SGLTable03.fits')

    # Load data from a FITS file
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")

    # Initialize LensCosmo
    lens_cosmo = LensCosmo(
        params=params,
        param_errors=param_errors,
        free_params=free_params,
        free_params_guess=free_params_guess,
        evo=evo,
        prior=prior,
        lnprob_function=lnprob_func,
        filename=filename,
        save_path=save_path,
        nsteps=10000,
        nwalkers=200,
        ndim=len(free_params),
        ncpu=12,
        checkpoint=True,
        nsteps_per_checkpoint=2000
    )

    # Load and run
    data = lens_cosmo.load_fits_data(evo, filename, eta, eta_err)
    lens_cosmo.mcmc(evo, data, free_params, free_params_guess)

if __name__ == "__main__":
    # main_6D_w0wa_linear_DESa_gamma_evo()
    # main_6D_wphi_linear_DESa_gamma_evo()
    
#     main_4D_linear_sim_noevo()
    main_5D_linear_sim_gamma_evo()
#    main_2D_linear_DESa_noevo()
    # main_2D_linear_11_DESa_evo()
#    main_2D_linear_DESa_evo7()
    # main_2D_linear_tri()
#    main_2D_linear_22()
#    main_2D_linear_18()
#    main_2D_CPL_tri()
#    main_2D_CPL_22()
#    main_2D_CPL_18()
#    main_2D_linear_LD()
#    main_2D_linear_OL()
#    main_3D_linear_tri()  
#    main_3D_linear_22()
#    main_3D_linear_18()
    # main_3D_CPL_tri()
    # main_3D_CPL_22()
#    main_3D_CPL_18()
#    main_3D_linear_LD()
#    main_3D_linear_OL()
#    main_kw_linear_tri()
#    main_kw_linear_22()
#    main_kw_linear_18()
#    main_kw_CPL_tri()
#    main_kw_CPL_22()
#    main_kw_CPL_18()
#    main_mk_linear_tri()
#    main_mk_linear_22()
#    main_mk_linear_18()
#    main_mk_CPL_tri()
#    main_mk_CPL_22()
#    main_mk_CPL_18()
    
