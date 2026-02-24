#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 12:17:13 2024

@author: gengs
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import pandas as pd

# data reading
prior = 'ANNfull'
output_path = r"/home/gengs/Codes/projects/GLS_Gamma/Gamma_codes/sgl_gamma/output3"

p_med_mad = r"/home/gengs/Codes/projects/GLS_Gamma/Gamma_codes/sgl_gamma/Output/individual_rec/ANNfull_gamma-zl_individual_nw_200_ns_20000/SGL_ANN_gammaMCMC.csv"


df_tmp = pd.read_csv(p_med_mad)


z_list = df_tmp['zl']
median_list = df_tmp['Gamma_median_ANN']
mad_tmp = df_tmp['Gamma_MAD_ANN']



# z_badguys05 = [0.121,0.438]
# median_badguys05 = [1.523,1.601]
# mad_badguys05 = [0.127,0.135]

# plt.xlim(0,1.0)
# plt.ylim(1.5,2.3)
# plt.errorbar(z_list, median_list,yerr = mad_tmp)


#log_likelihood
def linear_log_likelihood(theta, x, y, yerr):
    m, b = theta
    model = m * x + b
    sigma2 = yerr**2
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(2 * np.pi * sigma2))


def log_prior(theta):
    m, b, = theta
    if -2 < m < 2 and 1.5 < b < 2.5:
        return 0.0
    return -np.inf

def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + linear_log_likelihood(theta, x, y, yerr)



x = np.array(z_list)
y = np.array(median_list)
yerr = np.array(mad_tmp)

# x = np.array(z_list+z_badguys05)[1:]
# y = np.array(median_list+median_badguys05)[1:]
# yerr = np.array(mad_tmp+mad_badguys05)[1:]


from scipy.optimize import minimize

# np.random.seed(42)
# nll = lambda *args: -linear_log_likelihood(*args)
initial = np.array([0.0,2.0])
# soln = minimize(nll, initial, args=(x, y, yerr))
# m_ml, b_ml = soln.x

import emcee
from multiprocessing import Pool

ncpu = 12
checkpoint = True
nsteps = 6000
nsteps_per_checkpoint = 1000
burnin = None
save_outputs = True
chain_info_list = []
all_samples = []


with Pool(ncpu) as pool:
    # initial sampler
    ndim = 2
    nwalkers = 200

    p0 = [np.random.normal(loc=initial, scale=1e-4, size=ndim) for _ in range(nwalkers)]
    args = (x,y,yerr)
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                args = args,
                pool=pool
               )
    
    if checkpoint:   
        print(f'Running with checkpoints every {nsteps_per_checkpoint} ') 
        # Checkpoint system
        for _ in range(0, nsteps, nsteps_per_checkpoint):
            sampler.run_mcmc(p0, nsteps_per_checkpoint, store=True,progress=True)

            # Evaluate the current state of the chain
            try:
                tau = sampler.get_autocorr_time(tol=0)
                burnin = int(2 * np.max(tau))
                thin = 1
                print(f"Checkpoint: tau={tau}, burnin={burnin}, thin={thin}")

                # If chain is long enough, break
                if sampler.iteration / np.max(tau) > 50:
                    print("Sufficient chain length achieved. Stopping run.")
                    break
            except emcee.autocorr.AutocorrError as e:
                # Handle the case where autocorrelation time can't be reliably estimated
                print(str(e))
            p0 = sampler.get_last_sample()

    else:
        sampler.run_mcmc(p0, nsteps, progress=True)
    
        # only take one for each group of 10 for correlations between samples
        thin = 1
        
        if burnin is None:
            # autocorrelation check
            tau = sampler.get_autocorr_time()
            # Suggested burn-in
            burnin = int(2 * np.max(tau))
            # Suggested thinning
            thin = int(0.5 * np.min(tau))

    chain_info = {
        'nsteps': len(sampler.get_chain()),
        'nsteps_without_burn-in': len(sampler.get_chain(discard=burnin, thin=thin)),
        'burn-in': burnin,
        'thin': thin}
    
    chain_info_list.append(chain_info)

    if save_outputs:
        pd.DataFrame(chain_info_list).to_csv(os.path.join(output_path,'mcmc_chain_logfile.csv' ))

    # Append the samples for this 'z_l' bin to the list of all samples
    all_samples.append(sampler.get_chain(discard=burnin, thin=thin))
    #all_ln_probs.append(sampler.get_log_prob())

all_samples = np.array(all_samples) 
np.save(os.path.join(output_path,f'linear_z.npy'),all_samples)

print("Done!")

# plotting

samples = np.load(os.path.join(output_path,f'linear_z.npy'))[0]

def calculate_marginal_median_and_mad(samples):
    """
    Calculate the marginal median and median absolute deviation for each parameter.

    :param samples: A numpy array of MCMC samples with shape (nwalkers, nsteps, ndim)
    :return: Two numpy arrays containing the marginal medians and MADs for each parameter
    """
    # Reshape the samples to a 2D array (nwalkers*nsteps, ndim)
    nwalkers, nsteps, ndim = samples.shape
    flattened_samples = samples.reshape(-1, ndim)

    # Calculate marginal medians
    marginal_medians = np.median(flattened_samples, axis=0)

    # Calculate median absolute deviations
    mad = np.median(np.abs(flattened_samples - marginal_medians), axis=0)

    return marginal_medians, mad

def calculate_marginal_peak_and_mad(samples):
    """
    Calculate the marginal peak (mode) and median absolute deviation for each parameter.

    :param samples: A numpy array of MCMC samples with shape (nwalkers, nsteps, ndim)
    :return: Two numpy arrays containing the marginal peaks and MADs for each parameter
    """
    # Reshape the samples to a 2D array (nwalkers*nsteps, ndim)
    nwalkers, nsteps, ndim = samples.shape
    flattened_samples = samples.reshape(-1, ndim)

    marginal_peaks = np.empty(ndim)
    for i in range(ndim):
        # Calculate histogram for the i-th parameter
        hist, bin_edges = np.histogram(flattened_samples[:, i], bins='auto', density=True)
        # Find the bin with the highest frequency
        max_bin_index = np.argmax(hist)
        # Estimate the mode as the midpoint of the bin with the highest frequency
        marginal_peaks[i] = 0.5 * (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1])

    # Calculate median absolute deviations
    mad = np.median(np.abs(flattened_samples - np.median(flattened_samples, axis=0)), axis=0)

    return marginal_peaks, mad

marginal_medians, mads = calculate_marginal_median_and_mad(samples)

def plot_linear_uncertainty(x_list,a, da, b, db, line_style, line_color, line_label, 
                            region_color, region_alpha, region_label,zorder,hatch):
    """
    Plot a linear equation with its uncertainty region based on uncertainties in slope and intercept.
    Parameters:
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - title (str): Title of the plot.
    - a (float): Slope of the line.
    - da (float): Uncertainty in the slope.
    - b (float): Intercept of the line.
    - db (float): Uncertainty in the intercept.
    - line_style (str): Style of the line (e.g., '-', '--', ':').
    - line_color (str): Color of the line.
    - region_color (str): Color of the uncertainty region.
    - region_alpha (float): Transparency of the uncertainty region.
    - line_label (str): Label for the line in the legend.
    - region_label (str): Label for the uncertainty region in the legend.
    """
    # Generate a range of x values
    z = x_list
    # Calculate the central line y
    central_line = a * z + b
    # Calculate the upper and lower bounds of the region
    upper_bound = central_line+np.sqrt(z**2.0*da**2.0+db**2.0)
    lower_bound = central_line-np.sqrt(z**2.0*da**2.0+db**2.0)
    # Plotting
    # plt.figure(figsize=(8, 6))
    plt.plot(z, central_line, line_style, color=line_color, label=r'Gamma best fit: $\rm y = \rm %0.3fx (\pm %0.3f) + %0.3f (\pm %0.3f)$'%(a,da,b,db))  # Central line
    plt.fill_between(z, lower_bound, upper_bound, color=region_color, alpha=region_alpha, label=region_label, zorder=zorder,hatch=hatch)

# 
x_list= np.linspace(0., 1.2, 100)

plt.figure(figsize=(12,8))
plot_linear_uncertainty(x_list=x_list,
    a=marginal_medians[0], da=mads[0], b=marginal_medians[1], db=mads[1],
    line_style='-', line_color='#D00000', line_label='linear fit',
    region_color='#D00000', region_alpha=0.4,
    region_label=r'$1\sigma$ uncertainty',
    zorder=0,hatch=''
)
plt.xlim(0,1.2)
plt.ylim(1.5,2.5)
plt.errorbar(x, y,yerr = yerr,fmt='o')

plt.title('Redshift evolution of Total mass density slope in %s'%(prior))
plt.legend(fontsize=20)

plt.savefig(os.path.join(output_path,f'linear_z_{prior}.png'))
print("Figure saved!")