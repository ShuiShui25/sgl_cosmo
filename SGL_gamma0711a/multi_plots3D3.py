# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 15:52:23 2024

@author: poilo
"""

import numpy as np
import pandas as pd
from getdist import plots, MCSamples
import matplotlib.pyplot as plt
import os
from scipy.stats import mode

def add_dollar_signs(labels):
    # This function will wrap each label in dollar signs for LaTeX formatting
    return [f"${label}$" for label in labels]

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

def plot_tri_GetDist_from_files(samples1, samples2, samples3, param_names, contour_label,
                                output_folder, plot_name, marker = True):
    # Convert your samples to MCSamples objects
    mcmc_samples1 = MCSamples(samples=samples1, names=param_names, label=contour_label[0])
    mcmc_samples2 = MCSamples(samples=samples2, names=param_names, label=contour_label[1])
    mcmc_samples3 = MCSamples(samples=samples3, names=param_names, label=contour_label[2])

    # Initialize the GetDist plotter
    g = plots.getSubplotPlotter(width_inch=10)
    g.settings.title_limit_fontsize = 15
    g.settings.axes_fontsize = 15
    g.settings.axes_labelsize = 20
    g.settings.legend_fontsize = 20
    
    color_list = ['#4169E1', '#E3B23C','#D00000']
    marker_sample = samples1
    marker_coler = color_list[0]

    # Triangle plot for both sample sets
    g.triangle_plot([mcmc_samples1, mcmc_samples2, mcmc_samples3], param_names,
                    contour_colors=color_list, alpha=0.1,
                    filled=[True, False, False], 
                    zorder=[1,2,3],
                    contour_ls=['-', '-', '--'],
                    contour_lws=[1,2,2])

    # Triangle plot with filled contours
    # g.triangle_plot(mcmc_samples1, param_names, filled=True)

    marginal_medians, mads = calculate_marginal_median_and_mad(marker_sample)
    
    if marker:
        marginal_medians, mads = calculate_marginal_median_and_mad(marker_sample)
        # Maker reference lines
        for i, val in enumerate(marginal_medians):
            g.subplots[i, 0].axvline(2.0, color='black', ls='--')
            if i+1 >= len(marginal_medians):
              continue
            g.subplots[i+1, 1].axvline(0.0, color='black', ls='--')
            if i+3 >= len(marginal_medians):
              continue
            g.subplots[i+3, 3].axvline(0.0, color='black', ls='--')
        
        # # Mark the median
        # # Add markers and lines for best fit values
        # n_params = len(marginal_medians)
        # for i, val in enumerate(marginal_medians):
        #     # Add red dashed line to 1D subplots if the subplot exists
        #     if g.subplots[i, i] is not None:
        #         g.subplots[i, i].axvline(val, color=marker_coler, ls='--')
    
        #     # Add lines to 2D subplots if the subplot exists
        #     for j in range(n_params):
        #         if j != i:
        #             if g.subplots[i, j] is not None:
        #                 g.subplots[i, j].axhline(val, color=marker_coler, ls='--')
        #             if g.subplots[j, i] is not None:
        #                 g.subplots[j, i].axvline(val, color=marker_coler, ls='--')
    
        # Manually set titles for the 1D marginal distributions
        for i, param_name in enumerate(param_names):
            median = marginal_medians[i]
            mad = mads[i]
            title = f"{param_name} = {round(median,3)} ± {round(mad,3)}"
            ax = g.subplots[i, i]
            ax.set_title(title, fontsize=10)

    # Optionally, save the plot
    if output_folder is not None and not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plt.savefig(os.path.join(output_folder, plot_name), bbox_inches="tight", dpi=200)


samples_file1 = "/home/gengs/Codes/projects/GLS_Gamma/Gamma_codes/sgl_gamma/Output/ANNfull-Tri_gamma-zl_Koopmans_3D_nw_200_ns_20000/mcmc_samples_Koopmans_3D.npy"  # Update this path
samples_file2 = "/home/gengs/Codes/projects/GLS_Gamma/Gamma_codes/sgl_gamma/Output/ANNfull-Guerrini2024_gamma-zl_Koopmans_3D_nw_200_ns_20000/mcmc_samples_Koopmans_3D.npy" # Update this path
samples_file3 = "/home/gengs/Codes/projects/GLS_Gamma/Gamma_codes/sgl_gamma/Output/ANNfull-Bolton2006_gamma-zl_Koopmans_3D_nw_200_ns_20000/mcmc_samples_Koopmans_3D.npy"

contour_label = ["Triangular Prior","Guerrini2024 Prior","Bolton2006 prior"]
param_names = [r'$\gamma$', r'$\delta$', r'$\beta$']
# param_names = [r'\gamma_0', r'\gamma_S', r'\delta_0', r'\delta_S', r'\beta']
output_folder = "/home/gengs/Codes/projects/GLS_Gamma/Gamma_codes/sgl_gamma/output3"  # Update with your actual output folder path
# output_folder = None
plot_name = 'Posterior_Dist_3D_full.pdf'

ndim = 3
# Load the samples from .npy files
samples1 = np.load(samples_file1)[0]
samples2 = np.load(samples_file2)[0]
samples3 = np.load(samples_file3)[0]

plot_tri_GetDist_from_files(samples1, samples2,samples3, param_names, contour_label,
                            output_folder, plot_name,marker = True)

print(calculate_marginal_median_and_mad(samples1))
print(calculate_marginal_median_and_mad(samples2))
print(calculate_marginal_median_and_mad(samples3))