# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 11:43:41 2025

@author: poilo
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from getdist import plots, MCSamples

def calculate_marginal_median_and_mad(samples):
    """
    Flatten (nwalkers, nsteps, ndim) → (nwalkers*nsteps, ndim)
    and return (median, MAD) arrays of length ndim.
    """
    nwalkers, nsteps, ndim = samples.shape
    flat = samples.reshape(-1, ndim)
    med = np.median(flat, axis=0)
    mad = np.median(np.abs(flat - med), axis=0)
    return med, mad

def calculate_marginal_peak_and_mad(samples, bins='auto'):
    """
    Estimate the marginal peak (mode) and the median absolute deviation (MAD)
    around that peak for each parameter.

    Parameters
    ----------
    samples : array_like, shape (nwalkers, nsteps, ndim)
        The raw MCMC chain.
    bins : int or str or sequence, optional
        Passed to np.histogram for mode estimation.  Defaults to 'auto'.

    Returns
    -------
    peaks : ndarray, shape (ndim,)
        The mode estimate for each parameter.
    mads : ndarray, shape (ndim,)
        The median absolute deviation from that mode for each parameter.
    """
    nwalkers, nsteps, ndim = samples.shape
    flat = samples.reshape(-1, ndim)

    peaks = np.empty(ndim)
    mads  = np.empty(ndim)

    for i in range(ndim):
        data = flat[:, i]

        # 1) Histogram-based mode estimate
        hist, edges = np.histogram(data, bins=bins, density=True)
        # find the bin with maximum density
        idx_peak = np.argmax(hist)
        # mode ≈ midpoint of that bin
        peaks[i] = 0.5*(edges[idx_peak] + edges[idx_peak+1])

        # 2) MAD around the mode
        mads[i] = np.median(np.abs(data - peaks[i]))

    return peaks, mads

def plot_triangle_multiple(samples_list, param_names, model_names,
                           output_folder, plot_name, colors=None, filled=False):
    """
    Make a GetDist triangle plot for multiple MCMC chains, but
    only annotate the FIRST chain’s 1D marginals with dashed lines & values.
    
    samples_list : list of arrays, each shape (nwalkers, nsteps, ndim)
    param_names  : list of LaTeX strings, length = ndim
    model_names  : list of labels, one per chain
    """
    n = len(samples_list)
    if len(model_names) != n:
        raise ValueError("samples_list and model_names must have same length")
    
    # Wrap into GetDist MCSamples objects
    mcs = [
        MCSamples(samples=s, names=param_names, label=lbl)
        for s, lbl in zip(samples_list, model_names)
    ]
    
    # Pick colors
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if colors is None:
        colors = default_colors
    if len(colors) < n:
        raise ValueError("Need at least as many colors as models")
    
    # Set up the plotter
    g = plots.getSubplotPlotter(width_inch=8)
    g.settings.title_limit_fontsize = 14
    g.settings.axes_fontsize       = 14
    g.settings.axes_labelsize      = 14
    
    # Triangle plot of all chains
    g.triangle_plot(
        mcs, param_names,
        contour_colors = colors[:n],
        legend_labels  = model_names,
        filled         = [filled]*n,
        alphas         = [0.3]*n,
        contour_lws    = [1.5]*n
    )
    
    # === Annotate only the FIRST chain’s medians, with colored titles ===
#    med_first, mad_first = calculate_marginal_median_and_mad(samples_list[0])
    med_first, mad_first = calculate_marginal_peak_and_mad(samples_list[0])
    c0 = colors[0]
    ndim = len(param_names)
    for i in range(ndim):
        ax = g.subplots[i, i]
        # draw dashed line at the median
        ax.axvline(med_first[i], color=c0, ls='--', lw=1)
        # set a colored title: "param = median ± MAD"
        title = f"{param_names[i]} = {med_first[i]:.3f} ± {mad_first[i]:.3f}"
        ax.set_title(title, color=c0, fontsize=12)
    # ================================================================
    
    # Save and show
    os.makedirs(output_folder, exist_ok=True)
    outpath = os.path.join(output_folder, plot_name)
    plt.savefig(outpath, bbox_inches='tight', dpi=200)
    # plt.show()
    print(f"Saved triangle plot to {outpath}")


if __name__ == "__main__":
    # ——— Usage example ———
    
    # Base directory (script folder or cwd in notebooks)
    if "__file__" in globals():
        base_dir = r"/home/astrodust/SG/sgl_cosmo/GLS_cosmo01/"
    else:
        base_dir = os.getcwd()
    
    # 1) Parameter names (ndim = 3 here)
    param_names = [r'$\Omega_m$', r'$w$', r'$\beta$']
    
    # 2) List your models (any length ≥ 1)
    model_names = [
        'CPL-all-Tri-bfree1',
        'CPL-all-Guer-bfree1',
        'CPL-all-Bolton-bfree',
        'linear-all-Tri-bfree1',
        'linear-all-Guer-bfree1',
        'linear-all-Bolton-bfree1',
    ]
    
    # 3) Load each MCMC chain from .npy files
    samples_list = []
    for name in model_names:
        fn = os.path.join(base_dir, 'Output', name, f'mcmc_cosmop3D-{name}.npy')
        samples = np.load(fn)  # shape (nwalkers, nsteps, ndim)
        samples_list.append(samples)
    
    # 4) Plot them, but annotate only the first chain’s medians
    plot_triangle_multiple(
        samples_list=samples_list,
        param_names=param_names,
        model_names=model_names,
        output_folder=os.path.join(base_dir, 'Output', 'comparisons'),
        plot_name='triangle_compare_bfree1.png',
        # Optionally pass your own colors:
        # colors = ['#4169E1','#E3B23C','#D00000','#2E8B57'],
        filled=True
    )
