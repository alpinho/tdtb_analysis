"""
P-values profile of results obtained across different
 individualizations of ROIs

Author: Ana Luisa Pinho

Created: 14th of March, 2025
Last update: March 2025

Compatibility: Python 3.10.14
"""

import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


# ############################ FUNCTIONS ################################

def plot_pvalues(weights, pvals, output_path='./'):
    """
    Plots corrected p-values as a function of the weighting factor for
    ROI individualization.

    Parameters:
    -----------
    weights : array-like
        Weighting factors for individualization.
    pvals : array-like
        Corrected p-values corresponding to the weighting factors.
    output_path : str, optional (default='./')
        File path where the plot will be saved.

    Returns:
    --------
    None
    """

    plt.figure(figsize=(8, 5))
    plt.scatter(weights, pvals, color='b')
    plt.plot(weights, pvals, linestyle='--', alpha=0.7)

    plt.xlabel(r'Weighting Factor for ROI Individualization ($w$)')
    plt.ylabel(r'Corrected p-value $p_{\mathrm{FWE}}(w)$')
    # plt.title(
    #     r'Effect of ROI Individualization on p-value: '
    #     r'$p_{\mathrm{corr}} = f(w)$')

    # Remove top and right axis
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Extend x-axis slightly to prevent edge clipping
    plt.xlim(-0.05, 1.05)  # Small padding to keep end dots fully visible

    # Explicitly set x-ticks and y-ticks
    plt.xticks(np.linspace(0, 1, 11))  # Ensures x-ticks at [0, 0.1, ..., 1]
    plt.yticks(np.linspace(.025, .05, 6))  # Ensures y-ticks at [0, 0.01, ..., 0.05]

    plt.grid(True)
    # plt.legend()
    plt.savefig(output_path, dpi=300)
    plt.close()


# ############################# INPUTS ##################################

encoding_mask_type = 'all'
anova_type = '2way-anova_cat2rois_auditory'

tags = ['i', 'i9a', 'i8a', 'i7a', 'i6a', 'a', 'a4g', 'a3g', 'a2g', 'a1g', 'g']
hemisphere = 'bh'
task = 'allmain_tasks'

# ########################### PARAMETERS ################################

working_dir = os.path.dirname(os.path.abspath(__file__))
anovas_dir = os.path.join(working_dir , 'roi_analyses_rwls_hrf128',
                          encoding_mask_type, anova_type)
output_dir = os.path.join(working_dir, 'control_contrasts')

# ############################## RUN ####################################

if __name__ == '__main__':

    reversed_tags = tags[::-1]

    posthoc_paths = [
        os.path.join(
            anovas_dir,
            rtag + '_' + hemisphere + '_' + '2w-' + task + '_posthoc.tsv')
        for rtag in reversed_tags]

    df_list = [pd.read_csv(posthoc_path, sep='\t')
               for posthoc_path in posthoc_paths]

    p_corr_vals = [round(df.loc[df['ROI'] == 'dstr', 'p-corr'].iloc[-1], 3)
                   for df in df_list]

    p_uncorr_vals = [round(df.loc[df['ROI'] == 'dstr', 'p-unc'].iloc[-1], 3)
                     for df in df_list]

    # Plot
    ws = np.round(np.arange(0, 1.1, 0.1), 1)  # Ensures numeric values
    pcorr_path = os.path.join(output_dir, 'pcorr_plot.png')
    plot_pvalues(ws, p_corr_vals, output_path=pcorr_path)
    puncorr_path = os.path.join(output_dir, 'punc_plot.png')
    plot_pvalues(ws, p_corr_vals, output_path=puncorr_path)
