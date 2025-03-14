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

def plot_pvalues(weights, pvals, output_path='./pcorr_plot_final.png'):
    """
    Plots corrected p-values as a function of the weighting factor for
    ROI individualization, ensuring the markers at the boundaries are
    fully visible, increasing the spacing between axes and grid, and
    adjusting the layout so the x-label is not cut off.

    Parameters:
    -----------
    weights : array-like
        Weighting factors for individualization (numeric values).
    pvals : array-like
        Corrected p-values corresponding to the weighting factors.
    output_path : str, optional (default='./pcorr_plot_final.png')
        File path where the plot will be saved.

    Returns:
    --------
    None
    """
    plt.figure(figsize=(8, 5))
    # Use clip_on=False to ensure markers are fully drawn even if...
    # ... slightly outside the axes.
    plt.scatter(weights, pvals, color='b', s=70, edgecolors='black',
                linewidth=1.2, clip_on=False)
    plt.plot(weights, pvals, linestyle='--', alpha=.7, linewidth=2.5)

    plt.xlabel(r'Weighting Factor for ROI Individualization ($w$)',
               labelpad=12, fontsize=14)
    plt.ylabel(r'Corrected p-value $p_{\mathrm{FWE}}(w)$', labelpad=12,
               fontsize=14)

    ax = plt.gca()
    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Increase space between axes and grid by moving spines outward
    ax.spines["bottom"].set_position(("outward", 10))
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_bounds(0, 1)

    # Set axis limits and ticks
    plt.xlim(0, 1)
    plt.ylim(0.025, 0.05)
    plt.xticks(np.linspace(0, 1, 11), fontsize=14)
    plt.yticks(np.linspace(0.025, 0.05, 6), fontsize=14)

    # Add margin to avoid clipping markers at the edges
    plt.margins(x=0.05)

    # Set grid
    plt.grid(True)

    # Adjust layout to prevent the x-label from being cut off.
    plt.subplots_adjust(bottom=0.2)

    # Remove spaces in the borders
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=300)
    # plt.show()


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
