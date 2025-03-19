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

def plot_pvalues(weights, pvals, ylabel, ylim_min, ylim_max, y_step,
                 output_path='./pval_plot.png'):
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
        p-values corresponding to the weighting factors.
    output_path : str, optional (default='./pcorr_plot_final.png')
        File path where the plot will be saved.

    Returns:
    --------
    None
    """
    plt.figure(figsize=(8, 5))
    # Use clip_on=False to ensure markers are fully drawn even if...
    # ... slightly outside the axes.
    plt.scatter(weights, pvals, color='k', s=80, edgecolors='black',
                linewidth=1.5, clip_on=False)
    plt.plot(weights, pvals, linestyle='-', alpha=.7, linewidth=3., color='k')

    plt.xlabel(r'$w_{i}$', labelpad=12, fontsize=14)
    plt.ylabel(ylabel, labelpad=12, fontsize=14)

    ax = plt.gca()
    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Increase space between axes and grid by moving spines outward
    ax.spines["bottom"].set_position(('outward', 10))
    ax.spines["left"].set_position(('outward', 10))
    ax.spines["bottom"].set_bounds(0, 1)

    # Set axis limits and ticks
    plt.xlim(0., 1.)
    plt.ylim(ylim_min, ylim_max)
    plt.xticks(np.linspace(0, 1., 11), fontsize=14)
    plt.yticks(np.linspace(ylim_min, ylim_max, y_step), fontsize=14)

    # Set y-axis labels to exactly three decimals
    ax.set_yticklabels([f'{tick:.3f}' for tick in ax.get_yticks()],
                       fontsize=14)

    # Add margin to avoid clipping markers at the edges
    plt.margins(x=.05)

    # Set grid
    plt.grid(True)

    # Adjust layout to prevent the x-label from being cut off.
    plt.subplots_adjust(bottom=.2)

    # Remove spaces in the borders
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=300)
    # plt.show()


# ############################# INPUTS ##################################

# ### Mask Type
encoding_mask_type = 'all' # all, auditory, visual

# ### Tasks Modality Included
# anova_type = '2way-anova_cat2rois'        # all
anova_type = '2way-anova_cat2rois_auditory' # auditory
# anova_type = '2way-anova_cat2rois_visual'   # visual

tags = ['i', 'i9a', 'i8a', 'i7a', 'i6a', 'a', 'a4g', 'a3g', 'a2g', 'a1g', 'g']
hemisphere = 'bh'
task = 'allmain_tasks'

step = .005

# ########################### PARAMETERS ################################

working_dir = os.path.dirname(os.path.abspath(__file__))
anovas_dir = os.path.join(working_dir , 'roi_analyses_rwls_hrf128',
                          encoding_mask_type, anova_type)
output_dir = os.path.join(working_dir, 'results/pvalues_stability_plots')

# ############################## RUN ####################################

if __name__ == '__main__':

    # Create output_dir if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    reversed_tags = tags[::-1]

    posthoc_paths = [
        os.path.join(
            anovas_dir,
            rtag + '_' + hemisphere + '_' + '2w-' + task + '_posthoc.tsv')
        for rtag in reversed_tags]

    df_list = [pd.read_csv(posthoc_path, sep='\t')
               for posthoc_path in posthoc_paths]

    p_corr_vals = np.array(
        [
            round(df.loc[df['ROI'] == 'dstr', 'p-corr'].iloc[-1], 3)
            for df in df_list
        ]
    )

    p_uncorr_vals = np.array(
        [
            round(df.loc[df['ROI'] == 'dstr', 'p-unc'].iloc[-1], 3)
            for df in df_list
        ]
    )

    # #### Plot ####
    ws = np.round(np.arange(0, 1.1, .1), 1)  # Ensures numeric values

    if anova_type == '2way-anova_cat2rois':
        pcorr_path = os.path.join(
            output_dir,
            f'pcorrected_mask-{encoding_mask_type}_task-all_plot.png'
        )
        puncorr_path = os.path.join(
            output_dir,
            f'puncorrected_mask-{encoding_mask_type}_task-all_plot.png'
        )
    elif anova_type == '2way-anova_cat2rois_auditory':
        pcorr_path = os.path.join(
            output_dir,
            f'pcorrected_mask-{encoding_mask_type}_task-auditory_plot.png'
        )
        puncorr_path = os.path.join(
            output_dir,
            f'puncorrected_mask-{encoding_mask_type}_task-auditory_plot.png'
        )
    else:
        assert anova_type == '2way-anova_cat2rois_visual'
        pcorr_path = os.path.join(
            output_dir,
            f'pcorrected_mask-{encoding_mask_type}_task-visual_plot.png'
        )
        puncorr_path = os.path.join(
            output_dir,
            f'puncorrected_mask-{encoding_mask_type}_task-visual_plot.png'
        )

    # Determine the nearest lower multiple of 0.005 for y_min
    ycorr_min = np.round(
        np.floor((np.amin(p_corr_vals) - 1e-8) / step) * step, 3)
    yunc_min = np.round(
        np.floor((np.amin(p_uncorr_vals) - 1e-8) / step) * step, 3)

    # Determine the nearest higher multiple of 0.005 for y_max
    ycorr_max = np.round(
        np.ceil((np.amax(p_corr_vals) + 1e-8) / step) * step, 3)
    yunc_max = np.round(
        np.ceil((np.amax(p_uncorr_vals) + 1e-8) / step) * step, 3)

    # Steps
    ycorr_step = int((ycorr_max - ycorr_min) / step)
    yunc_step = int((yunc_max - yunc_min) / step)

    pcorr_label = r'$p_{\mathrm{FWE}}(w_{i})$'
    plot_pvalues(ws, p_corr_vals, pcorr_label, ycorr_min, ycorr_max,
                 4, output_path=pcorr_path)

    puncorr_label = r'$p_{\mathrm{uncorr}}(w_{i})$'
    plot_pvalues(ws, p_uncorr_vals, puncorr_label, yunc_min, yunc_max,
                 4, output_path=puncorr_path)
