"""
This script computes the mean activity in ROIS
for a given set of contrasts of the Music-SDTB Project.

Author: Ana Luisa Pinho

Created: September 2023
Last update: September 2023

Compatibility: Python 3.10.10

"""

import os
import numpy as np

from scipy import ndimage, stats

from nilearn.plotting import plot_glass_brain
from nilearn.image import load_img, new_img_like, resample_to_img
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker

from matplotlib import pyplot as plt


# ############################ FUNCTIONS ################################

def plot_mask(lh, rh, mask_description, output_file, cb=True,
              color_map='viridis'):

    fig = plt.figure(figsize=(6, 2.75))
    # left, bottom, width, height
    axes = plt.axes([0., 0., 1., 1.])

    display = plot_glass_brain(None, display_mode='lzr', black_bg=False,
                               alpha=1., axes=axes, title=mask_description,
                               vmin=0., vmax=1., symmetric_cbar=False)

    cmap = plt.get_cmap(color_map)
    display.add_overlay(lh, cmap=cmap)
    display.add_overlay(rh, cmap=cmap, colorbar=cb)

    fig.savefig(output_file, dpi=600)


def binarize(mask_path, threshold = .8):

    # Load
    mask = load_img(mask_path)

    # Threshold
    thresholded_mask_val = mask.get_fdata()
    thresholded_mask_val[thresholded_mask_val < threshold] = 0

    # Binarization
    bin_mask_val = (thresholded_mask_val != 0)

    # Dilation
    dil_bin_mask_val = ndimage.binary_dilation(bin_mask_val)
    dil_bin_mask_val = dil_bin_mask_val.astype(int)
    dil_bin_mask = new_img_like(mask, dil_bin_mask_val)

    return dil_bin_mask


# ############################# INPUTS ##################################

working_dir = os.path.dirname(os.path.abspath(__file__))
atlases_dir = os.path.join(working_dir, 'atlases')
data_dir = '/home/analu/diedrichsen_data/data/Cerebellum/music-sdtb/derivatives'
mask_gm = os.path.join(data_dir, 'group/anat/group_mask_gray.nii')

# SUBJECTS = [3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21,
#             22, 23, 26, 28, 29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 
#             44, 45, 46, 47]

SUBJECTS = [3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 23, 28, 29, 32,
            34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

TASKS = ['prod', 'percep', 'ntfd', 'allmain_tasks']
contrasts = {1: 'Encoding',
             2: 'Auditory Encoding',
             3: 'Visual Encoding',
             4: 'Auditory vs. Visual Encoding',
             5: 'Visual vs. Auditory Encoding',
             6: 'Beat vs. Interval',
             7: 'Auditory Beat vs. Auditory Interval',
             8: 'Visual Beat vs. Visual Interval',
             9: 'Interval vs. Beat',
             10: 'Auditory Interval vs. Auditory Beat',
             11: 'Visual Interval vs. Visual Beat',
             12: 'Decision'}

subjects_dir = [os.path.join(data_dir, 'sub-%02d') % sbj for sbj in SUBJECTS]
estimates_dir = [os.path.join(subject_dir, 'estimates')
                 for subject_dir in subjects_dir]

# ATAG
atag = os.path.join(atlases_dir, 'atag')

atag_masks = os.path.join(atag, 'Final_Neuroimage_2014_ATAG_prop_masks')
atag_linear = os.path.join(atag_masks, 'Linear')
atag_linear_norm = os.path.join(atag_linear, 'normalized')
str_atag_lh_ln = os.path.join(
    atag_linear_norm, 'Linear_MP2RAGE_STR_interrater_prop_L_normalized.nii.gz')
str_atag_rh_ln = os.path.join(
    atag_linear_norm, 'Linear_MP2RAGE_STR_interrater_prop_R_normalized.nii.gz')

atag_plots = os.path.join(atag, 'masks_plots')
str_atag_ln_plot = os.path.join(atag_plots, 'striatum_atag_ln.png')
str_atag_ln_resampled_bin_plot = os.path.join(
    atag_plots, 'striatum_atag_ln_resampled_bin.png')
str_atag_lh_ln_roi = os.path.join(atag_plots, 'striatum_atag_lh_ln_roi.png')




# ############################## RUN ####################################
  
if __name__ == '__main__':

    # Plot ATAG mask for striatum
    # plot_mask(str_atag_lh_ln, str_atag_rh_ln,
    #           'Striatum: ATAG Linear normalized',
    #           str_atag_ln_plot)

    # Resample masks
    resampled_str_atag_lh_ln = resample_to_img(str_atag_lh_ln, mask_gm)
    resampled_str_atag_rh_ln = resample_to_img(str_atag_rh_ln, mask_gm)

    # Binarize masks
    str_atag_lh_ln_bin = binarize(resampled_str_atag_lh_ln)
    str_atag_rh_ln_bin = binarize(resampled_str_atag_rh_ln)
    # Plot
    # plot_mask(str_atag_lh_ln_bin, str_atag_rh_ln_bin,
    #           'Striatum: ATAG Linear normalized binarized',
    #           str_atag_ln_resampled_bin_plot,
    #           cb=False, color_map='viridis_r')

    # Extract data from ROI
    masker = NiftiLabelsMasker(labels_img=str_atag_lh_ln_bin, mask_img=mask_gm)
    masker.fit()

    # Load contrasts
    # for task in TASKS:

    # left, bottom, width, height
    # ax = plt.axes([0., 0., 1., 1.])
    contrasts_mean = []
    pvalues = []
    for key in contrasts.keys():
        contrast_fname = 'wcon_%04d_desc-sm8gmmasked.nii' % key
        print(contrast_fname)
        masked_con = [os.path.join(estimate_dir, 'prod',
                                   'masked_derivatives_rwls',
                                   contrast_fname)
                      for estimate_dir in estimates_dir]
        print(np.array(masked_con))

        # Extract mean average of contrasts effect-size in ROI...
        # ... for every participant
        mask_data = [masker.transform(mcon)[0][0] for mcon in masked_con]

        # Compute mean
        mean_roi = np.mean(mask_data)

        # Compute stat
        tstat, pval = stats.ttest_1samp(mask_data, popmean=0.,
                                        alternative='greater')

        contrasts_mean.append(mean_roi)
        pvalues.append(pval)

    fig = plt.figure(figsize=(12., 6))
    cnames = list(contrasts.values())
    y_pos = np.arange(len(cnames))
    ax = plt.axes([.35, .1, .6, .85])

    # * For "star" text_format: `[[1e-4, "****"], [1e-3, "***"],
    #                         [1e-2, "**"], [0.05, "*"],
    #                         [1, "ns"]]`.

    pval_labels = []
    for pval in pvalues:
        if pval <= .0001:
            pval_labels.append('****')
        elif pval > .0001 and pval <= .001:
            pval_labels.append('***')
        elif pval > .001 and pval <= .01:
            pval_labels.append('**')
        elif pval > .01 and pval <= .05:
            pval_labels.append('*')
        else:
            pval_labels.append('ns')

    rects = ax.barh(y_pos, contrasts_mean, align='center')
    ax.bar_label(rects, labels=pval_labels, padding=3)
    ax.set_yticks(y_pos, labels=cnames, fontsize=16)
    plt.xticks(fontsize=16)
    # Hide the right and top spines
    ax.spines[['right', 'top']].set_visible(False)
    plt.title('Production', size=16, x=.5, fontweight='semibold')
    fig.savefig(str_atag_lh_ln_roi, dpi=300)
