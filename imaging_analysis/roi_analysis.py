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

from scipy import ndimage

from nilearn.plotting import plot_glass_brain
from nilearn.image import load_img, new_img_like

from matplotlib import pyplot as plt


# ############################ FUNCTIONS ################################

def plot_probmask(lh, rh, mask_description, output_file, cb=True,
                  color_map='viridis'):

    fig = plt.figure(figsize=(6, 2.5))
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

# ATAG
atag = os.path.join(atlases_dir, 'atag')

atag_masks = os.path.join(atag, 'Final_Neuroimage_2014_ATAG_prop_masks')
atag_linear = os.path.join(atag_masks, 'Linear')
atag_linear_norm = os.path.join(atag_linear, 'normalized')
str_lh_ln = os.path.join(
    atag_linear_norm, 'Linear_MP2RAGE_STR_interrater_prop_L_normalized.nii.gz')
str_rh_ln = os.path.join(
    atag_linear_norm, 'Linear_MP2RAGE_STR_interrater_prop_R_normalized.nii.gz')

atag_plots = os.path.join(atag, 'masks_plots')
striatum_atag_ln_plot = os.path.join(atag_plots, 'striatum_atag_ln.png')
striatum_atag_ln_bin_plot = os.path.join(atag_plots, 'striatum_atag_bin_ln.png')


# ############################## RUN ####################################

if __name__ == '__main__':

    # Plot ATAG mask for striatum
    plot_probmask(str_lh_ln, str_rh_ln,
                  'Striatum: ATAG Linear normalized',
                  striatum_atag_ln_plot)

    # Binarize mask
    str_atag_lh_ln_bin = binarize(str_lh_ln)
    str_atag_rh_ln_bin = binarize(str_rh_ln)
    plot_probmask(str_atag_lh_ln_bin, str_atag_rh_ln_bin,
                  'Striatum: ATAG Linear normalized binarized',
                  striatum_atag_ln_bin_plot, cb=False,
                  color_map='viridis_r')

