"""
This script creates masks of the Striatum from the ATAG Linear atlas.

Author: Ana Luisa Pinho

Created: November 2023
Last update: December 2023

Compatibility: Python 3.10.8

"""

import os
import numpy as np

from nilearn.image import load_img, new_img_like


# ############################ FUNCTIONS ################################


def binarize(mask_path, threshold):

    # Load
    mask = load_img(mask_path)

    # Threshold
    thresholded_mask_val = mask.get_fdata().astype(np.uint8)
    thresholded_mask_val[thresholded_mask_val < threshold] = 0

    # Binarization
    bin_mask_val = (thresholded_mask_val != 0)

    # Convert in Niimg file
    bin_mask = new_img_like(mask, bin_mask_val)

    return bin_mask


# ############################# INPUTS ##################################

working_dir = os.path.dirname(os.path.abspath(__file__))
atlases_dir = os.path.join(working_dir, 'atlases')
atag_dir = os.path.join(atlases_dir, 'atag_atlas')

atag_masks = os.path.join(atag_dir, 'Final_Neuroimage_2014_ATAG_prop_masks')
atag_linear = os.path.join(atag_masks, 'Linear')
atag_lnorm = os.path.join(atag_linear, 'normalized')

atag_lnorm_str_lh_probpath = os.path.join(
    atag_lnorm, 'Linear_MP2RAGE_STR_interrater_prop_L_normalized.nii.gz')
atag_lnorm_str_rh_probpath = os.path.join(
    atag_lnorm, 'Linear_MP2RAGE_STR_interrater_prop_R_normalized.nii.gz')

atag_lnorm_str_lh_maskpath = os.path.join(
    atag_dir, 'atag-lnorm_str_lh_mask.nii.gz')
atag_lnorm_str_rh_maskpath = os.path.join(
    atag_dir, 'atag-lnorm_str_rh_mask.nii.gz')

# ############################## RUN ####################################
  
if __name__ == '__main__':

    # Binarize masks
    atag_lnorm_lh_bin = binarize(atag_lnorm_str_lh_probpath, .5)
    atag_lnorm_rh_bin = binarize(atag_lnorm_str_rh_probpath, .5)

    # Save masks
    atag_lnorm_lh_bin.to_filename(atag_lnorm_str_lh_maskpath)
    atag_lnorm_rh_bin.to_filename(atag_lnorm_str_rh_maskpath)
