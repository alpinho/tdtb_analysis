"""
This script creates masks of the Striatum from the ATAG Linear atlas.

Author: Ana Luisa Pinho

Created: November 2023
Last update: November 2023

Compatibility: Python 3.10.8

"""

import os

from nilearn.image import load_img, new_img_like


# ############################ FUNCTIONS ################################


def binarize(mask_path, threshold):

    # Load
    mask = load_img(mask_path)

    # Threshold
    thresholded_mask_val = mask.get_fdata()
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
atag_linear_norm = os.path.join(atag_linear, 'normalized')

atag_linear_norm_striatum_lh = os.path.join(
    atag_linear_norm, 'Linear_MP2RAGE_STR_interrater_prop_L_normalized.nii.gz')
atag_linear_norm_striatum_rh = os.path.join(
    atag_linear_norm, 'Linear_MP2RAGE_STR_interrater_prop_R_normalized.nii.gz')

atag_linear_norm_striatum_lh_maskpath = os.path.join(
    atag_dir, 'atag_linear_striatum_lh_mask.nii.gz')
atag_linear_norm_striatum_rh_maskpath = os.path.join(
    atag_dir, 'atag_linear_striatum_rh_mask.nii.gz')

# ############################## RUN ####################################
  
if __name__ == '__main__':

    # Binarize masks
    str_atag_lh_ln_bin = binarize(atag_linear_norm_striatum_lh, .5)
    str_atag_rh_ln_bin = binarize(atag_linear_norm_striatum_rh, .5)

    # Save masks
    str_atag_lh_ln_bin.to_filename(atag_linear_norm_striatum_lh_maskpath)
    str_atag_rh_ln_bin.to_filename(atag_linear_norm_striatum_rh_maskpath)
