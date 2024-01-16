"""
This script creates masks of the Striatum from the ATAG Linear atlas.

Author: Ana Luisa Pinho

Created: November 2023
Last update: January 2024

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


def combine_masks(maskpath1, maskpath2, combined_maskpath):

    # Load
    mask1 = load_img(maskpath1)
    mask2 = load_img(maskpath2)

    # Get data
    mask1_val = mask1.get_fdata().astype(np.uint8)
    mask2_val = mask2.get_fdata().astype(np.uint8)

    # Merge masks in one single file
    combined_mask_val = mask1_val + mask2_val
    combined_mask_val[combined_mask_val == 2] = 1
    combined_mask = new_img_like(mask1, combined_mask_val)

    # Save file
    combined_mask.to_filename(combined_maskpath)


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

atag_lnorm_str_bh_maskpath = os.path.join(
    atag_dir, 'atag-lnorm_str_bh_mask.nii.gz')

# ############################## RUN ####################################

if __name__ == '__main__':

    # Binarize masks
    atag_lnorm_lh_bin = binarize(atag_lnorm_str_lh_probpath, 1.)
    atag_lnorm_rh_bin = binarize(atag_lnorm_str_rh_probpath, 1.)

    # Save masks
    atag_lnorm_lh_bin.to_filename(atag_lnorm_str_lh_maskpath)
    atag_lnorm_rh_bin.to_filename(atag_lnorm_str_rh_maskpath)

    # Mask of Dorsal Striatum
    combine_masks(atag_lnorm_str_lh_maskpath, atag_lnorm_str_rh_maskpath,
                  atag_lnorm_str_bh_maskpath)
