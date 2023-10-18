"""
This script creates masks from Harvard-Oxford subcortical atlas.

Author: Ana Luisa Pinho

Created: October 2023
Last update: October 2023

Compatibility: Python 3.10.10

"""

import os
import numpy as np

from nilearn.image import load_img, new_img_like


# ############################ FUNCTIONS ################################

def binarize_bigger(mask_path, threshold):

    # Load
    mask = load_img(mask_path)

    # Threshold
    thresholded_mask_val = mask.get_fdata()
    thresholded_mask_val[thresholded_mask_val < threshold] = 0

    # Binarization
    bin_mask_val = (thresholded_mask_val != 0)

    # Dilation
    # dil_bin_mask_val = ndimage.binary_dilation(bin_mask_val)
    # dil_bin_mask_val = dil_bin_mask_val.astype(int)

    bin_mask = new_img_like(mask, bin_mask_val)

    return bin_mask


# ############################# INPUTS ##################################

working_dir = os.path.dirname(os.path.abspath(__file__))
atlases_dir = os.path.join(working_dir, 'atlases')
hos_dir = os.path.join(atlases_dir, 'harvardoxford_subcortical')

hos_putamen_lh_probmap = os.path.join(
    hos_dir,
    'harvardoxford-subcortical_prob_left_putamen.nii.gz')
hos_putamen_rh_probmap = os.path.join(
    hos_dir,
    'harvardoxford-subcortical_prob_right_putamen.nii.gz')

hos_putamen_lh_maskpath = os.path.join(hos_dir,
                                       'hos_putamen_lh_mask.nii.gz')
hos_putamen_rh_maskpath = os.path.join(hos_dir,
                                       'hos_putamen_rh_mask.nii.gz')

# ############################## RUN ####################################

if __name__ == '__main__':

    hos_putamen_lh_bin = binarize_bigger(hos_putamen_lh_probmap, 50.)
    hos_putamen_rh_bin = binarize_bigger(hos_putamen_rh_probmap, 50.)

    # Save maks
    hos_putamen_lh_bin.to_filename(hos_putamen_lh_maskpath)
    hos_putamen_rh_bin.to_filename(hos_putamen_rh_maskpath)

