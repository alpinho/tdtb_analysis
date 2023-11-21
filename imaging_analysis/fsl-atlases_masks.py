"""
This script creates masks from Harvard-Oxford subcortical atlas.

Author: Ana Luisa Pinho

Created: October 2023
Last update: November 2023

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
    thresholded_mask_val[thresholded_mask_val < threshold] = 0.

    # Binarization
    bin_mask_val = (thresholded_mask_val != 0.)

    # Create bin mask
    bin_mask = new_img_like(mask, bin_mask_val)

    return bin_mask


# ############################# INPUTS ##################################

working_dir = os.path.dirname(os.path.abspath(__file__))
atlases_dir = os.path.join(working_dir, 'atlases')
fsl_dir = os.path.join(atlases_dir, 'fsl_atlases')

hos_putamen_lh_probmap = os.path.join(
    fsl_dir, 'harvardoxford-subcortical_prob_left_putamen.nii.gz')
hos_putamen_rh_probmap = os.path.join(
    fsl_dir, 'harvardoxford-subcortical_prob_right_putamen.nii.gz')
hos_putamen_lh_maskpath = os.path.join(
    fsl_dir, 'hos_putamen_lh_mask.nii.gz')
hos_putamen_rh_maskpath = os.path.join(
    fsl_dir, 'hos_putamen_rh_mask.nii.gz')

mniflirt_cereb6_lh_probmap = os.path.join(
    fsl_dir, 'cerebellum_mniflirt_prob_leftVI.nii.gz')
mniflirt_cereb6_rh_probmap = os.path.join(
    fsl_dir, 'cerebellum_mniflirt_prob_rightVI.nii.gz')
mniflirt_cereb6_lh_maskpath = os.path.join(
    fsl_dir, 'mniflirt_cereb6_lh_mask.nii.gz')
mniflirt_cereb6_rh_maskpath = os.path.join(
    fsl_dir, 'mniflirt_cereb6_rh_mask.nii.gz')

mniflirt_crus1_lh_probmap = os.path.join(
    fsl_dir, 'cerebellum_mniflirt_prob_leftcrusI.nii.gz')
mniflirt_crus1_rh_probmap = os.path.join(
    fsl_dir, 'cerebellum_mniflirt_prob_rightcrusI.nii.gz')
mniflirt_crus1_lh_maskpath = os.path.join(
    fsl_dir, 'mniflirt_crus1_lh_mask.nii.gz')
mniflirt_crus1_rh_maskpath = os.path.join(
    fsl_dir, 'mniflirt_crus1_rh_mask.nii.gz')

mniflirt_cereb7b_lh_probmap = os.path.join(
    fsl_dir, 'cerebellum_mniflirt_prob_leftVIIb.nii.gz')
mniflirt_cereb7b_rh_probmap = os.path.join(
    fsl_dir, 'cerebellum_mniflirt_prob_rightVIIb.nii.gz')
mniflirt_cereb7b_lh_maskpath = os.path.join(
    fsl_dir, 'mniflirt_cereb7b_lh_mask.nii.gz')
mniflirt_cereb7b_rh_maskpath = os.path.join(
    fsl_dir, 'mniflirt_cereb7b_rh_mask.nii.gz')

mniflirt_cereb8a_lh_probmap = os.path.join(
    fsl_dir, 'cerebellum_mniflirt_prob_leftVIIIa.nii.gz')
mniflirt_cereb8a_rh_probmap = os.path.join(
    fsl_dir, 'cerebellum_mniflirt_prob_rightVIIIa.nii.gz')
mniflirt_cereb8a_lh_maskpath = os.path.join(
    fsl_dir, 'mniflirt_cereb8a_lh_mask.nii.gz')
mniflirt_cereb8a_rh_maskpath = os.path.join(
    fsl_dir, 'mniflirt_cereb8a_rh_mask.nii.gz')

mniflirt_cereb7b8a_lh_maskpath = os.path.join(
    fsl_dir, 'mniflirt_cereb7b8a_lh_mask.nii.gz')
mniflirt_cereb7b8a_rh_maskpath = os.path.join(
    fsl_dir, 'mniflirt_cereb7b8a_rh_mask.nii.gz')

# ############################## RUN ####################################

if __name__ == '__main__':

    # # ######################## PUTAMEN ##################################
    hos_putamen_lh_bin = binarize_bigger(hos_putamen_lh_probmap, 50.)
    hos_putamen_rh_bin = binarize_bigger(hos_putamen_rh_probmap, 50.)

    # Save maks
    hos_putamen_lh_bin.to_filename(hos_putamen_lh_maskpath)
    hos_putamen_rh_bin.to_filename(hos_putamen_rh_maskpath)

    # # ##################### CEREBELLUM VI ###############################
    mniflirt_cereb6_lh_bin = binarize_bigger(mniflirt_cereb6_lh_probmap, 50.)
    mniflirt_cereb6_rh_bin = binarize_bigger(mniflirt_cereb6_rh_probmap, 50.)

    # Save maks
    mniflirt_cereb6_lh_bin.to_filename(mniflirt_cereb6_lh_maskpath)
    mniflirt_cereb6_rh_bin.to_filename(mniflirt_cereb6_rh_maskpath)

    # # ##################### CEREBELLUM CRUS I ###########################
    mniflirt_crus1_lh_bin = binarize_bigger(mniflirt_crus1_lh_probmap, 50.)
    mniflirt_crus1_rh_bin = binarize_bigger(mniflirt_crus1_rh_probmap, 50.)

    # Save maks
    mniflirt_crus1_lh_bin.to_filename(mniflirt_crus1_lh_maskpath)
    mniflirt_crus1_rh_bin.to_filename(mniflirt_crus1_rh_maskpath)

    # ###################### CEREBELLUM VIIb ##############################
    mniflirt_cereb7b_lh_bin = binarize_bigger(mniflirt_cereb7b_lh_probmap, 45.)
    mniflirt_cereb7b_rh_bin = binarize_bigger(mniflirt_cereb7b_rh_probmap, 45.)

    # Save maks
    mniflirt_cereb7b_lh_bin.to_filename(mniflirt_cereb7b_lh_maskpath)
    mniflirt_cereb7b_rh_bin.to_filename(mniflirt_cereb7b_rh_maskpath)

    # ##################### CEREBELLUM VIIIa ##############################
    mniflirt_cereb8a_lh_bin = binarize_bigger(mniflirt_cereb8a_lh_probmap, 45.)
    mniflirt_cereb8a_rh_bin = binarize_bigger(mniflirt_cereb8a_rh_probmap, 45.)

    # Save maks
    mniflirt_cereb8a_lh_bin.to_filename(mniflirt_cereb8a_lh_maskpath)
    mniflirt_cereb8a_rh_bin.to_filename(mniflirt_cereb8a_rh_maskpath)

    # ############## COMBINE CEREBELLUM VIIb W/ VIIIa #####################
    mniflirt_cereb7b_lh_bin_val = mniflirt_cereb7b_lh_bin.get_fdata()
    mniflirt_cereb8a_lh_bin_val = mniflirt_cereb8a_lh_bin.get_fdata()
    combined_mask_lh_val = \
        mniflirt_cereb7b_lh_bin_val + mniflirt_cereb8a_lh_bin_val
    combined_mask_lh_val[combined_mask_lh_val == 2.] = 1.
    combined_mask_lh = new_img_like(
        mniflirt_cereb7b_lh_bin, combined_mask_lh_val)

    mniflirt_cereb7b_rh_bin_val = mniflirt_cereb7b_rh_bin.get_fdata()
    mniflirt_cereb8a_rh_bin_val = mniflirt_cereb8a_rh_bin.get_fdata()
    combined_mask_rh_val = \
        mniflirt_cereb7b_rh_bin_val + mniflirt_cereb8a_rh_bin_val
    combined_mask_rh_val[combined_mask_rh_val == 2.] = 1.
    combined_mask_rh = new_img_like(
        mniflirt_cereb7b_rh_bin, combined_mask_rh_val)

    # Save maks
    combined_mask_lh.to_filename(mniflirt_cereb7b8a_lh_maskpath)
    combined_mask_rh.to_filename(mniflirt_cereb7b8a_rh_maskpath)
