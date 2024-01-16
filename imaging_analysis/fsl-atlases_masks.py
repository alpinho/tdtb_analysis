"""
This script creates masks from Harvard-Oxford subcortical atlas.

Author: Ana Luisa Pinho

Created: October 2023
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
    thresholded_mask_val[thresholded_mask_val < threshold] = 0.

    # Binarization
    bin_mask_val = (thresholded_mask_val != 0.)

    # Create bin mask
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
fsl_dir = os.path.join(atlases_dir, 'fsl_atlases')

hos_putamen_lh_probmap = os.path.join(
    fsl_dir, 'harvardoxford-subcortical_prob_left_putamen.nii.gz')
hos_putamen_rh_probmap = os.path.join(
    fsl_dir, 'harvardoxford-subcortical_prob_right_putamen.nii.gz')
hos_putamen_lh_maskpath = os.path.join(
    fsl_dir, 'hos_putamen_lh_mask.nii.gz')
hos_putamen_rh_maskpath = os.path.join(
    fsl_dir, 'hos_putamen_rh_mask.nii.gz')

hos_caudate_lh_probmap = os.path.join(
    fsl_dir, 'harvardoxford-subcortical_prob_left_caudate.nii.gz')
hos_caudate_rh_probmap = os.path.join(
    fsl_dir, 'harvardoxford-subcortical_prob_right_caudate.nii.gz')
hos_caudate_lh_maskpath = os.path.join(
    fsl_dir, 'hos_caudate_lh_mask.nii.gz')
hos_caudate_rh_maskpath = os.path.join(
    fsl_dir, 'hos_caudate_rh_mask.nii.gz')

hos_dstriatum_lh_maskpath = os.path.join(
    fsl_dir, 'hos_dstr_lh_mask.nii.gz')
hos_dstriatum_rh_maskpath = os.path.join(
    fsl_dir, 'hos_dstr_rh_mask.nii.gz')

hos_dstriatum_bh_maskpath = os.path.join(
    fsl_dir, 'hos_dstr_bh_mask.nii.gz')

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

    # # ######################## PUTAMEN ################################
    hos_putamen_lh_bin = binarize(hos_putamen_lh_probmap, 50.)
    hos_putamen_rh_bin = binarize(hos_putamen_rh_probmap, 50.)

    # Save maks
    hos_putamen_lh_bin.to_filename(hos_putamen_lh_maskpath)
    hos_putamen_rh_bin.to_filename(hos_putamen_rh_maskpath)

    # # ######################## CAUDATE ###############################
    hos_caudate_lh_bin = binarize(hos_caudate_lh_probmap, 50.)
    hos_caudate_rh_bin = binarize(hos_caudate_rh_probmap, 50.)

    # Save maks
    hos_caudate_lh_bin.to_filename(hos_caudate_lh_maskpath)
    hos_caudate_rh_bin.to_filename(hos_caudate_rh_maskpath)

    # # ###################### DORSAL STRIATUM #########################
    combine_masks(hos_putamen_lh_maskpath, hos_caudate_lh_maskpath,
                  hos_dstriatum_lh_maskpath)

    combine_masks(hos_putamen_rh_maskpath, hos_caudate_rh_maskpath,
                  hos_dstriatum_rh_maskpath)

    combine_masks(hos_dstriatum_lh_maskpath, hos_dstriatum_rh_maskpath,
                  hos_dstriatum_bh_maskpath)

    # # ##################### CEREBELLUM VI ############################
    mniflirt_cereb6_lh_bin = binarize(mniflirt_cereb6_lh_probmap, 50.)
    mniflirt_cereb6_rh_bin = binarize(mniflirt_cereb6_rh_probmap, 50.)

    # Save maks
    mniflirt_cereb6_lh_bin.to_filename(mniflirt_cereb6_lh_maskpath)
    mniflirt_cereb6_rh_bin.to_filename(mniflirt_cereb6_rh_maskpath)

    # # ##################### CEREBELLUM CRUS I ########################
    mniflirt_crus1_lh_bin = binarize(mniflirt_crus1_lh_probmap, 50.)
    mniflirt_crus1_rh_bin = binarize(mniflirt_crus1_rh_probmap, 50.)

    # Save maks
    mniflirt_crus1_lh_bin.to_filename(mniflirt_crus1_lh_maskpath)
    mniflirt_crus1_rh_bin.to_filename(mniflirt_crus1_rh_maskpath)

    # ###################### CEREBELLUM VIIb ###########################
    mniflirt_cereb7b_lh_bin = binarize(mniflirt_cereb7b_lh_probmap, 50.)
    mniflirt_cereb7b_rh_bin = binarize(mniflirt_cereb7b_rh_probmap, 50.)

    # Save maks
    mniflirt_cereb7b_lh_bin.to_filename(mniflirt_cereb7b_lh_maskpath)
    mniflirt_cereb7b_rh_bin.to_filename(mniflirt_cereb7b_rh_maskpath)

    # ##################### CEREBELLUM VIIIa ###########################
    mniflirt_cereb8a_lh_bin = binarize(mniflirt_cereb8a_lh_probmap, 50.)
    mniflirt_cereb8a_rh_bin = binarize(mniflirt_cereb8a_rh_probmap, 50.)

    # Save maks
    mniflirt_cereb8a_lh_bin.to_filename(mniflirt_cereb8a_lh_maskpath)
    mniflirt_cereb8a_rh_bin.to_filename(mniflirt_cereb8a_rh_maskpath)

    # ############## COMBINE CEREBELLUM VIIb W/ VIIIa ##################
    combine_masks(mniflirt_cereb7b_lh_maskpath, mniflirt_cereb8a_lh_maskpath,
                  mniflirt_cereb7b8a_lh_maskpath)

    combine_masks(mniflirt_cereb7b_rh_maskpath, mniflirt_cereb8a_rh_maskpath,
                  mniflirt_cereb7b8a_rh_maskpath)
