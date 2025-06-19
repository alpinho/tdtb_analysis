"""
This script creates masks from Harvard-Oxford subcortical atlas.

Author: Ana Luisa Pinho

Created: October 2023
Last update: June 2025

Compatibility: Python 3.10.14

"""

import os
import numpy as np

from nilearn.image import load_img, new_img_like


# ############################ FUNCTIONS ###############################

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


# ############################# INPUTS #################################

working_dir = os.path.dirname(os.path.abspath(__file__))
atlases_dir = os.path.join(working_dir, 'atlases')
fsl_dir = os.path.join(atlases_dir, 'fsl_atlases')

dstr_dir = os.path.join(fsl_dir, 'dstr')
cerebellum_dir = os.path.join(fsl_dir, 'cerebellum')
auditory_cortex_dir = os.path.join(fsl_dir, 'heschl_gyrus')
visual_cortex_dir = os.path.join(fsl_dir, 'occipital_lobe')

hos_putamen_lh_probmap = os.path.join(
    dstr_dir, 'harvardoxford-subcortical_prob_left_putamen.nii.gz')
hos_putamen_rh_probmap = os.path.join(
    dstr_dir, 'harvardoxford-subcortical_prob_right_putamen.nii.gz')
hos_putamen_lh_maskpath = os.path.join(
    dstr_dir, 'hos_putamen_lh_mask.nii.gz')
hos_putamen_rh_maskpath = os.path.join(
    dstr_dir, 'hos_putamen_rh_mask.nii.gz')

hos_caudate_lh_probmap = os.path.join(
    dstr_dir, 'harvardoxford-subcortical_prob_left_caudate.nii.gz')
hos_caudate_rh_probmap = os.path.join(
    dstr_dir, 'harvardoxford-subcortical_prob_right_caudate.nii.gz')
hos_caudate_lh_maskpath = os.path.join(
    dstr_dir, 'hos_caudate_lh_mask.nii.gz')
hos_caudate_rh_maskpath = os.path.join(
    dstr_dir, 'hos_caudate_rh_mask.nii.gz')

hos_dstriatum_lh_maskpath = os.path.join(
    dstr_dir, 'hos_dstr_lh_mask.nii.gz')
hos_dstriatum_rh_maskpath = os.path.join(
    dstr_dir, 'hos_dstr_rh_mask.nii.gz')

hos_dstriatum_bh_maskpath = os.path.join(
    dstr_dir, 'hos_dstr_bh_mask.nii.gz')

mniflirt_cereb6_lh_probmap = os.path.join(
    cerebellum_dir, 'cerebellum_mniflirt_prob_leftVI.nii.gz')
mniflirt_cereb6_rh_probmap = os.path.join(
    cerebellum_dir, 'cerebellum_mniflirt_prob_rightVI.nii.gz')
mniflirt_cereb6_lh_maskpath = os.path.join(
    cerebellum_dir, 'mniflirt_cereb6_lh_mask.nii.gz')
mniflirt_cereb6_rh_maskpath = os.path.join(
    cerebellum_dir, 'mniflirt_cereb6_rh_mask.nii.gz')

mniflirt_crus1_lh_probmap = os.path.join(
    cerebellum_dir, 'cerebellum_mniflirt_prob_leftcrusI.nii.gz')
mniflirt_crus1_rh_probmap = os.path.join(
    cerebellum_dir, 'cerebellum_mniflirt_prob_rightcrusI.nii.gz')
mniflirt_crus1_lh_maskpath = os.path.join(
    cerebellum_dir, 'mniflirt_crus1_lh_mask.nii.gz')
mniflirt_crus1_rh_maskpath = os.path.join(
    cerebellum_dir, 'mniflirt_crus1_rh_mask.nii.gz')

mniflirt_cereb7b_lh_probmap = os.path.join(
    cerebellum_dir, 'cerebellum_mniflirt_prob_leftVIIb.nii.gz')
mniflirt_cereb7b_rh_probmap = os.path.join(
    cerebellum_dir, 'cerebellum_mniflirt_prob_rightVIIb.nii.gz')
mniflirt_cereb7b_lh_maskpath = os.path.join(
    cerebellum_dir, 'mniflirt_cereb7b_lh_mask.nii.gz')
mniflirt_cereb7b_rh_maskpath = os.path.join(
    cerebellum_dir, 'mniflirt_cereb7b_rh_mask.nii.gz')

mniflirt_cereb8a_lh_probmap = os.path.join(
    cerebellum_dir, 'cerebellum_mniflirt_prob_leftVIIIa.nii.gz')
mniflirt_cereb8a_rh_probmap = os.path.join(
    cerebellum_dir, 'cerebellum_mniflirt_prob_rightVIIIa.nii.gz')
mniflirt_cereb8a_lh_maskpath = os.path.join(
    cerebellum_dir, 'mniflirt_cereb8a_lh_mask.nii.gz')
mniflirt_cereb8a_rh_maskpath = os.path.join(
    cerebellum_dir, 'mniflirt_cereb8a_rh_mask.nii.gz')

mniflirt_cereb7b8a_lh_maskpath = os.path.join(
    cerebellum_dir, 'mniflirt_cereb7b8a_lh_mask.nii.gz')
mniflirt_cereb7b8a_rh_maskpath = os.path.join(
    cerebellum_dir, 'mniflirt_cereb7b8a_rh_mask.nii.gz')

heschl_lh_probmap = os.path.join(
    auditory_cortex_dir,
    'harvardoxford-cortical-lateralized_prob_Left_Heschl_Gyrus.nii.gz')
heschl_lh_maskpath = os.path.join(
    auditory_cortex_dir, 'hos_heschl_lh_mask.nii.gz')

heschl_rh_probmap = os.path.join(
    auditory_cortex_dir,
    'harvardoxford-cortical-lateralized_prob_Right_Heschl_Gyrus.nii.gz')
heschl_rh_maskpath = os.path.join(
    auditory_cortex_dir, 'hos_heschl_rh_mask.nii.gz')

occipital_pole_lh_probmap = os.path.join(
    visual_cortex_dir,
    'harvardoxford-cortical-lateralized_prob_Left_Occipital_Pole.nii.gz')
occipital_pole_lh_maskpath = os.path.join(
    visual_cortex_dir, 'hos_occipital-pole_lh_mask.nii.gz')

occipital_pole_rh_probmap = os.path.join(
    visual_cortex_dir,
    'harvardoxford-cortical-lateralized_prob_Right_Occipital_Pole.nii.gz')
occipital_pole_rh_maskpath = os.path.join(
    visual_cortex_dir, 'hos_occipital-pole_rh_mask.nii.gz')

occipital_locsup_lh_probmap = os.path.join(
    visual_cortex_dir,
    'harvardoxford-cortical-lateralized_prob_Left_Lateral_Occipital_Cortex_' +
    'superior_division.nii.gz')
occipital_locsup_lh_maskpath = os.path.join(
    visual_cortex_dir, 'hos_loc-sup_lh_mask.nii.gz')

occipital_locsup_rh_probmap = os.path.join(
    visual_cortex_dir,
    'harvardoxford-cortical-lateralized_prob_Right_Lateral_Occipital_Cortex_' +
    'superior_division.nii.gz')
occipital_locsup_rh_maskpath = os.path.join(
    visual_cortex_dir, 'hos_loc-sup_rh_mask.nii.gz')

occipital_locinf_lh_probmap = os.path.join(
    visual_cortex_dir,
    'harvardoxford-cortical-lateralized_prob_Left_Lateral_Occipital_Cortex_' +
    'inferior_division.nii.gz')
occipital_locinf_lh_maskpath = os.path.join(
    visual_cortex_dir, 'hos_loc-inf_lh_mask.nii.gz')

occipital_locinf_rh_probmap = os.path.join(
    visual_cortex_dir,
    'harvardoxford-cortical-lateralized_prob_Right_Lateral_Occipital_Cortex_' +
    'inferior_division.nii.gz')
occipital_locinf_rh_maskpath = os.path.join(
    visual_cortex_dir, 'hos_loc-inf_rh_mask.nii.gz')

occipital_fusiform_lh_probmap = os.path.join(
    visual_cortex_dir,
    'harvardoxford-cortical-lateralized_prob_Left_Occipital_' +
    'Fusiform_Gyrus.nii.gz')
occipital_fusiform_lh_maskpath = os.path.join(
    visual_cortex_dir, 'hos_fusiform_lh_mask.nii.gz')

occipital_fusiform_rh_probmap = os.path.join(
    visual_cortex_dir,
    'harvardoxford-cortical-lateralized_prob_Right_Occipital_' +
    'Fusiform_Gyrus.nii.gz')
occipital_fusiform_rh_maskpath = os.path.join(
    visual_cortex_dir, 'hos_fusiform_rh_mask.nii.gz')

occipital_loc_lh_maskpath = os.path.join(
    visual_cortex_dir, 'hos_loc_lh_mask.nii.gz')
occipital_loc_rh_maskpath = os.path.join(
    visual_cortex_dir, 'hos_loc_rh_mask.nii.gz')
occipital_locpole_lh_maskpath = os.path.join(
    visual_cortex_dir, 'hos_locpole_lh_mask.nii.gz')
occipital_locpole_rh_maskpath = os.path.join(
    visual_cortex_dir, 'hos_locpole_rh_mask.nii.gz')
occipital_lh_maskpath = os.path.join(
    visual_cortex_dir, 'hos_occipital_lh_mask.nii.gz')
occipital_rh_maskpath = os.path.join(
    visual_cortex_dir, 'hos_occipital_rh_mask.nii.gz')

# ############################## RUN ###################################

if __name__ == '__main__':

    # ########################## PUTAMEN ###############################
    hos_putamen_lh_bin = binarize(hos_putamen_lh_probmap, 50.)
    hos_putamen_rh_bin = binarize(hos_putamen_rh_probmap, 50.)

    # Save masks
    hos_putamen_lh_bin.to_filename(hos_putamen_lh_maskpath)
    hos_putamen_rh_bin.to_filename(hos_putamen_rh_maskpath)

    # ######################## CAUDATE #################################
    hos_caudate_lh_bin = binarize(hos_caudate_lh_probmap, 50.)
    hos_caudate_rh_bin = binarize(hos_caudate_rh_probmap, 50.)

    # Save masks
    hos_caudate_lh_bin.to_filename(hos_caudate_lh_maskpath)
    hos_caudate_rh_bin.to_filename(hos_caudate_rh_maskpath)

    # ******************** DORSAL STRIATUM *****************************
    combine_masks(hos_putamen_lh_maskpath, hos_caudate_lh_maskpath,
                  hos_dstriatum_lh_maskpath)

    combine_masks(hos_putamen_rh_maskpath, hos_caudate_rh_maskpath,
                  hos_dstriatum_rh_maskpath)

    combine_masks(hos_dstriatum_lh_maskpath, hos_dstriatum_rh_maskpath,
                  hos_dstriatum_bh_maskpath)

    # ##################### CEREBELLUM VI ##############################
    mniflirt_cereb6_lh_bin = binarize(mniflirt_cereb6_lh_probmap, 50.)
    mniflirt_cereb6_rh_bin = binarize(mniflirt_cereb6_rh_probmap, 50.)

    # Save masks
    mniflirt_cereb6_lh_bin.to_filename(mniflirt_cereb6_lh_maskpath)
    mniflirt_cereb6_rh_bin.to_filename(mniflirt_cereb6_rh_maskpath)

    # ##################### CEREBELLUM CRUS I ##########################
    mniflirt_crus1_lh_bin = binarize(mniflirt_crus1_lh_probmap, 50.)
    mniflirt_crus1_rh_bin = binarize(mniflirt_crus1_rh_probmap, 50.)

    # Save masks
    mniflirt_crus1_lh_bin.to_filename(mniflirt_crus1_lh_maskpath)
    mniflirt_crus1_rh_bin.to_filename(mniflirt_crus1_rh_maskpath)

    # ###################### CEREBELLUM VIIb ###########################
    mniflirt_cereb7b_lh_bin = binarize(mniflirt_cereb7b_lh_probmap, 50.)
    mniflirt_cereb7b_rh_bin = binarize(mniflirt_cereb7b_rh_probmap, 50.)

    # Save masks
    mniflirt_cereb7b_lh_bin.to_filename(mniflirt_cereb7b_lh_maskpath)
    mniflirt_cereb7b_rh_bin.to_filename(mniflirt_cereb7b_rh_maskpath)

    # ##################### CEREBELLUM VIIIa ###########################
    mniflirt_cereb8a_lh_bin = binarize(mniflirt_cereb8a_lh_probmap, 50.)
    mniflirt_cereb8a_rh_bin = binarize(mniflirt_cereb8a_rh_probmap, 50.)

    # Save masks
    mniflirt_cereb8a_lh_bin.to_filename(mniflirt_cereb8a_lh_maskpath)
    mniflirt_cereb8a_rh_bin.to_filename(mniflirt_cereb8a_rh_maskpath)

    # ************** COMBINE CEREBELLUM VIIb W/ VIIIa ******************
    combine_masks(mniflirt_cereb7b_lh_maskpath, mniflirt_cereb8a_lh_maskpath,
                  mniflirt_cereb7b8a_lh_maskpath)

    combine_masks(mniflirt_cereb7b_rh_maskpath, mniflirt_cereb8a_rh_maskpath,
                  mniflirt_cereb7b8a_rh_maskpath)

    # ####################### HESCHL'S GYRUS ###########################
    hos_heschl_lh_bin = binarize(heschl_lh_probmap, 30.)
    hos_heschl_rh_bin = binarize(heschl_rh_probmap, 30.)

    # Save mask
    hos_heschl_lh_bin.to_filename(heschl_lh_maskpath)
    hos_heschl_rh_bin.to_filename(heschl_rh_maskpath)

    # ####################### OCCIPITAL LOBE ###########################
    hos_occpole_lh_bin = binarize(occipital_pole_lh_probmap, 30.)
    hos_occpole_rh_bin = binarize(occipital_pole_rh_probmap, 30.)
    hos_locsup_lh_bin = binarize(occipital_locsup_lh_probmap, 30.)
    hos_locsup_rh_bin = binarize(occipital_locsup_rh_probmap, 30.)
    hos_locinf_lh_bin = binarize(occipital_locinf_lh_probmap, 30.)
    hos_locinf_rh_bin = binarize(occipital_locinf_rh_probmap, 30.)
    hos_fusiform_lh_bin = binarize(occipital_fusiform_lh_probmap, 30.)
    hos_fusiform_rh_bin = binarize(occipital_fusiform_rh_probmap, 30.)

    # Save mask
    hos_occpole_lh_bin.to_filename(occipital_pole_lh_maskpath)
    hos_occpole_rh_bin.to_filename(occipital_pole_rh_maskpath)
    hos_locsup_lh_bin.to_filename(occipital_locsup_lh_maskpath)
    hos_locsup_rh_bin.to_filename(occipital_locsup_rh_maskpath)
    hos_locinf_lh_bin.to_filename(occipital_locinf_lh_maskpath)
    hos_locinf_rh_bin.to_filename(occipital_locinf_rh_maskpath)
    hos_fusiform_lh_bin.to_filename(occipital_fusiform_lh_maskpath)
    hos_fusiform_rh_bin.to_filename(occipital_fusiform_rh_maskpath)

    # ******************** COMBINE OCCIPITAL MASKS **********************
    combine_masks(occipital_locsup_lh_maskpath, occipital_locinf_lh_maskpath,
                  occipital_loc_lh_maskpath)
    combine_masks(occipital_locsup_rh_maskpath, occipital_locinf_rh_maskpath,
                  occipital_loc_rh_maskpath)
    combine_masks(occipital_loc_lh_maskpath, occipital_pole_lh_maskpath,
                  occipital_locpole_lh_maskpath)
    combine_masks(occipital_loc_rh_maskpath, occipital_pole_rh_maskpath,
                  occipital_locpole_rh_maskpath)
    combine_masks(occipital_locpole_lh_maskpath, occipital_fusiform_lh_maskpath,
                  occipital_lh_maskpath)
    combine_masks(occipital_locpole_rh_maskpath, occipital_fusiform_rh_maskpath,
                  occipital_rh_maskpath)
