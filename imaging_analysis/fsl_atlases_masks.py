"""
This script creates masks from Harvard-Oxford subcortical atlas.

Author: Ana Luisa Pinho

Created: October 2023
Last update: May 2026

Compatibility: Python 3.10.14

"""

import os
import numpy as np

from nilearn.image import load_img, new_img_like


# ############################ FUNCTIONS ##############################

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


def combine_masks(*args):

    *inputs, combined_maskpath = args

    # Load reference image and accumulate masks;
    # inputs can be file paths or in-memory image objects
    ref_img = None
    combined = None

    for inp in inputs:
        img = load_img(inp) if isinstance(inp, str) else inp
        data = img.get_fdata().astype(np.uint8)
        if ref_img is None:
            ref_img = img
            combined = np.zeros(ref_img.shape, dtype=np.uint8)
        combined += data

    # Binarize
    combined[combined > 1] = 1

    # Save
    new_img_like(ref_img, combined).to_filename(combined_maskpath)


# ############################# PATHS #################################

working_dir = os.path.dirname(os.path.abspath(__file__))
atlases_dir = os.path.join(working_dir, 'atlases')
fsl_dir = os.path.join(atlases_dir, 'fsl_atlases')

dstr_dir = os.path.join(fsl_dir, 'dstr')
cerebellum_dir = os.path.join(fsl_dir, 'cerebellum')
auditory_cortex_dir = os.path.join(fsl_dir, 'auditory_cortex')
visual_cortex_dir = os.path.join(fsl_dir, 'visual_cortex')

hos_putamen_lh_probmap = os.path.join(
    dstr_dir, 'harvardoxford-subcortical_prob_left_putamen.nii.gz')
hos_putamen_rh_probmap = os.path.join(
    dstr_dir, 'harvardoxford-subcortical_prob_right_putamen.nii.gz')

hos_caudate_lh_probmap = os.path.join(
    dstr_dir, 'harvardoxford-subcortical_prob_left_caudate.nii.gz')
hos_caudate_rh_probmap = os.path.join(
    dstr_dir, 'harvardoxford-subcortical_prob_right_caudate.nii.gz')

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

mniflirt_cereb8a_lh_probmap = os.path.join(
    cerebellum_dir, 'cerebellum_mniflirt_prob_leftVIIIa.nii.gz')
mniflirt_cereb8a_rh_probmap = os.path.join(
    cerebellum_dir, 'cerebellum_mniflirt_prob_rightVIIIa.nii.gz')

mniflirt_cereb7b8a_lh_maskpath = os.path.join(
    cerebellum_dir, 'mniflirt_cereb7b8a_lh_mask.nii.gz')
mniflirt_cereb7b8a_rh_maskpath = os.path.join(
    cerebellum_dir, 'mniflirt_cereb7b8a_rh_mask.nii.gz')

heschl_lh_probmap = os.path.join(
    auditory_cortex_dir,
    'harvardoxford-cortical-lateralized_prob_Left_Heschl_Gyrus.nii.gz')
heschl_rh_probmap = os.path.join(
    auditory_cortex_dir,
    'harvardoxford-cortical-lateralized_prob_Right_Heschl_Gyrus.nii.gz')

planum_temporale_lh_probmap = os.path.join(
    auditory_cortex_dir,
    'harvardoxford-cortical-lateralized_prob_Left_Planum_Temporale.nii.gz')
planum_temporale_rh_probmap = os.path.join(
    auditory_cortex_dir,
    'harvardoxford-cortical-lateralized_prob_Right_Planum_Temporale.nii.gz')

superior_temporal_gyrus_posterior_division_lh_probmap = os.path.join(
    auditory_cortex_dir,
    'harvardoxford-cortical-lateralized_prob_Left_Superior_Temporal_Gyrus_' +
    'posterior_division.nii.gz')
superior_temporal_gyrus_posterior_division_rh_probmap = os.path.join(
    auditory_cortex_dir,
    'harvardoxford-cortical-lateralized_prob_Right_Superior_Temporal_Gyrus_' +
    'posterior_division.nii.gz')

auditory_cortex_lh_maskpath = os.path.join(
    auditory_cortex_dir, 'hos_auditory_cortex_lh_mask.nii.gz')
auditory_cortex_rh_maskpath = os.path.join(
    auditory_cortex_dir, 'hos_auditory_cortex_rh_mask.nii.gz')
auditory_cortex_bh_maskpath = os.path.join(
    auditory_cortex_dir, 'hos_auditory_cortex_bh_mask.nii.gz')

occipital_pole_lh_probmap = os.path.join(
    visual_cortex_dir,
    'harvardoxford-cortical-lateralized_prob_Left_Occipital_Pole.nii.gz')
occipital_pole_rh_probmap = os.path.join(
    visual_cortex_dir,
    'harvardoxford-cortical-lateralized_prob_Right_Occipital_Pole.nii.gz')

occipital_locsup_lh_probmap = os.path.join(
    visual_cortex_dir,
    'harvardoxford-cortical-lateralized_prob_Left_Lateral_Occipital_Cortex_' +
    'superior_division.nii.gz')
occipital_locsup_rh_probmap = os.path.join(
    visual_cortex_dir,
    'harvardoxford-cortical-lateralized_prob_Right_Lateral_Occipital_Cortex_' +
    'superior_division.nii.gz')

occipital_locinf_lh_probmap = os.path.join(
    visual_cortex_dir,
    'harvardoxford-cortical-lateralized_prob_Left_Lateral_Occipital_Cortex_' +
    'inferior_division.nii.gz')
occipital_locinf_rh_probmap = os.path.join(
    visual_cortex_dir,
    'harvardoxford-cortical-lateralized_prob_Right_Lateral_Occipital_Cortex_' +
    'inferior_division.nii.gz')

occipital_fusiform_lh_probmap = os.path.join(
    visual_cortex_dir,
    'harvardoxford-cortical-lateralized_prob_Left_Occipital_' +
    'Fusiform_Gyrus.nii.gz')
occipital_fusiform_rh_probmap = os.path.join(
    visual_cortex_dir,
    'harvardoxford-cortical-lateralized_prob_Right_Occipital_' +
    'Fusiform_Gyrus.nii.gz')

visual_cortex_lh_maskpath = os.path.join(
    visual_cortex_dir, 'hos_visual_cortex_lh_mask.nii.gz')
visual_cortex_rh_maskpath = os.path.join(
    visual_cortex_dir, 'hos_visual_cortex_rh_mask.nii.gz')
visual_cortex_maskpath = os.path.join(
    visual_cortex_dir, 'hos_visual_cortex_bh_mask.nii.gz')

# ############################# INPUTS ################################

THRESHOLD = 25

# ############################## RUN ##################################

if __name__ == '__main__':

    # ******************** DORSAL STRIATUM ****************************
    combine_masks(
        binarize(hos_putamen_lh_probmap, THRESHOLD),
        binarize(hos_caudate_lh_probmap, THRESHOLD),
        hos_dstriatum_lh_maskpath)
    combine_masks(
        binarize(hos_putamen_rh_probmap, THRESHOLD),
        binarize(hos_caudate_rh_probmap, THRESHOLD),
        hos_dstriatum_rh_maskpath)
    combine_masks(hos_dstriatum_lh_maskpath, hos_dstriatum_rh_maskpath,
                  hos_dstriatum_bh_maskpath)

    # ##################### CEREBELLUM VI #############################
    binarize(mniflirt_cereb6_lh_probmap, THRESHOLD).to_filename(
        mniflirt_cereb6_lh_maskpath)
    binarize(mniflirt_cereb6_rh_probmap, THRESHOLD).to_filename(
        mniflirt_cereb6_rh_maskpath)

    # ##################### CEREBELLUM CRUS I #########################
    binarize(mniflirt_crus1_lh_probmap, THRESHOLD).to_filename(
        mniflirt_crus1_lh_maskpath)
    binarize(mniflirt_crus1_rh_probmap, THRESHOLD).to_filename(
        mniflirt_crus1_rh_maskpath)

    # **************** CEREBELLUM VIIb + VIIIa ************************
    combine_masks(
        binarize(mniflirt_cereb7b_lh_probmap, THRESHOLD),
        binarize(mniflirt_cereb8a_lh_probmap, THRESHOLD),
        mniflirt_cereb7b8a_lh_maskpath)
    combine_masks(
        binarize(mniflirt_cereb7b_rh_probmap, THRESHOLD),
        binarize(mniflirt_cereb8a_rh_probmap, THRESHOLD),
        mniflirt_cereb7b8a_rh_maskpath)

    # ####################### AUDITORY CORTEX #########################
    combine_masks(
        binarize(heschl_lh_probmap, THRESHOLD),
        binarize(planum_temporale_lh_probmap, THRESHOLD),
        binarize(superior_temporal_gyrus_posterior_division_lh_probmap,
                 THRESHOLD),
        auditory_cortex_lh_maskpath)
    combine_masks(
        binarize(heschl_rh_probmap, THRESHOLD),
        binarize(planum_temporale_rh_probmap, THRESHOLD),
        binarize(superior_temporal_gyrus_posterior_division_rh_probmap,
                 THRESHOLD),
        auditory_cortex_rh_maskpath)
    combine_masks(auditory_cortex_lh_maskpath, auditory_cortex_rh_maskpath,
                  auditory_cortex_bh_maskpath)

    # ####################### VISUAL CORTEX ###########################
    combine_masks(
        binarize(occipital_locsup_lh_probmap, THRESHOLD),
        binarize(occipital_locinf_lh_probmap, THRESHOLD),
        binarize(occipital_pole_lh_probmap, THRESHOLD),
        binarize(occipital_fusiform_lh_probmap, THRESHOLD),
        visual_cortex_lh_maskpath)
    combine_masks(
        binarize(occipital_locsup_rh_probmap, THRESHOLD),
        binarize(occipital_locinf_rh_probmap, THRESHOLD),
        binarize(occipital_pole_rh_probmap, THRESHOLD),
        binarize(occipital_fusiform_rh_probmap, THRESHOLD),
        visual_cortex_rh_maskpath)
    combine_masks(visual_cortex_lh_maskpath, visual_cortex_rh_maskpath,
                  visual_cortex_maskpath)