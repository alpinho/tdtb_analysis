"""
Script to compute ffx group-level analysis

Author: Ana Luisa Pinho
Created: April 2023
Last Update: April 2023

Compatibility: Python 3.10.10
"""

import os
import glob
import numpy as np

from nilearn.maskers import NiftiMasker


# %%
# ========================== FUNCTIONS =================================

def compute_fixed_effects_params(contrasts, variances,
                                 precision_weighted=False):
    """Compute the fixed effects t-statistic, contrast, variance, \
    given arrays of effects and variance."""
    tiny = 1.e-16
    contrasts, variances = np.asarray(contrasts), np.asarray(variances)
    variances = np.maximum(variances, tiny)

    if precision_weighted:
        weights = 1. / variances
        fixed_fx_variance = 1. / np.sum(weights, 0)
        fixed_fx_contrasts = np.sum(contrasts * weights, 0) * fixed_fx_variance
    else:
        fixed_fx_variance = np.mean(variances, 0) / len(variances)
        fixed_fx_contrasts = np.mean(contrasts, 0)

    fixed_fx_stat = fixed_fx_contrasts / np.sqrt(fixed_fx_variance)
    return fixed_fx_contrasts, fixed_fx_variance, fixed_fx_stat


# %%
# =========================== INPUTS ===================================

SUBJECTS_NUMBERS = [3, 4, 7, 8, 10]
maindir = '/home/analu/diedrichsen_data/data/Cerebellum/music-sdtb/derivatives'

design = 'allmain_tasks'
ffx_model = 'ffx_onesample_t_nilearn'

# %%
# ========================= PARAMETERS =================================

group_dir = os.path.join(maindir, 'group')
group_anat = os.path.join(group_dir, 'anat')
group_wholebrain_mask = os.path.join(group_anat, 'group_mask_noskull.nii')
group_cmask = os.path.join(group_anat, 'group_mask_gray.nii')

# %%
# ============================ RUN =====================================

subjects = ['sub-%02d' % s for s in SUBJECTS_NUMBERS]

# Get list of individual contrasts and variances
individual_contrasts_paths = []
individual_variances_paths = []
for subject in subjects:
    individual_contrast_path = os.path.join(maindir, subject, 'estimates',
                                            'allmain_tasks', 'snorm_standard',
                                            'swcon_0001_masked.nii')
    individual_variance_path = os.path.join(maindir, subject, 'estimates',
                                            'allmain_tasks', 'snorm_standard',
                                            'swResMS_masked.nii')
    individual_contrasts_paths.append(individual_contrast_path)
    individual_variances_paths.append(individual_variance_path)

# Define and create output dir, if does not exist
contrast_dir = os.path.join(group_dir, design, ffx_model, 'con_01_Encoding')
if not os.path.exists(contrast_dir):
    os.makedirs(contrast_dir)
else:
    for f in glob.glob(contrast_dir + '/*.nii'):
        os.remove(f)

# Transform nifti files in numpy arrays
nifti_masker = NiftiMasker(mask_img=group_wholebrain_mask)
individual_contrasts_arr = nifti_masker.fit_transform(
    individual_contrasts_paths)
individual_variances_arr = nifti_masker.fit_transform(
    individual_variances_paths)

# Compute ffx
groupffx_contrasts_arr, groupffx_variance_arr, groupffx_tstat_arr = \
    compute_fixed_effects_params(individual_contrasts_arr,
                                 individual_variances_arr)

# Transform resulting numpy arrays into nifti files
group_contrasts_maps = nifti_masker.inverse_transform(groupffx_contrasts_arr)
group_variances_maps = nifti_masker.inverse_transform(groupffx_variance_arr)
group_tsat_maps = nifti_masker.inverse_transform(groupffx_tstat_arr)

# Define paths of output nifti files
gfc_path = os.path.join(contrast_dir, 'contrast_0001.nii')
gfv_path = os.path.join(contrast_dir, 'variance_0001.nii')
gfs_path = os.path.join(contrast_dir, 'tstat_0001.nii')

# Save output nifti files
group_contrasts_maps.to_filename(gfc_path)
group_variances_maps.to_filename(gfv_path)
group_tsat_maps.to_filename(gfs_path)
