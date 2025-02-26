"""
Script to do the volume to surface projection of data from the
 Music-SDTB project + smoothing

Author: Ana Luisa Pinho
Email: agrilopi@uwo.ca

Creation: 24th of February 2025
Last Update: February 2025

Compatibility: Python 3.10.14, nilearn 0.11.1
"""

import os
import numpy as np

import nibabel as nib
import nitools as nt
import surfAnalysisPy as surf

from scipy import stats
from nilearn.surface import load_surf_data


# %%
# ========================== FUNCTIONS =================================

def imeshes(derivatives_dir, subjects, surfspace='fsaverage'):

    if surfspace == 'fsaverage':
        surfspace_dir = os.path.join(os.path.dirname(derivatives_dir),
                                     'surfaceFreeSurfer')
        white_left = [os.path.join(
            surfspace_dir, 'sub-%02d' % sub, 'surf', 'lh.white.surf.gii')
                      for sub in subjects]
        white_right = [os.path.join(
            surfspace_dir, 'sub-%02d' % sub, 'surf', 'rh.white.surf.gii')
                       for sub in subjects]

        pial_left = [os.path.join(
            surfspace_dir, 'sub-%02d' % sub, 'surf', 'lh.pial.surf.gii')
                      for sub in subjects]
        pial_right = [os.path.join(
            surfspace_dir, 'sub-%02d' % sub, 'surf', 'rh.pial.surf.gii')
                       for sub in subjects]
    else:
        assert surfspace == 'fslr32k'
        
        surfspace_dir = os.path.join(os.path.dirname(derivatives_dir),
                                     'surfaceWB', 'data')
        subjects_dir = [os.path.join(surfspace_dir, 'sub-%02d' % sub)
                        for sub in subjects]
        
        white_left = [os.path.join(subjects_dir[s],
                                   'sub-%02d' % sub + '.L.white.32k.surf.gii')
                      for s, sub in enumerate(subjects)]
        white_right = [os.path.join(subjects_dir[s],
                                    'sub-%02d' % sub + '.R.white.32k.surf.gii')
                       for s, sub in enumerate(subjects)]

        pial_left = [os.path.join(subjects_dir[s],
                                  'sub-%02d' % sub + '.L.pial.32k.surf.gii')
                     for s, sub in enumerate(subjects)]
        pial_right = [os.path.join(subjects_dir[s],
                                   'sub-%02d' % sub + '.R.pial.32k.surf.gii')
                      for s, sub in enumerate(subjects)]

    return white_left, white_right, pial_left, pial_right


def individual_surf(derivatives_dir, subjects, task_key, contrast_key,
                    output_folder, surfspace='fsaverage', save='gifti'):

    # Paths of non-normalized individual contrast map for all subjects
    encoding_maps = [os.path.join(derivatives_dir, 'sub-%02d' % sub,
                                  'estimates', task_key, 'ffx_rwls_dbb_hrf128',
                                  'con_%04d' % contrast_key + '.nii')
                     for sub in subjects]

    # Paths of individual meshes per hemisphere
    white_left, white_right, pial_left, pial_right = imeshes(
        derivatives_dir, subjects, surfspace=surfspace)


    # For each subject...
    for emap, wl, wr, pl, pr, sb in zip(encoding_maps, white_left, white_right,
                                        pial_left, pial_right, SUBJECTS):

        # Map individual functional data from  Nifti to the surface of...
        # ... left and right hemispheres
        DL = surf.map.vol_to_surf([emap], wl, pl)
        DR = surf.map.vol_to_surf([emap], wr, pr)

        # Transform numpy arrays in gifti files
        contrast = contrast_tag.replace(" ", "-")
        GIFTIL = nt.gifti.make_func_gifti(DL, anatomical_struct='CortexLeft',
                                          column_names=[contrast])
        GIFTIR = nt.gifti.make_func_gifti(DR, anatomical_struct='CortexRight',
                                          column_names=[contrast])

        # Create output folder if does not exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Save output
        if save == 'gifti':
            # Save Gifti files
            nib.save(
                GIFTIL,
                os.path.join(
                    output_folder,
                    "sub-{sb:02d}_".format(sb=sb)
                    + contrast.lower()
                    + "_"
                    + surfspace
                    + ".hem-L.func.gii",
                ),
            )
            nib.save(
                GIFTIR,
                os.path.join(
                    output_folder,
                    "sub-{sb:02d}_".format(sb=sb)
                    + contrast.lower()
                    + "_"
                    + surfspace
                    + ".hem-R.func.gii",
                ),
            )
        else:
            assert save == 'cifti'
            # Create CIFTI
            CIFTI = nt.cifti.join_giftis_to_cifti([GIFTIL, GIFTIR],
                                                  mask=[None, None])
            # Save CIFT file
            nib.save(CIFTI, os.path.join(
                output_folder,
                'sub-%02d_' % sb + contrast.lower() + '.dscalar.nii'))


def zval_conversion(tval, dof):
    pval = stats.t.sf(tval, dof)
    one_minus_pval = stats.t.cdf(tval, dof)
    zval_sf = stats.norm.isf(pval)
    zval_cdf = stats.norm.ppf(one_minus_pval)
    zval = np.empty(pval.shape)
    use_cdf = zval_sf < 0
    use_sf = np.logical_not(use_cdf)
    zval[np.atleast_1d(use_cdf)] = zval_cdf[use_cdf]
    zval[np.atleast_1d(use_sf)] = zval_sf[use_sf]

    return zval


def threshold(z_vals, p_vals, alpha, height_control='fdr'):
    """
    Return the Benjamini-Hochberg FDR or Bonferroni threshold for
    the input correlations + corresponding p-values.
    """
    if alpha < 0 or alpha > 1:
        raise ValueError(
            'alpha should be between 0 and 1. {} was provided'.format(alpha))

    p_vals_ = np.sort(p_vals)
    idx = np.argsort(p_vals)

    z_vals_abs = np.abs(z_vals)
    z_vals_ = z_vals_abs[idx]

    n_samples = len(p_vals_)

    if height_control == 'fdr':
        pos = p_vals_ < alpha * np.linspace(1 / n_samples, 1, n_samples)
    elif height_control == 'bonferroni':
        pos = p_vals_ < alpha / n_samples
    else:
        raise ValueError('Height-control method not valid.')

    return (z_vals_[pos][-1] - 1.e-12) if pos.any() else np.infty


def group_surf(input_dir, subjects, contrast_key):

    contrast = contrast_key.lower().replace(" ", "-")

    # Paths of individual gifti files per hemisphere
    gifti_left = [
        os.path.join(
            surf_dir,
            f'sub-{sub:02d}_{contrast}.hem-L.func.gii'
        ) 
        for sub in subjects
    ]

    gifti_right = [
        os.path.join(
            surf_dir,
            f'sub-{sub:02d}_{contrast}.hem-R.func.gii'
        ) 
        for sub in subjects
    ]

    # Load individual surface data
    data_left = np.array([load_surf_data(gl) for gl in gifti_left])
    data_right = np.array([load_surf_data(gr) for gr in gifti_right])

    # Stack data
    data = np.hstack((data_left, data_right))

    # Calculate the one sample t-test
    tvals, pvals = stats.ttest_1samp(data, 0, axis=0, nan_policy='propagate',
                                     alternative='two-sided')

    # Compute z-values from t-values
    zvals = zval_conversion(tvals, len(subjects)-1)

    # Threshold z-values, ...
    # ... because pvalues are two-sided, alpha is corrected by a factor of 2
    fdr_thresh = threshold(zvals, pvals, 0.05/2, height_control='fdr')




# %%
# =========================== INPUTS ===================================

# Subjects without pilot
SUBJECTS = [3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26, 28,
            29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

# Output Path
surf_dir = 'surface_files'

task_tag = 'All Tasks'
contrast_tag = 'Auditory Encoding'

# %%
# ========================= PARAMETERS =================================

# Parent directories
home = os.path.expanduser("~")
music = os.path.join(home, 'diedrichsen_data/data/Cerebellum/music-sdtb')
derivatives_folder = os.path.join(music, 'derivatives')

tasks = {'prod': 'Production', 'percep': 'Perception', 'ntfd': 'NTFD',
         'allmain_tasks': 'All Tasks'}

all_contrasts = {1: 'Encoding',
                 2: 'Auditory Encoding',
                 3: 'Visual Encoding',
                 4: 'Auditory vs Visual Encoding',
                 5: 'Visual vs Auditory Encoding',
                 6: 'Beat',
                 7: 'Interval',
                 8: 'Beat vs Interval',
                 9: 'Interval vs Beat',
                 10: 'Auditory Beat',
                 11: 'Auditory Interval',
                 12: 'Auditory Beat vs Auditory Interval',
                 13: 'Auditory Interval vs Auditory Beat',
                 14: 'Visual Beat',
                 15: 'Visual Interval',
                 16: 'Visual Beat vs Visual Interval',
                 17: 'Visual Interval vs Visual Beat',
                 18: 'Decision'}

task_id = {v: k for k, v in tasks.items()}.get(task_tag)
contrast_id = {v: k for k, v in all_contrasts.items()}.get(contrast_tag)

# %%
# ============================ RUN =====================================

if __name__ == '__main__':

    # Get individual cifti files with the volume to surface projection of...
    # ... the contrast map per participant
    individual_surf(derivatives_folder, SUBJECTS, task_id, contrast_id,
                    surf_dir, surfspace='fslr32k')

    # Compute group func gifti
    # group_surf(surf_dir, SUBJECTS, contrast_tag, surfspace='fsaverage')
