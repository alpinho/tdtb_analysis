"""
Script to do the volume to suit projection of data from the
 Music-SDTB project

Author: Ana Luisa Pinho
Email: agrilopi@uwo.ca

Creation: 27th of February 2025
Last Update: February 2025

Compatibility: Python 3.10.14, SUITPy 1.3.2
"""

import os

import numpy as np
import nibabel as nib
import nitools as nt

import matplotlib.pyplot as plt

from SUITPy import flatmap
from scipy import stats


# %%
# ========================== FUNCTIONS =================================

def individual_suit(derivatives_dir, subjects, task_key, contrast_key,
                    suit_dir):

    # Paths of non-normalized individual contrast map for all subjects
    encoding_maps = [os.path.join(derivatives_dir, 'sub-%02d' % sub,
                                  'estimates', task_key, 'ffx_rwls_dbb_hrf128',
                                  'con_%04d' % contrast_key + '.nii')
                     for sub in subjects]

    # Maps volume-based data onto the suit surface as numpy arrfays
    suit_maps = [flatmap.vol_to_surf(emap, space='SUIT')
                 for emap in encoding_maps]

    # Transform numpy arrays in gifti files
    contrast = all_contrasts[contrast_key].replace(' ', '-')    
    giftis = [nt.gifti.make_func_gifti(suit_map,
                                       anatomical_struct='Cerebellum',
                                       column_names=[contrast])
              for suit_map in suit_maps]

    # Create directory to save outputs if does not exist
    if not os.path.exists(suit_dir):
        os.makedirs(suit_dir)

    # Save the data
    for s, sb in enumerate(subjects):
        nib.save(
            giftis[s],
            os.path.join(
                suit_dir,
                'sub-{sb:02d}_'.format(sb=sb)
                + contrast.lower()
                + '_'
                + 'suit.func.gii',
            ),
        )


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


def group_suit(surf_dir, subjects, contrast_tag):

    contrast = contrast_tag.lower().replace(' ', '-')

    # Get individual functional data projected on suit space
    igiftis_paths = [
        os.path.join(
            surf_dir,
            f'sub-{sub:02d}_{contrast}_suit.func.gii'
        )
        for sub in subjects
    ]

    # Load individual suit data
    igiftis = [nib.load(gl) for gl in igiftis_paths]
    data = np.array([nt.get_gifti_data_matrix(igifti)
                     for igifti in igiftis])
    data = np.squeeze(data, axis=-1)

    # Substitute nan's by 0's
    data[np.isnan(data)] = 0

    # Calculate the one sample t-test
    tvals, pvals = stats.ttest_1samp(data, 0, axis=0,
                                     alternative='two-sided')

    # Compute z-values from t-values
    zvals = zval_conversion(tvals, len(subjects)-1)

    # Threshold z-values, ...
    # ... because pvalues are two-sided, ...
    # ... alpha is corrected by a factor of 2
    fdr_thresh = threshold(zvals, pvals, 0.05/2, height_control='fdr')
    print(fdr_thresh)

    return zvals, fdr_thresh


def plot_suitflat(stats, threshold, contrast_tag):

    contrast = contrast_tag.lower().replace(' ', '-')

    # Define color limits
    vmin, vmax = 0, 10

    # Do the flatmap
    flatmap.plot(stats,
                 cmap='jet',
                 #cscale=[vmin, vmax],
                 #underscale=[-1.5, 1],
                 threshold=vmin,
                 colorbar=True,
                 render='matplotlib')

    # Get the current figure created by flatmap.plot()
    fig = plt.gcf()  # Get the figure from the active Matplotlib state

    # Save figure
    output_name = f'group_{contrast}_suit.png'
    fig.savefig(output_name, dpi=300)


# %%
# =========================== INPUTS ===================================

# Subjects without pilot
SUBJECTS = [3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26, 28,
            29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

# Relative path for output folder
suit_folder = 'suit_files'

task_tag = 'All Tasks'
contrast_name = 'Auditory Encoding'

# %%
# ========================= PARAMETERS =================================

# Parent directories
home = os.path.expanduser('~')
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
contrast_id = {v: k for k, v in all_contrasts.items()}.get(contrast_name)

# %%
# ============================ RUN =====================================

if __name__ == '__main__':

    # Compute individual gifti files with the volume to suit...
    # ... projection of the contrast map
    # individual_suit(derivatives_folder, SUBJECTS, task_id, contrast_id,
    #                 suit_folder)

    # Compute group func gifti
    z_values, thresh = group_suit(suit_folder, SUBJECTS, contrast_name)

    # Plot cerebellum flatmap
    plot_suitflat(z_values, thresh, contrast_name)
