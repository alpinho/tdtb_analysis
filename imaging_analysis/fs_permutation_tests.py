"""
Script to do the sign-flipping permutation tests on volume data

Author: Ana Luisa Pinho
Email: agrilopi@uwo.ca

Creation: 6th of August 2025
Last Update: Augusst 2025

Compatibility: Python 3.10.14, nilearn 0.11.1

"""

import os
import numpy as np
import nibabel as nib
from scipy.stats import norm
import matplotlib.pyplot as plt


# ========================== FUNCTIONS ==================================

def get_volume_data(derivatives_dir, subjects, task_key,
                    active_ckey, passive_ckey, derivative_type='sm8wbmasked'):
    """
    Get paths of masked, smoothed or unsmoothed volume data for a specific
    task and contrast.

    Parameters
    ----------
    derivatives_dir : str
        Path to the directory containing the derivatives.
    subjects : list
        List of subject IDs.
    task_key : str
        Key for the task (e.g., 'prod', 'percep', 'ntfd', 'allmain_tasks').
    active_ckey : int
        Key for the active contrast (e.g., 6 for 'Beat').
    passive_ckey : int
        Key for the passive contrast (e.g., 7 for 'Interval')
    derivative_type : str, optional
        Type of derivative to load (default is 'sm8wbmasked').

    Returns
    -------
    active_conpaths : list
        List of paths to the active condition contrast maps for all subjects.
    passive_conpaths : list
        List of patives to the passive condition contrast maps for all subjects.
    """

    active_fname = \
        'wcon_%04d' % active_ckey + '_desc-' + derivative_type + '.nii'
    
    passive_fname = \
        'wcon_%04d' % passive_ckey + '_desc-' + derivative_type + '.nii'

    # Paths of individual contrast map of active condition for all subjects
    active_conpaths = [
        os.path.join(
            derivatives_dir,
            'sub-%02d' % sub,
            'estimates',
            task_key,
            'masked_derivatives_rwls_dbb_hrf128',
            active_fname
        )
        for sub in subjects
    ]

    # Paths of individual contrast map of passive condition for all subjects
    passive_conpaths = [
        os.path.join(
            derivatives_dir, 
            'sub-%02d' % sub,
            'estimates', 
            task_key, 
            'masked_derivatives_rwls_dbb_hrf128',
            passive_fname
        )
        for sub in subjects
    ]
    
    return active_conpaths, passive_conpaths


def compute_affine(contrasts_paths):
    """
    Load one of the individual contrasts to get the affine 
    transformation matrix.

    Returns
    -------
    affine : np.ndarray
        Affine transformation matrix of the NIfTI image.
    """

    # Load one of the contrast maps to get the affine
    image = nib.load(contrasts_paths[0])
    affine = image.affine

    return affine


def squared_difference(active_conpaths, passive_conpaths):
    """
    Load contrast maps of active and passsive conditions and compute
    the squared difference between them for each subject.

    Parameters
    ----------
    active_conpaths : list
        List of paths to the active condition contrast maps for all 
        subjects.
    passive_conpaths : list
        List of paths to the passive condition contrast maps for all 
        subjects.

    Returns
    -------
    diff_squared_maps : np.ndarray
        4D array containing the squared difference maps for each 
        subject.
        Shape: (n_subjects, x, y, z)
    """

    diff_squared_maps = []

    for active_conpath, passive_conpath in zip(active_conpaths, 
                                               passive_conpaths):
        active_img = nib.load(active_conpath)
        passive_img = nib.load(passive_conpath)

        active_data = active_img.get_fdata()
        passive_data = passive_img.get_fdata()

        diff_squared = (active_data - passive_data) ** 2
        diff_squared_maps.append(diff_squared)

    diff_squared_maps = np.stack(diff_squared_maps, axis=0)

    return diff_squared_maps


# ============================ INPUTS ===================================

# Subjects without pilot
SUBJECTS = [3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26, 28,
            29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

task_tag = 'All Tasks'
main_cname = 'Beat'
control_cname = 'Interval' # Set to None if not used

n_permutations = 5000

# ========================= PARAMETERS =================================

# Path to volume data
if os.path.isdir('/home/analu/diedrichsen_data/data'):
    base_dir = '/home/analu/diedrichsen_data/data'
else:
    base_dir = '/cifs/diedrichsen/data'

music = os.path.join(base_dir, 'Cerebellum/music-sdtb')
derivatives_folder = os.path.join(music, 'derivatives')

# Path to output folders
permut_folder = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'permutation_tests')

contrasts_folder = os.path.join(
    permut_folder,
    (
        f"{main_cname.lower().replace(' ', '-')}_vs_"
        f"{control_cname.lower().replace(' ', '-')}"
    )
)

# Task and contrast definitions
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

main_contrast_id = {v: k for k, v in all_contrasts.items()}.get(main_cname)
control_contrast_id = {v: k for k, v in all_contrasts.items()}.get(control_cname)


# ============================ RUN =====================================

if __name__ == '__main__':

    # Get paths of volume maps
    main_conpaths, control_conpaths = get_volume_data(
        derivatives_folder, SUBJECTS, task_id, main_contrast_id,
        control_contrast_id)
    
    # Compute affine transformation matrix
    affine_mtx = compute_affine(main_conpaths)

    # Compute squared difference maps for each subject
    individual_diff_squared_maps = squared_difference(main_conpaths, 
                                                      control_conpaths)

    # Compute observed mean Euclidean distance map across subjects
    # shape: (x, y, z)
    observed_map = np.mean(individual_diff_squared_maps, axis=0)  

    # Permutation Procedure (Sign-Flipping)
    # For each subject, randomly decide whether to swap their main and 
    # control maps (i.e., apply a random sign flip to the difference)    
    perm_maps = np.zeros((n_permutations,) + observed_map.shape)

    # Create a new generator object
    rng = np.random.default_rng(seed=42)

    for k in range(n_permutations):
        signs = rng.choice([1, -1], size=len(SUBJECTS))
        signed_maps = \
            signs[:, None, None, None] * individual_diff_squared_maps
        perm_maps[k] = np.mean(signed_maps, axis=0)

    # Compute voxel-wise p-values (one-sided test: observed > permuted)
    p_map = np.mean(perm_maps >= observed_map[None, ...], axis=0)

    # Convert to z-scores
    z_map = norm.isf(p_map)

    # Save results
    os.makedirs(contrasts_folder, exist_ok=True)
    zmap_path = os.path.join(
        contrasts_folder,
        (
            f"{main_cname.lower().replace(' ', '-')}_vs_"
            f"{control_cname.lower().replace(' ', '-')}"
            f"_eudist_zmap.nii.gz"
        )
    )
    nib.save(nib.Nifti1Image(z_map, affine_mtx), zmap_path)