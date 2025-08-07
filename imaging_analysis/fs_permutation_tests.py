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

def individual_squared_difference(derivatives_dir, subjects, task_key,
                                  contrast_key, derivative_type='sm8wbmasked'):
    
    wcon_0001_desc-sm8wbmasked.nii
    
    contrast_fname = f'{filetype}_{key:04d}_desc-{derivative_type}.nii'

    # Paths of the NORMALIZED individual contrast map of active condition for all subjects
    active_con_maps = [os.path.join(derivatives_dir, 'sub-%02d' % sub,
                                    'estimates', task_key, 'ffx_rwls_dbb_hrf128',
                                    'wcon_%04d' % contrast_key + '.nii')
                       for sub in subjects]
    
    # Paths of the NORMALIZED individual contrast map of passive condition for all subjects
    passive_con_maps = [os.path.join(derivatives_dir, 'sub-%02d' % sub,
                                     'estimates', task_key, 'ffx_rwls_dbb_hrf128',
                                     'wcon_%04d' % contrast_key + '.nii')
                        for sub in subjects]
    
    0/0


# ============================ INPUTS ===================================

# Subjects without pilot
SUBJECTS = [3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26, 28,
            29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

# Relative path for output folders
surf_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'results', 'surface_files')
contrasts_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'results', 'surface_images')

task_tag = 'All Tasks'
contrast_name = 'Beat'
contrast_name2 = 'Interval' # Set to None if not used

# ========================= PARAMETERS =================================

# Parent directories
if os.path.isdir('/home/analu/diedrichsen_data/data'):
    base_dir = '/home/analu/diedrichsen_data/data'
else:
    base_dir = '/cifs/diedrichsen/data'

music = os.path.join(base_dir, 'Cerebellum/music-sdtb')
derivatives_folder = os.path.join(music, 'derivatives')

model = 'rwls' # 'rwls'; or 'standard' (no rwls)
masking = 'wb' # 'wb' for whole-brain; 'gm' for grey matter
design = 'dbb' # 'dbb' if decision and response are modeled together;
               # 'drbb' if otherwise
hrf_cutoff = 'hrf128' # 'hrf128' or 'hrf42'
# hrf_cutoff = 'hrf128_timederiv'
# hrf_cutoff = 'hrf128_timedispderiv'

individual_derivatives_folder = 'masked_derivatives_' + model + '_' + \
    design + '_' + hrf_cutoff
contrast_type = 'wbmasked' # 'sm8wbmasked'

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


# ============================ RUN =====================================

if __name__ == '__main__':

    individual_squared_difference(derivatives_folder, SUBJECTS, task_id,
                                  contrast_id)

# -----------------------------
# 2. Load maps and compute per-subject squared difference
# -----------------------------
diff_squared_maps = []

for beat_file, interval_file in zip(beat_paths, interval_paths):
    beat_img = nib.load(beat_file)
    interval_img = nib.load(interval_file)

    beat_data = beat_img.get_fdata()
    interval_data = interval_img.get_fdata()

    diff_squared = (beat_data - interval_data) ** 2
    diff_squared_maps.append(diff_squared)

diff_squared_maps = np.stack(diff_squared_maps, axis=0)  # shape: (n_subjects, x, y, z)

# -----------------------------
# 3. Compute observed mean Euclidean distance map
# -----------------------------
observed_map = np.mean(diff_squared_maps, axis=0)  # shape: (x, y, z)

# -----------------------------
# 4. Permutation Test (Sign-Flipping)
# -----------------------------
n_perm = 5000
perm_maps = np.zeros((n_perm,) + observed_map.shape)

rng = np.random.default_rng(seed=42)

for k in range(n_perm):
    signs = rng.choice([1, -1], size=n_subjects)
    signed_maps = signs[:, None, None, None] * diff_squared_maps
    perm_maps[k] = np.mean(signed_maps, axis=0)

# -----------------------------
# 5. Compute voxel-wise p-values (one-sided test: observed > permuted)
# -----------------------------
p_map = np.mean(perm_maps >= observed_map[None, ...], axis=0)

# Optional: convert to z-scores
z_map = norm.isf(p_map)

# -----------------------------
# 6. Save NIfTI results
# -----------------------------
affine = beat_img.affine
nib.save(nib.Nifti1Image(observed_map, affine), "euclidean_distance_observed.nii.gz")
nib.save(nib.Nifti1Image(p_map, affine), "euclidean_distance_pmap.nii.gz")
nib.save(nib.Nifti1Image(z_map, affine), "euclidean_distance_zmap.nii.gz")