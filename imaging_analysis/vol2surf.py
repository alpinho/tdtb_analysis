"""
Script to do the volume to surface projection of data from the
 Music-SDTB project + smoothing

Author: Ana Luisa Pinho
Email: agrilopi@uwo.ca

Creation: 24th of February 2025
Last Update: February 2025

Compatibility: Python 3.10.14
"""

import os

import nibabel as nib
import nitools as nt
import surfAnalysisPy as surf


# %%
# ========================== FUNCTIONS =================================

def individual_surf(derivatives_dir, subjects, task_key, contrast_key,
                    surfspace_dir):

    # Paths of non-normalized individual contrast map for all subjects
    encoding_maps = [os.path.join(derivatives_dir, 'sub-%02d' % sub,
                                  'estimates', task_key, 'ffx_rwls_dbb_hrf128',
                                  'con_%04d' % contrast_key + '.nii')
                     for sub in subjects]

    # Paths of individual meshes per hemisphere
    white_left = [os.path.join(surfspace_dir, 'sub-%02d' % sub,
                               'sub-%02d' % sub + '.L.white.32k.surf.gii')
                  for sub in subjects]

    white_right = [os.path.join(surfspace_dir, 'sub-%02d' % sub,
                                'sub-%02d' % sub + '.R.white.32k.surf.gii')
                   for sub in subjects]

    pial_left = [os.path.join(surfspace_dir, 'sub-%02d' % sub,
                              'sub-%02d' % sub + '.L.pial.32k.surf.gii')
                 for sub in subjects]

    pial_right = [os.path.join(surfspace_dir, 'sub-%02d' % sub,
                               'sub-%02d' % sub + '.R.pial.32k.surf.gii')
                  for sub in subjects]

    # For each subject...
    for emap, wl, wr, pl, pr, sb in zip(encoding_maps, white_left, white_right,
                                        pial_left, pial_right, SUBJECTS):

        # Map individual functional data from  Nifti to the surface of...
        # ... left and right hemispheres
        DL = surf.map.vol_to_surf([emap], wl, pl)
        DR = surf.map.vol_to_surf([emap], wr, pr)

        # Transform numpy arrays in gifti files
        GIFTIL = nt.gifti.make_func_gifti(DL, anatomical_struct='CortexLeft',
                                          column_names=['Auditory-Encoding'])
        GIFTIR = nt.gifti.make_func_gifti(DR, anatomical_struct='CortexRight',
                                          column_names=['Auditory-Encoding'])

        # Create CIFTI
        CIFTI = nt.cifti.join_giftis_to_cifti([GIFTIL, GIFTIR],
                                              mask=[None, None])

        # Create output folder if does not exist
        output_folder = 'cifti_files'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Save CIFT file
        contrast = contrast_tag.lower().replace(" ", "-")
        nib.save(CIFTI, os.path.join(
            output_folder, 'sub-%02d_' % sb + contrast + '.dscalar.nii'))


# %%
# =========================== INPUTS ===================================

# Subjects without pilot
SUBJECTS = [3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26, 28,
            29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

# Relative path for Individual fsl32K meshes
surfWB_folder = os.path.join('surfaceWB', 'data')

task_tag = 'All Tasks'
contrast_tag = 'Auditory Encoding'

# %%
# ========================= PARAMETERS =================================

# Parent directories
home = os.path.expanduser("~")
music = os.path.join(home, 'diedrichsen_data/data/Cerebellum/music-sdtb')
derivatives_folder = os.path.join(music, 'derivatives')

# Individual fsl32K meshes
surfWB_dir = os.path.join(music, surfWB_folder)

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
                    surfWB_dir)
