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

import nibabel as nib
import nitools as nt

from SUITPy import flatmap


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

    individual_suit(derivatives_folder, SUBJECTS, task_id, contrast_id,
                    suit_folder)
