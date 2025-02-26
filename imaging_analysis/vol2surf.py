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
import numpy as np

import nibabel as nib
import nitools as nt
import surfAnalysisPy as surf

import matplotlib.pyplot as plt


# Subjects without pilot
SUBJECTS = [3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26, 28,
            29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

# Load non-normalized individual contrast map for all subjects
derivatives_folder = '/home/analu/diedrichsen_data/data/Cerebellum/music-sdtb/derivatives'

encoding_maps = [
    os.path.join(derivatives_folder,
                 'sub-%02d' %s,
                 'estimates/allmain_tasks/ffx_rwls_dbb_hrf128/con_0002.nii')
    for s in SUBJECTS]

# Load individual fsl32K meshes
surfWB_folder = '/home/analu/diedrichsen_data/data/Cerebellum/music-sdtb/surfaceWB/data'

white_left = [os.path.join(surfWB_folder, 'sub-%02d' % s,
                           'sub-%02d' % s + '.L.white.32k.surf.gii')
              for s in SUBJECTS]

white_right = [os.path.join(surfWB_folder, 'sub-%02d' % s,
                            'sub-%02d' % s + '.R.white.32k.surf.gii')
               for s in SUBJECTS]

pial_left = [os.path.join(surfWB_folder, 'sub-%02d' % s,
                          'sub-%02d' % s + '.L.pial.32k.surf.gii')
             for s in SUBJECTS]

pial_right = [os.path.join(surfWB_folder, 'sub-%02d' % s,
                           'sub-%02d' % s + '.R.pial.32k.surf.gii')
              for s in SUBJECTS]

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
    CIFTI = nt.cifti.join_giftis_to_cifti([GIFTIL, GIFTIR], mask=[None, None])

    # Save CIFT file
    nib.save(CIFTI, 'cifti_files/sub-%02d_' % sb + 'auditory-encoding' + '.dscalar.nii')
