"""
This script performs several ANOVA analysis using ROIS extracted from 
contrasts of the Music-SDTB Project, previously projected on fs_LR32k 
surface. 

Author: Ana Luisa Pinho
email: agrilopi@uwo.ca

Created: 24th of September 2025
Last update: September 2025

Compatibility: Python 3.10.14

"""

import os
import numpy as np


# ############################ FUNCTIONS ################################


# ############################# INPUTS ##################################

# Subjects without pilot
SUBJECTS = [3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26, 28,
            29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

main_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'roi_analyses_rwls_hrf128_wb_puncorr_unsmoothed',
    'bothmod_allmain_tasks'
)

folder_name = 'main_tasks' # 'rand_ntfd'
contype = 'psc'

# ### Define number of ROIs of the analysis ###
# All ROIs: 6 cortical ROIs
region_names = ['motor_area', 'motor_area', 'motor_area', 'motor_area', 
                'heschl_gyrus', 
                'occipital_lobe']
atlas_names = ['hmat', 'hmat', 'hmat', 'hmat',
               'hos', 
               'hos']
roi_names = ['pmd', 'pmv', 'sma', 'presma',
             'heschl',
             'occipital']

# #############

tags = ['i', 'i9a', 'i8a', 'i7a', 'i6a', 'a', 'a4g', 'a3g', 'a2g', 'a1g', 'g']

# ########################### PARAMETERS ################################

working_dir = os.path.dirname(os.path.abspath(__file__))

# ############################## RUN ####################################

if __name__ == '__main__':

    for region_name, atlas_name, roi_name in zip(
        region_names, atlas_names, roi_names):

        # Create both hemispheres (bh) files
        roi_folder = os.path.join(main_dir, folder_name, region_name, 
                                  atlas_name, roi_name)
        roi_surf_folder = os.path.join(roi_folder, 'rois_surf_extraction')
        
        for tag in tags:
            roi_hems_fname = tag + '_' + roi_name + '_' + contype + '.npy'
            roi_hems_path = os.path.join(roi_surf_folder, roi_hems_fname)
            roi_hems = np.load(roi_hems_path)

            roi_bh = roi_hems.mean(axis=0)
            roi_bh_fname = tag + '_' + roi_name + '_bh_' + contype + '.npy'
            roi_bh_path = os.path.join(roi_surf_folder, roi_bh_fname)
            np.save(roi_bh_path, roi_bh, allow_pickle=False)

