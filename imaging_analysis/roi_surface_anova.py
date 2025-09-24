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

def create_bh_surf_rois(roi_bh_path):

    # Split into (directory, filename)
    surf_dir, bh_fname = os.path.split(roi_bh_path)

    # Separate extension
    bh_name, ext = os.path.splitext(bh_fname)

    # Two-hemisphere filenames
    hems_name = bh_name.replace('_bh', '')
    
    # Load left and right hemisphere ROIs
    roi_hems_path = os.path.join(surf_dir, hems_name + ext)
    roi_hems = np.load(roi_hems_path)

    # Average across hemispheres (axis 0)
    roi_bh_arr = roi_hems.mean(axis=0)

    # Save both hemispheres ROI
    np.save(roi_bh_path, roi_bh_arr, allow_pickle=False)

    return roi_bh_arr


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

        roi_folder = os.path.join(main_dir, folder_name, region_name, 
                                  atlas_name, roi_name)
        roi_surf_folder = os.path.join(roi_folder, 'rois_surf_extraction')
        
        for tag in tags:

            roi_bh_fname = tag + '_' + roi_name + '_bh_' + contype + '.npy'
            roi_bh_path = os.path.join(roi_surf_folder, roi_bh_fname)
            if os.path.exists(roi_bh_path):
                roi_bh = np.load(roi_bh_path)
            else:
                # Create both hemispheres (bh) files
                roi_bh = create_bh_surf_rois(roi_bh_path)


