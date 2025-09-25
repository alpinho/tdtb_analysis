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
import pandas as pd

from roi_anova_msdtb import (dataframe, threeway_rmanova, 
                             twoway_rmanova_task, oneway_rmanova,
                             twoway_rmanova_catroi, posthoc_catroi,
                             twoway_rmanova_timingroi, posthoc_timingroi,
                             threeway_rmanova_timing)


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

base_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'roi_analyses_rwls_hrf128_wb_puncorr_unsmoothed',
    'bothmod_allmain_tasks'
)

folder_name = 'main_tasks'  # 'main_tasks' or 'rand_ntfd'
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

n_rois = 6
tags = ['i', 'i9a', 'i8a', 'i7a', 'i6a', 'a', 'a4g', 'a3g', 'a2g', 'a1g', 'g']

# ########################### PARAMETERS ################################

working_dir = os.path.dirname(os.path.abspath(__file__))

if folder_name == 'main_tasks':
    tasks = {
        'prod': 'Production', 
        'percep': 'Perception', 
        'ntfd': 'NTFD',
        'allmain_tasks': 'All Tasks'
    }
    selected_contrasts = {
        10: 'Auditory Beat',
        11: 'Auditory Interval',
        14: 'Visual Beat',
        15: 'Visual Interval'
    }
    task_roidef_id = 'allmain_tasks'   
else:
    assert folder_name == 'rand_ntfd'
    tasks = {
        'rand_ntfd': 'NTFD Random'
    }
    selected_contrasts = {
        18: 'Auditory Beat',
        19: 'Auditory Interval',
        21: 'Auditory Random',
        30: 'Visual Beat',
        31: 'Visual Interval',
        33: 'Visual Random'
    }
    task_roidef_id = 'allmain_tasks'

main_dir = os.path.join(base_dir, folder_name)

# ############################## RUN ####################################

if __name__ == '__main__':

    for tag in tags:
        dfrois = pd.DataFrame()
        for region_name, atlas_name, roi_name in \
                zip(region_names, atlas_names, roi_names):

            roi_folder = os.path.join(main_dir, region_name, atlas_name, 
                                      roi_name)
            roi_surf_folder = os.path.join(roi_folder, 'rois_surf_extraction')
            roi_anovasurf_folder = os.path.join(roi_folder, 'anova_surf')        
        
            # Load both hemispheres (bh) ROI file
            roi_hems_fname = tag + '_' + roi_name + '_' + contype + '.npy'
            roi_bh_fname = tag + '_' + roi_name + '_bh_' + contype + '.npy'
            roi_hems_path = os.path.join(roi_surf_folder, roi_hems_fname)
            roi_bh_path = os.path.join(roi_surf_folder, roi_bh_fname)

            roi_hems = np.load(roi_hems_path)
            if os.path.exists(roi_bh_path):
                roi_bh = np.load(roi_bh_path)
            else:
                # Create both hemispheres (bh) files
                roi_bh = create_bh_surf_rois(roi_bh_path)

            # Combine lh and rh with bh in the same array
            # (hemisphere, tasks, contrasts, subjects)
            # where hemispheres: lh, rh, bh
            roi_bh_reshaped = np.expand_dims(roi_bh, axis=0)
            roi_hems_combined = np.concatenate([roi_hems, roi_bh_reshaped], 
                                               axis=0)

            # Create dataframe for ANOVAs
            df_path = os.path.join(
                roi_anovasurf_folder, tag + '_' + roi_name + '_df.tsv'
            )
            dfroi = dataframe(roi_hems_combined,
                              ['lh', 'rh', 'bh'],
                              list(tasks.values()),
                              list(selected_contrasts.values()),
                              SUBJECTS,
                              df_path)
            
            # Add roi column to dataframe
            roi_arr = np.repeat(roi_name, len(dfroi.index))
            dfroi['ROI'] = roi_arr
            # Append dataframe
            dfrois = pd.concat([dfrois, dfroi], ignore_index=True, sort=False)

            # # ############## Run ANOVAs per ROI #####################

            # 3-way RM-ANOVA
            three_anova_dir = os.path.join(roi_anovasurf_folder, '3way-anova')
            threeway_rmanova(df_path, three_anova_dir, tag, roi_name)

            # 2-way RM-ANOVA for modality and beat/interval
            twoway_anova_task_dir = os.path.join(roi_anovasurf_folder, 
                                                 '2way-anova_task')
            twoway_rmanova_task(df_path, tasks, twoway_anova_task_dir, tag, 
                                roi_name)

            # 1-way RM-ANOVA for beat/interval
            oneway_anova_task_dir = os.path.join(
                roi_anovasurf_folder, '1way-anova')
            oneway_rmanova(df_path, tasks, oneway_anova_task_dir, tag, 
                           roi_name)
            
        # Save dataframe with all ROIs
        dfrois_surf_folder = os.path.join(main_dir, 'df_rois_surface')
        if not os.path.exists(dfrois_surf_folder):
            os.makedirs(dfrois_surf_folder)
        dfrois.to_csv(
            os.path.join(
                main_dir, 'df_rois_surface',
                'dfrois_' + tag + '_' + str(n_rois) + '-rois.tsv'),
            sep='\t', index=False)
        
        # ##################### 6 ROIs ################################

        # ################# CATROI ANALYSES ###################
        # 2-way RM-ANOVA for roi and category and...
        # ... for both modalities
        twoway_anova_catroi_dir = os.path.join(
            main_dir, '2way-anova_surf_cat6rois')
        twoway_rmanova_catroi(dfrois, tasks, twoway_anova_catroi_dir, tag)
        posthoc_catroi(
            dfrois, tasks, twoway_anova_catroi_dir, tag, n_rois, roi_names)
        
        # ... for the auditory modality
        twoway_anova_catroi_dir = os.path.join(
            main_dir, '2way-anova_surf_cat6rois_auditory')
        twoway_rmanova_catroi(
            dfrois, tasks, twoway_anova_catroi_dir, tag, modality='auditory')
        posthoc_catroi(
            dfrois, tasks, twoway_anova_catroi_dir, tag, n_rois, roi_names, 
            modality='auditory')
        
        # ... for the visual modality
        twoway_anova_catroi_dir = os.path.join(
            main_dir, '2way-anova_surf_cat6rois_visual')
        twoway_rmanova_catroi(
            dfrois, tasks, twoway_anova_catroi_dir, tag, modality='visual')
        posthoc_catroi(
            dfrois, tasks, twoway_anova_catroi_dir, tag, n_rois, roi_names, 
            modality='visual')

        # ##### EXPLICIT/IMPLICIT TIMING ROI ANALYSES ######
        # 2-way RM-ANOVA for roi and timing type tasks ...
        # ...for both modalities
        twoway_anova_timingroi_dir = os.path.join(
            main_dir, '2way-anova_surf_timing6rois')
        twoway_rmanova_timingroi(
            dfrois, twoway_anova_timingroi_dir, tag)
        posthoc_timingroi(
            dfrois, twoway_anova_timingroi_dir, tag, n_rois, roi_names)
        
        # ... for the auditory modality
        twoway_anova_timingroi_dir = os.path.join(
            main_dir, '2way-anova_surf_timing6rois_auditory')
        twoway_rmanova_timingroi(
            dfrois, twoway_anova_timingroi_dir, tag, modality='auditory')
        posthoc_timingroi(
            dfrois, twoway_anova_timingroi_dir, tag, n_rois, roi_names, 
            modality='auditory')
        
        # ... for the visual modality
        twoway_anova_timingroi_dir = os.path.join(
            main_dir, '2way-anova_surf_timing6rois_visual')
        twoway_rmanova_timingroi(
            dfrois, twoway_anova_timingroi_dir, tag, modality='visual')
        posthoc_timingroi(
            dfrois, twoway_anova_timingroi_dir, tag, n_rois, roi_names, 
            modality='visual')

        # ####### 3-WAY ROI × TASK × MODALITY ANOVA #######
        threeway_anova_roi_task_modality_dir = os.path.join(
            main_dir, '3way-anova_surf_timing6rois')

        threeway_rmanova_timing(
            dfrois, threeway_anova_roi_task_modality_dir, tag)

