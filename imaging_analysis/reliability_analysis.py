"""
Script to do the reliability analysis across conditions.

Author: Ana Luisa Pinho
Email: agrilopi@uwo.ca

Creation: 23rd of March 2026
Last Update: March 2026

Compatibility: Python 3.10.16
"""

import os
import re
import pandas as pd
import numpy as np

from nitools import spm

# Prevent DataFrame truncation when printing
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# =========================== FUNCTIONS ================================

def reliability_dataframe(subjects, task_models, base_dir, cond_mapping, 
                          output_path):
    """Builds the inputs DataFrame, saves it, and returns it."""

    derivatives_dir = os.path.join(
        base_dir, 'data', 'Cerebellum', 'music-sdtb', 'derivatives')

    rows = []
    for subj in subjects:
        # Format subject with leading zero if needed, e.g., "sub-03"
        subj_str = f"sub-{subj:02d}"
        for model in task_models:

            spm_dir = os.path.join(
                derivatives_dir, subj_str, 'estimates', model,
                'ffx_rwls_dbb_hrf128'
            )
            masked_derivatives_dir = os.path.join(
                derivatives_dir, subj_str, 'estimates', model,
                'masked_derivatives_rwls_dbb_hrf128'
            )

            # Load SPM.mat using nitools
            SPM = spm.SpmGlm(spm_dir)
            SPM.get_info_from_spm_mat()  # retrieve SPM.mat info
            
            # Retrieve beta names, rawdata_files, and run_numbers...
            # ... as numpy arrays
            beta_names = np.array(SPM.beta_names)
            rawdata_files = np.array(SPM.rawdata_files)
            run_numbers = np.array(SPM.run_number)

            # Remove volume numbers from rawdata_files
            rawdata_files_cleaned = np.array(
                [re.sub(r', \d+\s*$', '', rawdata_file)
                 for rawdata_file in rawdata_files])
            # Get unique elements while preserving the original order...
            _, unique_indices = np.unique(rawdata_files_cleaned,
                                          return_index=True)
            # ... and sort by original order
            rawdata_unique = rawdata_files_cleaned[np.sort(unique_indices)]
            # Match its length with the number of encoding entries...
            # ... in beta_names
            rawdata_repeat = np.repeat(rawdata_unique, 4)

            # Filter beta_names to keep only encoding-related ones
            mask = np.char.find(beta_names, 'encoding') >= 0
            # Apply mask to get filtered names and runs
            filtered_beta_names = beta_names[mask]
            filtered_run_numbers = run_numbers[mask] 

            # Corresponding full list of beta files
            beta_files = np.array([f"beta_{i+1:04d}.nii"
                                   for i in range(len(beta_names))])
            # Apply the same mask to filter beta_files
            filtered_beta_files = beta_files[mask]
            
            # Loop over the original beta_names with their index
            for i, name in enumerate(filtered_beta_names):
                # Remove the suffix to get the condition type
                cond = name[:-len('_encoding*bf(1)')]

                # Get task_id
                match = re.search(r'task-(.*?)_run', rawdata_repeat[i])
                task_id = match.group(1) if match else None

                # Map condition type to its abbreviation and combine...
                # ... with task_id
                cond_abbr = cond_mapping.get(cond, cond)
                condition_name = f"{cond_abbr}_{task_id}"
               
                # Use the original beta index (1-indexed) for the...
                # ... betamap filename
                betamap_path = os.path.join(spm_dir, filtered_beta_files[i])

                # Get the corresponding run number
                run_num = filtered_run_numbers[i]

                # Get the regressor id and build path of...
                # ... corresponding masked derivative
                reg_number = os.path.splitext(filtered_beta_files[i])[0][5:]
                pscmap_path = os.path.join(
                    masked_derivatives_dir, 
                    'wpsc_maps_' + reg_number + '_desc-wbmasked.nii')

                # Convert full paths to paths relative to base_dir
                relative_betamap_path = os.path.relpath(
                    betamap_path, base_dir)
                relative_pscmap_path = os.path.relpath(
                    pscmap_path, base_dir)
                
                rows.append({
                    'subject': subj,
                    'task_id': task_id,
                    'run_number': run_num,
                    'condition_type': cond,
                    'condition_name': condition_name,
                    'betamap_path': relative_betamap_path,
                    'wmasked_pscmap_path': relative_pscmap_path
                })

    # Create the DataFrame
    df = pd.DataFrame(rows)
    print(df)

    # Save the DataFrame in the rsa_folder
    df.to_csv(output_path, index=False, sep='\t')

    return df


# =========================== INPUTS ===================================

# Subjects without pilot
# SUBJECTS = [3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26, 28,
#             29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
SUBJECTS = [43]

# Parent directories
home = os.path.expanduser('~')

if os.path.isdir('/home/analu/diedrichsen_data'):
    data_storage = '/home/analu/diedrichsen_data'
else:
    assert os.path.isdir('/home/UWO/agrilopi')
    data_storage = '/cifs/diedrichsen'

# Define tasks in the glm
# glm_tasks = ['prod', 'percep', 'ntfd', 'rand_ntfd']
glm_tasks = ['prod', 'percep', 'ntfd']

# Path for output folders
reliability_folder = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'results', 'reliability')

# Define mapping for condition type abbreviations
conditions_mapping = {
    'auditory_beat': 'abeat',
    'auditory_interval': 'ainterval',
    'visual_beat': 'vbeat',
    'visual_interval': 'vinterval'
}

thresh_type = 'puncorr'  # 'puncorr' or 'pcorr'
smooth = 'unsmoothed'  # 'smoothed'

# ####################### ROIs ##############################
atlas_names = ['hos', 
               'ntk_symmni128', 
               'hmat', 'hmat', 'hmat', 'hmat',
               'hos',
               'hos']
region_names = ['dorsal_striatum', 
                'cerebellum', 
                'motor_area', 'motor_area', 'motor_area', 'motor_area',
                'heschl_gyrus',
                'occipital_lobe']
roi_names = ['dstr', 
             'cereb', 
             'pmd', 'pmv', 'sma', 'presma',
             'heschl',
             'occipital']

# ============================ RUN =====================================

if __name__ == '__main__':
    # Create output folder if it does not exist
    os.makedirs(reliability_folder, exist_ok=True)

    # Paths of dataframes
    db_taskglm_path = os.path.join(
        reliability_folder, 
        f'reliability_taskglm_{thresh_type}_{smooth}.tsv')

    reliability_dataframe(SUBJECTS, glm_tasks, data_storage, 
                          conditions_mapping, db_taskglm_path)