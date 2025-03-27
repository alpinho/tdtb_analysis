"""
Script to do the rsa analysis to calculate similarities of beat conditions
as well as interval conditions across tasks

Author: Ana Luisa Pinho
Email: agrilopi@uwo.ca

Creation: 26th of March 2025
Last Update: March 2025

Compatibility: Python 3.10.14
"""

import os
import numpy as np
import pandas as pd

from nitools import spm


# =========================== FUNCTIONS ================================

def create_rsa_dataframe(subjects, task_ids, derivatives_dir, cond_mapping,
                         output_path):
    """Builds the RSA inputs DataFrame, saves it, and returns it."""
    rows = []
    for subj in subjects:
        # Format subject with leading zero if needed, e.g., "sub-03"
        subj_str = f"sub-{subj:02d}"
        for task_id in task_ids:
            if task_id == 'allmain_tasks':
                continue  # Skip the grand GLM of all tasks together

            spm_dir = os.path.join(
                derivatives_dir, subj_str, 'estimates', task_id,
                'ffx_rwls_dbb_hrf128'
            )
            resms_path = os.path.join(spm_dir, 'ResMS.nii')

            # Load SPM.mat using nitools
            SPM = spm.SpmGlm(spm_dir)
            SPM.get_info_from_spm_mat()  # retrieve SPM.mat info

            # Retrieve beta names and run_numbers as numpy arrays
            beta_names = np.array(SPM.beta_names)
            run_numbers = np.array(SPM.run_number)

            # Loop over the original beta_names with their index
            for i, name in enumerate(beta_names):
                if name.endswith('_encoding*bf(1)'):
                    # Remove the suffix to get the condition type
                    cond = name[:-len('_encoding*bf(1)')]
                    # Map condition type to its abbreviation and combine...
                    # ... with task_id
                    cond_abbr = cond_mapping.get(cond, cond)
                    condition_name = f"{cond_abbr}_{task_id}"
                    # Use the original beta index (1-indexed) for the...
                    # ... betamap filename
                    betamap_fname = f"beta_{i+1:04d}.nii"
                    betamap_path = os.path.join(spm_dir, betamap_fname)
                    # Get the corresponding run number
                    run_num = run_numbers[i]
                    rows.append({
                        'subject': subj,
                        'task_id': task_id,
                        'run_number': run_num,
                        'condition_type': cond,
                        'condition_name': condition_name,
                        'resms_path': resms_path,
                        'betamap_path': betamap_path
                    })

    # Create the DataFrame
    df = pd.DataFrame(rows)
    print(df)

    # Save the DataFrame in the rsa_folder
    df.to_csv(output_path, index=False, sep='\t')

    return df


# =========================== INPUTS ===================================

# Subjects without pilot
SUBJECTS = [3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26, 28,
            29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

# Relative path for output folders
rsa_folder = 'results/rsa'

# ========================= PARAMETERS =================================

# Parent directories
home = os.path.expanduser('~')
music = os.path.join(home, 'diedrichsen_data', 'data', 'Cerebellum',
                     'music-sdtb')
derivatives_folder = os.path.join(music, 'derivatives')

# Define tasks
tasks = ['prod', 'percep', 'ntfd', 'allmain_tasks']

# Define mapping for condition type abbreviations
conditions_mapping = {
    'auditory_beat': 'abeat',
    'auditory_interval': 'ainterval',
    'visual_beat': 'vbeat',
    'visual_interval': 'vinterval'
}

# ============================ RUN =====================================

if __name__ == '__main__':

    # Create output folder if it does not exist
    os.makedirs(rsa_folder, exist_ok=True)
    
    # Create dataframe
    db_path = os.path.join(rsa_folder, 'rsa_inputs.tsv')
    db = create_rsa_dataframe(SUBJECTS, tasks, derivatives_folder,
                              conditions_mapping, db_path)
