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
import re

import numpy as np
import pandas as pd
import PcmPy as pcm

import matplotlib.pyplot as plt

from nitools import spm
from nilearn import image
from nilearn.input_data import NiftiMasker


# =========================== FUNCTIONS ================================

def rsa_dataframe(subjects, task_models, base_dir, cond_mapping, output_path,
                  glm_type='task_glm'):
    """Builds the inputs DataFrame, saves it, and returns it."""

    derivatives_dir = os.path.join(
        base_dir, 'data', 'Cerebellum', 'music-sdtb', 'derivatives')

    if glm_type == 'task_glm':
        filtered_models = [tm for tm in task_models if tm != 'allmain_tasks']
    else:
        assert glm_type == 'grand_glm'
        filtered_models = ['allmain_tasks']

    rows = []
    for subj in subjects:
        # Format subject with leading zero if needed, e.g., "sub-03"
        subj_str = f"sub-{subj:02d}"
        for model in filtered_models:

            spm_dir = os.path.join(
                derivatives_dir, subj_str, 'estimates', model,
                'ffx_rwls_dbb_hrf128'
            )
            resms_path = os.path.join(spm_dir, 'ResMS.nii')

            # Load SPM.mat using nitools
            SPM = spm.SpmGlm(spm_dir)
            SPM.get_info_from_spm_mat()  # retrieve SPM.mat info

            # Retrieve beta names, rawdata_files, and run_numbers...
            # ... as numpy arrays
            beta_names = np.array(SPM.beta_names)
            rawdata_files = np.array(SPM.rawdata_files)

            if glm_type == 'task_glm':
                run_numbers = np.array(SPM.run_number)
            else:
                assert glm_type == 'grand_glm'
                run_numbers = np.repeat(
                    [1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4], 4)

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
            # Apply mask to get filtered names
            filtered_beta_names = beta_names[mask] 

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
                run_num = run_numbers[i]

                # Convert full paths to paths relative to base_dir
                relative_resms_path = os.path.relpath(resms_path, base_dir)
                relative_betamap_path = os.path.relpath(betamap_path,
                                                        base_dir)
                rows.append({
                    'subject': subj,
                    'task_id': task_id,
                    'run_number': run_num,
                    'condition_type': cond,
                    'condition_name': condition_name,
                    'resms_path': relative_resms_path,
                    'betamap_path': relative_betamap_path
                })

    # Create the DataFrame
    df = pd.DataFrame(rows)
    print(df)

    # Save the DataFrame in the rsa_folder
    df.to_csv(output_path, index=False, sep='\t')

    return df


def prewhiten_betas(df_input, subjects, base_dir, output_path):
    """
    Loads the input DataFrame or uses the given DataFrame,
    processes each beta map by dividing it by the square root of the
    ResMS map, and saves the prewhitened beta map as a new NIfTI file
    with the suffix '_desc-prewhitened' in the same directory.

    Parameters
    ----------
    df_input : str or pd.DataFrame
        Either the path to the input DataFrame (TSV file) or a
        DataFrame object.

    Returns
    -------
    None
    """
    
    # Load the DataFrame if a path is provided
    if isinstance(df_input, str):
        df_unfiltered = pd.read_csv(df_input, sep='\t')
    elif isinstance(df_input, pd.DataFrame):
        df_unfiltered = df_input.copy()
    else:
        raise ValueError(
            f"df_input must be either a path to a TSV file or"
            f"a pandas DataFrame."
        )

    # Filter the DataFrame according to subjects list
    df = df_unfiltered[df_unfiltered['subject'].isin(subjects)]

    # List to store new file paths
    swmasked_paths = []

    # Process each row in the DataFrame
    for _, row in df.iterrows():
        # The paths stored in the DataFrame are relative to base_dir
        rel_beta_map_path = row['betamap_path']
        rel_resms_path = row['resms_path']

        # Build full paths for opening the files
        full_beta_map_path = os.path.join(base_dir, rel_beta_map_path)
        full_resms_path = os.path.join(base_dir, rel_resms_path)

        # Load beta map using nilearn and get data as a numpy array
        beta_img = image.load_img(full_beta_map_path)
        beta_data = beta_img.get_fdata()

        # Load ResMS map using nilearn and get data as a numpy array
        resms_img = image.load_img(full_resms_path)
        resms_data = resms_img.get_fdata()

        # Compute square root of ResMS and avoid division by zero
        sqrt_resms = np.sqrt(resms_data)
        epsilon = 1e-6
        sqrt_resms[sqrt_resms < epsilon] = epsilon

        # Compute the prewhitened beta map
        beta_prewhitened = beta_data / sqrt_resms

        # Create a new image with the prewhitened beta data,
        # preserving the affine and header of the original beta map.
        new_img = image.new_img_like(beta_img, beta_prewhitened)

        # Build new full filename with the suffix '_desc-prewhitened'
        full_base, ext = os.path.splitext(full_beta_map_path)
        if ext == '.gz':
            full_base, ext2 = os.path.splitext(full_base)
            ext = ext2 + ext  # ext becomes '.nii.gz'
        new_full_fname = full_base + '_desc-prewhitened' + ext

        # Save the new prewhitened beta map using nilearn
        new_img.to_filename(new_full_fname)

        # Convert the new full filename back to a relative path...
        # ... (relative to base_dir)
        new_rel_fname = os.path.relpath(new_full_fname, base_dir)

        # Append the relative path to the list
        swmasked_paths.append(new_rel_fname)

    # Register the new paths in a new column
    df['swmasked_betamap_path'] = swmasked_paths

    # Save the DataFrame in the rsa_folder
    df.to_csv(output_path, index=False, sep='\t')

    return df


def extract_roi_signals_from_dataframe(df_input, subjects, rois, tags):
    # Load DataFrame if needed
    if isinstance(df_input, str):
        df_unfiltered = pd.read_csv(df_input, sep='\t')
    elif isinstance(df_input, pd.DataFrame):
        df_unfiltered = df_input.copy()
    else:
        raise ValueError("df_input must be a path or a pandas DataFrame.")

    # Filter the DataFrame according to subjects list
    df = df_unfiltered[df_unfiltered['subject'].isin(subjects)]

    # Filter tasks
    tasks = [t for t in tasks if t != 'allmain_tasks']
    
    # Loop over each tag and each ROI.
    # for tag, wpair in zip(tags, weights_list):
    #     for atlas_dirname, atlas_name, region_name, roi_name in zip(
    #             atlas_dirnames, atlas_names, region_names, roi_names):
        

# =========================== INPUTS ===================================

# Subjects without pilot
SUBJECTS = [3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26, 28,
            29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
# SUBJECTS = [3]

# Path for output folders
rsa_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'results', 'rsa')

# ########################### ROIs ######################################
# All ROIs: 7 ROIs
region_names = ['dorsal_striatum', 'cerebellum', 'cerebellum', 'cerebellum',
                'motor_area', 'motor_area', 'motor_area']
atlas_names = ['hos', 'ntk_symmni128', 'ntk_symmni128', 'ntk_symmni128',
               'hmat', 'hmat', 'hmat']
roi_names = ['dstr', 'cereb-s', 'cereb-i', 'cereb',
             'pmd', 'sma', 'presma']

# Paths of ROIs
# irois_paths = os.path.join(os.path.dirname(os.path.abspath(__file__)),
#                            'roi_analyses_rwls_hrf128_wb_puncorr',
#                            'all',
#                            region_name,
#                            atlas_name)

# irois_masks_paths = os.path.join(irois_paths, 'individual_roi_masks')
# irois_signals_paths = os.path.join(irois_paths, 'rois_extraction')

itags = ['i', 'i9a', 'i8a', 'i7a', 'i6a', 'a', 'a4g', 'a3g', 'a2g', 'a1g', 'g']

# #######################################################################

# Parent directories
home = os.path.expanduser('~')

if home == '/home/analu':
    data_storage = os.path.join(home, 'diedrichsen_data')
else:
    assert home == '/home/ROBARTS/agrilopi'
    data_storage = '/srv/diedrichsen'

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

    # Paths of dataframes
    db_taskglm_path = os.path.join(rsa_folder, 'rsa_taskglm_inputs.tsv')
    db_grandglm_path = os.path.join(rsa_folder, 'rsa_grandglm_inputs.tsv')
   
    # Create dataframes
    # db_taskglm = rsa_dataframe(
    #     SUBJECTS, tasks, data_storage, conditions_mapping, db_taskglm_path,
    #     glm_type='task_glm')
    # db_grandglm = rsa_dataframe(
    #     SUBJECTS, tasks, data_storage, conditions_mapping, db_grandglm_path,
    #     glm_type='grand_glm')

    # Prewhiten task glm beta maps and save them
    # db_taskglm = prewhiten_betas(
    #     db_taskglm_path, SUBJECTS, data_storage, db_taskglm_path)
    db_grandglm = prewhiten_betas(
        db_grandglm_path, SUBJECTS, data_storage, db_grandglm_path)

    # Prewhiten grand glm beta maps and save them

    # Note: The next steps rely on prewhiten_beta_maps that were normalized,
    #       smoothed and masked. These steps were done in MATLAB.

    # Extract signals from prewhitened data using the individualized ROIs
    # extract_roi_signals_from_dataframe(db_taskglm_path, SUBJECTS, roi_names,
    #                                    itags)
