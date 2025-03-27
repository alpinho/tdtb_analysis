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
import PcmPy as pcm

import matplotlib.pyplot as plt

from nitools import spm
from nilearn import image


# =========================== FUNCTIONS ================================

def create_rsa_dataframe(subjects, task_ids, derivatives_dir, cond_mapping,
                         output_path):
    """Builds the inputs DataFrame, saves it, and returns it."""
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


def prewhiten_beta_maps(df_input, subjects):
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

    # Process each row in the DataFrame
    for _, row in df.iterrows():
        beta_map_path = row['betamap_path']
        resms_path = row['resms_path']

        # Load beta map using nilearn and get data as a numpy array
        beta_img = image.load_img(beta_map_path)
        beta_data = beta_img.get_fdata()

        # Load ResMS map using nilearn and get data as a numpy array
        resms_img = image.load_img(resms_path)
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

        # Build new filename with the suffix '_desc-prewhitened'
        base, ext = os.path.splitext(beta_map_path)
        if ext == '.gz':
            base, ext2 = os.path.splitext(base)
            ext = ext2 + ext  # ext becomes '.nii.gz'
        new_fname = base + '_desc-prewhitened' + ext

        # Save the new prewhitened beta map using nilearn
        new_img.to_filename(new_fname)


def update_dataframe(df_input, subjects, output_path=None):
    """
    Loads the input DataFrame (or uses the provided DataFrame) and adds
    a new column 'swmasked_betamap_path'. The new path is constructed by
    replacing the original betamap_path's directory and filename with the
    masked derivatives version. For example, if the original betamap_path
    is:
    
        /home/analu/diedrichsen_data/data/Cerebellum/music-sdtb/
            derivatives/sub-03/estimates/prod/ffx_rwls_dbb_hrf128/
            beta_0001.nii

    the new swmasked_betamap_path will be:
    
        /home/analu/diedrichsen_data/data/Cerebellum/music-sdtb/
            derivatives/sub-03/estimates/prod/masked_derivatives/
            wbeta_0001_desc-prewhitened_desc-sm8wbmasked.nii

    Parameters
    ----------
    df_input : str or pd.DataFrame
        Either the path to the RSA input DataFrame (TSV file) or a
        DataFrame object.
    output_path : str, optional
        If provided, the updated DataFrame will be saved to this path
        (TSV format).

    Returns
    -------
    df : pd.DataFrame
        DataFrame with the new column 'swmasked_betamap_path'.
    """
    
    # Load the DataFrame if a path is provided
    if isinstance(df_input, str):
        df_unfiltered = pd.read_csv(df_input, sep='\t')
    elif isinstance(df_input, pd.DataFrame):
        df_unfiltered = df_input.copy(deep=True)
    else:
        raise ValueError("df_input must be a path or a pandas DataFrame.")

    # Filter the DataFrame according to subjects list
    df = df_unfiltered[df_unfiltered['subject'].isin(subjects)]

    # Create a new DataFrame by copying the old one
    new_df = df.copy(deep=True)
    
    swmasked_paths = []
    for idx, row in new_df.iterrows():
        betamap_path = row['betamap_path']
        # Assume the original path is of the form:
        # .../sub-XX/estimates/task_id/ffx_rwls_dbb_hrf128/beta_XXXX.nii
        # We want to change it to:
        # .../sub-XX/estimates/task_id/masked_derivatives/wbeta_XXXX_desc-prewhitened_desc-sm8wbmasked.nii

        # Get the folder of the betamap (i.e., the ffx folder)
        old_folder = os.path.dirname(betamap_path)
        # Get the parent folder (e.g., .../estimates/prod)
        base_folder = os.path.dirname(old_folder)
        # Construct new folder: masked_derivatives inside the estimates/task folder
        new_folder = os.path.join(base_folder, 'masked_derivatives')
        if not os.path.exists(new_folder):
            os.makedirs(new_folder, exist_ok=True)
        
        orig_fname = os.path.basename(betamap_path)
        # Process the filename: expect something like "beta_0001.nii"
        if orig_fname.startswith("beta_"):
            # Remove the "beta_" prefix and split extension
            number_part = orig_fname[5:]  # e.g., "0001.nii"
            number, ext = os.path.splitext(number_part)  # "0001", ".nii"
            new_fname = f"wbeta_{number}_desc-prewhitened_desc-sm8wbmasked{ext}"
        else:
            # Fallback: if file name does not start with "beta_"
            base_name, ext = os.path.splitext(orig_fname)
            new_fname = f"w{base_name}_desc-prewhitened_desc-sm8wbmasked{ext}"
        
        new_path = os.path.join(new_folder, new_fname)
        swmasked_paths.append(new_path)
    
    # Add the new column to the new DataFrame
    new_df['swmasked_betamap_path'] = swmasked_paths
    
    # Save the DataFrame if an output path is provided
    if output_path is not None:
        new_df.to_csv(output_path, index=False, sep='\t')
    
    return df


# =========================== INPUTS ===================================

# Subjects without pilot
SUBJECTS = [3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26, 28,
            29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
# SUBJECTS = [3]

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
    # db = create_rsa_dataframe(SUBJECTS, tasks, derivatives_folder,
    #                           conditions_mapping, db_path)

    # Prewhiten beta maps and save them
    # prewhiten_beta_maps(db_path, SUBJECTS)

    # Note: The next steps rely on prewhiten_beta_maps that were normalized,
    #       smoothed and masked. These steps were done in MATLAB.

    # Add paths of beta maps that were normalized, smoothed and masked...
    # ... to a new dataframe
    updated_db_path = os.path.join('results', 'rsa',
                                   'rsa_inputs_with_swmasked.tsv')
    _ = update_dataframe(db_path, SUBJECTS, output_path=updated_db_path)
