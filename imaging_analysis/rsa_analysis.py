"""
Script to do the rsa analysis to calculate similarities of beat conditions
and interval conditions across tasks

Author: Ana Luisa Pinho
Email: agrilopi@uwo.ca

Creation: 26th of March 2025
Last Update: June 2025

Compatibility: Python 3.10.16
"""

import os
import re

import numpy as np
import pandas as pd
import PcmPy as pcm

import matplotlib.pyplot as plt

from nitools import spm
from nilearn import image
from nilearn.maskers import NiftiMasker


# =========================== FUNCTIONS ================================

def rsa_dataframe(subjects, task_models, base_dir, cond_mapping, output_path,
                  glm_type='grand_glm'):
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
    beta_paths = []
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
        beta_paths.append(new_rel_fname)

        # Create the relative path for the normalized, smoothed...
        # ... and masked pre-whiten beta maps
        swmasked_path = os.path.join(
            os.path.relpath(
                os.path.dirname(os.path.dirname(new_full_fname)), base_dir),
            'masked_derivatives_rwls_dbb_hrf128',
            'w'
            + os.path.basename(new_full_fname)[:-4]
            + '_desc-sm8wbmasked.nii',
        )

        # Append the relative path of the swmasked files
        swmasked_paths.append(swmasked_path)

    # Register the new paths in a new column
    df['betamap_prewhitened_path'] = beta_paths
    df['swmasked_betamap_prewhitened_path'] = swmasked_paths

    # Save the DataFrame in the rsa_folder
    df.to_csv(output_path, index=False, sep='\t')

    return df


def grandglm_roi_extraction(df_input, base_dir, task_models, subjects, tags, 
                            regions, atlases, rois, hems, iroi_mask_dir):
    """
    Extracts ROI signals for given subjects, tags, regions, and
    hemispheres, saving a 5D array per tag.

    The output array has shape:
        (hemispheres, conditions, runs, subjects, voxels)

    Parameters
    ----------
    df_input : str or pandas.DataFrame
        Path to a tab-separated input file or a DataFrame containing
        experimental data. Must include columns:
        ['subject', 'condition_name', 'run_number',
         'swmasked_betamap_prewhitened_path'].
    base_dir: str
        Path of home dir
    task_models : list of str
        List of task model names. 'allmain_tasks' will be excluded if
        present.
    subjects : list of int
        List of subject identifiers.
    tags : list of str
        List of tags for which separate output arrays will be saved.
    regions : list of str
        List of region names to process
        (matched with `atlases` and `rois`).
    atlases : list of str
        List of atlas names (matched with `regions` and `rois`).
    rois : list of str
        List of ROI names (matched with `regions` and `atlases`).
    hems : list of str
        List of hemisphere identifiers (e.g., ['lh', 'rh']).

    Notes
    -----
    For each combination of region, atlas, roi, and tag, the function:
        - Loads the corresponding ROI mask for each subject and
          hemisphere.
        - Extracts voxel values from the provided beta maps for each
          subject, condition, and run.
        - Saves the resulting array to disk as a .npy file under
          ./results/rsa/grandglm_roi_signals/<region>/
          grandglm_roi_signals_<roi>_<tag>.npy

    The axes of the output array are ordered as:
        (hemisphere, condition, run, subject, voxel)
    where 'voxel' is the number of voxels in the ROI mask.

    The extraction is performed in the order of `condition_names`,
    `run_numbers`, `subjects`, then `hems` as they first appear
    in the input data.

    Missing data (e.g., missing beta map or mask) results in NaNs in
    the output array.

    Raises
    ------
    ValueError
        If `df_input` is not a string or pandas DataFrame.
    """
    task_models = [t for t in task_models if t != 'allmain_tasks']

    if isinstance(df_input, str):
        df_unfiltered = pd.read_csv(df_input, sep='\t')
    elif isinstance(df_input, pd.DataFrame):
        df_unfiltered = df_input.copy()
    else:
        raise ValueError("df_input must be a path or a pandas DataFrame.")

    for region, atlas, roi in zip(regions, atlases, rois):
        for tag in tags:
            print(f"Processing tag: {tag}")

            # Preserve user-specified order
            condition_names = [c for c in df_unfiltered['condition_name'].unique()]
            run_numbers = [r for r in df_unfiltered['run_number'].unique()]
            # subjects and hems are already user-specified sequences

            n_hems = len(hems)
            n_conditions = len(condition_names)
            n_runs = len(run_numbers)
            n_subjects = len(subjects)

            # Placeholder to get voxel dimensionality
            if region == 'dorsal_striatum':
                first_mask_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    iroi_mask_dir, 'all', region, atlas, 
                    'individual_roi_masks',
                    (f'{tag}_sub-{subjects[0]:02d}_'
                    f'{roi}_{hems[0]}_mask.nii.gz')
                )
            else:
                first_mask_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    iroi_mask_dir, 'all', region, atlas, roi,
                    'individual_roi_masks',
                    (f'{tag}_sub-{subjects[0]:02d}_'
                     f'{roi}_{hems[0]}_mask.nii.gz')
                )

            first_masker = NiftiMasker(
                mask_img=first_mask_path, standardize=False
            )
            first_df = df_unfiltered[df_unfiltered['subject'] == subjects[0]]
            first_img = image.load_img(
                os.path.join(base_dir, first_df.iloc[0][
                    'swmasked_betamap_prewhitened_path'])
            )
            example_voxels = first_masker.fit_transform(first_img)
            voxel_dim = example_voxels.shape[1]

            # Initialize the 5D array
            # Shape = (hemispheres, conditions, runs, subjects, voxels)
            data_array = np.full(
                (n_hems, n_conditions, n_runs, n_subjects, voxel_dim),
                np.nan
            )

            # Loop over all combinations according to the user-specified order
            for h, hem in enumerate(hems):
                for c, condition in enumerate(condition_names):
                    for r, run in enumerate(run_numbers):
                        for s, subject in enumerate(subjects):
                            if region == 'dorsal_striatum':
                                iroi_mask_path = os.path.join(
                                    os.path.dirname(
                                        os.path.abspath(__file__)),
                                    iroi_mask_dir, 'all', region, atlas,
                                    'individual_roi_masks',
                                    (f'{tag}_sub-{subject:02d}_'
                                    f'{roi}_{hem}_mask.nii.gz')
                                )
                            else:
                                iroi_mask_path = os.path.join(
                                    os.path.dirname(
                                        os.path.abspath(__file__)),
                                    iroi_mask_dir, 'all', region, atlas, roi,
                                    'individual_roi_masks',
                                    (f'{tag}_sub-{subject:02d}_'
                                     f'{roi}_{hem}_mask.nii.gz')
                                )

                            masker = NiftiMasker(
                                mask_img=iroi_mask_path, standardize=False
                            )

                            # Find the beta map row matching subject,
                            # condition, run
                            row_match = df_unfiltered[
                                (df_unfiltered['subject'] == subject) &
                                (df_unfiltered['condition_name'] == condition) &
                                (df_unfiltered['run_number'] == run)
                            ]
                            if row_match.empty:
                                continue
                            betamap_path = os.path.join(
                                base_dir, 
                                row_match.iloc[0][
                                    'swmasked_betamap_prewhitened_path']
                            )
                            betamap = image.load_img(betamap_path)
                            voxels = masker.fit_transform(betamap)
                            data_array[h, c, r, s, :] = voxels

            # Save array to disk
            output_folder = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'results', 'rsa', 'grandglm_roi_signals', region
            )
            os.makedirs(output_folder, exist_ok=True)

            output_path = os.path.join(
                output_folder,
                f'grandglm_roi_signals_{roi}_{tag}.npy'
            )
            np.save(output_path, data_array)
            print(
                f"Saved 5D voxel array for tag '{tag}' to {output_path}"
            )


def compute_euclidean_distances(Y, tasks_list, conditions_list,
                                hem_index=None):
    """
    Compute cross-validated Euclidean distances between conditions
    for all subjects in the dataset.

    Parameters
    ----------
    Y : np.ndarray
        5D array with shape 
        (hemisphere, condition, run, subject, voxel).
    tasks_list : list of str
        List of task labels (e.g., ['prod', 'percep', 'ntfd']).
    conditions_list : dict
        Dictionary mapping full condition names to abbreviations
        (e.g., {'auditory_beat': 'abeat'}).
    hem_index : int or None, optional
        If specified, selects the hemisphere index along the first axis.
        If None, the first hemisphere (index 0) is used.

    Returns
    -------
    distances : np.ndarray
        Array of Euclidean distance matrices with shape
        (n_subjects, n_conditions, n_conditions).
    """
    condition_labels = [
        f"{abbr}_{task}"
        for abbr in conditions_list.values()
        for task in tasks_list
    ]

    n_subjects = Y.shape[3]
    n_conditions = len(condition_labels)
    n_runs = Y.shape[2]
    n_voxels = Y.shape[-1]

    cond_vec = np.tile(condition_labels, n_runs)
    part_vec = np.repeat(np.arange(1, n_runs + 1), n_conditions)

    # Select hemisphere
    Y_hem = Y[hem_index] if hem_index is not None else Y[0]

    # Reorder axes to (subject, run, condition, voxel)
    Y_reordered = np.swapaxes(Y_hem, 0, 2)

    # Reshape to (subject, run * condition, voxel)
    Y_reshaped = Y_reordered.reshape(n_subjects,
                                     n_runs * n_conditions,
                                     n_voxels)

    # Compute distance matrix for each subject
    distances = np.empty((n_subjects, n_conditions, n_conditions))
    for i, subj_data in enumerate(Y_reshaped):
        G_cv, _ = pcm.est_G_crossval(subj_data, cond_vec, part_vec)
        distances[i] = pcm.G_to_dist(G_cv)

    return distances


# =========================== INPUTS ===================================

# Subjects without pilot
SUBJECTS = [3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26, 28,
            29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
# SUBJECTS = [3]

# Path for output folders
rsa_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'results', 'rsa')

# ########################### ROIs ######################################

iroi_main_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'roi_analyses_rwls_hrf128_wb_puncorr')

# All ROIs: 7 ROIs
# region_names = ['dorsal_striatum', 'cerebellum', 'cerebellum', 'cerebellum',
#                 'motor_area', 'motor_area', 'motor_area']
# atlas_names = ['hos', 'ntk_symmni128', 'ntk_symmni128', 'ntk_symmni128',
#                'hmat', 'hmat', 'hmat']
# roi_names = ['dstr', 'cereb-s', 'cereb-i', 'cereb',
#              'pmd', 'sma', 'presma']

region_names = ['dorsal_striatum', 'cerebellum', 'motor_area', 'motor_area',
                'motor_area']
atlas_names = ['hos', 'ntk_symmni128', 'hmat', 'hmat', 'hmat']
roi_names = ['dstr', 'cereb', 'pmd', 'sma', 'presma']

# itags = ['i', 'i9a', 'i8a', 'i7a', 'i6a', 'a', 'a4g', 'a3g', 'a2g', 'a1g', 'g']
itags = ['i', 'i8a']

hemispheres = ['bh']  # Both hemispheres

thresh_type = 'puncorr'  # 'puncorr' or 'pcorr'

# #######################################################################

# Parent directories
home = os.path.expanduser('~')

if os.path.isdir('/home/analu/diedrichsen_data'):
    data_storage = '/home/analu/diedrichsen_data'
else:
    assert os.path.isdir('/home/UWO/agrilopi')
    data_storage = '/cifs/diedrichsen'

# Define tasks in the glm
glm_tasks = ['prod', 'percep', 'ntfd', 'allmain_tasks']

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
    # db_taskglm_path = os.path.join(rsa_folder, 'rsa_taskglm.tsv')
    # db_grandglm_path = os.path.join(rsa_folder, 'rsa_grandglm.tsv')
   
    # # Create dataframes
    # db_taskglm = rsa_dataframe(
    #     SUBJECTS, glm_tasks, data_storage, conditions_mapping,
    #     db_taskglm_path, glm_type='task_glm')
    # db_grandglm = rsa_dataframe(
    #     SUBJECTS, glm_tasks, data_storage, conditions_mapping,
    #     db_grandglm_path, glm_type='grand_glm')

    # # Prewhiten task glm beta maps and save them
    # db_taskglm = prewhiten_betas(
    #     db_taskglm_path, SUBJECTS, data_storage, db_taskglm_path)

    # # Prewhiten grand glm beta maps and save them
    # db_grandglm = prewhiten_betas(
    #     db_grandglm_path, SUBJECTS, data_storage, db_grandglm_path)

    # ##################################################################
    # Note: The next steps rely on prewhiten_beta_maps that were normalized,
    #       smoothed and masked. These steps were done in MATLAB.
    # ##################################################################

    # Open dataframes
    # db_taskglm = pd.read_csv(db_taskglm_path)
    # db_grandglm = pd.read_csv(db_grandglm_path)

    # Extract signals from prewhitened data using the individualized ROIs
    # Order of conditions: abeat_prod, ainterval_prod,
    #                      vbeat_prod, vinterval_prod, 
    #                      abeat_percep, ainterval_percep,
    #                      vbeat_percep, vinterval_percep, 
    #                      abeat_ntfd, ainterval_ntfd,
    #                      vbeat_ntfd, vinterval_ntfd
    # grandglm_roi_extraction(db_grandglm_path, data_storage, glm_tasks,
    #                         SUBJECTS, itags, region_names, atlas_names,
    #                         roi_names, hemispheres, iroi_main_dir)

    # Compute RSA within a region
    for itag in itags:
        print(f"Processing tag: {itag}")

        for region_name, roi_name in zip(region_names, roi_names):
            print(f"Processing ROI: {roi_name}")

            # Load the ROI signals for the current tag and ROI
            roi_signals_path = os.path.join(
                rsa_folder, 'grandglm_roi_signals', region_name,
                f'grandglm_roi_signals_{roi_name}_{itag}_{thresh_type}.npy'
            )
            if not os.path.exists(roi_signals_path):
                print(f"Skipping {roi_name} for tag {itag}: file not found.")
                continue

            roi_signals = np.load(roi_signals_path)

            # Compute Euclidean distances for the current ROI
            eucl_distances = compute_euclidean_distances(
                roi_signals, glm_tasks[:3], conditions_mapping)

            # Create output folder if it does not exist
            output_dir = os.path.join(rsa_folder, 'euclidean_distances')
            os.makedirs(output_dir, exist_ok=True)

            # Save the distances to a .npy file
            output_path = os.path.join(
                output_dir,
                f'eucl_distances_{roi_name}_{itag}_{thresh_type}.npy'
            )
            if os.path.exists(output_path):
                os.remove(output_path)
            np.save(output_path, eucl_distances)
            print(f"Saved distances to {output_path}")