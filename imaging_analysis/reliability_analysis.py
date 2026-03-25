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
from nilearn import image
from nilearn.maskers import NiftiMasker

# Prevent DataFrame truncation when printing
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# =========================== FUNCTIONS ================================

def reliability_dataframe(subjects, task_models, base_dir, cond_mapping, 
                          output_path, prefix, masking):
    """Builds the inputs DataFrame, saves it, and returns it."""

    derivatives_dir = os.path.join(
        base_dir, 'data', 'Cerebellum', 'music-sdtb', 'derivatives')

    rows = []
    for subj in subjects:
        # Format subject with leading zero if needed, e.g., "sub-03"
        subj_str = f"sub-{subj:02d}"
        for model in task_models:

            if model != 'rand_ntfd':
                cond_mapping.pop('auditory_random', None)
                cond_mapping.pop('visual_random', None)

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
            rawdata_repeat = np.repeat(rawdata_unique, len(cond_mapping))

            # Filter beta_names to keep only encoding-related ones
            mask = np.char.find(beta_names, 'encoding') >= 0
            # Apply mask to get filtered beta names and run numbers
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
                if model != 'rand_ntfd':
                    match = re.search(r'task-(.*?)_run', rawdata_repeat[i])
                    task_id = match.group(1) if match else None
                else:
                    assert model == 'rand_ntfd'
                    task_id = model

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
                psc_fname = prefix + '_' + reg_number + '_desc-' + masking + \
                    'masked.nii'
                spsc_fname = prefix + '_' + reg_number + '_desc-' + 'sm8' + \
                    masking + 'masked.nii'

                pscmap_path = os.path.join(masked_derivatives_dir, psc_fname)
                spscmap_path = os.path.join(masked_derivatives_dir, 
                                            spsc_fname)

                # Convert full paths to paths relative to base_dir
                relative_betamap_path = os.path.relpath(
                    betamap_path, base_dir)
                relative_pscmap_path = os.path.relpath(
                    pscmap_path, base_dir)
                relative_spscmap_path = os.path.relpath(
                    spscmap_path, base_dir)
                
                rows.append({
                    'subject': subj,
                    'task_id': task_id,
                    'run_number': run_num,
                    'condition_type': cond,
                    'condition_name': condition_name,
                    'betamap_path': relative_betamap_path,
                    'wmasked_pscmap_path': relative_pscmap_path,
                    'swmasked_pscmap_path': relative_spscmap_path
                })

    # Create the DataFrame
    df = pd.DataFrame(rows)
    print(df)

    # Save the DataFrame in the rsa_folder
    df.to_csv(output_path, index=False, sep='\t')

    return df


def taskglm_roi_extraction(df_input, base_dir, task_models, subjects, tags, 
                           regions, atlases, rois, hems, iroi_mask_dir, 
                           thresh, smoothing):
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

    `condition_names` are ordered as it follows:
    ['abeat_prod', 'ainterval_prod', 
     'vbeat_prod', 'vinterval_prod', 
     'abeat_percep', 'ainterval_percep', 
     'vbeat_percep', 'vinterval_percep', 
     'abeat_ntfd', 'ainterval_ntfd', 
     'vbeat_ntfd', 'vinterval_ntfd']

    Missing data (e.g., missing beta map or mask) results in NaNs in
    the output array.

    Raises
    ------
    ValueError
        If `df_input` is not a string or pandas DataFrame.
    """

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
            condition_names = [
                c for c in df_unfiltered['condition_name'].unique()]
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
                    iroi_mask_dir, 'bothmod_allmain_tasks', 'main_tasks', 
                    region, atlas, 'individual_roi_masks',
                    (f'{tag}_sub-{subjects[0]:02d}_'
                     f'{roi}_{hems[0]}_mask.nii.gz')
                )
            else:
                first_mask_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    iroi_mask_dir, 'bothmod_allmain_tasks', 'main_tasks', 
                    region, atlas, roi, 'individual_roi_masks',
                    (f'{tag}_sub-{subjects[0]:02d}_'
                     f'{roi}_{hems[0]}_mask.nii.gz')
                )

            first_masker = NiftiMasker(
                mask_img=first_mask_path, standardize=False
            )
            first_df = df_unfiltered[df_unfiltered['subject'] == subjects[0]]
            first_img = image.load_img(
                os.path.join(base_dir, first_df.iloc[0][
                    'wmasked_pscmap_path'])
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
                                    iroi_mask_dir, 'bothmod_allmain_tasks', 
                                    'main_tasks', region, atlas,
                                    'individual_roi_masks',
                                    (f'{tag}_sub-{subject:02d}_'
                                     f'{roi}_{hem}_mask.nii.gz')
                                )
                            else:
                                iroi_mask_path = os.path.join(
                                    os.path.dirname(
                                        os.path.abspath(__file__)),
                                    iroi_mask_dir, 'bothmod_allmain_tasks', 
                                    'main_tasks', region, atlas, roi,
                                    'individual_roi_masks',
                                    (f'{tag}_sub-{subject:02d}_'
                                     f'{roi}_{hem}_mask.nii.gz')
                                )

                            masker = NiftiMasker(
                                mask_img=iroi_mask_path, standardize=False
                            )

                            # Find the derivative map row matching subject,
                            # condition, run
                            row_match = df_unfiltered[
                                (df_unfiltered['subject'] == subject)
                                &
                                (df_unfiltered['condition_name'] == condition)
                                &
                                (df_unfiltered['run_number'] == run)
                            ]
                            if len(row_match) != 1:
                                raise ValueError(
                                    f"Expected 1 matching row, "
                                    f"but found {len(row_match)} "
                                    f"for subject={subject}, "
                                    f"condition='{condition}', run={run}."
                                )
                            if smoothing == 'unsmoothed':
                                betamap_path = os.path.join(
                                    base_dir, 
                                    row_match.iloc[0][
                                        'wmasked_pscmap_path']
                                )
                            else:
                                assert smoothing == 'smoothed'
                                betamap_path = os.path.join(
                                    base_dir,
                                    row_match.iloc[0][
                                        'swmasked_pscmap_path']
                                )
                            betamap = image.load_img(betamap_path)
                            voxels = masker.fit_transform(betamap)
                            data_array[h, c, r, s, :] = voxels

            # Save array to disk
            output_folder = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'results', 'reliability', 
                f'taskglm_roi_signals_{thresh}_{smoothing}', region
            )
            os.makedirs(output_folder, exist_ok=True)

            output_path = os.path.join(
                output_folder,
                f'taskglm_roi_signals_{roi}_{tag}_{thresh}_{smoothing}.npy'
            )
            np.save(output_path, data_array)
            print(
                f"Saved 5D voxel array for tag '{tag}' to {output_path}"
            )


# =========================== INPUTS ===================================

# Subjects without pilot
SUBJECTS = [3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26, 28,
            29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
# SUBJECTS = [46]

# Parent directories
home = os.path.expanduser('~')

if os.path.isdir('/home/analu/diedrichsen_data'):
    data_storage = '/home/analu/diedrichsen_data'
else:
    assert os.path.isdir('/home/UWO/agrilopi')
    data_storage = '/cifs/diedrichsen'

# Define tasks in the glm
glm_tasks = ['prod', 'percep', 'ntfd', 'rand_ntfd']
# glm_tasks = ['rand_ntfd']

# Path for output folders
reliability_folder = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'results', 'reliability')

# Define mapping for condition type abbreviations
conditions_mapping = {
    'auditory_beat': 'abeat',
    'auditory_interval': 'ainterval',
    'auditory_random': 'arandom',
    'visual_beat': 'vbeat',
    'visual_interval': 'vinterval',
    'visual_random': 'vrandom'
}

# Threshold of group-level contrast ued to generate ROI
thresh_type = 'puncorr'  # 'puncorr' or 'pcorr'
# Derivatives used are smoothed or unsmoothed?
smooth = 'unsmoothed'  # 'smoothed'
# What type of map?
derivative_type = 'wpsc'
# Whole-brain (wb) mask or gray-matter (gm) mask?
mask_type = 'wb'

# ####################### ROIs ##############################
# atlas_names = ['hos', 
#                'ntk_symmni128', 
#                'hmat', 'hmat', 'hmat', 'hmat',
#                'hos',
#                'hos']
# region_names = ['dorsal_striatum', 
#                 'cerebellum', 
#                 'motor_area', 'motor_area', 'motor_area', 'motor_area',
#                 'heschl_gyrus',
#                 'occipital_lobe']
# roi_names = ['dstr', 
#              'cereb', 
#              'pmd', 'pmv', 'sma', 'presma',
#              'heschl',
#              'occipital']

atlas_names = ['hos']
region_names = ['dorsal_striatum']
roi_names = ['dstr']

# itags = ['i', 'i9a', 'i8a', 'i7a', 'i6a', 'a', 'a4g', 'a3g', 'a2g', 'a1g', 
#          'g']
# itags = ['i', 'g']
itags = ['i']

hemispheres = ['bh']  # Both hemispheres

iroi_main_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    f'roi_analyses_rwls_hrf128_wb_{thresh_type}_{smooth}')

# ============================ RUN =====================================

if __name__ == '__main__':
    # Create output folder if it does not exist
    os.makedirs(reliability_folder, exist_ok=True)

    # Paths of dataframes
    db_taskglm_path = os.path.join(reliability_folder,
                                   'reliability_taskglm.tsv')

    # Create dataframes
    # reliability_dataframe(SUBJECTS, glm_tasks, data_storage,
    #                       conditions_mapping, db_taskglm_path,
    #                       derivative_type, mask_type)

    # Open dataframes
    # db_taskglm = pd.read_csv(db_taskglm_path, sep='\t')

    # Extract signals from derivatives using individualized ROIs
    # Order of conditions: abeat_prod, ainterval_prod,
    #                      vbeat_prod, vinterval_prod,
    #                      abeat_percep, ainterval_percep,
    #                      vbeat_percep, vinterval_percep,
    #                      abeat_ntfd, ainterval_ntfd,
    #                      vbeat_ntfd, vinterval_ntfd
    taskglm_roi_extraction(db_taskglm_path, data_storage, glm_tasks,
                           SUBJECTS, itags, region_names, atlas_names,
                           roi_names, hemispheres, iroi_main_dir,
                           thresh_type, smooth)