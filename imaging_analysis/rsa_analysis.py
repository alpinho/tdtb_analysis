"""
Script to do the rsa analysis to calculate similarities of beat conditions
and interval conditions across tasks

Author: Ana Luisa Pinho
Email: agrilopi@uwo.ca

Creation: 26th of March 2025
Last Update: March 2026

Compatibility: Python 3.10.16
"""

import os
import re

import numpy as np
import pandas as pd
import PcmPy as pcm

from scipy.stats import ttest_1samp

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import seaborn as sns

from nitools import spm
from nilearn import image
from nilearn.maskers import NiftiMasker

# Prevent DataFrame truncation when printing
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


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
                if subj == 14:
                    run_numbers = np.repeat(
                        [1, 2, 1, 2, 1, 2, 3, 3, 4, 3, 4, 4], 4)
                elif subj == 42:
                    run_numbers = np.repeat(
                        [1, 1, 2, 1, 2, 2, 3, 4, 3, 4, 3, 4], 4)
                elif subj == 43:
                    run_numbers = np.repeat(
                        [1, 2, 1, 2, 1, 2, 3, 3, 4, 3, 4, 4], 4)
                elif subj == 46:
                    run_numbers = np.repeat(
                        [1, 2, 1, 2, 1, 2, 3, 3, 3, 4, 4, 4], 4)
                else:
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
            # Apply mask to get filtered beta names and run numbers
            filtered_beta_names = beta_names[mask]

            if glm_type == 'task_glm':
                run_numbers = run_numbers[mask]

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


def prewhiten_betas(df_input, subjects, base_dir, output_path, 
                    prewhiten=True):
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
    wmasked_paths = []
    swmasked_paths = []

    # Process each row in the DataFrame
    for _, row in df.iterrows():
        # The paths stored in the DataFrame are relative to base_dir
        rel_beta_map_path = row['betamap_path']
        rel_resms_path = row['resms_path']

        # Build full paths for opening the files
        full_beta_map_path = os.path.join(base_dir, rel_beta_map_path)
        full_resms_path = os.path.join(base_dir, rel_resms_path)

        if prewhiten:
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

        if prewhiten:
            # Save the new prewhitened beta map using nilearn
            new_img.to_filename(new_full_fname)

        # Convert the new full filename back to a relative path...
        # ... (relative to base_dir)
        new_rel_fname = os.path.relpath(new_full_fname, base_dir)

        # Append the relative path to the list
        beta_paths.append(new_rel_fname)

        # Create the relative path for the normalized, (smoothed)...
        # ... and masked pre-whiten beta maps
        wmasked_path = os.path.join(
            os.path.relpath(
                os.path.dirname(os.path.dirname(new_full_fname)), base_dir),
            'masked_derivatives_rwls_dbb_hrf128',
            'w'
            + os.path.basename(new_full_fname)[:-4]
            + '_desc-wbmasked.nii',
        )
        swmasked_path = os.path.join(
            os.path.relpath(
                os.path.dirname(os.path.dirname(new_full_fname)), base_dir),
            'masked_derivatives_rwls_dbb_hrf128',
            'w'
            + os.path.basename(new_full_fname)[:-4]
            + '_desc-sm8wbmasked.nii',
        )

        # Append the relative path of the wmasked and swmasked files
        wmasked_paths.append(wmasked_path)
        swmasked_paths.append(swmasked_path)

    # Register the new paths in a new column
    df['betamap_prewhitened_path'] = beta_paths
    df['wmasked_betamap_prewhitened_path'] = wmasked_paths
    df['swmasked_betamap_prewhitened_path'] = swmasked_paths

    # Save the DataFrame in the rsa_folder
    df.to_csv(output_path, index=False, sep='\t')

    return df


def grandglm_roi_extraction(df_input, base_dir, task_models, subjects, tags, 
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
                                        'wmasked_betamap_prewhitened_path']
                                )
                            else:
                                assert smoothing == 'smoothed'
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
                'results', 'rsa', 
                f'grandglm_roi_signals_{thresh}_{smoothing}', region
            )
            os.makedirs(output_folder, exist_ok=True)

            output_path = os.path.join(
                output_folder,
                f'grandglm_roi_signals_{roi}_{tag}_{thresh}_{smoothing}.npy'
            )
            np.save(output_path, data_array)
            print(
                f"Saved 5D voxel array for tag '{tag}' to {output_path}"
            )


def compute_euclidean_distances(Y, tasks, conditions,
                                hem_index=None, mean_center=True):
    """
    Compute cross-validated Euclidean distances between conditions
    for all subjects in the dataset.

    Parameters
    ----------
    Y : np.ndarray
        5D array with shape 
        (hemisphere, condition, run, subject, voxel).
    tasks : list of str
        List of task labels (e.g., ['prod', 'percep', 'ntfd']).
    conditions : dict
        Dictionary mapping full condition names to abbreviations
        (e.g., {'auditory_beat': 'abeat'}).
    hem_index : int or None, optional
        If specified, selects the hemisphere index along the first axis.
        If None, the first hemisphere (index 0) is used.
    mean_center : bool, optional
        Whether to mean-center each run (across conditions) per voxel.

    Returns
    -------
    distances : np.ndarray
        Array of Euclidean distance matrices with shape
        (n_subjects, n_conditions, n_conditions).
    """

    # Select hemisphere
    Y_hem = Y[hem_index] if hem_index is not None else Y[0]

    # Filter conditions based on the specified task
    if tasks == ['prod']:
        Y_hem = Y_hem[:4]  # First 4 conditions
    elif tasks == ['percep']:
        Y_hem = Y_hem[4:8]  # Next 4 conditions
    elif tasks == ['ntfd']:
        Y_hem = Y_hem[8:]  # Last 4 conditions
    else:
        assert tasks == ['prod', 'percep', 'ntfd']
        pass

    # new (tasks X conds)
    # Order of the condition_labels should be:
    #  ['abeat_prod', 'ainterval_prod', 
    #  'vbeat_prod', 'vinterval_prod', 
    #  'abeat_percep', 'ainterval_percep', 
    #  'vbeat_percep', 'vinterval_percep', 
    #  'abeat_ntfd', 'ainterval_ntfd', 
    #  'vbeat_ntfd', 'vinterval_ntfd']
    condition_labels = [
        f"{abbr}_{task}"
        for task in tasks
        for abbr in conditions.values()
    ]

    n_subjects = Y.shape[3]
    n_conditions = len(condition_labels)
    n_runs = Y.shape[2]
    n_voxels = Y.shape[-1]

    cond_vec_alpha = np.tile(condition_labels, n_runs)
    label_to_index = \
        {label: idx + 1 for idx, label in enumerate(condition_labels)}
    cond_vec_numeric = np.array([
        label_to_index[label] for label in cond_vec_alpha])

    part_vec = np.repeat(np.arange(1, n_runs + 1), n_conditions)

    # Reorder axes from (condition, run, subject, voxel) to ...
    # # ... (subject, run, condition, voxel)
    Y_reordered = np.swapaxes(Y_hem, 0, 2)

    # Reshape to (subject, run * condition, voxel)
    Y_reshaped = Y_reordered.reshape(n_subjects,
                                     n_runs * n_conditions,
                                     n_voxels)

    # Mean-center per run if specified
    # Note: this might be redundant when computing afterwards the 
    #       crossvalidated second-moment estimation, because it removes
    #       run-specific means via the crossvalidation scheme.
    #       According to Diedrichsen & Kriegeskorte (2017):
    #       "Crossvalidated estimates of the second-moment matrix are 
    #        insensitive to additive run-specific components, as long 
    #        as those components are consistent across conditions 
    #        within a run."
    if mean_center:
        for s in range(n_subjects):
            for run in range(n_runs):
                start = run * n_conditions
                end = (run + 1) * n_conditions
                run_data = Y_reshaped[s, start:end, :]  # shape (12, n_voxels)
                run_mean = np.mean(run_data, axis=0, keepdims=True)
                Y_reshaped[s, start:end, :] -= run_mean

    # Compute distance matrix for each subject
    distances = np.empty((n_subjects, n_conditions, n_conditions))
    for s, subj_data in enumerate(Y_reshaped):
        G_cv, _ = pcm.est_G_crossval(subj_data, cond_vec_numeric, part_vec)
        distances[s] = pcm.G_to_dist(G_cv)

    return distances


def plot_rdms(i_dist, g_dist, subjects, tasks, conditions,
              output_folder, roi_label, i_label, thresh_label, smooth_label,
              color_scheme='PiYG', rescale=True, 
              truncate_to_zero=True):
    """
    Plot the individual and group Euclidean distance matrices.

    Parameters
    ----------
    i_dist : np.ndarray
        Individual distance matrices with shape 
        (n_subjects, n_conditions, n_conditions).
    g_dist : np.ndarray
        Group distance matrix with shape (n_conditions, n_conditions).
    tasks : list of str
        List of task labels (e.g., ['prod', 'percep', 'ntfd']).
    conditions : dict
        Dictionary mapping full condition names to abbreviations
        (e.g., {'auditory_beat': 'abeat'}).
    output_folder : str
        Folder to save the plot.
    roi_label : str
        ROI label for the filename.
    i_label : str
        Tag label for the filename.
    thresh_label : str
        Threshold type for the filename.
    smooth_label : str
        Smooth type for the filename.
    """ 

    # new (tasks X conds)
    # Order of the condition_labels should be:
    #  ['abeat_prod', 'ainterval_prod', 
    #  'vbeat_prod', 'vinterval_prod', 
    #  'abeat_percep', 'ainterval_percep', 
    #  'vbeat_percep', 'vinterval_percep', 
    #  'abeat_ntfd', 'ainterval_ntfd', 
    #  'vbeat_ntfd', 'vinterval_ntfd']
    condition_labels = [
        f"{abbr}_{task}"
        for task in tasks
        for abbr in conditions.values()
    ]
    
    # Rescale
    if rescale:
        i_dist = np.sign(i_dist) * np.sqrt(np.abs(i_dist))
        g_dist = np.sign(g_dist) * np.sqrt(np.abs(g_dist))

    # Compute global min/max across all individual and group distances
    all_vals = np.concatenate([i_dist.flatten(), g_dist.flatten()])
    abs_max = np.max(np.abs(all_vals))

    # Define colormap
    if truncate_to_zero:
        vmin, vmax = 0, abs_max
        # Use the top half of 'PiYG' (white to green)
        orig_cmap = cm.get_cmap(color_scheme)
        new_colors = orig_cmap(np.linspace(0.5, 1.0, 256))  # Top half
        cmap = ListedColormap(new_colors)
    else:
        vmin, vmax = -abs_max, abs_max
        cmap = color_scheme

    n_cols = 4
    n_rows = int(np.ceil((len(subjects) + 1) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for s, subject in enumerate(subjects):
        ax = axes[s]
        i_matrix = i_dist[s].copy()
        if truncate_to_zero:
            i_matrix[i_matrix < 0] = 0
        im = ax.imshow(i_matrix, cmap=cmap, aspect='equal', vmin=vmin,
                       vmax=vmax)
        ax.set_title(f'Subject {subject}')
        ax.set_xticks(np.arange(len(condition_labels)))
        ax.set_xticklabels(condition_labels, rotation=45, ha='right', 
                           fontsize=8)
        ax.set_yticks(np.arange(len(condition_labels)))
        ax.set_yticklabels(condition_labels, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Group RDM in the last subplot
    ax = axes[len(subjects)]
    g_matrix = g_dist.copy()
    if truncate_to_zero:
        g_matrix[g_matrix < 0] = 0
    im = ax.imshow(g_matrix, cmap=cmap, aspect='equal', vmin=vmin,
                   vmax=vmax)
    ax.set_title('Group')
    ax.set_xticks(np.arange(len(condition_labels)))
    ax.set_xticklabels(condition_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(np.arange(len(condition_labels)))
    ax.set_yticklabels(condition_labels, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused axes
    for j in range(len(subjects) + 1, len(axes)):
        axes[j].axis('off')

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    if len(tasks) > 1:
        task_tag = 'all'
    else:
        task_tag = tasks[0]
    plt.savefig(
        os.path.join(
            output_folder,
            f'eucl_distances_{roi_label}_{i_label}_{thresh_label}_'
            f'{smooth_label}_{task_tag}.png'
        ),
        dpi=300
    )

    # To show the plot, uncomment the next line:
    # plt.show()

    plt.close()


def rsa(tags, tasks, regions, rois, rsa_dir, thresh_label, smooth_label, 
        conditions, truncate_to_zero=True):
    for tag in tags:
        print(f"Processing tag: {tag}")

        if len(tasks) > 1:
            task_tag = 'all'
        else:
            task_tag = tasks[0]

        for region, roi in zip(regions, rois):
            print(f"Processing ROI: {roi}")

            # Load the ROI signals for the current tag and ROI
            roi_signals_path = os.path.join(
                rsa_dir, f'grandglm_roi_signals_{thresh_label}_{smooth_label}',
                region, 
                f'grandglm_roi_signals_{roi}_{tag}_{thresh_label}_'
                f'{smooth_label}.npy'
            )
            if not os.path.exists(roi_signals_path):
                print(f"Skipping {roi} for tag {tag}: file not found.")
                continue

            roi_signals = np.load(roi_signals_path)

            # Compute Euclidean distances for the current ROI
            individual_eucl_distances = compute_euclidean_distances(
                roi_signals, tasks, conditions)

            # Compute the mean of the distances across subjects
            group_eucl_distances = np.mean(individual_eucl_distances, axis=0)

            # Create output folder if it does not exist
            output_dir = os.path.join(
                rsa_folder, 
                f'euclidean_distances_{thresh_label}_{smooth_label}')
            os.makedirs(output_dir, exist_ok=True)

            # Save the distances to a .npy file
            individual_output_path = os.path.join(
                output_dir,
                f'individual_eucl_distances_{roi}_{tag}_{thresh_label}_'
                f'{smooth_label}_{task_tag}.npy'
            )
            group_output_path = os.path.join(
                output_dir,
                f'group_eucl_distances_{roi}_{tag}_{thresh_label}_'
                f'{smooth_label}_{task_tag}.npy'
            )
            if os.path.exists(individual_output_path):
                os.remove(individual_output_path)
            if os.path.exists(group_output_path):
                os.remove(group_output_path)
            np.save(individual_output_path, individual_eucl_distances)
            np.save(group_output_path, group_eucl_distances)
            print(
                f"Saved distances to {individual_output_path} "
                f"and {group_output_path}"
            )

            # Plot the RDMs
            plot_rdms(
                individual_eucl_distances, group_eucl_distances, SUBJECTS,
                tasks, conditions_mapping, output_dir, roi, tag, 
                thresh_type, smooth, truncate_to_zero=truncate_to_zero)
            print(f"Save rdm plots for {roi} for tag {tag}.")

    # Print completion message
    print("RSA analysis completed for all ROIs and tags.")


def periodicity_significance(output_dir, tasks, conditions, tags, regions, 
                             rois, rsa_dir, thresh_label, smooth_label, 
                             modalities, modality_dic):
    """    
    Compute the significance of periodicity in RDMs for each ROI and 
    tag. This function computes the significance of periodicity in RDMs
    for each ROI and tag by performing a one-sample t-test across 
    subjects. It filters the RDMs based on task and modality, computes 
    the mean value for each subject, and saves the results in a TSV 
    file.

    Parameters
    ----------
    output_dir : str
        Directory where the output files will be saved.
    tasks : list of str
        List of task names (e.g., ['prod', 'percep', 'ntfd']).
    conditions : dict
        Dictionary mapping full condition names to abbreviations
        (e.g., {'auditory_beat': 'abeat'}).
    tags : list of str
        List of tags for which the significance will be computed.
    regions : list of str
        List of region names to process (matched with `rois`).
    rois : list of str
        List of ROI names to process (matched with `regions`).
    rsa_dir : str
        Directory containing the RSA results.
    thresh_label : str
        Threshold type for the filename (e.g., 'thresh05').
    smooth_label : str
        Smooth type for the filename (e.g., 'sm8').
    modalities : list of str
        List of modalities to consider 
        (e.g., ['audio', 'visual', 'both']).
    modality_dic : dict
        Dictionary mapping modality names to their tags (e.g.,
        {'audio': 'a', 'visual': 'v', 'both': 'both'}).
    """
    
    # Create output folder if it does not exist
    output_dir = os.path.join(
        rsa_folder, f'periodicity_significance_{thresh_label}_{smooth_label}')
    os.makedirs(output_dir, exist_ok=True)

    # Only the three tasks (skip 'allmain_tasks')
    valid_tasks = [t for t in tasks if t != 'allmain_tasks']
    # Only audio & visual (skip 'both')
    valid_modalities = [m for m in modalities if m != 'both']
    task_order = valid_tasks
    modality_order = valid_modalities
    box_width = .4

    for tag in tags:
        all_results = []

        for region, roi in zip(regions, rois):
            tasks_results = []

            for task in valid_tasks:
                task_tag = task

                # load precomputed distances
                dist_dir = os.path.join(
                    rsa_dir,
                    f'euclidean_distances_{thresh_label}_{smooth_label}'
                )
                id_path = os.path.join(
                    dist_dir,
                    f'individual_eucl_distances_{roi}_{tag}_'
                    f'{thresh_label}_{smooth_label}_{task_tag}.npy'
                )
                if not os.path.exists(id_path):
                    print(f"Skipping {roi}/{tag}/{task}: missing {id_path}")
                    continue
                idissim = np.load(id_path)

                # build beat-vs-interval pairs for this task
                pairs = [
                    (f"abeat_{task}", f"ainterval_{task}"),
                    (f"vbeat_{task}", f"vinterval_{task}")
                ]

                # extract values per subject & pair
                records = []
                for subj in range(idissim.shape[0]):
                    mat = idissim[subj]
                    for l1, l2 in pairs:
                        # find indices
                        labels = [
                            f"{abbr}_{task}" for abbr in conditions.values()]
                        i, j = labels.index(l1), labels.index(l2)
                        val = mat[i, j] if i > j else mat[j, i]
                        records.append({
                            'subject': subj+1,
                            'label1': l1,
                            'label2': l2,
                            'value': val
                        })
                df = pd.DataFrame(records)

                # test each modality
                for modality in valid_modalities:
                    tag_m = modality_dic[modality]
                    df_mod = df[
                        df['label1'].str.startswith(tag_m) &
                        df['label2'].str.startswith(tag_m)
                    ]
                    if df_mod.empty:
                        continue

                    mp = df_mod.groupby('subject')['value']\
                               .mean().reset_index()
                    if len(mp) > 1:
                        t_stat, p_val = ttest_1samp(mp['value'], popmean=0)
                    else:
                        t_stat, p_val = np.nan, np.nan

                    for _, r in mp.iterrows():
                        tasks_results.append({
                            'task': task_tag,
                            'modality': modality,
                            'subject': r['subject'],
                            'mean_value': r['value'],
                            't_stat': t_stat,
                            'p_val': p_val
                        })

            df_res = pd.DataFrame(tasks_results)
            df_res['roi'] = roi
            df_res['region'] = region
            # save per-ROI TSV into output_dir
            tsv = os.path.join(
                output_dir,
                f'periodicity_significance_{roi}_{tag}_'
                f'{thresh_label}_{smooth_label}.tsv'
            )
            df_res.to_csv(tsv, sep='\t', index=False)
            all_results.append(df_res)

        # plotting
        df_all = pd.concat(all_results, ignore_index=True)
        fig, axes = plt.subplots(
            len(rois), 1, figsize=(24, 5 * len(rois)), sharex=True
        )
        if len(rois) == 1:
            axes = [axes]

        for ax, roi in zip(axes, rois):
            d = df_all[df_all['roi'] == roi]
            sns.boxplot(
                data=d, x='task', y='mean_value', hue='modality',
                order=task_order, hue_order=modality_order,
                width=box_width, notch=True, showfliers=False,
                meanline=True, showmeans=True,
                medianprops={"color": "k", "linewidth": 0},
                meanprops={"color": "tab:brown", "linewidth": 1.5},
                boxprops={"alpha": 0.5, "edgecolor": "black"},
                ax=ax
            )
            ax.margins(x=0.15)

            # compute y-limits
            y_vals = []
            for t in task_order:
                for m in modality_order:
                    sub = d[(d['task'] == t) & (d['modality'] == m)]
                    if sub.empty:
                        continue
                    q1, q3 = sub['mean_value'].quantile([0.25, 0.75])
                    y_vals.append(q3 + 1.5*(q3-q1))
            top = max(y_vals) if y_vals else 0
            bot = min(d['mean_value'].min(), 0)
            margin = (top-bot)*0.2 if top != bot else 0.05
            ax.set_ylim(bot-margin, top+margin)

            # annotations
            offsets = np.linspace(-box_width/4, box_width/4,
                                  len(valid_modalities))
            for i, t in enumerate(task_order):
                for j, m in enumerate(modality_order):
                    sub = d[(d['task'] == t) & (d['modality'] == m)]
                    if sub.empty:
                        continue
                    p = sub['p_val'].iloc[0]
                    sig = (
                        '****' if p < 1e-4 else
                        '***' if p < 1e-3 else
                        '**' if p < 1e-2 else
                        '*' if p < 0.05 else
                        'n.s.'
                    )
                    x = i + offsets[j]
                    ax.text(x, top, sig, ha='center', va='bottom',
                            fontsize=10, weight='bold')

            if ax is not axes[-1]:
                ax.set_xticks([])
                ax.spines['bottom'].set_visible(False)
                # Remove the x-ticks
                ax.tick_params(axis='x', which='both', bottom=False,
                               top=False)
            else:
                ax.set_xlabel('Task')
                ax.set_xticks(range(len(task_order)))
                ax.set_xticklabels(task_order)

            ax.set_title(roi)
            ax.set_ylabel('Mean Dissimilarity')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend_.remove()

        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, title='Modality',
                   loc='upper right', bbox_to_anchor=(.95, 1))
        plt.tight_layout(rect=[.05, .05, .9, .95])

        png = os.path.join(
            output_dir,
            f'periodicity_significance_allrois_{tag}_'
            f'{thresh_label}_{smooth_label}.png'
        )
        plt.savefig(png, dpi=300)
        plt.close()
        print(f"Saved plot: {png}")


def encoding_significance(output_dir, conditions, tags,
                          regions, rois, rsa_dir,
                          thresh_label, smooth_label,
                          modalities):
    """
    Compute significance for encoding: comparisons between task pairs
    (prod vs percep, prod vs ntfd, percep vs ntfd), combining both
    beat and interval distances, stratified by modality.
    Saves per-ROI TSVs and a combined PNG per tag.

    Parameters
    ----------
    output_dir : str
        Directory where output TSVs and plots will be saved.
    conditions : dict
        Mapping of full condition names to abbreviations
        (e.g., {'auditory_beat':'abeat'}).
    tags : list of str
        List of tags identifying different ROI extractions.
    regions : list of str
        Region names corresponding to each ROI.
    rois : list of str
        ROI identifiers matching `regions`.
    rsa_dir : str
        Directory containing the precomputed distance matrices.
    thresh_label : str
        Threshold label used in filenames (e.g., 'puncorr').
    smooth_label : str
        Smoothing label used in filenames (e.g., 'unsmoothed').
    modalities : list of str
        Modalities to include ('audio', 'visual').
    """
    # Setup output folder
    out_folder = os.path.join(
        output_dir,
        f'encoding_significance_{thresh_label}_{smooth_label}'
    )
    os.makedirs(out_folder, exist_ok=True)

    # Define the three task-pairs
    task_pairs = [('prod', 'percep'), ('prod', 'ntfd'), ('percep', 'ntfd')]
    pair_labels = [f"{a}_vs_{b}" for a, b in task_pairs]
    box_width = 0.5

    # Only audio & visual (drop 'both')
    valid_modalities = [m for m in modalities if m != 'both']

    for tag in tags:
        all_roi_dfs = []

        for region, roi in zip(regions, rois):
            records = []
            dist_dir = os.path.join(
                rsa_dir,
                f'euclidean_distances_{thresh_label}_{smooth_label}'
            )
            fname = (
                f'individual_eucl_distances_{roi}_{tag}_'
                f'{thresh_label}_{smooth_label}_all.npy'
            )
            path = os.path.join(dist_dir, fname)
            if not os.path.exists(path):
                print(f"Missing distances: {path}")
                continue
            idist = np.load(path)

            # Build long DataFrame of off-diagonals
            n_subj, n_cond, _ = idist.shape
            cond_labels = [
                f"{abbr}_{t}"
                for t in ['prod', 'percep', 'ntfd']
                for abbr in conditions.values()
            ]
            idx = np.tril_indices(n_cond, k=-1)
            rows = []
            for s in range(n_subj):
                vals = idist[s][idx]
                for (i, j), v in zip(zip(*idx), vals):
                    rows.append({
                        'subject': s+1,
                        'label1': cond_labels[i],
                        'label2': cond_labels[j],
                        'value': v
                    })
            df_all = pd.DataFrame(rows)

            # derive task and modality
            df_all['task1'] = df_all['label1'].str.split('_').str[-1]
            df_all['task2'] = df_all['label2'].str.split('_').str[-1]
            df_all['mod'] = df_all['label1'].str[0]
            df_all = df_all[df_all['mod'] == df_all['label2'].str[0]]
            df_all['modality'] = \
                df_all['mod'].map({'a': 'audio', 'v': 'visual'})

            # collect significance
            for t1, t2 in task_pairs:
                for modality in valid_modalities:
                    sub = df_all[
                        ((df_all['task1'] == t1) & (df_all['task2'] == t2)) |
                        ((df_all['task1'] == t2) & (df_all['task2'] == t1))
                    ]
                    sub = sub[sub['modality'] == modality]
                    if sub.empty:
                        continue
                    mps = sub.groupby('subject')['value'].mean().reset_index()
                    if len(mps) > 1:
                        t_stat, p_val = ttest_1samp(mps['value'], popmean=0)
                    else:
                        t_stat, p_val = np.nan, np.nan
                    for _, r in mps.iterrows():
                        records.append({
                            'region': region,
                            'roi': roi,
                            'tag': tag,
                            'pair': f"{t1}_vs_{t2}",
                            'modality': modality,
                            'subject': r['subject'],
                            'mean_value': r['value'],
                            't_stat': t_stat,
                            'p_val': p_val
                        })

            df_roi = pd.DataFrame(records)
            tsv = os.path.join(
                out_folder,
                f'encoding_significance_{roi}_{tag}_'
                f'{thresh_label}_{smooth_label}.tsv'
            )
            df_roi.to_csv(tsv, sep='	', index=False)
            all_roi_dfs.append(df_roi)

        # plot all ROIs
        concat = pd.concat(all_roi_dfs, ignore_index=True)
        if concat.empty:
            print(f"No data to plot for tag {tag}")
            continue
        valid_rois = concat['roi'].unique()
        fig, axes = plt.subplots(
            len(valid_rois), 1,
            figsize=(24, 5*len(valid_rois)), sharex=True
        )
        if len(valid_rois) == 1:
            axes = [axes]
        for ax, roi in zip(axes, valid_rois):
            # remove x-axis tick marks
            ax.tick_params(axis='x', which='both', bottom=False, top=False)
            dfp = concat[concat['roi'] == roi]
            sns.boxplot(
                data=dfp,
                x='pair', y='mean_value', hue='modality',
                order=pair_labels, hue_order=valid_modalities,
                width=box_width, notch=True,
                showfliers=False, meanline=True, showmeans=True,
                medianprops={'color': 'k', 'linewidth': 0},
                meanprops={'color': 'tab:brown', 'linewidth': 1.5},
                boxprops={'alpha': 0.5, 'edgecolor': 'black'},
                ax=ax
            )
            ax.margins(x=0.15)

            # --- annotations ---
            # compute whisker-based y-limit
            y_vals = []
            for pair in pair_labels:
                for mod in valid_modalities:
                    sub = \
                        dfp[(dfp['pair'] == pair) & (dfp['modality'] == mod)]
                    if sub.empty:
                        continue
                    q1, q3 = sub['mean_value'].quantile([0.25, 0.75])
                    y_vals.append(q3 + 1.5*(q3-q1))
            top = max(y_vals) if y_vals else dfp['mean_value'].max()
            bottom = min(dfp['mean_value'].min(), 0)
            margin = (top - bottom) * .05 if top != bottom else 0.02
            y_annot = top + margin

            # symmetric offsets for modalities
            offsets = np.linspace(-box_width/4, box_width/4,
                                  len(valid_modalities))
            for i, pair in enumerate(pair_labels):
                for j, mod in enumerate(valid_modalities):
                    sub = \
                        dfp[(dfp['pair'] == pair) & (dfp['modality'] == mod)]
                    if sub.empty: 
                        continue
                    p = sub['p_val'].iloc[0]
                    if np.isnan(p):
                        sig = 'n.s.'
                    elif p < 1e-4:
                        sig = '****'
                    elif p < 1e-3:
                        sig = '***'
                    elif p < 1e-2:
                        sig = '**'
                    elif p < 0.05:
                        sig = '*'
                    else:
                        sig = 'n.s.'
                    x = i + offsets[j]
                    ax.text(x, y_annot, sig,
                            ha='center', va='bottom',
                            fontsize=10, fontweight='bold')
            # set x-label only on bottom
            if ax is not axes[-1]:
                ax.set_xticks([])
                ax.spines['bottom'].set_visible(False)
                # Remove the x-ticks
                ax.tick_params(axis='x', which='both', bottom=False,
                               top=False)
            else:
                ax.set_xlabel('Task Pair')
                ax.set_xticks(range(len(pair_labels)))
                ax.set_xticklabels(pair_labels, rotation=45, ha='right')
            ax.set_title(f"{roi}", pad=30)
            ax.set_ylabel('Mean Dissimilarity')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend_.remove()
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, title='Modality', loc='upper right')
        plt.tight_layout(rect=[.05, .05, .9, .95])
        png = os.path.join(
            out_folder,
            f'encoding_significance_allrois_{tag}_'
            f'{thresh_label}_{smooth_label}.png'
        )
        plt.savefig(png, dpi=300)
        plt.close()
        print(f"Saved plot: {png}")


# =========================== INPUTS ===================================

# Subjects without pilot
# SUBJECTS = [3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26, 28,
#             29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
SUBJECTS = [42]

# Path for output folders
rsa_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'results', 'rsa')

# ########################### ROIs ######################################

# # All ROIs: 10 ROIs
# atlas_names = ['hos', 
#                'ntk_symmni128', 'ntk_symmni128', 'ntk_symmni128',
#                'hmat', 'hmat', 'hmat', 'hmat',
#                'hos', 
#                'hos']
# region_names = ['dorsal_striatum', 
#                 'cerebellum', 'cerebellum', 'cerebellum',
#                 'motor_area', 'motor_area', 'motor_area', 'motor_area', 
#                 'heschl_gyrus', 
#                 'occipital_lobe']
# roi_names = ['dstr', 
#              'cereb-s', 'cereb-i', 'cereb'
#              'pmd', 'pmv', 'sma', 'presma',
#              'heschl',
#              'occipital']

# 8 ROIs
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

# atlas_names = ['hos',
#                'hos']
# region_names = ['heschl_gyrus',
#                 'occipital_lobe']
# roi_names = ['heschl',
#              'occipital']

# itags = ['i', 'i9a', 'i8a', 'i7a', 'i6a', 'a', 'a4g', 'a3g', 'a2g', 'a1g', 
#          'g']
itags = ['i', 'i8a']

hemispheres = ['bh']  # Both hemispheres

thresh_type = 'puncorr'  # 'puncorr' or 'pcorr'
smooth = 'unsmoothed'  # 'smoothed'

iroi_main_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    f'roi_analyses_rwls_hrf128_wb_{thresh_type}_{smooth}')

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

modality_list = ['both', 'audio', 'visual']
modality_map = {'both': '', 'audio': 'a', 'visual': 'v'}

# ============================ RUN =====================================

if __name__ == '__main__':

    # Create output folder if it does not exist
    os.makedirs(rsa_folder, exist_ok=True)

    # Paths of dataframes
    # db_taskglm_path = os.path.join(rsa_folder, 'rsa_taskglm.tsv')
    db_grandglm_path = os.path.join(
        rsa_folder, f'rsa_grandglm_{thresh_type}_{smooth}.tsv')
   
    # Create dataframes
    # db_grandglm = rsa_dataframe(
    #     SUBJECTS, glm_tasks, data_storage, conditions_mapping,
    #     db_grandglm_path, glm_type='task_glm')

    # Prewhiten grand glm beta maps and save them, ...
    # db_grandglm = prewhiten_betas(
    #     db_grandglm_path, SUBJECTS, data_storage, db_grandglm_path)
    # ... or just add paths of derivatives to dataframe
    # db_grandglm = prewhiten_betas(
    #     db_grandglm_path, SUBJECTS, data_storage, db_grandglm_path,
    #     prewhiten=False)

    # ##################################################################
    # Note: The next steps rely on prewhiten_beta_maps that were normalized,
    #       smoothed and masked. These steps were done in MATLAB.
    # ##################################################################

    # Open dataframes
    # db_taskglm = pd.read_csv(db_taskglm_path)
    db_grandglm = pd.read_csv(db_grandglm_path, sep='\t')

    # Extract signals from prewhitened data using the individualized ROIs
    # Order of conditions: abeat_prod, ainterval_prod,
    #                      vbeat_prod, vinterval_prod, 
    #                      abeat_percep, ainterval_percep,
    #                      vbeat_percep, vinterval_percep, 
    #                      abeat_ntfd, ainterval_ntfd,
    #                      vbeat_ntfd, vinterval_ntfd
    # grandglm_roi_extraction(db_grandglm_path, data_storage, glm_tasks,
    #                         SUBJECTS, itags, region_names, atlas_names,
    #                         roi_names, hemispheres, iroi_main_dir, 
    #                         thresh_type, smooth)

    # Compute RSA within a region and plot RDMs
    # For all tasks together
    # rsa(itags, glm_tasks[:3], region_names, roi_names, rsa_folder, 
    #     thresh_type, smooth, conditions_mapping, truncate_to_zero=False)
    # # For production task only
    # rsa(itags, [glm_tasks[0]], region_names, roi_names, rsa_folder,
    #     thresh_type, smooth, conditions_mapping, truncate_to_zero=False)
    # # For perception task only
    # rsa(itags, [glm_tasks[1]], region_names, roi_names, rsa_folder,
    #     thresh_type, smooth, conditions_mapping, truncate_to_zero=False)
    # # For ntfd task only
    # rsa(itags, [glm_tasks[2]], region_names, roi_names, rsa_folder,
    #     thresh_type, smooth, conditions_mapping, truncate_to_zero=False)

    # Compute RDM significance of beat vs interval pairs
    # periodicity_significance(
    #     output_dir=rsa_folder,
    #     tasks=glm_tasks,
    #     conditions=conditions_mapping,
    #     tags=itags,
    #     regions=region_names,
    #     rois=roi_names,
    #     rsa_dir=rsa_folder,
    #     thresh_label=thresh_type,
    #     smooth_label=smooth,
    #     modalities=modality_list,
    #     modality_dic=modality_map
    # )

    # Compute RDM significance of task pairs
    encoding_significance(
        output_dir=rsa_folder,
        conditions=conditions_mapping,
        tags=itags,
        regions=region_names,
        rois=roi_names,
        rsa_dir=rsa_folder,
        thresh_label=thresh_type,
        smooth_label=smooth,
        modalities=modality_list
    )