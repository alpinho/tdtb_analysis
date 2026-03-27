"""
Script to do the reliability analysis across conditions.

Author: Ana Luisa Pinho
Email: agrilopi@uwo.ca

Creation: 23rd of March 2026
Last Update: March 2026

Compatibility: Python 3.10.16
"""

import os
import pandas as pd
import numpy as np


from nilearn import image
from nilearn.input_data import NiftiLabelsMasker

# Prevent DataFrame truncation when printing
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# =========================== FUNCTIONS ================================

def reliability_dataframe(
        subjects, task_models, base_dir,
        contrasts_main, contrasts_random,
        output_path, prefix, masking, smoothing):
    """Builds the inputs DataFrame, saves it, and returns it."""

    derivatives_dir = os.path.join(
        base_dir, 'data', 'Cerebellum', 'music-sdtb', 'derivatives')

    rows = []
    for subj in subjects:
        # Format subject with leading zero if needed, e.g., "sub-03"
        subj_str = f"sub-{subj:02d}"
        for model in task_models:

            if model != 'rand_ntfd':
                contrasts_mapping = contrasts_main
            else:
                assert model == 'rand_ntfd'
                contrasts_mapping = contrasts_random        

            masked_derivatives_dir = os.path.join(
                derivatives_dir, subj_str, 'estimates', model,
                'masked_derivatives_rwls_dbb_hrf128'
            )

            keys = list(contrasts_mapping)
            n_runs = keys[1] - keys[0]

            for key, value in contrasts_mapping.items():
                con_name = value.lower().replace(' ', '_')
                cond_name = con_name + '_' + model
                for rn in np.arange(n_runs):

                    if smoothing == 'unsmoothed':
                        psc_fname = f'{prefix}_{key + rn:04d}_' + \
                                    f'desc-{masking}masked.nii'
                        pscpath_colname = 'wmasked_pscmap_path'
                    else:
                        assert smoothing == 'smoothed'
                        psc_fname = f'{prefix}_{key + rn:04d}_' + \
                                    f'desc-sm8{masking}masked.nii'
                        pscpath_colname = 'swmasked_pscmap_path'
                    pscmap_path = os.path.join(masked_derivatives_dir,
                                               psc_fname)
                    relative_pscmap_path = os.path.relpath(pscmap_path,
                                                           base_dir)
                    rows.append({
                        'subject': subj,
                        'task_id': model,
                        'contrast_name': con_name,
                        'condition_name': cond_name,
                        'run_number': rn + 1,
                        pscpath_colname: relative_pscmap_path,
                    })

    # Create the DataFrame
    df = pd.DataFrame(rows)
    print(df)

    # Save the DataFrame in the rsa_folder
    df.to_csv(output_path, index=False, sep='\t')

    return df


def taskglm_roi_extraction(df_input, base_dir, tags,
                           regions, atlases, rois, hems, iroi_mask_dir,
                           smoothing):
    """
    Extract mean PSC within an ROI for each contrast.

    Input table is read row-by-row. Each row must contain at least:
        - subject
        - condition_name
        - run_number
        - wmasked_pscmap_path   (for smoothing='unsmoothed')
    or:
        - swmasked_pscmap_path  (for smoothing='smoothed')

    Output array shape:
    (hemisphere, conditions, runs, subjects)
    """

    if isinstance(df_input, str):
        df_unfiltered = pd.read_csv(df_input, sep='\t')
    elif isinstance(df_input, pd.DataFrame):
        df_unfiltered = df_input.copy()
    else:
        raise ValueError("df_input must be a path or a pandas DataFrame.")

    if smoothing == 'unsmoothed':
        path_col = 'wmasked_pscmap_path'
    elif smoothing == 'smoothed':
        path_col = 'swmasked_pscmap_path'
    else:
        raise ValueError(
            "smoothing must be either 'unsmoothed' or 'smoothed'."
        )

    if path_col not in df_unfiltered.columns:
        raise ValueError(
            f"Column '{path_col}' was not found in the input table."
        )

    required_cols = ['subject', 'condition_name',
                     'run_number', path_col]
    missing_cols = [col for col in required_cols
                    if col not in df_unfiltered.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in input table: {missing_cols}"
        )

    subjects = sorted(df_unfiltered['subject'].unique())
    condition_names = list(df_unfiltered['condition_name'].unique())
    run_numbers = sorted(df_unfiltered['run_number'].unique())

    subject_to_idx = {sub: i for i, sub in enumerate(subjects)}
    condition_to_idx = {
        cond: i for i, cond in enumerate(condition_names)
    }
    run_to_idx = {rn: i for i, rn in enumerate(run_numbers)}

    n_hems = len(hems)
    n_conditions = len(condition_names)
    n_runs = len(run_numbers)
    n_subjects = len(subjects)

    for region, atlas, roi in zip(regions, atlases, rois):
        for tag in tags:
            print(f"Processing tag: {tag}")

            data_array = np.full(
                (n_hems, n_conditions, n_runs, n_subjects),
                np.nan
            )

            for h, hem in enumerate(hems):
                for _, row in df_unfiltered.iterrows():
                    subject = row['subject']
                    condition = row['condition_name']
                    run_number = row['run_number']

                    s = subject_to_idx[subject]
                    c = condition_to_idx[condition]
                    r = run_to_idx[run_number]

                    if region == 'dorsal_striatum':
                        iroi_mask_path = os.path.join(
                            os.path.dirname(os.path.abspath(__file__)),
                            iroi_mask_dir,
                            'bothmod_allmain_tasks',
                            'main_tasks',
                            region,
                            atlas,
                            'individual_roi_masks',
                            (f'{tag}_sub-{subject:02d}_'
                             f'{roi}_{hem}_mask.nii.gz')
                        )
                    else:
                        iroi_mask_path = os.path.join(
                            os.path.dirname(os.path.abspath(__file__)),
                            iroi_mask_dir,
                            'bothmod_allmain_tasks',
                            'main_tasks',
                            region,
                            atlas,
                            roi,
                            'individual_roi_masks',
                            (f'{tag}_sub-{subject:02d}_'
                             f'{roi}_{hem}_mask.nii.gz')
                        )

                    masker = NiftiLabelsMasker(
                        labels_img=iroi_mask_path
                    )

                    map_path = os.path.join(
                        base_dir, row[path_col]
                    )
                    print(map_path)

                    img = image.load_img(map_path)
                    val = masker.fit_transform(img)

                    # extract scalar explicitly
                    data_array[h, c, r, s] = val[0, 0]

            output_folder = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'results',
                'reliability',
                f'taskglm_roi_signals_{smoothing}'
            )
            os.makedirs(output_folder, exist_ok=True)

            output_path = os.path.join(
                output_folder,
                f'taskglm_roi_signals_{roi}_{tag}_{smoothing}.npy'
            )
            np.save(output_path, data_array)

            print(
                f"Saved 4D voxel array for tag '{tag}' to {output_path}"
            )


# =========================== INPUTS ===================================

# Subjects without pilot
SUBJECTS = [3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26, 28,
            29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
# SUBJECTS = [3]

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

# Threshold of group-level contrast ued to generate ROI
thresh_type = 'puncorr'  # 'puncorr' or 'pcorr'
# Derivatives used are smoothed or unsmoothed?
smooth = 'unsmoothed'  # 'smoothed'
# What type of map?
derivative_type = 'wpsc'
# Whole-brain (wb) mask or gray-matter (gm) mask?
mask_type = 'wb'

# -------- All contrast dictionaries --------

ALL_CONTRASTS_MAIN_RUN1 = {
    19: 'Encoding',
    23: 'Auditory Encoding',
    27: 'Visual Encoding',
    31: 'Auditory vs Visual Encoding',
    35: 'Visual vs Auditory Encoding',
    39: 'Beat',
    43: 'Interval',
    47: 'Beat vs Interval',
    51: 'Interval vs Beat',
    55: 'Auditory Beat',
    59: 'Auditory Interval',
    63: 'Auditory Beat vs Auditory Interval',
    67: 'Auditory Interval vs Auditory Beat',
    71: 'Visual Beat',
    75: 'Visual Interval',
    79: 'Visual Beat vs Visual Interval',
    83: 'Visual Interval vs Visual Beat',
    87: 'Decision'
}

ALL_CONTRASTS_RAND_RUN1 = {
    43: 'Encoding',
    45: 'Auditory Encoding',
    47: 'Visual Encoding',
    49: 'Auditory vs Visual Encoding',
    51: 'Visual vs Auditory Encoding',
    53: 'Beat',
    55: 'Interval',
    57: 'Non-Random',
    59: 'Random',
    61: 'Beat vs Interval',
    63: 'Interval vs Beat',
    65: 'Beat vs Random',
    67: 'Random vs Beat',
    69: 'Interval vs Random',
    71: 'Random vs Interval',
    73: 'Non-Random vs Random',
    75: 'Random vs Non-Random',
    77: 'Auditory Beat',
    79: 'Auditory Interval',
    81: 'Auditory Non-Random',
    83: 'Auditory Random',
    85: 'Auditory Beat vs Auditory Interval',
    87: 'Auditory Interval vs Auditory Beat',
    89: 'Auditory Beat vs Auditory Random',
    91: 'Auditory Random vs Auditory Beat',
    93: 'Auditory Interval vs Auditory Random',
    95: 'Auditory Random vs Auditory Interval',
    97: 'Auditory Non-Random vs Auditory Random',
    99: 'Auditory Random vs Auditory Non-Random',
    101: 'Visual Beat',
    103: 'Visual Interval',
    105: 'Visual Non-Random',
    107: 'Visual Random',
    109: 'Visual Beat vs Visual Interval',
    111: 'Visual Interval vs Visual Beat',
    113: 'Visual Beat vs Visual Random',
    115: 'Visual Random vs Visual Beat',
    117: 'Visual Interval vs Visual Random',
    119: 'Visual Random vs Visual Interval',
    121: 'Visual Non-Random vs Visual Random',
    123: 'Visual Random vs Visual Non-Random',
    125: 'Decision'
}

selected_contrasts_main = {
    55: 'Auditory Beat',
    59: 'Auditory Interval',
    71: 'Visual Beat',
    75: 'Visual Interval'
}

selected_contrasts_random = {
    77: 'Auditory Beat',
    79: 'Auditory Interval',
    83: 'Auditory Random',
    101: 'Visual Beat',
    103: 'Visual Interval',
    107: 'Visual Random'
}

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

    # Paths of dataframe
    db_taskglm_path = os.path.join(reliability_folder,
                                   f'reliability_taskglm_{smooth}.tsv')

    # Create dataframes
    # reliability_dataframe(SUBJECTS, glm_tasks, data_storage,
    #                       selected_contrasts_main, selected_contrasts_random, 
    #                       db_taskglm_path, derivative_type, mask_type, smooth)

    # Open dataframe
    # db_taskglm = pd.read_csv(db_taskglm_path, sep='\t')

    # Extract signals from derivatives using individualized ROIs
    # Order of conditions: 
    #   'auditory_beat_prod', 'auditory_interval_prod', 
    #   'visual_beat_prod', 'visual_interval_prod',
    #   'auditory_beat_percep', 'auditory_interval_percep', 
    #   'visual_beat_percep', 'visual_interval_percep',
    #   'auditory_beat_ntfd', 'auditory_interval_ntfd', 
    #   'visual_beat_ntfd', 'visual_interval_ntfd',
    #   'auditory_beat_rand_ntfd', 'auditory_interval_rand_ntfd', 
    #   'auditory_random_rand_ntfd',
    #   'visual_beat_rand_ntfd', 'visual_interval_rand_ntfd', 
    #   'visual_random_rand_ntfd'
    taskglm_roi_extraction(db_taskglm_path, data_storage, itags, 
                           region_names, atlas_names, roi_names, hemispheres, 
                           iroi_main_dir, smooth)
    
    # Compute the relibility analysis