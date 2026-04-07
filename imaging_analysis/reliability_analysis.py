"""
Script to do the reliability analysis across conditions.

Author: Ana Luisa Pinho
Email: agrilopi@uwo.ca

Creation: 23rd of March 2026
Last Update: April 2026

Compatibility: Python 3.10.16
"""

import os
from pathlib import Path
import pypandoc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
from nilearn import image
from nilearn.input_data import NiftiLabelsMasker

from roi_similarity_encoding import plot_matrix

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
                        psc_fname = (
                            f'{prefix}_{key + rn:04d}_'
                            f'desc-{masking}masked.nii'
                        )
                        pscpath_colname = 'wmasked_pscmap_path'
                    else:
                        assert smoothing == 'smoothed'
                        psc_fname = (
                            f'{prefix}_{key + rn:04d}_'
                            f'desc-sm8{masking}masked.nii'
                        )
                        pscpath_colname = 'swmasked_pscmap_path'
                    pscmap_path = os.path.join(
                        masked_derivatives_dir,
                        psc_fname,
                    )
                    relative_pscmap_path = os.path.relpath(
                        pscmap_path,
                        base_dir,
                    )
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


def taskglm_roi_extraction(df_input, base_dir, script_dir, tags,
                           regions, atlases, rois, hems, iroi_mask_dir,
                           smoothing, output_folder):
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

                    if tag == 'g':
                        if region == 'dorsal_striatum':
                            iroi_mask_path = os.path.join(
                                script_dir,
                                iroi_mask_dir,
                                'bothmod_allmain_tasks',
                                'main_tasks',
                                region,
                                atlas,
                                'group_roi_masks',
                                (f'{tag}_msdtb_{atlas}_{roi}_{hem}_'
                                 f'mask.nii.gz')
                            )
                        else:
                            iroi_mask_path = os.path.join(
                                script_dir,
                                iroi_mask_dir,
                                'bothmod_allmain_tasks',
                                'main_tasks',
                                region,
                                atlas,
                                roi,
                                'group_roi_masks',
                                (f'{tag}_msdtb_{atlas}_{roi}_{hem}_'
                                 f'mask.nii.gz')
                            )
                    else:
                        if region == 'dorsal_striatum':
                            iroi_mask_path = os.path.join(
                                script_dir,
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
                                script_dir,
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

                    map_path = os.path.join(base_dir, row[path_col])
                    print(map_path)

                    img = image.load_img(map_path)
                    val = masker.fit_transform(img)

                    data_array[h, c, r, s] = val[0, 0]

            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(
                output_folder,
                f'taskglm_roi_signals_{roi}_{tag}_{smoothing}.npy'
            )
            np.save(output_path, data_array)

            print(
                f"Saved 4D voxel array for tag '{tag}' to {output_path}"
            )


def load_roi_signal_arrays(input_dir, indiv):
    """
    Load ROI signal arrays for one individualization level.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing the .npy files.
    indiv : str
        Individualization label, e.g. 'i' or 'g'.

    Returns
    -------
    data_by_roi : dict
        Maps ROI name to array of shape
        (n_hemi, n_cond, n_run, n_subj).
    """
    input_dir = Path(input_dir)

    roi_names = ROI_NAMES

    data_by_roi = {}

    for roi in roi_names:
        fname = (
            f'taskglm_roi_signals_{roi}_{indiv}_unsmoothed.npy'
        )
        fpath = input_dir / fname
        data_by_roi[roi] = np.load(fpath)

    return data_by_roi


def plot_reliability_distributions(
    rel_scheme_subj,
    roi_names,
    output_dir,
    hemi_label,
    tag,
):
    """
    Plot scheme-level and subject-mean reliability distributions per ROI.

    Parameters
    ----------
    rel_scheme_subj : np.ndarray
        Array of shape (n_subj, n_schemes, n_roi).
    roi_names : list of str
        ROI labels.
    output_dir : str
        Directory where plots will be saved.
    hemi_label : str
        Hemisphere label.
    tag : str
        Individualization tag.
    """
    os.makedirs(output_dir, exist_ok=True)

    n_subj, n_schemes, n_roi = rel_scheme_subj.shape

    # Plot 1: all subject x scheme reliabilities per ROI
    rel_all = rel_scheme_subj.reshape(n_subj * n_schemes, n_roi)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(
        [rel_all[:, i][np.isfinite(rel_all[:, i])] for i in range(n_roi)],
        tick_labels=roi_names,
        patch_artist=False,
    )

    for i in range(n_roi):
        y = rel_all[:, i]
        y = y[np.isfinite(y)]
        x = np.random.normal(i + 1, 0.04, size=len(y))
        ax.plot(x, y, 'o', alpha=0.5)

    ax.axhline(0, linestyle='--', linewidth=1)
    ax.set_ylabel('Scheme-level reliability')
    ax.set_title(
        f'Reliability distribution across subjects and schemes\n'
        f'Hemisphere: {hemi_label} | Tag: {tag}'
    )
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_dir,
            f'reliability_distribution_all_{hemi_label}_{tag}.png',
        ),
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()

    # Plot 2: subject means across schemes per ROI
    rel_subj_mean = np.nanmean(rel_scheme_subj, axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(
        [rel_subj_mean[:, i][np.isfinite(rel_subj_mean[:, i])]
         for i in range(n_roi)],
        tick_labels=roi_names,
        patch_artist=False,
    )

    for i in range(n_roi):
        y = rel_subj_mean[:, i]
        y = y[np.isfinite(y)]
        x = np.random.normal(i + 1, 0.04, size=len(y))
        ax.plot(x, y, 'o', alpha=0.6)

    ax.axhline(0, linestyle='--', linewidth=1)
    ax.set_ylabel('Subject-mean reliability')
    ax.set_title(
        f'Subject-level reliability across schemes\n'
        f'Hemisphere: {hemi_label} | Tag: {tag}'
    )
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_dir,
            f'reliability_distribution_subjectmean_{hemi_label}_{tag}.png',
        ),
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()


def summarize_negative_reliabilities(
    rel_scheme_subj_raw,
    roi_names,
    output_dir,
    hemi_label,
    tag,
):
    """
    Compute and save the percentage of negative reliabilities per ROI.

    Parameters
    ----------
    rel_scheme_subj_raw : np.ndarray
        Array of shape (n_subj, n_schemes, n_roi) with raw reliabilities.
    roi_names : list of str
        ROI labels.
    output_dir : str
        Directory where outputs will be saved.
    hemi_label : str
        Hemisphere label.
    tag : str
        Individualization tag.

    Returns
    -------
    pd.Series
        Percentage of negative reliabilities per ROI.
    """
    os.makedirs(output_dir, exist_ok=True)

    perc_neg = []
    for i, roi in enumerate(roi_names):
        vals = rel_scheme_subj_raw[:, :, i]
        vals = vals[np.isfinite(vals)]

        if len(vals) == 0:
            perc_neg.append(np.nan)
        else:
            perc_neg.append(100 * np.mean(vals < 0))

    perc_neg = pd.Series(
        perc_neg,
        index=roi_names,
        name=f'perc_negative_reliability_{hemi_label}_{tag}',
    )

    out_path = os.path.join(
        output_dir,
        f'perc_negative_reliability_{hemi_label}_{tag}.tsv',
    )
    perc_neg.to_csv(out_path, sep='\t', header=True)

    return perc_neg


def plot_reliability_vs_corrected_similarity(
    ceiling_mat,
    corrected_mat,
    roi_names,
    output_dir,
    hemi_label,
    tag,
):
    """
    Plot pairwise reliability ceiling vs corrected similarity,
    restricted to pairs involving dstr and/or cerebellum.
    """
    os.makedirs(output_dir, exist_ok=True)

    xs = []
    ys = []
    labels = []

    n_roi = len(roi_names)

    # Target ROIs
    target_rois = {'dstr', 'cereb'}

    for i in range(n_roi):
        for j in range(i + 1, n_roi):

            roi_i = roi_names[i]
            roi_j = roi_names[j]

            # FILTER HERE
            if not (roi_i in target_rois or roi_j in target_rois):
                continue

            x = ceiling_mat[i, j]
            y = corrected_mat[i, j]

            if np.isfinite(x) and np.isfinite(y):
                xs.append(x)
                ys.append(y)
                labels.append(f'{roi_i}-{roi_j}')

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(xs, ys, 'o')

    for x, y, label in zip(xs, ys, labels):
        ax.annotate(
            label,
            (x, y),
            xytext=(3, 3),
            textcoords='offset points',
            fontsize=8,
        )

    ax.axhline(0, linestyle='--', linewidth=1)
    ax.set_xlabel('Reliability ceiling')
    ax.set_ylabel('Corrected similarity')
    ax.set_title(
        f'Reliability ceiling vs corrected similarity\n'
        f'(dstr / cereb pairs only)\n'
        f'Hemisphere: {hemi_label} | Tag: {tag}'
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_dir,
            f'reliability_vs_corrected_similarity_{hemi_label}_{tag}.png',
        ),
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()


def compute_roi_reliability_pipeline(
    data_by_roi,
    condition_names,
    output_dir,
    hemi_labels,
    tag,
):
    """
    Compute reliability-aware ROI similarity measures.

    This pipeline implements:
    - within-ROI split-half reliability across condition profiles
    - Spearman-Brown correction
    - crossed cross-validated ROI-ROI correlations
    - pairwise reliability ceilings
    - attenuation-corrected ROI-ROI similarity

    Parameters
    ----------
    data_by_roi : dict
        Maps ROI name to an array with shape
        (n_hemi, n_cond, n_run, n_subj).
    condition_names : list of str
        Names of the non-rest conditions, in array order.
    output_dir : str
        Directory where outputs will be saved.
    hemi_labels : list of str
        Hemisphere labels, e.g. ['bh'] or ['lh', 'rh', 'bh'].
    tag : str
        Individualization tag, e.g. 'i' or 'g'.

    Returns
    -------
    results : dict
        One entry per hemisphere label. Each entry contains:
            - 'roi_reliability'
            - 'cv_similarity'
            - 'ceiling'
            - 'corrected_similarity'
            - 'perc_negative_reliability'
    """
    roi_names = list(data_by_roi.keys())
    first = np.asarray(data_by_roi[roi_names[0]], dtype=float)

    n_hemi, n_cond, _, n_subj = first.shape
    if len(hemi_labels) != n_hemi:
        raise ValueError(
            f"len(hemi_labels)={len(hemi_labels)} does not match "
            f"n_hemi={n_hemi}."
        )

    n_roi = len(roi_names)
    full_conditions = condition_names + ['rest']
    n_full = len(full_conditions)

    os.makedirs(output_dir, exist_ok=True)

    schemes = [
        {'part4': ((0, 1), (2, 3)), 'part2': ((0,), (1,))},
        {'part4': ((0, 1), (2, 3)), 'part2': ((1,), (0,))},
        {'part4': ((0, 2), (1, 3)), 'part2': ((0,), (1,))},
        {'part4': ((0, 2), (1, 3)), 'part2': ((1,), (0,))},
        {'part4': ((1, 2), (0, 3)), 'part2': ((0,), (1,))},
        {'part4': ((1, 2), (0, 3)), 'part2': ((1,), (0,))},
    ]

    def safe_corr(x, y):
        """Return Pearson correlation or NaN if undefined."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            return np.nan
        if np.std(x) == 0 or np.std(y) == 0:
            return np.nan

        return np.corrcoef(x, y)[0, 1]

    results = {}

    for h, hemi_label in enumerate(hemi_labels):
        cond_n_runs = []

        # Check only the number of available runs for each condition
        # across all ROIs, irrespective of subject.
        for c in range(n_cond):
            n_runs_ref = None

            for roi in roi_names:
                arr = np.asarray(data_by_roi[roi], dtype=float)
                vals = arr[h, c, :, :]
                avail = np.any(np.isfinite(vals), axis=-1)
                runs = np.where(avail)[0]

                if n_runs_ref is None:
                    n_runs_ref = len(runs)
                elif len(runs) != n_runs_ref:
                    raise ValueError(
                        f"Mismatch in number of available runs for "
                        f"condition {condition_names[c]!r} in hemisphere "
                        f"{hemi_label!r}."
                    )

            if n_runs_ref not in [2, 4]:
                raise ValueError(
                    f"Condition {condition_names[c]!r} in hemisphere "
                    f"{hemi_label!r} has {n_runs_ref} runs; expected 2 or 4."
                )

            cond_n_runs.append(n_runs_ref)

        # Subject-level outputs after averaging across schemes.
        rel_subj = np.full((n_subj, n_roi), np.nan)
        rel_scheme_subj_raw = np.full((n_subj, len(schemes), n_roi), np.nan)
        rel_scheme_subj = np.full((n_subj, len(schemes), n_roi), np.nan)
        cv_subj = np.full((n_subj, n_roi, n_roi), np.nan)
        ceil_subj = np.full((n_subj, n_roi, n_roi), np.nan)
        corr_subj = np.full((n_subj, n_roi, n_roi), np.nan)

        for s in range(n_subj):
            cond_valid_runs_subj = []

            for c in range(n_cond):
                runs_ref = None

                for roi in roi_names:
                    arr = np.asarray(data_by_roi[roi], dtype=float)
                    vals = arr[h, c, :, s]
                    runs = np.where(np.isfinite(vals))[0]

                    if runs_ref is None:
                        runs_ref = runs
                    elif not np.array_equal(runs, runs_ref):
                        raise ValueError(
                            f"Subject-level mismatch in available runs for "
                            f"condition {condition_names[c]!r}, hemisphere "
                            f"{hemi_label!r}, subject index {s}, ROI "
                            f"{roi!r}. Expected {runs_ref}, got {runs}."
                        )

                if len(runs_ref) != cond_n_runs[c]:
                    raise ValueError(
                        f"Unexpected number of runs for condition "
                        f"{condition_names[c]!r}, hemisphere "
                        f"{hemi_label!r}, subject index {s}. "
                        f"Expected {cond_n_runs[c]}, got {len(runs_ref)}."
                    )

                cond_valid_runs_subj.append(runs_ref)

            rel_schemes = []
            cv_schemes = []
            ceil_schemes = []
            corr_schemes = []

            for k, scheme in enumerate(schemes):
                # Half profiles: shape = n_roi x (n_cond + 1)
                y1 = np.full((n_roi, n_full), np.nan)
                y2 = np.full((n_roi, n_full), np.nan)

                for r, roi in enumerate(roi_names):
                    arr = np.asarray(data_by_roi[roi], dtype=float)

                    for c, valid_runs in enumerate(cond_valid_runs_subj):
                        vals = arr[h, c, :, s]

                        if len(valid_runs) == 4:
                            idx1 = valid_runs[list(scheme['part4'][0])]
                            idx2 = valid_runs[list(scheme['part4'][1])]
                        elif len(valid_runs) == 2:
                            idx1 = valid_runs[list(scheme['part2'][0])]
                            idx2 = valid_runs[list(scheme['part2'][1])]
                        else:
                            raise ValueError(
                                f"Condition {condition_names[c]!r} in "
                                f"hemisphere {hemi_label!r} has "
                                f"{len(valid_runs)} valid runs; "
                                f"expected 2 or 4."
                            )

                        y1[r, c] = np.nanmean(vals[idx1])
                        y2[r, c] = np.nanmean(vals[idx2])

                    # Add Rest = 0 as the final condition.
                    y1[r, -1] = 0.0
                    y2[r, -1] = 0.0

                # Step 6-7: within-ROI split-half reliability
                rel_vec_raw = np.full(n_roi, np.nan)
                rel_vec = np.full(n_roi, np.nan)
                for r in range(n_roi):
                    r_half = safe_corr(y1[r, :], y2[r, :])
                    if np.isfinite(r_half) and r_half > -1:
                        rel_val = (2 * r_half) / (1 + r_half)
                        rel_vec_raw[r] = rel_val
                        if rel_val >= 0:
                            rel_vec[r] = rel_val

                rel_scheme_subj_raw[s, k, :] = rel_vec_raw
                rel_scheme_subj[s, k, :] = rel_vec

                # Step 8: crossed ROI-ROI correlations
                cv_mat = np.full((n_roi, n_roi), np.nan)
                for i in range(n_roi):
                    for j in range(n_roi):
                        c12 = safe_corr(y1[i, :], y2[j, :])
                        c21 = safe_corr(y2[i, :], y1[j, :])
                        cv_mat[i, j] = np.nanmean([c12, c21])

                # Step 9: ceilings
                ceil_mat = np.full((n_roi, n_roi), np.nan)
                for i in range(n_roi):
                    for j in range(n_roi):
                        if (
                            np.isfinite(rel_vec[i])
                            and np.isfinite(rel_vec[j])
                            and rel_vec[i] >= 0
                            and rel_vec[j] >= 0
                        ):
                            ceil_mat[i, j] = np.sqrt(
                                rel_vec[i] * rel_vec[j]
                            )

                # Step 10: corrected similarity
                corr_mat = np.full((n_roi, n_roi), np.nan)
                valid = np.isfinite(ceil_mat) & (ceil_mat > 0)
                corr_mat[valid] = cv_mat[valid] / ceil_mat[valid]

                rel_schemes.append(rel_vec)
                cv_schemes.append(cv_mat)
                ceil_schemes.append(ceil_mat)
                corr_schemes.append(corr_mat)

            # Step 11: average across schemes within subject
            rel_subj[s, :] = np.nanmean(np.stack(rel_schemes, axis=0), axis=0)
            cv_subj[s, :, :] = np.nanmean(
                np.stack(cv_schemes, axis=0),
                axis=0,
            )
            ceil_subj[s, :, :] = np.nanmean(
                np.stack(ceil_schemes, axis=0),
                axis=0,
            )
            corr_subj[s, :, :] = np.nanmean(
                np.stack(corr_schemes, axis=0),
                axis=0,
            )

        # Step 12: average across subjects
        rel_group = np.nanmean(rel_subj, axis=0)
        cv_group = np.nanmean(cv_subj, axis=0)
        ceil_group = np.nanmean(ceil_subj, axis=0)
        corr_group = np.nanmean(corr_subj, axis=0)

        # Save subject-level outputs
        np.save(
            os.path.join(output_dir, 
                         f"roi_reliability_subj_{hemi_label}_{tag}.npy"),
            rel_subj,
        )
        np.save(
            os.path.join(
                output_dir,
                f"roi_reliability_scheme_subj_raw_{hemi_label}_{tag}.npy",
            ),
            rel_scheme_subj_raw,
        )
        np.save(
            os.path.join(
                output_dir,
                f"roi_reliability_scheme_subj_{hemi_label}_{tag}.npy",
            ),
            rel_scheme_subj,
        )
        np.save(
            os.path.join(output_dir, 
                         f"cv_similarity_subj_{hemi_label}_{tag}.npy"),
            cv_subj,
        )
        np.save(
            os.path.join(output_dir, f"ceiling_subj_{hemi_label}_{tag}.npy"),
            ceil_subj,
        )
        np.save(
            os.path.join(
                output_dir,
                f"corrected_similarity_subj_{hemi_label}_{tag}.npy",
            ),
            corr_subj,
        )

        # Save group-level outputs
        np.save(
            os.path.join(output_dir, 
                         f"roi_reliability_{hemi_label}_{tag}.npy"),
            rel_group,
        )
        np.save(
            os.path.join(output_dir, f"cv_similarity_{hemi_label}_{tag}.npy"),
            cv_group,
        )
        np.save(
            os.path.join(output_dir, f"ceiling_{hemi_label}_{tag}.npy"),
            ceil_group,
        )
        np.save(
            os.path.join(
                output_dir,
                f"corrected_similarity_{hemi_label}_{tag}.npy",
            ),
            corr_group,
        )

        plot_dir = os.path.join(output_dir, 'reliability_distributions')
        plot_reliability_distributions(
            rel_scheme_subj_raw,
            roi_names,
            plot_dir,
            hemi_label,
            tag,
        )

        neg_dir = os.path.join(output_dir, 'negative_reliability_summary')
        perc_neg = summarize_negative_reliabilities(
            rel_scheme_subj_raw,
            roi_names,
            neg_dir,
            hemi_label,
            tag,
        )

        relcorr_dir = os.path.join(
            output_dir,
            'reliability_vs_corrected_similarity'
        )
        plot_reliability_vs_corrected_similarity(
            ceil_group,
            corr_group,
            roi_names,
            relcorr_dir,
            hemi_label,
            tag,
        )

        results[hemi_label] = {
            'roi_reliability': pd.Series(
                rel_group,
                index=roi_names,
                name=f'roi_reliability_{hemi_label}_{tag}',
            ),
            'cv_similarity': pd.DataFrame(
                cv_group,
                index=roi_names,
                columns=roi_names,
            ),
            'ceiling': pd.DataFrame(
                ceil_group,
                index=roi_names,
                columns=roi_names,
            ),
            'corrected_similarity': pd.DataFrame(
                corr_group,
                index=roi_names,
                columns=roi_names,
            ),
            'perc_negative_reliability': perc_neg,
        }

    return results


def save_results_to_rtf(results, output_path):
    """
    Save reliability results into an RTF file with proper tables.
    """

    sections = []

    for hemi, res in results.items():
        sections.append(f"# Hemisphere: {hemi}\n")

        # ROI reliability (convert Series to DataFrame for table format)
        rel_df = res["roi_reliability"].to_frame(name="Reliability")
        sections.append("## ROI reliability\n")
        sections.append(rel_df.round(3).to_markdown())
        sections.append("\n")

        sections.append("## Percentage of negative reliabilities\n")
        neg_df = res["perc_negative_reliability"].to_frame(
            name="Percentage"
        )
        sections.append(neg_df.round(3).to_markdown())
        sections.append("\n")

        # Cross-validated similarity
        sections.append("## Cross-validated ROI–ROI similarity\n")
        sections.append(res["cv_similarity"].round(3).to_markdown())
        sections.append("\n")

        # Ceiling
        sections.append("## Reliability ceiling\n")
        sections.append(res["ceiling"].round(3).to_markdown())
        sections.append("\n")

        # Corrected similarity
        sections.append("## Corrected similarity\n")
        sections.append(res["corrected_similarity"].round(3).to_markdown())
        sections.append("\n\n")

    text = "\n".join(sections)

    pypandoc.convert_text(
        text,
        'rtf',
        format='md',
        outputfile=output_path,
        extra_args=['--standalone']
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

# Threshold of group-level contrast ued to generate ROI
thresh_type = 'puncorr'  # 'puncorr' or 'pcorr'
# Derivatives used are smoothed or unsmoothed?
smooth = 'unsmoothed'  # 'smoothed'
# What type of map?
derivative_type = 'wpsc'
# Whole-brain (wb) mask or gray-matter (gm) mask?
mask_type = 'wb'


# Path for output folders
main_dir = os.path.dirname(os.path.abspath(__file__))
reliability_folder = os.path.join(main_dir, 'results', 'reliability')
roi_signals_folder = os.path.join(reliability_folder, 
                                  f'taskglm_roi_signals_{smooth}')
split_half_folder = os.path.join(reliability_folder, 'roi_profile_split_half')

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

condition_names = [
    'auditory_beat_prod',
    'auditory_interval_prod',
    'visual_beat_prod',
    'visual_interval_prod',
    'auditory_beat_percep',
    'auditory_interval_percep',
    'visual_beat_percep',
    'visual_interval_percep',
    'auditory_beat_ntfd',
    'auditory_interval_ntfd',
    'visual_beat_ntfd',
    'visual_interval_ntfd',
    'auditory_beat_rand_ntfd',
    'auditory_interval_rand_ntfd',
    'auditory_random_rand_ntfd',
    'visual_beat_rand_ntfd',
    'visual_interval_rand_ntfd',
    'visual_random_rand_ntfd',
]

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
ROI_NAMES = ['dstr',
             'cereb',
             'presma', 'sma', 'pmd', 'pmv',
             'heschl',
             'occipital']

# itags = ['i', 'i9a', 'i8a', 'i7a', 'i6a', 'a', 'a4g', 'a3g', 'a2g', 'a1g',
#          'g']
# itags = ['i', 'g']
itags = ['i', 'g']

hemispheres = ['bh']  # Both hemispheres

iroi_main_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    f'roi_analyses_rwls_hrf128_wb_{thresh_type}_{smooth}')

# ============================ RUN =====================================

if __name__ == '__main__':
    # Create output folder if it does not exist
    os.makedirs(reliability_folder, exist_ok=True)

    # Paths of dataframe
    db_taskglm_path = os.path.join(
        reliability_folder,
        f'reliability_taskglm_{smooth}.tsv',
    )

    # Create dataframes
    # reliability_dataframe(
    #     SUBJECTS, glm_tasks, data_storage,
    #     selected_contrasts_main, selected_contrasts_random,
    #     db_taskglm_path, derivative_type, mask_type, smooth)

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
    # taskglm_roi_extraction(db_taskglm_path, data_storage, main_dir, itags,
    #                        region_names, atlas_names, roi_names,
    #                        hemispheres, iroi_main_dir, smooth,
    #                        roi_signals_folder)

    # Compute the reliability analysis
    for itag in itags:
        data_by_roi = load_roi_signal_arrays(
            roi_signals_folder,
            indiv=itag,
        )

        results = compute_roi_reliability_pipeline(
            data_by_roi,
            condition_names,
            split_half_folder,
            hemispheres,
            itag,
        )

        for hemi in hemispheres:
            print(f"\nTag: {itag} | Hemisphere: {hemi}")

            print("\nROI reliability")
            print(results[hemi]['roi_reliability'])

            print("\nPercentage of negative reliabilities")
            print(results[hemi]['perc_negative_reliability'])

            print("\nCross-validated ROI-ROI similarity")
            print(results[hemi]['cv_similarity'])

            print("\nReliability ceiling")
            print(results[hemi]['ceiling'])

            print("\nCorrected similarity")
            print(results[hemi]['corrected_similarity'])

        rtf_path = os.path.join(
            split_half_folder,
            f"reliability_report_{itag}.rtf"
        )
        save_results_to_rtf(results, rtf_path)