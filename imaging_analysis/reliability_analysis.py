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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypandoc
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
    """Build the inputs DataFrame, save it, and return it."""

    derivatives_dir = os.path.join(
        base_dir, 'data', 'Cerebellum', 'music-sdtb', 'derivatives')

    rows = []
    for subj in subjects:
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

    df = pd.DataFrame(rows)
    print(df)
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

    data_by_roi = {}

    for roi in ROI_NAMES:
        fname = (
            f'taskglm_roi_signals_{roi}_{indiv}_unsmoothed.npy'
        )
        fpath = input_dir / fname
        data_by_roi[roi] = np.load(fpath)

    return data_by_roi


def plot_split_half_distributions(
    split_half_scheme_subj,
    roi_names,
    output_dir,
    hemi_label,
    tag,
):
    """
    Plot scheme-level and subject-median split-half correlations per ROI.

    Parameters
    ----------
    split_half_scheme_subj : np.ndarray
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

    n_subj, n_schemes, n_roi = split_half_scheme_subj.shape

    corr_all = split_half_scheme_subj.reshape(n_subj * n_schemes, n_roi)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(
        [corr_all[:, i][np.isfinite(corr_all[:, i])] for i in range(n_roi)],
        tick_labels=roi_names,
        patch_artist=False,
    )

    for i in range(n_roi):
        y = corr_all[:, i]
        y = y[np.isfinite(y)]
        x = np.random.normal(i + 1, 0.04, size=len(y))
        ax.plot(x, y, 'o', alpha=0.5)

    ax.axhline(0, linestyle='--', linewidth=1)
    ax.set_ylabel('Scheme-level split-half correlation')
    ax.set_title(
        f'Split-half correlation across subjects and schemes\n'
        f'Hemisphere: {hemi_label} | Tag: {tag}'
    )
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_dir,
            f'split_half_distribution_all_{hemi_label}_{tag}.png',
        ),
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()

    corr_subj_median = np.nanmedian(split_half_scheme_subj, axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(
        [corr_subj_median[:, i][np.isfinite(corr_subj_median[:, i])]
         for i in range(n_roi)],
        tick_labels=roi_names,
        patch_artist=False,
    )

    for i in range(n_roi):
        y = corr_subj_median[:, i]
        y = y[np.isfinite(y)]
        x = np.random.normal(i + 1, 0.04, size=len(y))
        ax.plot(x, y, 'o', alpha=0.6)

    ax.axhline(0, linestyle='--', linewidth=1)
    ax.set_ylabel('Subject-median split-half correlation')
    ax.set_title(
        f'Subject-level median split-half correlation across schemes\n'
        f'Hemisphere: {hemi_label} | Tag: {tag}'
    )
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_dir,
            f'split_half_distribution_subjectmedian_{hemi_label}_{tag}.png',
        ),
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()


def plot_spearman_brown_distributions(
    sb_subj,
    roi_names,
    output_dir,
    hemi_label,
    tag,
):
    """
    Plot subject-level Spearman-Brown values per ROI.

    Parameters
    ----------
    sb_subj : np.ndarray
        Array of shape (n_subj, n_roi).
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

    _, n_roi = sb_subj.shape

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(
        [sb_subj[:, i][np.isfinite(sb_subj[:, i])] for i in range(n_roi)],
        tick_labels=roi_names,
        patch_artist=False,
    )

    for i in range(n_roi):
        y = sb_subj[:, i]
        y = y[np.isfinite(y)]
        x = np.random.normal(i + 1, 0.04, size=len(y))
        ax.plot(x, y, 'o', alpha=0.6)

    ax.axhline(0, linestyle='--', linewidth=1)
    ax.set_ylabel('Subject-level Spearman-Brown')
    ax.set_title(
        f'Spearman-Brown across subjects\n'
        f'Hemisphere: {hemi_label} | Tag: {tag}'
    )
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_dir,
            f'spearman_brown_distribution_subject_{hemi_label}_{tag}.png',
        ),
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()


def summarize_negative_schemes(
    split_half_scheme_subj,
    roi_names,
    output_dir,
    hemi_label,
    tag,
):
    """
    Compute and save the percentage of negative split-half schemes per ROI.

    Parameters
    ----------
    split_half_scheme_subj : np.ndarray
        Array of shape (n_subj, n_schemes, n_roi).
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
        Percentage of negative scheme-level correlations per ROI.
    """
    os.makedirs(output_dir, exist_ok=True)

    perc_neg = []
    for i in np.arange(len(roi_names)):
        vals = split_half_scheme_subj[:, :, i]
        vals = vals[np.isfinite(vals)]

        if len(vals) == 0:
            perc_neg.append(np.nan)
        else:
            perc_neg.append(100 * np.mean(vals < 0))

    perc_neg = pd.Series(
        perc_neg,
        index=roi_names,
        name=f'perc_negative_schemes_{hemi_label}_{tag}',
    )

    out_path = os.path.join(
        output_dir,
        f'perc_negative_schemes_{hemi_label}_{tag}.tsv',
    )
    perc_neg.to_csv(out_path, sep='\t', header=True)

    return perc_neg


def summarize_spearman_brown_capping(
    sb_raw_subj,
    roi_names,
    output_dir,
    hemi_label,
    tag,
):
    """
    Compute and save percentages capped below -1 and above 1 per ROI.

    Parameters
    ----------
    sb_raw_subj : np.ndarray
        Array of shape (n_subj, n_roi) with unclipped values.
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
    tuple of pd.Series
        Percentage capped below -1 per ROI and percentage capped above
        1 per ROI.
    """
    os.makedirs(output_dir, exist_ok=True)

    perc_below = []
    perc_above = []

    for i in np.arange(len(roi_names)):
        vals = sb_raw_subj[:, i]
        vals = vals[np.isfinite(vals)]

        if len(vals) == 0:
            perc_below.append(np.nan)
            perc_above.append(np.nan)
        else:
            perc_below.append(100 * np.mean(vals < -1))
            perc_above.append(100 * np.mean(vals > 1))

    perc_below = pd.Series(
        perc_below,
        index=roi_names,
        name=f'perc_sb_capped_below_m1_{hemi_label}_{tag}',
    )
    perc_above = pd.Series(
        perc_above,
        index=roi_names,
        name=f'perc_sb_capped_above_p1_{hemi_label}_{tag}',
    )

    out_path_below = os.path.join(
        output_dir,
        f'perc_sb_capped_below_m1_{hemi_label}_{tag}.tsv',
    )
    out_path_above = os.path.join(
        output_dir,
        f'perc_sb_capped_above_p1_{hemi_label}_{tag}.tsv',
    )

    perc_below.to_csv(out_path_below, sep='\t', header=True)
    perc_above.to_csv(out_path_above, sep='\t', header=True)

    return perc_below, perc_above


def safe_corr(x, y):
    """Return Pearson correlation or NaN if undefined."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return np.nan
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return np.corrcoef(x, y)[0, 1]


def compute_spearman_brown(split_half_subj):
    """Apply Spearman-Brown and clip to [-1, 1]."""
    sb_raw_subj = np.full_like(split_half_subj, np.nan, dtype=float)

    valid = np.isfinite(split_half_subj) & (split_half_subj > -1)
    sb_raw_subj[valid] = (
        2 * split_half_subj[valid] /
        (1 + split_half_subj[valid])
    )
    sb_subj = np.clip(sb_raw_subj, -1.0, 1.0)

    return sb_raw_subj, sb_subj


def compute_subject_ceiling(sb_subj):
    """
    Compute subject-level reliability ceiling matrices.

    A ceiling entry is set to zero whenever one of the two
    Spearman-Brown values is negative or non-finite. Otherwise it is
    computed as the square root of the product of the two values.
    """
    n_subj, n_roi = sb_subj.shape
    ceil_subj = np.full((n_subj, n_roi, n_roi), np.nan)

    for s in range(n_subj):
        for i in range(n_roi):
            for j in range(n_roi):
                sb_i = sb_subj[s, i]
                sb_j = sb_subj[s, j]

                if (
                    np.isfinite(sb_i)
                    and np.isfinite(sb_j)
                    and sb_i >= 0
                    and sb_j >= 0
                ):
                    ceil_subj[s, i, j] = np.sqrt(sb_i * sb_j)
                else:
                    ceil_subj[s, i, j] = 0.0

    return ceil_subj


def compute_corrected_similarity(cv_subj, ceil_subj):
    """
    Compute subject-level corrected similarity matrices.

    If the ceiling is positive, corrected similarity is obtained by
    division. If the ceiling is zero, the corrected value is set to 1,
    -1, or 0 depending on the sign of the numerator. Final values are
    clipped to [-1, 1].
    """
    n_subj, n_roi, _ = cv_subj.shape
    corr_subj = np.full((n_subj, n_roi, n_roi), np.nan)

    for s in range(n_subj):
        for i in range(n_roi):
            for j in range(n_roi):
                num = cv_subj[s, i, j]
                den = ceil_subj[s, i, j]

                if not np.isfinite(num) or not np.isfinite(den):
                    corr_subj[s, i, j] = np.nan
                elif den > 0:
                    corr_subj[s, i, j] = num / den
                else:
                    if num > 0:
                        corr_subj[s, i, j] = 1.0
                    elif num < 0:
                        corr_subj[s, i, j] = -1.0
                    else:
                        corr_subj[s, i, j] = 0.0

        corr_subj[s, :, :] = np.clip(corr_subj[s, :, :], -1.0, 1.0)

    return corr_subj


def compute_cv_similarity_matrix(y1, y2):
    """
    Compute a symmetric cross-validated ROI-ROI similarity matrix.

    Formula:
        r_xy^CV = 1/2 * [
            corr(x^(1), y^(2)) + corr(x^(2), y^(1))
        ]
    """
    n_roi = y1.shape[0]
    cv_mat = np.full((n_roi, n_roi), np.nan)

    for i in range(n_roi):
        for j in range(n_roi):
            corr_12 = safe_corr(y1[i, :], y2[j, :])
            corr_21 = safe_corr(y2[i, :], y1[j, :])

            if np.isfinite(corr_12) and np.isfinite(corr_21):
                cv_mat[i, j] = 0.5 * (corr_12 + corr_21)
            elif np.isfinite(corr_12):
                cv_mat[i, j] = corr_12
            elif np.isfinite(corr_21):
                cv_mat[i, j] = corr_21
            else:
                cv_mat[i, j] = np.nan

    return cv_mat


def save_results_to_rtf(results, output_path):
    """
    Save selected reliability results into an RTF file.

    Includes only:
    - Spearman-Brown corrected ROI reliability
    - Cross-validated ROI-ROI similarity
    - Reliability ceiling
    - Corrected similarity
    """
    sections = []

    for hemi, res in results.items():
        sections.append(f"# Hemisphere: {hemi}\n")

        sections.append("## Spearman-Brown corrected ROI reliability\n")
        sb_df = res["spearman_brown"].to_frame(name="Reliability")
        sections.append(sb_df.round(3).to_markdown())
        sections.append("\n")

        sections.append("## Cross-validated ROI-ROI similarity\n")
        sections.append(res["cv_similarity"].round(3).to_markdown())
        sections.append("\n")

        sections.append("## Reliability ceiling\n")
        sections.append(res["ceiling"].round(3).to_markdown())
        sections.append("\n")

        sections.append("## Corrected similarity\n")
        sections.append(
            res["corrected_similarity"].round(3).to_markdown()
        )
        sections.append("\n\n")

    text = "\n".join(sections)

    pypandoc.convert_text(
        text,
        "rtf",
        format="md",
        outputfile=output_path,
        extra_args=["--standalone"],
    )


def plot_reliability_vs_corrected_similarity(
    ceiling_mat,
    corrected_mat,
    roi_names,
    output_dir,
    hemi_label,
    tag,
):
    """
    Plot reliability ceiling vs corrected similarity,
    restricted to pairs involving dstr and/or cereb.
    """
    os.makedirs(output_dir, exist_ok=True)

    xs = []
    ys = []
    labels = []

    n_roi = len(roi_names)
    target_rois = {'dstr', 'cereb'}

    for i in range(n_roi):
        for j in range(i + 1, n_roi):
            roi_i = roi_names[i]
            roi_j = roi_names[j]

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
    ax.axvline(0, linestyle='--', linewidth=1)
    ax.set_xlabel('Reliability ceiling')
    ax.set_ylabel('Corrected similarity')
    ax.set_title(
        f'Reliability ceiling vs corrected similarity\n'
        f'(pairs with dstr and/or cereb)\n'
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


def compute_split_half_pipeline(
    data_by_roi,
    condition_names,
    output_dir,
    hemi_labels,
    tag,
):
    """
    Compute split-half, Spearman-Brown, CV similarity, ceiling, and
    corrected similarity.

    Recipe
    ------
    1. For each hemisphere, determine for every condition whether it has
       4 valid runs or 2 valid runs, and verify that run availability is
       consistent across ROIs and within subjects.
    2. For each subject and each of the 6 mixed split schemes, build two
       half-profiles per ROI across all conditions:
       - 4-run conditions contribute as 2 runs vs 2 runs;
       - 2-run conditions contribute as 1 run vs 1 run.
       Rest is appended as a final condition with value 0 in both halves.
    3. For each ROI and scheme, compute the raw split-half correlation
       between the two half-profiles. Keep all scheme-level values.
    4. Summarize raw split-half correlations within subject using the
       median across the 6 schemes. Save scheme-level, subject-level,
       and group-level summaries. Plot all scheme-level values and the
       subject medians. Report the percentage of negative schemes.
    5. Apply the Spearman-Brown correction to the subject-level median
       split-half values as 2r / (1 + r). Keep the raw transformed
       values to quantify how often they would fall below -1 or above 1,
       then clip the final Spearman-Brown values to [-1, 1]. Save
       subject-level and group-level summaries, plot the subject-level
       distributions, and report percentages capped below -1 and above
       1 per ROI.
    6. For each subject and scheme, compute the cross-validated ROI-ROI
       similarity matrix using:
           r_xy^CV = 1/2 * [
               corr(x^(1), y^(2)) + corr(x^(2), y^(1))
           ]
       Summarize within subject using the median across schemes, and
       across subjects using the median.
    7. Compute the subject-level reliability ceiling from the clipped
       Spearman-Brown values. If either ROI in a pair has a negative or
       non-finite Spearman-Brown value, set that subject-level ceiling
       entry to 0. Otherwise compute the square root of the product of
       the two subject-level Spearman-Brown values. Summarize across
       subjects using the median.
    8. Compute corrected similarity at the subject level. If the
       subject-level ceiling is positive, divide CV similarity by the
       ceiling. If the ceiling is 0, assign 1, -1, or 0 according to
       the sign of the subject-level CV similarity. Clip final corrected
       similarities to [-1, 1]. Summarize across subjects using the
       median.

    Parameters
    ----------
    data_by_roi : dict
        Maps ROI name to an array with shape
        (n_hemi, n_cond, n_run, n_subj).
    condition_names : list of str
        Condition names in array order.
    output_dir : str
        Directory where outputs will be saved.
    hemi_labels : list of str
        Hemisphere labels.
    tag : str
        Individualization tag.

    Returns
    -------
    results : dict
        One entry per hemisphere label with split-half,
        Spearman-Brown, cross-validated similarity, reliability
        ceiling, corrected similarity, and percentages of negative
        schemes and Spearman-Brown capping.
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

    results = {}

    for h, hemi_label in enumerate(hemi_labels):
        cond_n_runs = []

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

        split_half_subj = np.full((n_subj, n_roi), np.nan)
        split_half_scheme_subj = np.full(
            (n_subj, len(schemes), n_roi),
            np.nan,
        )
        cv_scheme_subj = np.full(
            (n_subj, len(schemes), n_roi, n_roi),
            np.nan,
        )

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

            for k, scheme in enumerate(schemes):
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

                    y1[r, -1] = 0.0
                    y2[r, -1] = 0.0

                split_half_vec = np.full(n_roi, np.nan)
                for r in range(n_roi):
                    split_half_vec[r] = safe_corr(y1[r, :], y2[r, :])

                split_half_scheme_subj[s, k, :] = split_half_vec
                cv_scheme_subj[s, k, :, :] = compute_cv_similarity_matrix(
                    y1, y2
                )

            split_half_subj[s, :] = np.nanmedian(
                split_half_scheme_subj[s, :, :],
                axis=0,
            )

        sb_raw_subj, sb_subj = compute_spearman_brown(split_half_subj)
        cv_subj = np.nanmedian(cv_scheme_subj, axis=1)
        ceil_subj = compute_subject_ceiling(sb_subj)
        corr_subj = compute_corrected_similarity(cv_subj, ceil_subj)

        split_half_group = np.nanmedian(split_half_subj, axis=0)
        sb_group = np.nanmedian(sb_subj, axis=0)
        cv_group = np.nanmedian(cv_subj, axis=0)
        ceil_group = np.nanmedian(ceil_subj, axis=0)
        corr_group = np.nanmedian(corr_subj, axis=0)

        np.save(
            os.path.join(
                output_dir,
                f"split_half_corr_scheme_subj_{hemi_label}_{tag}.npy",
            ),
            split_half_scheme_subj,
        )
        np.save(
            os.path.join(
                output_dir,
                f"split_half_corr_subj_{hemi_label}_{tag}.npy",
            ),
            split_half_subj,
        )
        np.save(
            os.path.join(
                output_dir,
                f"split_half_corr_{hemi_label}_{tag}.npy",
            ),
            split_half_group,
        )
        np.save(
            os.path.join(
                output_dir,
                f"spearman_brown_raw_subj_{hemi_label}_{tag}.npy",
            ),
            sb_raw_subj,
        )
        np.save(
            os.path.join(
                output_dir,
                f"spearman_brown_subj_{hemi_label}_{tag}.npy",
            ),
            sb_subj,
        )
        np.save(
            os.path.join(
                output_dir,
                f"spearman_brown_{hemi_label}_{tag}.npy",
            ),
            sb_group,
        )
        np.save(
            os.path.join(
                output_dir,
                f"cv_similarity_scheme_subj_{hemi_label}_{tag}.npy",
            ),
            cv_scheme_subj,
        )
        np.save(
            os.path.join(
                output_dir,
                f"cv_similarity_subj_{hemi_label}_{tag}.npy",
            ),
            cv_subj,
        )
        np.save(
            os.path.join(
                output_dir,
                f"cv_similarity_{hemi_label}_{tag}.npy",
            ),
            cv_group,
        )
        np.save(
            os.path.join(
                output_dir,
                f"ceiling_subj_{hemi_label}_{tag}.npy",
            ),
            ceil_subj,
        )
        np.save(
            os.path.join(
                output_dir,
                f"ceiling_{hemi_label}_{tag}.npy",
            ),
            ceil_group,
        )
        np.save(
            os.path.join(
                output_dir,
                f"corrected_similarity_subj_{hemi_label}_{tag}.npy",
            ),
            corr_subj,
        )
        np.save(
            os.path.join(
                output_dir,
                f"corrected_similarity_{hemi_label}_{tag}.npy",
            ),
            corr_group,
        )

        plot_dir = os.path.join(output_dir, 'split_half_distributions')
        plot_split_half_distributions(
            split_half_scheme_subj,
            roi_names,
            plot_dir,
            hemi_label,
            tag,
        )

        sb_plot_dir = os.path.join(
            output_dir,
            'spearman_brown_distributions'
        )
        plot_spearman_brown_distributions(
            sb_subj,
            roi_names,
            sb_plot_dir,
            hemi_label,
            tag,
        )

        neg_dir = os.path.join(output_dir, 'negative_scheme_summary')
        perc_neg = summarize_negative_schemes(
            split_half_scheme_subj,
            roi_names,
            neg_dir,
            hemi_label,
            tag,
        )

        cap_dir = os.path.join(output_dir, 'spearman_brown_capping_summary')
        perc_sb_below, perc_sb_above = summarize_spearman_brown_capping(
            sb_raw_subj,
            roi_names,
            cap_dir,
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
            'split_half_corr': pd.Series(
                split_half_group,
                index=roi_names,
                name=f'split_half_corr_{hemi_label}_{tag}',
            ),
            'spearman_brown': pd.Series(
                sb_group,
                index=roi_names,
                name=f'spearman_brown_{hemi_label}_{tag}',
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
            'split_half_corr_subj': pd.DataFrame(
                split_half_subj,
                columns=roi_names,
            ),
            'spearman_brown_subj': pd.DataFrame(
                sb_subj,
                columns=roi_names,
            ),
            'perc_negative_schemes': perc_neg,
            'perc_sb_capped_below_m1': perc_sb_below,
            'perc_sb_capped_above_p1': perc_sb_above,
        }

    return results


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

# Threshold of group-level contrast used to generate ROI
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
roi_signals_folder = os.path.join(
    reliability_folder,
    f'taskglm_roi_signals_{smooth}'
)
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
    f'roi_analyses_rwls_hrf128_wb_{thresh_type}_{smooth}'
)

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
    #                        region_names, atlas_names, ROI_NAMES,
    #                        hemispheres, iroi_main_dir, smooth,
    #                        roi_signals_folder)

    for itag in itags:
        data_by_roi = load_roi_signal_arrays(
            roi_signals_folder,
            indiv=itag,
        )

        results = compute_split_half_pipeline(
            data_by_roi,
            condition_names,
            split_half_folder,
            hemispheres,
            itag,
        )

        for hemi in hemispheres:
            print(f"\nTag: {itag} | Hemisphere: {hemi}")

            print("\nSplit-half correlations")
            print(results[hemi]['split_half_corr'])

            print("\nSpearman-Brown")
            print(results[hemi]['spearman_brown'])

            print("\nPercentage of negative schemes")
            print(results[hemi]['perc_negative_schemes'])

            print("\nPercentage of Spearman-Brown capped below -1")
            print(results[hemi]['perc_sb_capped_below_m1'])

            print("\nPercentage of Spearman-Brown capped above 1")
            print(results[hemi]['perc_sb_capped_above_p1'])

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