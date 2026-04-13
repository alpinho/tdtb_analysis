#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Subject-wise ROI profile correlations with paired t-tests.

This script follows the same data-loading and wide-vector scheme used in
roi_similarity_encoding.py, but replaces repeated-measures correlation by
subject-wise Pearson correlations.

For the selected individualization X modality X hemisphere:
- Build Subject X Task X Category X Modality PSC vectors for each ROI
  pair.
- Optionally merge random-NTFD source.
- Optionally add one synthetic Rest row (PSC=0) per Subject X ROI.
- Compute Pearson correlations separately for each subject.
- Summarize all pairs involving cerebellum or dorsal striatum.
- For each cortical ROI, run a paired t-test comparing:
  target-cerebellum vs target-dstr subject-wise correlations.
- Write TSV outputs and simple summary plots.

Author: Ana Luisa Pinho
email: agrilopi@uwo.ca

Date of creation: 9th of March 2026
Last update: April 2026

Compatibility: Python 3.10+
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg

from matplotlib.colors import to_rgb
from plot_anovas import bootstrap_conf_intervals, _poly_xspan_at_y


# =========================== FUNCTIONS ============================= #


def _mod_mask(df: pd.DataFrame, modality: str) -> pd.Series:
    """
    Select modality rows. 'Both' stacks Auditory and Visual.
    """
    if 'Modality' not in df.columns:
        raise ValueError("Column 'Modality' not found in dataframe.")
    if modality == 'Both':
        return df['Modality'].isin(['Auditory', 'Visual'])
    return df['Modality'] == modality


def _add_rest_rows(sub: pd.DataFrame) -> pd.DataFrame:
    """
    Add one Rest row (PSC=0) per Subject X ROI. Tags set to 'Rest'.
    """
    cols = ['Subject', 'Task', 'Category', 'Modality', 'ROI', 'PSC']
    for col in cols:
        if col not in sub.columns:
            raise ValueError(f"Missing column '{col}' in sub-dataframe.")
    keys = sub[['Subject', 'ROI']].drop_duplicates().copy()
    keys['Task'] = 'Rest'
    keys['Category'] = 'Rest'
    keys['Modality'] = 'Rest'
    keys['PSC'] = 0.0
    rest = keys[cols].copy()
    out = pd.concat([sub[cols], rest], axis=0, ignore_index=True)
    out = out.drop_duplicates(subset=cols)
    return out


def p_to_stars(p_val: float) -> str:
    """
    Return significance stars for a p-value.
    """
    if not np.isfinite(p_val):
        return ''
    if p_val < 0.0001:
        return '****'
    if p_val < 0.001:
        return '***'
    if p_val < 0.01:
        return '**'
    if p_val < 0.05:
        return '*'
    return ''


def load_df(indiv: str) -> pd.DataFrame:
    """
    Load and optionally merge per-individualization dataframes.

    Sources
    -------
    - main_tasks/df_rois_volume (always)
    - rand_ntfd_pairs/df_rois_volume (when USE_RAND)

    Returns
    -------
    Non-empty dataframe filtered to allowed Tasks and Categories.
    """
    path_main = os.path.join(
        DF_DIR_MAIN, f"dfrois_{indiv}_{N_ROIS}-rois.tsv"
    )
    path_rand = os.path.join(
        DF_DIR_RAND, f"dfrois_{indiv}_{N_ROIS}-rois.tsv"
    )

    print(f"[LOAD] indiv={indiv} n_rois={N_ROIS}")
    print(f"[LOAD] main path : {path_main}")
    print(f"[LOAD] rand path : {path_rand} (USE_RAND={USE_RAND})")

    dtypes = {
        'Subject': str,
        'Task': str,
        'ROI': str,
        'Hemisphere': str,
        'PSC': float,
        'Modality': str,
        'Category': str,
    }

    if not os.path.exists(path_main):
        raise FileNotFoundError(f"[ERROR] main file not found: {path_main}")

    df_main = pd.read_csv(path_main, sep='\t', dtype=dtypes)
    dfs: List[pd.DataFrame] = [df_main]

    if USE_RAND:
        if not os.path.exists(path_rand):
            raise FileNotFoundError(
                f"[ERROR] rand file not found: {path_rand}"
            )
        df_rand = pd.read_csv(path_rand, sep='\t', dtype=dtypes)
        tasks_rand = (
            df_rand['Task'].astype(str).str.strip().unique().tolist()
        )
        bad_tasks = [task for task in tasks_rand if task != 'NTFD Random']
        if bad_tasks:
            raise ValueError(
                "[ERROR] rand source has non-'NTFD Random' Task values: "
                f"{sorted(set(bad_tasks))}"
            )

        cats = df_rand['Category'].astype(str).str.strip().unique()
        ok = {'Beat', 'Interval', 'Random'}
        bad_cats = sorted(set(cats) - ok)
        if bad_cats:
            raise ValueError(
                "[ERROR] rand source Category outside allowed set "
                f"{sorted(ok)}: {bad_cats}"
            )
        dfs.append(df_rand)

    df = pd.concat(dfs, axis=0, ignore_index=True)

    for col in ['Subject', 'Task', 'Category', 'Modality',
                'ROI', 'Hemisphere']:
        if col not in df.columns:
            raise ValueError(f"[ERROR] missing column '{col}' in data.")
        df[col] = df[col].astype(str).str.strip()

    if 'PSC' not in df.columns:
        raise ValueError("[ERROR] missing column 'PSC' in merged data.")

    df = df[df['Task'].isin(TASKS_NO_REST)]
    df = df[df['Category'].isin(CATS)]

    if df.empty:
        raise ValueError(
            "[ERROR] dataframe empty after Task/Category filtering."
        )

    return df


def wide_for_corr(
    df: pd.DataFrame,
    hemi: str,
    modality: str,
    roi1: str,
    roi2: str,
    add_rest: bool = False,
) -> pd.DataFrame:
    """
    Build Subject X Task X Category X Modality table for two ROIs.

    This follows the same recipe as wide_for_rmcorr in the original
    script.
    """
    if df.empty:
        raise ValueError("[ERROR] input dataframe to wide_for_corr is empty.")

    req = ['Hemisphere', 'ROI', 'Modality', 'Subject', 'Task',
           'Category', 'PSC']
    for col in req:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' for wide build.")

    mask = (
        (df['Hemisphere'] == hemi) &
        (df['ROI'].isin([roi1, roi2])) &
        _mod_mask(df, modality)
    )
    sub = df.loc[mask].copy()
    if sub.empty:
        raise ValueError(
            "[ERROR] selection for hemi/modality/ROIs returned empty."
        )

    sub = (
        sub.groupby(
            ['Subject', 'Task', 'Category', 'Modality', 'ROI'],
            as_index=False
        )['PSC'].mean()
    )

    if add_rest:
        sub = _add_rest_rows(sub)

    wide = (
        sub.pivot_table(
            index=['Subject', 'Task', 'Category', 'Modality'],
            columns='ROI',
            values='PSC'
        ).reset_index()
    )

    if roi1 not in wide.columns or roi2 not in wide.columns:
        raise ValueError(
            "[ERROR] pivot missing ROI columns "
            f"('{roi1}', '{roi2}')."
        )

    wide = wide.dropna(subset=[roi1, roi2])
    if wide.empty:
        raise ValueError(
            "[ERROR] both ROI columns contain only NaNs after pivot."
        )

    cols = ['Subject', 'Task', 'Category', 'Modality', roi1, roi2]
    wide = wide.loc[:, cols]
    wide.index.name = None
    wide.columns.name = None
    wide = wide.sort_values(
        ['Subject', 'Task', 'Category', 'Modality']
    ).reset_index(drop=True)

    return wide


def ordered_rois_in_df(df: pd.DataFrame) -> List[str]:
    """
    Return ROIs in the canonical paper order.
    """
    if 'ROI' not in df.columns:
        raise ValueError("[ERROR] missing 'ROI' column in dataframe.")

    present = set(df['ROI'].astype(str).str.strip().unique().tolist())

    ordered: List[str] = []
    missing: List[str] = []
    for group in ROI_ORDER_GROUPS:
        picked = next((key for key in group if key in present), None)
        if picked is None:
            missing.append('/'.join(group))
        else:
            ordered.append(picked)

    if missing:
        raise ValueError(
            "[ERROR] missing expected ROIs for ordering: "
            f"{missing}"
        )

    return ordered


def validate_plot_targets(
    plot_targets: List[str] | None,
    cortical_targets: List[str],
) -> List[str]:
    """
    Validate and return plot targets.
    """
    if plot_targets is None:
        return cortical_targets

    missing = [
        roi for roi in plot_targets if roi not in cortical_targets
    ]
    if missing:
        raise ValueError(
            "[ERROR] invalid ROI(s) in PLOT_TARGETS: "
            f"{missing}. Allowed values: {cortical_targets}"
        )

    return plot_targets


def compute_subject_corrs(
    wide: pd.DataFrame,
    roi1: str,
    roi2: str,
) -> pd.DataFrame:
    """
    Compute one Pearson correlation per subject for a ROI pair.
    """
    rows: List[Dict[str, object]] = []
    for subject, sub_df in wide.groupby('Subject', sort=True):
        if len(sub_df) < 2:
            continue
        r_val = sub_df[roi1].corr(sub_df[roi2], method='pearson')
        if pd.isna(r_val):
            continue
        rows.append({
            'Subject': subject,
            'roi1': roi1,
            'roi2': roi2,
            'pair': f'{roi1}-{roi2}',
            'r': float(r_val),
            'n_points': len(sub_df),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(
            f"[ERROR] no valid subject-wise correlations for {roi1}-{roi2}."
        )
    return out


def summarize_pair(corrs: pd.DataFrame, indiv: str) -> Dict[str, object]:
    """
    Summarize subject-wise correlations for one ROI pair.
    """
    r_vals = corrs['r'].to_numpy(dtype=float)
    roi1 = str(corrs['roi1'].iloc[0])
    roi2 = str(corrs['roi2'].iloc[0])

    return {
        'individualization': indiv,
        'pair': f'{roi1}-{roi2}',
        'roi1': roi1,
        'roi2': roi2,
        'mean_r': float(np.mean(r_vals)),
        'median_r': float(np.median(r_vals)),
        'sd_r': float(np.std(r_vals, ddof=1)) if len(r_vals) > 1 else np.nan,
        'min_r': float(np.min(r_vals)),
        'max_r': float(np.max(r_vals)),
        'n_subj': int(corrs['Subject'].nunique()),
        'n_points_per_subj': int(corrs['n_points'].iloc[0]),
    }


def paired_tests_from_subject_corrs(
    all_corrs: pd.DataFrame,
    cortical_targets: List[str],
    indiv: str,
) -> pd.DataFrame:
    """
    Compare target-cereb and target-dstr subject-wise correlations.
    """
    rows: List[Dict[str, object]] = []

    for target in cortical_targets:
        pair_c = f'cereb-{target}'
        pair_d = f'dstr-{target}'

        sub_c = all_corrs[all_corrs['pair'] == pair_c]
        sub_d = all_corrs[all_corrs['pair'] == pair_d]

        merged = pd.merge(
            sub_c[['Subject', 'r']],
            sub_d[['Subject', 'r']],
            on='Subject',
            suffixes=('_cereb', '_dstr')
        )
        if merged.empty:
            raise ValueError(
                f"[ERROR] no overlapping subjects for {target}."
            )

        t_res = pg.ttest(
            merged['r_cereb'],
            merged['r_dstr'],
            paired=True,
            alternative='two-sided'
        )

        p_col = 'p-val' if 'p-val' in t_res.columns else 'p-val'
        ci_col = 'CI95%'
        ci_vals = t_res[ci_col].iloc[0]
        if isinstance(ci_vals, (list, tuple, np.ndarray)):
            ci_low = float(ci_vals[0])
            ci_high = float(ci_vals[1])
        else:
            ci_low = np.nan
            ci_high = np.nan

        rows.append({
            'individualization': indiv,
            'target_roi': target,
            'pair_cereb': pair_c,
            'pair_dstr': pair_d,
            'mean_r_cereb': float(merged['r_cereb'].mean()),
            'mean_r_dstr': float(merged['r_dstr'].mean()),
            't': float(t_res['T'].iloc[0]),
            'dof': float(t_res['dof'].iloc[0]),
            'p_uncorr': float(t_res[p_col].iloc[0]),
            'cohen_d': float(t_res['cohen-d'].iloc[0]),
            'ci95_low': ci_low,
            'ci95_high': ci_high,
            'n_subj': int(len(merged)),
            'significant': bool(float(t_res[p_col].iloc[0]) < ALPHA),
        })

    return pd.DataFrame(rows)


def plot_seed_vs_target_boxplots(
    all_corrs: pd.DataFrame,
    paired_df: pd.DataFrame,
    targets: List[str],
    out_png: Path,
    indiv: str,
    hemi: str,
    modality: str,
    ylim: tuple[float, float] = (-0.7, 1.0),
) -> None:
    """
    Paired boxplots with bootstrap notches and mean lines.
    """
    # Match spacing of 6-target reference figure
    base_targets = 6
    base_width = 6.4
    width_scale = len(targets) / base_targets
    fig_width = base_width * width_scale

    # Keep constant pixels per y-unit across ylim choices
    base_ylim = (-0.7, 1.0)
    base_height = 4.8
    y_range = ylim[1] - ylim[0]
    base_y_range = base_ylim[1] - base_ylim[0]
    height_scale = y_range / base_y_range
    fig_height = base_height * height_scale

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    dstr_color = '#E69F00'
    cereb_color = '#56B4E9'

    positions = []
    data = []
    centers = []

    pos = 1.0
    rng = np.random.default_rng(7)

    for target in targets:
        vals_d = all_corrs.loc[
            all_corrs['pair'] == f'dstr-{target}', 'r'
        ].to_numpy(dtype=float)
        vals_c = all_corrs.loc[
            all_corrs['pair'] == f'cereb-{target}', 'r'
        ].to_numpy(dtype=float)

        positions.extend([pos, pos + 0.17])
        data.extend([vals_d, vals_c])
        centers.append(pos + 0.12)
        pos += 0.6

    conf_intervals = bootstrap_conf_intervals(
        data=data,
        n_boot=5000,
        alpha=0.05,
        seed=12345,
    )

    box_w = 0.14
    box_lw = 1.2

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=box_w,
        notch=True,
        patch_artist=True,
        showfliers=False,
        showmeans=False,
        meanline=False,
        whis=1.5,
        medianprops={'linewidth': 0, 'color': 'none'},
        conf_intervals=conf_intervals,
    )

    for i, box in enumerate(bp['boxes']):
        color = dstr_color if i % 2 == 0 else cereb_color
        face_color = (*plt.matplotlib.colors.to_rgb(color), 0.35)
        box.set_facecolor(face_color)
        box.set_edgecolor(color)
        box.set_linewidth(box_lw)

    for i, whisker in enumerate(bp['whiskers']):
        color = dstr_color if (i // 2) % 2 == 0 else cereb_color
        whisker.set_color(color)
        whisker.set_linewidth(1.0)

    for i, cap in enumerate(bp['caps']):
        color = dstr_color if (i // 2) % 2 == 0 else cereb_color
        cap.set_color(color)
        cap.set_linewidth(1.0)

    for patch, vals in zip(bp['boxes'], data):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue

        mean_val = float(np.mean(vals))
        edge_color = patch.get_edgecolor()

        span = _poly_xspan_at_y(patch, mean_val)
        if span is None:
            verts = patch.get_path().vertices
            x_left = float(verts[:, 0].min())
            x_right = float(verts[:, 0].max())
        else:
            x_left, x_right = span

        ax.plot(
            [x_left, x_right],
            [mean_val, mean_val],
            color=edge_color,
            lw=patch.get_linewidth(),
            zorder=3,
            solid_capstyle='butt',
        )

    for i, (xpos, vals) in enumerate(zip(positions, data)):
        color = dstr_color if i % 2 == 0 else cereb_color
        jitter = rng.normal(0.0, 0.025, size=len(vals))

        dot_color = tuple(np.array(to_rgb(color)) * 0.75)
        ax.scatter(
            np.full(len(vals), xpos) + jitter,
            vals,
            s=12,
            facecolors=dot_color,
            edgecolors='none',
            alpha=.65,
            zorder=3,
        )

    ax.axhline(
        0.0,
        linewidth=1.0,
        color='0.4',
        linestyle='--',
        dashes=(4, 3),
    )
    ax.set_ylim(ylim)
    # ax.set_yticks(np.arange(-0.75, 1.01, 0.25))

    labels = [ROI_LABELS.get(t, t) for t in targets]
    labels = ['PreSMA' if lab == 'preSMA' else lab for lab in labels]

    ax.set_xticks(centers)
    ax.set_xticklabels(labels)
    ax.set_xlim(positions[0] - 0.225, positions[-1] + 0.225)
    ax.margins(x=0)

    ax.set_ylabel('Subject-wise Correlation ($r_s$)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for idx, target in enumerate(targets):
        row = paired_df.loc[paired_df['target_roi'] == target]
        if row.empty:
            continue

        stars = p_to_stars(float(row['p_uncorr'].iloc[0]))
        if not stars:
            continue

        ax.text(
            centers[idx],
            ylim[1] + 0.05 * y_range,
            stars,
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold',
            clip_on=False,
        )

    handles = [
        plt.Line2D(
            [0], [0],
            marker='s',
            linestyle='',
            markerfacecolor=dstr_color,
            markeredgecolor=dstr_color,
            label='Dorsal Striatum',
            markersize=7,
        ),
        plt.Line2D(
            [0], [0],
            marker='s',
            linestyle='',
            markerfacecolor=cereb_color,
            markeredgecolor=cereb_color,
            label='Cerebellum',
            markersize=7,
        ),
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        loc='upper right',
        bbox_to_anchor=(0.995, 1.02),
    )

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {out_png}")


# =========================== USER INPUTS =========================== #

ALPHA: float = 0.05
N_ROIS: int = 8

ADD_REST: bool = True
USE_RAND: bool = True

INDIVID_LEVELS: List[str] = ['i']
HEMIS: List[str] = ['bh']
MODALITIES: List[str] = ['Both']
YLIM: tuple[float, float] = (-0.2, 1.0)

BASE_TASKS: List[str] = ['Production', 'Perception', 'NTFD']
TASKS_NO_REST: List[str] = (
    BASE_TASKS + ['NTFD Random'] if USE_RAND else BASE_TASKS
)
TASKS: List[str] = TASKS_NO_REST + ['Rest']

CATS: List[str] = (
    ['Beat', 'Interval', 'Random'] if USE_RAND else ['Beat', 'Interval']
)

ROI_LABELS: Dict[str, str] = {
    'dstr': 'Dorsal Striatum',
    'cereb': 'Cerebellum',
    'pmv': 'PMV',
    'pmd': 'PMD',
    'presma': 'preSMA',
    'sma': 'SMA',
    'heschl': "Heschl's\nGyrus",
    'occipital': 'Occipital\nLobe',
    'occipital_lobe': 'Occipital\nLobe',
}

ROI_ORDER_GROUPS: List[List[str]] = [
    ['dstr'],
    ['cereb'],
    ['presma'],
    ['sma'],
    ['pmd'],
    ['pmv'],
    ['heschl'],
    ['occipital_lobe', 'occipital'],
]

SEEDS: List[str] = ['cereb', 'dstr']

# Set to None to include all cortical targets in the plot.
# Example excluding heschl and occipital:
# PLOT_TARGETS: List[str] | None = ['presma', 'sma', 'pmd', 'pmv',
#                                   'heschl', 'occipital']
PLOT_TARGETS: List[str] | None = ['presma', 'sma', 'pmd', 'pmv']


# =============================== PATHS ============================== #

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL = 'rwls'
MASKING = 'wb'
HRF = 'hrf128'

BASE_ALL = os.path.join(
    WORKING_DIR,
    f"roi_analyses_{MODEL}_{HRF}_{MASKING}_puncorr_unsmoothed",
    'bothmod_allmain_tasks',
)

BASE_DIR_MAIN = os.path.join(BASE_ALL, 'main_tasks')
BASE_DIR_RAND = os.path.join(BASE_ALL, 'rand_ntfd_pairs')

DF_DIR_MAIN = os.path.join(BASE_DIR_MAIN, 'df_rois_volume')
DF_DIR_RAND = os.path.join(BASE_DIR_RAND, 'df_rois_volume')

if USE_RAND and ADD_REST:
    SUBJECTCORR_DIRNAME = 'subjectcorr_paired_restrand'
    FILETAG = 'withrestrand'
elif USE_RAND and not ADD_REST:
    SUBJECTCORR_DIRNAME = 'subjectcorr_paired_rand'
    FILETAG = 'withrand'
elif not USE_RAND and ADD_REST:
    SUBJECTCORR_DIRNAME = 'subjectcorr_paired_rest'
    FILETAG = 'withrest'
else:
    SUBJECTCORR_DIRNAME = 'subjectcorr_paired'
    FILETAG = 'norand'

OUT_ROOT = Path(BASE_ALL) / 'profile_similarity' / SUBJECTCORR_DIRNAME


# ================================ RUN =============================== #

if __name__ == '__main__':

    for indiv in INDIVID_LEVELS:
        df_all = load_df(indiv)
        rois = ordered_rois_in_df(df_all)
        cortical_targets = [roi for roi in rois if roi not in SEEDS]
        plot_targets = validate_plot_targets(
            PLOT_TARGETS,
            cortical_targets
        )
        n_plot_targets = len(plot_targets)

        indiv_root = OUT_ROOT / indiv
        os.makedirs(indiv_root, exist_ok=True)

        for modality in MODALITIES:
            for hemi in HEMIS:
                all_corr_rows: List[pd.DataFrame] = []
                pair_rows: List[Dict[str, object]] = []

                seed_targets = []
                for seed in SEEDS:
                    for target in rois:
                        if target == seed:
                            continue
                        seed_targets.append((seed, target))

                for roi1, roi2 in seed_targets:
                    wide = wide_for_corr(
                        df_all,
                        hemi=hemi,
                        modality=modality,
                        roi1=roi1,
                        roi2=roi2,
                        add_rest=ADD_REST,
                    )

                    # Diagnostic: print vector length per subject
                    vec_lengths = wide.groupby('Subject').size()
                    print(
                        f"[VECTOR LENGTH] {roi1}-{roi2} | "
                        f"{indiv} | {modality} | {hemi} | "
                        f"min={vec_lengths.min()} "
                        f"max={vec_lengths.max()} "
                        f"unique={sorted(vec_lengths.unique())}"
                    )

                    corr_df = compute_subject_corrs(wide, roi1, roi2)
                    corr_df['individualization'] = indiv
                    corr_df['modality'] = modality
                    corr_df['hemisphere'] = hemi
                    all_corr_rows.append(corr_df)
                    pair_rows.append(summarize_pair(corr_df, indiv))

                all_corrs = pd.concat(all_corr_rows, axis=0, ignore_index=True)
                pair_summary = pd.DataFrame(pair_rows)
                paired_df = paired_tests_from_subject_corrs(
                    all_corrs=all_corrs,
                    cortical_targets=cortical_targets,
                    indiv=indiv,
                )

                reject, p_corr = pg.multicomp(
                    paired_df['p_uncorr'].to_numpy(dtype=float),
                    alpha=ALPHA,
                    method='holm',
                )

                paired_df.insert(
                    paired_df.columns.get_loc('p_uncorr') + 1,
                    'p_holm',
                    p_corr
                )

                paired_df['significant_holm'] = reject
                paired_df['modality'] = modality
                paired_df['hemisphere'] = hemi

                stem = (
                    f'{indiv}_{modality}_{hemi}_'
                    f'{n_plot_targets}-targets_{FILETAG}'
                )

                out_corrs = indiv_root / f'subject_corrs_{stem}.tsv'
                out_pairs = indiv_root / f'pair_summary_{stem}.tsv'
                out_ttest = indiv_root / f'paired_tests_{stem}.tsv'

                all_corrs.to_csv(out_corrs, sep='\t', index=False)
                pair_summary.to_csv(out_pairs, sep='\t', index=False)
                paired_df.to_csv(out_ttest, sep='\t', index=False)
                print(f"[SAVED] {out_corrs}")
                print(f"[SAVED] {out_pairs}")
                print(f"[SAVED] {out_ttest}")

                out_png = indiv_root / f'paired_boxplots_{stem}.png'
                plot_seed_vs_target_boxplots(
                    all_corrs=all_corrs,
                    paired_df=paired_df,
                    targets=plot_targets,
                    out_png=out_png,
                    indiv=indiv,
                    hemi=hemi,
                    modality=modality,
                    ylim=YLIM,
                )