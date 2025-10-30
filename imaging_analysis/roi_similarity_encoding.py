#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Profile similarity (encoding) with repeated-measures correlation.

For each individualization X modality X hemisphere:
- Build Subject X Task X Category X Modality PSC for two ROIs.
- Optionally merge random-NTFD source:
  main_tasks/df_rois_volume and
  rand_ntfd_pairs/df_rois_volume (when USE_RAND).
  Random source is validated:
  Task must be 'NTFD Random' and Category must be one of
  {'Beat', 'Interval', 'Random'}. Otherwise, raise an error.
- Optional synthetic Rest row (PSC=0) per Subject X ROI:
  Task='Rest', Category='Rest', Modality='Rest'.
- Compute rm-corr (Pingouin) for all ROI pairs (alphabetical).
- Plot ONLY significant pairs (p < ALPHA).

Outputs per individualization under:
BASE_ALL/profile_similarity/<encoding|encoding_rest|encoding_rand|
encoding_restrand>/ with buckets:
- cereb_only: includes cereb, excludes dstr
- dstr_only : includes dstr, excludes cereb
- hmat      : remaining (incl. cereb-dstr)

Also writes:
- A TSV per (modality X hemisphere) with r, p, n_subj, n_points and
  'significant'.
- ROI X ROI matrices (r/p/sig TSVs + PNG). Matrix PNG shows all r;
  significant cells have stars; NaNs are not expected in strict mode.

Author: Ana Luisa Pinho
email: agrilopi@uwo.ca

Created: 7th of October 2025
Last Update: October 2025

Compatibility: Python 3.10.16
"""

from __future__ import annotations

import os
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg


# ============================== HELPERS ============================= #

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
    for c in cols:
        if c not in sub.columns:
            raise ValueError(f"Missing column '{c}' in sub-dataframe.")
    keys = sub[['Subject', 'ROI']].drop_duplicates().copy()
    keys['Task'] = 'Rest'
    keys['Category'] = 'Rest'
    keys['Modality'] = 'Rest'
    keys['PSC'] = 0.0
    rest = keys[cols].copy()
    out = pd.concat([sub[cols], rest], axis=0, ignore_index=True)
    out = out.drop_duplicates(subset=cols)
    return out


STAR_THRESHOLDS: List[Tuple[float, str]] = [
    (0.0001, '****'),
    (0.001, '***'),
    (0.01, '**'),
    (0.05, '*'),
]


def p_to_stars(p: float) -> str:
    """
    Return significance stars for a p-value.
    """
    if not np.isfinite(p):
        return ''
    for thr, sym in STAR_THRESHOLDS:
        if p < thr:
            return sym
    return ''


# =========================== PUBLIC HELPERS ========================== #

def load_df(indiv: str) -> pd.DataFrame:
    """
    Load and optionally merge per-individualization dataframes.

    Sources
    -------
    - main_tasks/df_rois_volume (always)
    - rand_ntfd_pairs/df_rois_volume (when USE_RAND)

    Random source validation
    ------------------------
    - Task must be exactly 'NTFD Random' for all rows.
    - Category must be in {'Beat', 'Interval', 'Random'}.
    - No relabeling or coercion is performed.

    Returns
    -------
    Non-empty dataframe filtered to allowed Tasks and Categories.

    Raises
    ------
    FileNotFoundError if a required file is missing.
    ValueError if validation or filtering fails.
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
        'Subject': str, 'Task': str, 'ROI': str, 'Hemisphere': str,
        'PSC': float, 'Modality': str, 'Category': str,
    }

    if not os.path.exists(path_main):
        raise FileNotFoundError(
            f"[ERROR] main file not found: {path_main}"
        )

    df_main = pd.read_csv(path_main, sep='\t', dtype=dtypes)
    print(f"[LOAD] main rows: {len(df_main)}")

    dfs: List[pd.DataFrame] = [df_main]

    if USE_RAND:
        if not os.path.exists(path_rand):
            raise FileNotFoundError(
                f"[ERROR] rand file not found: {path_rand}"
            )
        df_rand = pd.read_csv(path_rand, sep='\t', dtype=dtypes)

        # --- strict validation: do NOT change the data ---
        for col in ['Task', 'Category']:
            if col not in df_rand.columns:
                raise ValueError(
                    f"[ERROR] rand source missing '{col}' column."
                )

        tasks_rand = (
            df_rand['Task'].astype(str).str.strip().unique().tolist()
        )
        bad_tasks = [t for t in tasks_rand if t != 'NTFD Random']
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
        # --------------------------------------------------

        dfs.append(df_rand)

    df = pd.concat(dfs, axis=0, ignore_index=True)

    for col in ['Subject', 'Task', 'Category', 'Modality',
                'ROI', 'Hemisphere']:
        if col not in df.columns:
            raise ValueError(
                f"[ERROR] missing column '{col}' in merged data."
            )
        df[col] = df[col].astype(str).str.strip()

    if 'PSC' not in df.columns:
        raise ValueError("[ERROR] missing column 'PSC' in merged data.")

    df = df[df['Task'].isin(TASKS_NO_REST)]
    df = df[df['Category'].isin(CATS)]

    if df.empty:
        raise ValueError(
            "[ERROR] dataframe empty after Task/Category filtering."
        )

    exp_tasks = set(TASKS_NO_REST)
    got_tasks = set(df['Task'].unique())
    if not exp_tasks.issubset(got_tasks):
        missing = sorted(exp_tasks - got_tasks)
        raise ValueError(
            f"[ERROR] missing expected Task levels: {missing}"
        )

    exp_cats = set(CATS)
    got_cats = set(df['Category'].unique())
    if not exp_cats.issubset(got_cats):
        missing = sorted(exp_cats - got_cats)
        raise ValueError(
            f"[ERROR] missing expected Category levels: {missing}"
        )

    return df


def wide_for_rmcorr(
    df: pd.DataFrame,
    hemi: str,
    modality: str,
    roi1: str,
    roi2: str,
    add_rest: bool = False,
) -> pd.DataFrame:
    """
    Build Subject X Task X Category X Modality table for two ROIs.

    Returns a non-empty dataframe with both ROI columns present.

    Raises
    ------
    ValueError if any required condition is not satisfied.
    """
    if df.empty:
        raise ValueError("[ERROR] input dataframe to wide_for_rmcorr is "
                         "empty.")

    for col in ['Hemisphere', 'ROI', 'Modality', 'Subject', 'Task',
                'Category', 'PSC']:
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


def group_means_for_plot(
    df: pd.DataFrame,
    hemi: str,
    modality: str,
    roi1: str,
    roi2: str,
    add_rest: bool = False,
) -> pd.DataFrame:
    """
    Compute group-mean PSC per Task for roi1 and roi2.

    Raises
    ------
    ValueError if selection is empty or pivot fails.
    """
    mask = (
        (df['Hemisphere'] == hemi) &
        (df['ROI'].isin([roi1, roi2])) &
        _mod_mask(df, modality)
    )
    sub = df.loc[mask].copy()
    if sub.empty:
        raise ValueError(
            "[ERROR] selection for group means returned empty."
        )
    if add_rest:
        sub = _add_rest_rows(sub)
    grp = sub.groupby(['ROI', 'Task'])['PSC'].mean().reset_index()
    idx = TASKS if add_rest else TASKS_NO_REST
    mat = grp.pivot(index='Task', columns='ROI', values='PSC').reindex(idx)
    if mat.isna().all(axis=None):
        raise ValueError(
            "[ERROR] group-mean pivot resulted only in NaNs."
        )
    return mat


def bucket_for_pair(roi_a: str, roi_b: str) -> str:
    """
    Decide bucket for a ROI pair (alphabetical order).
    """
    a, b = sorted([roi_a, roi_b])
    if 'cereb' in (a, b) and 'dstr' not in (a, b):
        return 'cereb_only'
    if 'dstr' in (a, b) and 'cereb' not in (a, b):
        return 'dstr_only'
    return 'hmat'


def plot_profiles(
    mat: pd.DataFrame,
    r_val: float,
    p_val: float,
    indiv: str,
    modality: str,
    hemi: str,
    roi1: str,
    roi2: str,
    tasks_order: List[str],
    out_png: Path,
) -> None:
    """
    Plot group-mean ROI profiles with rm-corr annotation.

    Raises
    ------
    ValueError if matrix is invalid for plotting.
    """
    if mat.empty:
        raise ValueError("[ERROR] empty matrix for plot_profiles.")
    if roi1 not in mat.columns or roi2 not in mat.columns:
        raise ValueError(
            "[ERROR] plot_profiles missing ROI columns."
        )

    plt.figure(figsize=(5.0, 4.0))
    lbl1 = ROI_LABELS.get(roi1, roi1)
    lbl2 = ROI_LABELS.get(roi2, roi2)
    plt.plot(tasks_order, mat[roi1], marker='o', label=lbl1)
    plt.plot(tasks_order, mat[roi2], marker='s', label=lbl2)
    plt.title(f'{hemi} | {modality} | {indiv}')
    plt.xlabel('Task')
    plt.ylabel('PSC (%)')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, loc=LEG_LOC)
    ax.text(
        ANNO_X, ANNO_Y,
        rf"$r_{{rm}}={r_val:.3f},\ p={p_val:.3f}$",
        transform=ax.transAxes, va='top',
        bbox=dict(boxstyle="round,pad=.3", fc="white",
                  ec="gray", alpha=0.7),
    )
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {out_png}")


def roi_matrix_stats(
    df: pd.DataFrame,
    rois: List[str],
    hemi: str,
    modality: str,
    add_rest: bool = False,
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build ROI X ROI matrices for r, p and significance (p < alpha).

    Raises
    ------
    ValueError if any ROI pair cannot build a valid wide table.
    """
    rois_sorted = sorted(rois)
    if len(rois_sorted) < 2:
        raise ValueError("[ERROR] fewer than two ROIs available.")

    n = len(rois_sorted)
    R = np.full((n, n), np.nan, dtype=float)
    P = np.full((n, n), np.nan, dtype=float)

    for i in range(n):
        R[i, i] = 1.0
        P[i, i] = 0.0
        for j in range(i + 1, n):
            r1, r2 = rois_sorted[i], rois_sorted[j]
            wide = wide_for_rmcorr(
                df, hemi=hemi, modality=modality,
                roi1=r1, roi2=r2, add_rest=add_rest
            )
            res = pg.rm_corr(data=wide, x=r1, y=r2, subject='Subject')
            r_val = float(res['r'].iloc[0])
            p_col = 'pval' if 'pval' in res.columns else 'p'
            p_val = float(res[p_col].iloc[0])
            R[i, j] = R[j, i] = r_val
            P[i, j] = P[j, i] = p_val

    r_mat = pd.DataFrame(R, index=rois_sorted, columns=rois_sorted)
    p_mat = pd.DataFrame(P, index=rois_sorted, columns=rois_sorted)
    sig_mat = (p_mat < alpha).astype(int)

    if r_mat.isna().any(axis=None) or p_mat.isna().any(axis=None):
        raise ValueError(
            "[ERROR] NaNs found in r/p matrices in strict mode."
        )

    return r_mat, p_mat, sig_mat


def plot_matrix(
    mat: pd.DataFrame,
    title: str,
    out_png: Path,
    p_mat: pd.DataFrame | None = None,
    alpha_thr: float = 0.05,
) -> None:
    """
    Plot r-matrix heatmap with stars on significant cells.

    Raises
    ------
    ValueError if matrix is empty.
    """
    if mat.empty:
        raise ValueError("[ERROR] empty matrix for plot_matrix.")

    plt.figure(figsize=(6.2, 5.4))
    im = plt.imshow(mat.values, vmin=-1.0, vmax=1.0, cmap='coolwarm')

    xt = np.arange(mat.shape[1])
    yt = np.arange(mat.shape[0])
    plt.xticks(ticks=xt, labels=mat.columns, rotation=45, ha='right')
    plt.yticks(ticks=yt, labels=mat.index)
    plt.title(title)
    cbar = plt.colorbar(im)
    cbar.set_label('r (rmcorr)')

    if p_mat is not None:
        n = mat.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                p = p_mat.values[i, j]
                if np.isfinite(p) and p < alpha_thr:
                    stars = p_to_stars(p)
                    if stars:
                        plt.text(
                            j, i, stars, ha='center', va='center',
                            fontsize=9, color='k', fontweight='bold'
                        )

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {out_png}")


# ============================ USER INPUTS ============================ #

ALPHA: float = 0.05
N_ROIS: int = 8

ADD_REST: bool = True
USE_RAND: bool = True

INDIVID_LEVELS: List[str] = [
    'i', 'i9a', 'i8a', 'i7a', 'i6a',
    'a', 'a4g', 'a3g', 'a2g', 'a1g', 'g',
]

HEMIS: List[str] = ['bh', 'lh', 'rh']

BASE_TASKS: List[str] = ['Production', 'Perception', 'NTFD']
TASKS_NO_REST: List[str] = (
    BASE_TASKS + ['NTFD Random'] if USE_RAND else BASE_TASKS
)
TASKS: List[str] = TASKS_NO_REST + ['Rest']

CATS: List[str] = (
    ['Beat', 'Interval', 'Random'] if USE_RAND else ['Beat', 'Interval']
)

MODALITIES: List[str] = ['Both', 'Auditory', 'Visual']

if USE_RAND and ADD_REST:
    ENCODING_DIRNAME = 'encoding_restrand'
elif USE_RAND and not ADD_REST:
    ENCODING_DIRNAME = 'encoding_rand'
elif not USE_RAND and ADD_REST:
    ENCODING_DIRNAME = 'encoding_rest'
else:
    ENCODING_DIRNAME = 'encoding'

if USE_RAND and ADD_REST:
    FILETAG = 'withrestrand'
elif USE_RAND and not ADD_REST:
    FILETAG = 'withrand'
elif not USE_RAND and ADD_REST:
    FILETAG = 'withrest'
else:
    FILETAG = 'norand'

ROI_LABELS: Dict[str, str] = {
    'dstr': 'Dorsal Striatum',
    'cereb': 'Cerebellum',
    'pmv': 'PMV',
    'pmd': 'PMD',
    'presma': 'preSMA',
    'sma': 'SMA',
    'heschl': 'Heschl Gyrus',
    'occipital': 'Occipital Lobe',
    'occipital_lobe': 'Occipital Lobe',
}

LEG_LOC: str = 'upper right'
ANNO_X, ANNO_Y = 0.05, 0.10


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


# ================================ RUN =============================== #

if __name__ == "__main__":

    for indiv in INDIVID_LEVELS:
        df_all = load_df(indiv)

        present = sorted(set(df_all['ROI'].unique()) & set(ROI_LABELS))
        if len(present) < 2:
            raise ValueError(
                f"[ERROR] fewer than two labeled ROIs for '{indiv}'."
            )

        indiv_root = (
            Path(BASE_ALL) / 'profile_similarity' /
            ENCODING_DIRNAME / indiv
        )
        for sub in ('cereb_only', 'dstr_only', 'hmat', 'matrices'):
            os.makedirs(indiv_root / sub, exist_ok=True)

        for modality in MODALITIES:
            for hemi in HEMIS:
                summary_rows: List[Dict[str, object]] = []

                for roi_a, roi_b in combinations(present, 2):
                    roi1, roi2 = sorted([roi_a, roi_b])

                    wide = wide_for_rmcorr(
                        df_all, hemi=hemi, modality=modality,
                        roi1=roi1, roi2=roi2, add_rest=ADD_REST
                    )

                    rmc = pg.rm_corr(
                        data=wide, x=roi1, y=roi2, subject='Subject'
                    )
                    r_val = float(rmc['r'].iloc[0])
                    p_col = 'pval' if 'pval' in rmc.columns else 'p'
                    p_val = float(rmc[p_col].iloc[0])
                    is_sig = bool(p_val < ALPHA)

                    summary_rows.append({
                        'individualization': indiv,
                        'modality': modality,
                        'hemisphere': hemi,
                        'pair': f"{roi1}-{roi2}",
                        'r': r_val,
                        'p': p_val,
                        'n_subj': wide['Subject'].nunique(),
                        'n_points': len(wide),
                        'significant': is_sig,
                    })

                    if is_sig:
                        mat = group_means_for_plot(
                            df_all, hemi=hemi, modality=modality,
                            roi1=roi1, roi2=roi2, add_rest=ADD_REST
                        )
                        bucket = bucket_for_pair(roi1, roi2)
                        out_dir = indiv_root / bucket
                        fname = (
                            f"rmcorr_profile_{indiv}_{modality}_{hemi}_"
                            f"{roi1}-{roi2}_{N_ROIS}-rois_{FILETAG}.png"
                        )
                        out_png = out_dir / fname
                        tasks_order = TASKS if ADD_REST else TASKS_NO_REST
                        plot_profiles(
                            mat=mat, r_val=r_val, p_val=p_val,
                            indiv=indiv, modality=modality, hemi=hemi,
                            roi1=roi1, roi2=roi2,
                            tasks_order=tasks_order,
                            out_png=out_png
                        )

                sum_name = (
                    f"summary_{indiv}_{modality}_{hemi}_"
                    f"{N_ROIS}-rois_{FILETAG}.tsv"
                )
                out_tsv = indiv_root / sum_name
                pd.DataFrame(summary_rows).to_csv(
                    out_tsv, sep='\t', index=False
                )
                print(f"[SAVED] {out_tsv}")

                mat_dir = indiv_root / 'matrices'
                r_mat, p_mat, sig_mat = roi_matrix_stats(
                    df_all, rois=present, hemi=hemi, modality=modality,
                    add_rest=ADD_REST, alpha=ALPHA
                )

                r_tsv = mat_dir / (
                    f"matrix_r_{indiv}_{modality}_{hemi}_"
                    f"{N_ROIS}-rois_{FILETAG}.tsv"
                )
                p_tsv = mat_dir / (
                    f"matrix_p_{indiv}_{modality}_{hemi}_"
                    f"{N_ROIS}-rois_{FILETAG}.tsv"
                )
                s_tsv = mat_dir / (
                    f"matrix_sig_{indiv}_{modality}_{hemi}_"
                    f"{N_ROIS}-rois_{FILETAG}.tsv"
                )
                r_mat.to_csv(r_tsv, sep='\t')
                p_mat.to_csv(p_tsv, sep='\t')
                sig_mat.to_csv(s_tsv, sep='\t')
                print(f"[SAVED] {r_tsv}")
                print(f"[SAVED] {p_tsv}")
                print(f"[SAVED] {s_tsv}")

                mat_png = mat_dir / (
                    f"matrix_{indiv}_{modality}_{hemi}_"
                    f"{N_ROIS}-rois_{FILETAG}.png"
                )
                plot_matrix(
                    mat=r_mat,
                    title=f"{indiv} | {modality} | {hemi} ({FILETAG})",
                    out_png=mat_png,
                    p_mat=p_mat,
                    alpha_thr=ALPHA,
                )