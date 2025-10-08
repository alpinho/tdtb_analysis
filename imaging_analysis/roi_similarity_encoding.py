#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Profile similarity (encoding) using repeated-measures correlation.

- For each individualization × modality × hemisphere:
  * Build Subject×Task×Category×Modality wide PSC for two ROIs.
  * Keep 12 points per subject for 'Both', 6 for Aud/Vis.
  * Compute rm-corr (within-subject) with Pingouin.
  * Plot group-mean task profiles for the two ROIs and annotate
    r_rm and p.

Outputs:
- PNG figures under:
  BASE_DIR/profile_similarity/encoding_matrices

Author: Ana Luisa Pinho
Email: agrilopi@uwo.ca

Created: 7 Oct 2025
Last Update: Oct 2025
Compatibility: Python 3.10.16
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pingouin as pg


# ============================== HELPERS ============================= #

def _expected_n(modality: str) -> int:
    """Expected points per subject."""
    return 12 if modality == 'Both' else 6


def _mod_mask(df: pd.DataFrame, modality: str) -> pd.Series:
    """
    Select modality rows. 'Both' stacks Auditory and Visual.
    """
    if 'Modality' not in df.columns:
        return pd.Series(True, index=df.index)
    if modality == 'Both':
        return df['Modality'].isin(['Auditory', 'Visual'])
    return df['Modality'] == modality


def _load_df(indiv: str) -> pd.DataFrame:
    """
    Load one dataframe and subset to desired tasks.
    """
    path = os.path.join(DF_DIR, f"dfrois_{indiv}_{N_ROIS}-rois.tsv")
    if not os.path.exists(path):
        print(f"[WARN] Missing: {path}")
        return pd.DataFrame()

    dtypes = {
        'Subject': str, 'Task': str, 'ROI': str, 'Hemisphere': str,
        'PSC': float, 'Modality': str, 'Category': str,
    }
    df = pd.read_csv(path, sep='\t', dtype=dtypes)

    if 'Task' in df.columns:
        df = df[df['Task'].isin(TASKS)]
    if 'Category' in df.columns:
        df = df[df['Category'].isin(CATS)]

    return df


def _wide_for_rmcorr(
    df: pd.DataFrame,
    hemi: str,
    modality: str,
    roi1: str,
    roi2: str,
) -> pd.DataFrame:
    """
    Build Subject×Task×Category×Modality wide table for two ROIs.

    - Keeps 12 points per subject for 'Both' (3×2×2).
    - Keeps 6 points for Auditory/Visual (3×2×1).
    - Does not collapse across Category or Modality.
    """
    if df.empty:
        return df

    mask = (
        (df['Hemisphere'] == hemi) &
        (df['ROI'].isin([roi1, roi2])) &
        _mod_mask(df, modality)
    )
    sub = df.loc[mask].copy()
    if sub.empty:
        return pd.DataFrame()

    # Average only if exact duplicates exist within a cell.
    sub = (
        sub.groupby(
            ['Subject', 'Task', 'Category', 'Modality', 'ROI'],
            as_index=False
        )['PSC'].mean()
    )

    wide = (
        sub.pivot_table(
            index=['Subject', 'Task', 'Category', 'Modality'],
            columns='ROI',
            values='PSC'
        )
        .reset_index()
    )

    # Ensure both ROI columns exist and are complete
    if roi1 not in wide.columns or roi2 not in wide.columns:
        return pd.DataFrame()
    wide = wide.dropna(subset=[roi1, roi2])

    # Enforce per-subject completeness
    need = _expected_n(modality)
    cnt = wide.groupby('Subject').size().rename('n').reset_index()
    keep = set(cnt.loc[cnt['n'] == need, 'Subject'])
    wide = wide[wide['Subject'].isin(keep)].copy()

    # Final tidy columns for pg.rm_corr
    cols = ['Subject', 'Task', 'Category', 'Modality', roi1, roi2]
    wide = wide.loc[:, cols]
    wide.index.name = None
    wide.columns.name = None
    wide = wide.sort_values(
        ['Subject', 'Task', 'Category', 'Modality']
    ).reset_index(drop=True)
    return wide


def _group_means_for_plot(
    df: pd.DataFrame,
    hemi: str,
    modality: str,
    roi1: str,
    roi2: str,
) -> pd.DataFrame:
    """
    Group-mean PSC per Task for roi1 and roi2 in a hemisphere.

    Note: plot mirrors your original script (Task-only means),
    while rmcorr uses 6/12 points per subject internally.
    """
    if df.empty:
        return pd.DataFrame()

    mask = (
        (df['Hemisphere'] == hemi) &
        (df['ROI'].isin([roi1, roi2])) &
        _mod_mask(df, modality)
    )
    sub = df.loc[mask].copy()
    if sub.empty:
        return pd.DataFrame()

    grp = (
        sub.groupby(['ROI', 'Task'])['PSC']
        .mean()
        .reset_index()
    )

    mat = (
        grp.pivot(index='Task', columns='ROI', values='PSC')
        .reindex(index=TASKS)
    )
    return mat


def _plot_profiles(
    mat: pd.DataFrame,
    r_val: float,
    p_val: float,
    indiv: str,
    modality: str,
    hemi: str,
    roi1: str,
    roi2: str,
    out_png: Path,
) -> None:
    """
    Plot group-mean ROI task profiles with rm-corr annotation.
    """
    if mat.empty or (roi1 not in mat.columns) or (roi2 not in mat.columns):
        return

    plt.figure(figsize=(5.0, 4.0))
    lbl1 = ROI_LABELS.get(roi1, roi1)
    lbl2 = ROI_LABELS.get(roi2, roi2)

    plt.plot(TASKS, mat[roi1], marker='o', label=lbl1)
    plt.plot(TASKS, mat[roi2], marker='s', label=lbl2)

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
        transform=ax.transAxes,
        va='top',
        bbox=dict(
            boxstyle="round,pad=.3",
            fc="white", ec="gray", alpha=0.7
        ),
    )

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {out_png}")


# ============================ USER INPUTS ============================ #

N_ROIS: int = 8

# INDIVID_LEVELS: List[str] = [
#     'i', 'i9a', 'i8a', 'i7a', 'i6a',
#     'a', 'a4g', 'a3g', 'a2g', 'a1g', 'g',
# ]
INDIVID_LEVELS: List[str] = ['i8a']

ROI1, ROI2 = 'dstr', 'sma'

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

HEMIS: List[str] = ['bh', 'lh', 'rh']
TASKS: List[str] = ['Production', 'Perception', 'NTFD']
CATS: List[str] = ['Beat', 'Interval']
MODALITIES: List[str] = ['Both', 'Auditory', 'Visual']

LEG_LOC: str = 'upper right'
ANNO_X, ANNO_Y = 0.05, 0.10


# =============================== PATHS ============================== #

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL = 'rwls'
MASKING = 'wb'
HRF = 'hrf128'

BASE_DIR = os.path.join(
    WORKING_DIR,
    f"roi_analyses_{MODEL}_{HRF}_{MASKING}_puncorr_unsmoothed",
    'bothmod_allmain_tasks',
    'main_tasks',
)

DF_DIR = os.path.join(BASE_DIR, 'df_rois_volume')

OUT_DIR = os.path.join(
    BASE_DIR, 'profile_similarity', 'encoding_matrices'
)
os.makedirs(OUT_DIR, exist_ok=True)


# ================================ RUN =============================== #

if __name__ == "__main__":
    rois = [ROI1, ROI2]

    for indiv in INDIVID_LEVELS:
        df_all = _load_df(indiv)
        if df_all.empty:
            continue

        for modality in MODALITIES:
            for hemi in HEMIS:
                wide = _wide_for_rmcorr(
                    df_all, hemi=hemi, modality=modality,
                    roi1=ROI1, roi2=ROI2
                )
                if wide.empty:
                    print(f"[SKIP] {indiv} {modality} {hemi}: no data")
                    continue
                
                rmc = pg.rm_corr(
                    data=wide, x=ROI1, y=ROI2, subject='Subject'
                )
                r_val = float(rmc['r'].iloc[0])
                p_col = 'pval' if 'pval' in rmc.columns else 'p'
                p_val = float(rmc[p_col].iloc[0])

                mat = _group_means_for_plot(
                    df_all, hemi=hemi, modality=modality,
                    roi1=ROI1, roi2=ROI2
                )
                if mat.empty:
                    print(f"[SKIP] {indiv} {modality} {hemi}: no plot")
                    continue

                fname = (
                    f"rmcorr_profile_{indiv}_{modality}_{hemi}_"
                    f"{ROI1}-{ROI2}_{N_ROIS}-rois.png"
                )
                out_png = Path(OUT_DIR) / fname
                _plot_profiles(
                    mat=mat, r_val=r_val, p_val=p_val,
                    indiv=indiv, modality=modality, hemi=hemi,
                    roi1=ROI1, roi2=ROI2, out_png=out_png
                )