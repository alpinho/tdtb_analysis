#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Profile similarity (encoding) with repeated-measures correlation.

- For each individualization × modality × hemisphere:
  * Build Subject×Task×Category×Modality PSC for two ROIs (no collapse).
  * Optionally add a single synthetic Rest row (PSC=0) per Subject×ROI:
      Task='Rest', Category='Rest', Modality='Rest'.
  * Points per subject (with Rest): Both=13 (12+1), Aud/Vis=7 (6+1).
  * Compute rm-corr (Pingouin) for all ROI pairs (alphabetical).
  * Plot ONLY significant pairs (p < ALPHA).
  * Store under:
      BASE_DIR/profile_similarity/encoding/<indiv>/
    with buckets:
      - cereb_only: includes cereb, excludes dstr
      - dstr_only : includes dstr, excludes cereb
      - hmat      : remaining (incl. cereb–dstr)

Writes a TSV per (modality × hemisphere) with r, p, n_subj, n_points,
and a 'significant' column.
"""

from __future__ import annotations

import os
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg


# ============================== HELPERS ============================= #

def _expected_n(modality: str, add_rest: bool) -> int:
    """
    Expected points per subject. Adds +1 for Rest when enabled.
    """
    base = 12 if modality == 'Both' else 6
    return base + 1 if add_rest else base


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
    Load one dataframe and subset to desired tasks/categories.
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
        df = df[df['Task'].isin([t for t in TASKS if t != 'Rest'])]
    if 'Category' in df.columns:
        df = df[df['Category'].isin(CATS)]

    return df


def _add_rest_rows(sub: pd.DataFrame) -> pd.DataFrame:
    """
    Add one Rest row (PSC=0) per Subject×ROI with tags set to 'Rest'.
    Skip if already present.
    """
    cols = ['Subject', 'Task', 'Category', 'Modality', 'ROI', 'PSC']
    for c in cols:
        if c not in sub.columns:
            sub[c] = np.nan

    keys = sub[['Subject', 'ROI']].drop_duplicates().copy()
    keys['Task'] = 'Rest'
    keys['Category'] = 'Rest'
    keys['Modality'] = 'Rest'
    keys['PSC'] = 0.0

    rest = keys[cols].copy()
    out = pd.concat([sub[cols], rest], axis=0, ignore_index=True)
    out = out.drop_duplicates(subset=cols)
    return out


def _wide_for_rmcorr(
    df: pd.DataFrame,
    hemi: str,
    modality: str,
    roi1: str,
    roi2: str,
    add_rest: bool = False,
) -> pd.DataFrame:
    """
    Subject×Task×Category×Modality table for two ROIs.

    Keeps 12/6 points without Rest, 13/7 with Rest.
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
        )
        .reset_index()
    )

    if roi1 not in wide.columns or roi2 not in wide.columns:
        return pd.DataFrame()

    wide = wide.dropna(subset=[roi1, roi2])

    need = _expected_n(modality, add_rest=add_rest)
    cnt = wide.groupby('Subject').size().rename('n').reset_index()
    keep = set(cnt.loc[cnt['n'] == need, 'Subject'])
    wide = wide[wide['Subject'].isin(keep)].copy()

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
    add_rest: bool = False,
) -> pd.DataFrame:
    """
    Group-mean PSC per Task for roi1 and roi2 in a hemisphere.
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

    if add_rest:
        sub = _add_rest_rows(sub)

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


def _bucket_for_pair(roi_a: str, roi_b: str) -> str:
    """
    Decide bucket for a pair (alphabetical order).
    """
    a, b = sorted([roi_a, roi_b])
    if 'cereb' in (a, b) and 'dstr' not in (a, b):
        return 'cereb_only'
    if 'dstr' in (a, b) and 'cereb' not in (a, b):
        return 'dstr_only'
    return 'hmat'


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

ALPHA: float = 0.05
N_ROIS: int = 8

INDIVID_LEVELS: List[str] = [
    'i', 'i9a', 'i8a', 'i7a', 'i6a',
    'a', 'a4g', 'a3g', 'a2g', 'a1g', 'g',
]

HEMIS: List[str] = ['bh', 'lh', 'rh']
TASKS: List[str] = ['Production', 'Perception', 'NTFD', 'Rest']
CATS: List[str] = ['Beat', 'Interval']
MODALITIES: List[str] = ['Both', 'Auditory', 'Visual']

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

BASE_DIR = os.path.join(
    WORKING_DIR,
    f"roi_analyses_{MODEL}_{HRF}_{MASKING}_puncorr_unsmoothed",
    'bothmod_allmain_tasks',
    'main_tasks',
)

DF_DIR = os.path.join(BASE_DIR, 'df_rois_volume')


# ================================ RUN =============================== #

if __name__ == "__main__":
    for indiv in INDIVID_LEVELS:
        df_all = _load_df(indiv)
        if df_all.empty:
            continue

        # per-individualization output root and buckets
        indiv_root = Path(BASE_DIR) / 'profile_similarity' / 'encoding' / indiv
        for sub in ('cereb_only', 'dstr_only', 'hmat'):
            os.makedirs(indiv_root / sub, exist_ok=True)

        present = sorted(set(df_all['ROI'].unique()) & set(ROI_LABELS))
        if len(present) < 2:
            print(f"[SKIP] {indiv}: <2 ROIs present")
            continue

        for modality in MODALITIES:
            for hemi in HEMIS:
                summary_rows: List[Dict[str, object]] = []

                for roi_a, roi_b in combinations(present, 2):
                    roi1, roi2 = sorted([roi_a, roi_b])

                    wide = _wide_for_rmcorr(
                        df_all, hemi=hemi, modality=modality,
                        roi1=roi1, roi2=roi2, add_rest=False
                    )
                    if wide.empty:
                        summary_rows.append({
                            'individualization': indiv,
                            'modality': modality,
                            'hemisphere': hemi,
                            'pair': f"{roi1}-{roi2}",
                            'r': np.nan, 'p': np.nan,
                            'n_subj': 0, 'n_points': 0,
                            'significant': False,
                        })
                        continue

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
                        'r': r_val, 'p': p_val,
                        'n_subj': wide['Subject'].nunique(),
                        'n_points': len(wide),
                        'significant': is_sig,
                    })

                    if not is_sig:
                        continue

                    mat = _group_means_for_plot(
                        df_all, hemi=hemi, modality=modality,
                        roi1=roi1, roi2=roi2, add_rest=False
                    )
                    if mat.empty:
                        continue

                    bucket = _bucket_for_pair(roi1, roi2)
                    out_dir = indiv_root / bucket
                    fname = (
                        f"rmcorr_profile_{indiv}_{modality}_{hemi}_"
                        f"{roi1}-{roi2}_{N_ROIS}-rois.png"
                    )
                    out_png = out_dir / fname
                    _plot_profiles(
                        mat=mat, r_val=r_val, p_val=p_val,
                        indiv=indiv, modality=modality, hemi=hemi,
                        roi1=roi1, roi2=roi2, out_png=out_png
                    )

                sum_name = (
                    f"summary_{indiv}_{modality}_{hemi}_{N_ROIS}-rois.tsv"
                )
                out_tsv = indiv_root / sum_name
                pd.DataFrame(summary_rows).to_csv(
                    out_tsv, sep='\t', index=False
                )
                print(f"[SAVED] {out_tsv}")