#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute ROI-similarity (rmcorr via Pingouin) for the first ROI pair.

Author: Ana Luisa Pinho
email: agrilopi@uwo.ca

Created: 7 Oct 2025
Last Update: Oct 2025

Compatibility: Python 3.10.16

Notes
-----
- Loads per-individualization ROI data.
- Builds Subject×Task×Category×Modality tables for the first
  ROI pair.
- Ensures subject vectors have 12 (Both) or 6 (Aud/Vis) points.
- Computes repeated-measures correlation with Pingouin.
- Saves TSV summary and PNG plots under:
  profile_similarity/encoding_matrices
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pingouin as pg


# =========================== CORE HELPERS =========================== #

def first_roi_pair(roi_labels: Dict[str, str]) -> Tuple[str, str]:
    """
    Return the first ROI pair (alphabetical by key).
    """
    keys = sorted(roi_labels.keys())
    if len(keys) < 2:
        raise ValueError("Need at least two ROIs to form a pair.")
    return keys[0], keys[1]


def load_df_for_indiv(indiv: str) -> pd.DataFrame:
    """
    Load one per-individualization dataframe and subset to chosen tasks.
    """
    df_dir = os.path.join(BASE_DIR, 'df_rois_volume')
    df_path = os.path.join(df_dir, f"dfrois_{indiv}_{N_ROIS}-rois.tsv")
    if not os.path.exists(df_path):
        print(f"[WARN] Missing file for {indiv}: {df_path}")
        return pd.DataFrame()

    dtypes = {
        'Subject': str,
        'Task': str,
        'ROI': str,
        'Hemisphere': str,
        'Category': str,
        'Modality': str,
        'PSC': float,
    }
    df = pd.read_csv(df_path, sep='\t', dtype=dtypes)
    return df[df['Task'].isin(TASKS)]


def expected_n(modality: str) -> int:
    """
    Expected number of rows per subject for a given modality.
    """
    return 12 if modality == 'Both' else 6


def modality_mask(df: pd.DataFrame, modality: str) -> pd.Series:
    """
    Boolean mask for Modality selection (Both stacks Aud + Vis).
    """
    if modality == 'Both':
        return df['Modality'].isin(['Auditory', 'Visual'])
    return df['Modality'] == modality


def prepare_wide_pair(
        df: pd.DataFrame,
        roi1: str,
        roi2: str,
        modality: str,
        hemisphere: str,
    ) -> pd.DataFrame:
    """
    Build Subject×Task×Category×Modality wide table for the ROI pair.

    - If modality == 'Both', include Auditory and Visual as separate
      repeated measures (stack).
    - Averages duplicates within Subject×Task×Category×Modality×ROI.
    - Keeps only subjects with complete vectors:
      12 (Both) or 6 (Auditory/Visual).
    """
    if df.empty:
        return df

    mask = (
        modality_mask(df, modality) &
        (df['Hemisphere'] == hemisphere) &
        (df['ROI'].isin([roi1, roi2])) &
        (df['Category'].isin(CATS))
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

    wide = sub.pivot_table(
        index=['Subject', 'Task', 'Category', 'Modality'],
        columns='ROI',
        values='PSC'
    ).reset_index()

    if roi1 not in wide.columns or roi2 not in wide.columns:
        return pd.DataFrame()

    wide = wide.dropna(subset=[roi1, roi2]).rename(
        columns={roi1: 'x', roi2: 'y'}
    )

    # enforce complete vectors per subject
    need = expected_n(modality)
    cnt = (
        wide.groupby('Subject')
        .size()
        .rename('n')
        .reset_index()
    )
    keep_subjects = set(cnt.loc[cnt['n'] == need, 'Subject'])
    wide = wide[wide['Subject'].isin(keep_subjects)].copy()

    # sort rows for stable plotting/inspection
    wide['Task'] = pd.Categorical(wide['Task'], categories=TASKS, ordered=True)
    wide['Category'] = pd.Categorical(
        wide['Category'], categories=CATS, ordered=True
    )
    wide['Modality'] = pd.Categorical(
        wide['Modality'], categories=['Auditory', 'Visual'], ordered=True
    )

    wide.index.name = None
    wide.columns.name = None
    wide = wide.sort_values(['Subject', 'Task', 'Category', 'Modality'])
    wide = wide.reset_index(drop=True)

    return wide


def run_rmcorr_pingouin(wide: pd.DataFrame) -> dict:
    """
    Compute repeated-measures correlation with Pingouin.

    Robust to non-zero/non-integer DataFrame index in the result.
    """
    if wide.empty or ('Subject' not in wide.columns):
        return {
            'r': np.nan, 'r2': np.nan, 'p': np.nan, 'dof': np.nan,
            'slope': np.nan, 'intercept': np.nan,
            'n_subjects': 0, 'n_points': 0,
        }

    res = pg.rm_corr(data=wide, x='x', y='y', subject='Subject')
    # Make indexing position-based, not label-based
    row = res.iloc[0]

    # Column names can vary by version; handle both cases
    pcol = 'pval' if 'pval' in res.columns else (
        'p' if 'p' in res.columns else None
    )

    r = float(row.get('r', np.nan))
    p = float(row[pcol]) if pcol else np.nan
    dof = float(row.get('dof', np.nan))
    slope = float(row.get('slope', np.nan))
    icpt = float(row.get('intercept', np.nan))

    return {
        'r': r, 'r2': r * r, 'p': p, 'dof': dof,
        'slope': slope, 'intercept': icpt,
        'n_subjects': wide['Subject'].nunique(),
        'n_points': len(wide),
    }


def plot_rmcorr(
        wide: pd.DataFrame,
        indiv: str,
        modality: str,
        hemisphere: str,
        roi1: str,
        roi2: str,
        roi_labels: Dict[str, str],
        slope: float,
        out_png: Path,
    ) -> None:
    """
    Scatter by subject and parallel lines with the common slope.
    """
    if wide.empty:
        return

    means = (
        wide.groupby('Subject')[['x', 'y']]
        .mean()
        .rename(columns={'x': 'mx', 'y': 'my'})
    )

    plt.figure(figsize=(5.2, 5.0))

    for s in sorted(wide['Subject'].unique()):
        d = wide[wide['Subject'] == s]
        plt.scatter(d['x'], d['y'], label=s, s=26, alpha=0.9)
        x0, x1 = d['x'].min(), d['x'].max()
        mx, my = means.loc[s, 'mx'], means.loc[s, 'my']
        xs = np.linspace(x0, x1, 50)
        ys = slope * (xs - mx) + my
        plt.plot(xs, ys, linewidth=1.4, alpha=0.9)

    xl = f"{roi_labels.get(roi1, roi1)} (PSC)"
    yl = f"{roi_labels.get(roi2, roi2)} (PSC)"
    ttl = (
        f"Rmcorr — {indiv} | {modality} | {hemisphere}\n"
        f"{roi1} vs {roi2}"
    )

    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.title(ttl)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


# =========================== USER INPUTS ============================ #

ALPHA = 0.05  # reserved for future filtering

N_ROIS = 8
INDIVID_LEVELS = [
    'i', 'i9a', 'i8a', 'i7a', 'i6a',
    'a', 'a4g', 'a3g', 'a2g', 'a1g', 'g',
]

ROI_LABELS: Dict[str, str] = {
    'dstr': 'Dorsal Striatum',
    'sma': 'SMA',
    'cereb': 'Cerebellum',
    'pmv': 'PMV',
    'pmd': 'PMD',
    'presma': 'PreSMA',
    'heschl': 'Heschl Gyrus',
    'occipital': 'Occipital Lobe',
}

HEMIS = ['bh', 'lh', 'rh']
TASKS = ['Production', 'Perception', 'NTFD']
CATS = ['Beat', 'Interval']
MODALITIES = ['Both', 'Auditory', 'Visual']

HEMI_ORDER = {h: i for i, h in enumerate(HEMIS)}
MOD_ORDER = {m: i for i, m in enumerate(MODALITIES)}
INDIV_ORDER = {k: i for i, k in enumerate(INDIVID_LEVELS)}


# ============================== PATHS =============================== #

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

OUTPUT_DIR = os.path.join(
    BASE_DIR, 'profile_similarity', 'encoding_matrices'
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================== RUN ================================= #

if __name__ == "__main__":
    roi1, roi2 = first_roi_pair(ROI_LABELS)

    rows: List[Dict[str, object]] = []

    for indiv in INDIVID_LEVELS:
        df_all = load_df_for_indiv(indiv)
        if df_all.empty:
            continue

        for modality in MODALITIES:
            for hemisphere in HEMIS:
                wide = prepare_wide_pair(
                    df_all, roi1, roi2, modality, hemisphere
                )

                if wide.empty:
                    rows.append({
                        'individualization': indiv,
                        'modality': modality,
                        'hemisphere': hemisphere,
                        'roi1': roi1, 'roi2': roi2,
                        'r': np.nan, 'r2': np.nan, 'p': np.nan,
                        'dof': np.nan, 'slope': np.nan,
                        'intercept': np.nan,
                        'n_subjects': 0, 'n_points': 0,
                    })
                    continue

                stats = run_rmcorr_pingouin(wide)
                rows.append({
                    'individualization': indiv,
                    'modality': modality,
                    'hemisphere': hemisphere,
                    'roi1': roi1, 'roi2': roi2,
                    **stats,
                })

                if np.isfinite(stats.get('slope', np.nan)):
                    png_name = (
                        f"rmcorr_{indiv}_{modality}_{hemisphere}_"
                        f"{roi1}-{roi2}.png"
                    )
                    out_png = Path(OUTPUT_DIR) / png_name
                    plot_rmcorr(
                        wide=wide,
                        indiv=indiv,
                        modality=modality,
                        hemisphere=hemisphere,
                        roi1=roi1,
                        roi2=roi2,
                        roi_labels=ROI_LABELS,
                        slope=stats['slope'],
                        out_png=out_png,
                    )

    df_sum = pd.DataFrame(rows)

    df_sum = df_sum.sort_values(
        by=['individualization', 'modality', 'hemisphere'],
        key=lambda s: s.map({
            **INDIV_ORDER, **MOD_ORDER, **HEMI_ORDER
        }).fillna(1e9)
    )

    out_tsv = Path(OUTPUT_DIR) / (
        f"rmcorr_summary_firstpair_{roi1}-{roi2}.tsv"
    )
    df_sum.to_csv(out_tsv, sep='\t', index=False)

    print(f"[INFO] Summary saved to: {out_tsv}")
    print(f"[INFO] Figures saved under: {OUTPUT_DIR}")