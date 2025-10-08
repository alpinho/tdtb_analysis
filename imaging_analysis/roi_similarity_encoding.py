#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Profile similarity (encoding) with repeated-measures correlation.

- For each individualization × modality × hemisphere:
  * Build Subject × Task × Category × Modality PSC for two ROIs
    (no collapse).
  * Optional synthetic Rest row (PSC=0) per Subject × ROI:
    Task='Rest', Category='Rest', Modality='Rest'.
  * Points per subject with Rest: Both=13 (12+1), Aud/Vis=7 (6+1).
  * Compute rm-corr (Pingouin) for all ROI pairs (alphabetical).
  * Plot ONLY significant pairs (p < ALPHA).
  * Store under:
      BASE_DIR/profile_similarity/encoding/<indiv>/
    with buckets:
      - cereb_only: includes cereb, excludes dstr
      - dstr_only : includes dstr, excludes cereb
      - hmat      : remaining (incl. cereb–dstr)

Also writes:
- A TSV per (modality × hemisphere) with r, p, n_subj, n_points,
  and a 'significant' column.
- An ROI × ROI r-matrix (TSV + PNG) per (modality × hemisphere).

The global toggle ADD_REST controls whether the synthetic Rest
condition is included across the pipeline and filenames.
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


def _add_rest_rows(sub: pd.DataFrame) -> pd.DataFrame:
    """
    Add one Rest row (PSC=0) per Subject×ROI (tags set to 'Rest').
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


# =========================== PUBLIC HELPERS ========================== #

def load_df(indiv: str) -> pd.DataFrame:
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
        df = df[df['Task'].isin(TASKS_NO_REST)]
    if 'Category' in df.columns:
        df = df[df['Category'].isin(CATS)]
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


def group_means_for_plot(
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
    idx = TASKS if add_rest else TASKS_NO_REST
    mat = grp.pivot(index='Task', columns='ROI', values='PSC').reindex(idx)
    return mat


def bucket_for_pair(roi_a: str, roi_b: str) -> str:
    """
    Decide bucket for a pair (alphabetical order).
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
    Plot group-mean ROI task profiles with rm-corr annotation.
    """
    if mat.empty or (roi1 not in mat.columns) or (roi2 not in mat.columns):
        return
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


def roi_matrix(
    df: pd.DataFrame,
    rois: List[str],
    hemi: str,
    modality: str,
    add_rest: bool = False,
) -> pd.DataFrame:
    """
    Build an ROI×ROI matrix of rmcorr r values for a given combo.
    """
    rois_sorted = sorted(rois)
    n = len(rois_sorted)
    M = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        M[i, i] = 1.0
        for j in range(i + 1, n):
            r1, r2 = rois_sorted[i], rois_sorted[j]
            wide = wide_for_rmcorr(
                df, hemi=hemi, modality=modality,
                roi1=r1, roi2=r2, add_rest=add_rest
            )
            if wide.empty:
                r_val = np.nan
            else:
                res = pg.rm_corr(data=wide, x=r1, y=r2, subject='Subject')
                r_val = float(res['r'].iloc[0])
            M[i, j] = r_val
            M[j, i] = r_val
    mat = pd.DataFrame(M, index=rois_sorted, columns=rois_sorted)
    return mat


def plot_matrix(
    mat: pd.DataFrame,
    title: str,
    out_png: Path,
) -> None:
    """
    Plot an r-matrix heatmap with ROI codes on axes.
    """
    if mat.empty:
        return
    plt.figure(figsize=(6.0, 5.2))
    im = plt.imshow(mat.values, vmin=-1.0, vmax=1.0, cmap='coolwarm')
    plt.xticks(
        ticks=np.arange(mat.shape[1]),
        labels=mat.columns, rotation=45, ha='right'
    )
    plt.yticks(ticks=np.arange(mat.shape[0]), labels=mat.index)
    plt.title(title)
    cbar = plt.colorbar(im)
    cbar.set_label('r (rmcorr)')
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {out_png}")


# ============================ USER INPUTS ============================ #

ALPHA: float = 0.05
N_ROIS: int = 8

# Toggle: include synthetic Rest (Task/Category/Modality = 'Rest')
ADD_REST: bool = True  # set True to include Rest

INDIVID_LEVELS: List[str] = [
    'i', 'i9a', 'i8a', 'i7a', 'i6a',
    'a', 'a4g', 'a3g', 'a2g', 'a1g', 'g',
]

HEMIS: List[str] = ['bh', 'lh', 'rh']

# With and without Rest orders
TASKS: List[str] = ['Production', 'Perception', 'NTFD', 'Rest']
TASKS_NO_REST: List[str] = ['Production', 'Perception', 'NTFD']

CATS: List[str] = ['Beat', 'Interval']
MODALITIES: List[str] = ['Both', 'Auditory', 'Visual']

# Directory name switches with Rest toggle
ENCODING_DIRNAME: str = 'encoding_rest' if ADD_REST else 'encoding'

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
    rest_tag = "withrest" if ADD_REST else "norest"

    for indiv in INDIVID_LEVELS:
        df_all = load_df(indiv)
        if df_all.empty:
            continue

        indiv_root = (
            Path(BASE_DIR) / 'profile_similarity' / 'encoding_rest' / indiv
        )
        for sub in ('cereb_only', 'dstr_only', 'hmat', 'matrices'):
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

                    wide = wide_for_rmcorr(
                        df_all, hemi=hemi, modality=modality,
                        roi1=roi1, roi2=roi2, add_rest=ADD_REST
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

                    mat = group_means_for_plot(
                        df_all, hemi=hemi, modality=modality,
                        roi1=roi1, roi2=roi2, add_rest=ADD_REST
                    )
                    if mat.empty:
                        continue

                    bucket = bucket_for_pair(roi1, roi2)
                    out_dir = indiv_root / bucket
                    fname = (
                        f"rmcorr_profile_{indiv}_{modality}_{hemi}_"
                        f"{roi1}-{roi2}_{N_ROIS}-rois_{rest_tag}.png"
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
                    f"{N_ROIS}-rois_{rest_tag}.tsv"
                )
                out_tsv = indiv_root / sum_name
                pd.DataFrame(summary_rows).to_csv(
                    out_tsv, sep='\t', index=False
                )
                print(f"[SAVED] {out_tsv}")

                # ROI×ROI matrix (TSV + PNG) for this combo
                mat_dir = indiv_root / 'matrices'
                roi_list = present
                mat_df = roi_matrix(
                    df_all, rois=roi_list, hemi=hemi,
                    modality=modality, add_rest=ADD_REST
                )
                mat_tsv = mat_dir / (
                    f"matrix_{indiv}_{modality}_{hemi}_"
                    f"{N_ROIS}-rois_{rest_tag}.tsv"
                )
                mat_df.to_csv(mat_tsv, sep='\t')
                print(f"[SAVED] {mat_tsv}")

                mat_png = mat_dir / (
                    f"matrix_{indiv}_{modality}_{hemi}_"
                    f"{N_ROIS}-rois_{rest_tag}.png"
                )
                plot_matrix(
                    mat=mat_df,
                    title=f"{indiv} | {modality} | {hemi} ({rest_tag})",
                    out_png=mat_png
                )