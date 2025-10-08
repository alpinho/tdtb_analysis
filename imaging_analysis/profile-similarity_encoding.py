"""
Encoding profile similarity (within-subject rmcorr) across Modality.

For each individualization level, hemisphere, and unordered ROI pair:
- Compute repeated-measures correlation (r_rm) on Encoding (Beat+Interval
  combined) across tasks, within each Modality (Both, Auditory, Visual).
- Exclude ROI pairs that contain 'heschl' or 'occipital'.
- Save a TSV summary over all settings (bucket column removed).
- For each (roi1, roi2, hemisphere, modality), pick the single "winner"
  by the smallest UNCORRECTED p across individualizations (p < ALPHA;
  tie by larger |r_rm|), then:
    * save the single-panel figure (ROI1/ROI2 across tasks);
    * save line plots of p across all individualization levels
      (uncorrected and Holm–Bonferroni via Pingouin);
    * write winners_p_arrays.tsv (sorted: pair α-order, modality order,
      hemisphere order).
- Console winners are printed in the same sorted order.

Author: Ana Luisa Pinho
email: agrilopi@uwo.ca

Created: 1st of October 2025
Last Update: 2nd of October 2025

Compatibility: Python 3.10.16
"""

from __future__ import annotations

import os
import itertools
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pingouin as pg  # required


# ============================== INPUTS =================================
ANNO_X, ANNO_Y = 0.05, 0.15
LOC_LEG = 'best'
ALPHA = 0.05

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

# Enforced listing/sorting order
HEMIS = ['bh', 'lh', 'rh']
TASKS = ['Production', 'Perception', 'NTFD']
MODALITIES = ['Both', 'Auditory', 'Visual']
EXCLUDE_ROIS = {'heschl', 'occipital'}

HEMI_ORDER = {h: i for i, h in enumerate(HEMIS)}
MOD_ORDER = {m: i for i, m in enumerate(MODALITIES)}


# ============================ UTIL FUNCS ===============================
def get_pcol(rmc_df: pd.DataFrame) -> str:
    for k in ('p-val', 'pval', 'p'):
        if k in rmc_df.columns:
            return k
    raise KeyError(f"No p column in rm_corr: {list(rmc_df.columns)}")


def unordered_pairs_alpha(keys: List[str]) -> List[Tuple[str, str]]:
    """All unordered, non-identical pairs from sorted ROI keys."""
    keys = sorted(keys)  # enforce alphabetical start
    return list(itertools.combinations(keys, 2))


def make_wide_rmcorr(df: pd.DataFrame,
                     roi1: str,
                     roi2: str) -> pd.DataFrame:
    """Subject×Task wide table for rmcorr, averaging duplicates."""
    base = (
        df.query("ROI in [@roi1, @roi2] and Task in @TASKS")
        .groupby(['Subject', 'Task', 'ROI'], as_index=False)['PSC']
        .mean()
    )
    wide = (
        base.pivot_table(
            index=['Subject', 'Task'],
            columns='ROI',
            values='PSC',
            aggfunc='mean'
        )
        .reset_index()
        .dropna(subset=[roi1, roi2])
    )
    return wide


def mat_for_plot(sub_grp: pd.DataFrame,
                 roi1: str,
                 roi2: str) -> pd.DataFrame:
    mat = (
        sub_grp.pivot_table(
            index='Task',
            columns='ROI',
            values='PSC',
            aggfunc='mean'
        )
        .reindex(index=TASKS)
    )
    return mat


def bucket_for_pair(roi1: str, roi2: str) -> str:
    s = {roi1, roi2}
    if 'dstr' in s and 'cereb' in s:
        return 'dstr_cereb'
    if 'dstr' in s:
        return 'dstr_only'
    if 'cereb' in s:
        return 'cereb_only'
    return 'hmat'


def ensure_dirs_encoding(base_out: str) -> str:
    enc_dir = os.path.join(base_out, 'profile_similarity', 'encoding')
    os.makedirs(enc_dir, exist_ok=True)
    for b in ('dstr_only', 'cereb_only', 'dstr_cereb', 'hmat'):
        os.makedirs(os.path.join(enc_dir, b), exist_ok=True)
    return enc_dir


# ============================ PATHS/FILES ==============================
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


# ============================== CORE RUN ==============================
def compute_rmcorr_for_setting(
    df_all: pd.DataFrame,
    grp: pd.DataFrame,
    hemi: str,
    roi1: str,
    roi2: str,
    mod: str,
) -> Dict[str, object]:
    if mod == 'Both':
        df_mod = df_all[df_all['Hemisphere'] == hemi]
        grp_mod = grp[grp['Hemisphere'] == hemi]
    else:
        df_mod = df_all[
            (df_all['Hemisphere'] == hemi) &
            (df_all['Modality'] == mod)
        ]
        grp_mod = grp[
            (grp['Hemisphere'] == hemi) &
            (grp['Modality'] == mod)
        ]

    wide = make_wide_rmcorr(df_mod, roi1, roi2)
    if wide.empty:
        return {'r_rm': np.nan, 'p': np.nan, 'n_subjects': 0}

    rmc = pg.rm_corr(data=wide, x=roi1, y=roi2, subject='Subject')
    pcol = get_pcol(rmc)
    r_val = float(rmc['r'].iloc[0])
    p_val = float(rmc[pcol].iloc[0])

    return {
        'r_rm': r_val,
        'p': p_val,
        'n_subjects': int(wide['Subject'].nunique()),
    }


def render_encoding_panel(
    df_all: pd.DataFrame,
    grp: pd.DataFrame,
    hemi: str,
    roi1: str,
    roi2: str,
    mod: str,
    out_dir: str,
    indiv: str,
    r_val: float,
    p_val: float,
) -> str:
    if mod == 'Both':
        grp_mod = grp[grp['Hemisphere'] == hemi]
    else:
        grp_mod = grp[
            (grp['Hemisphere'] == hemi) & (grp['Modality'] == mod)
        ]

    sub_grp = grp_mod[
        (grp_mod['ROI'].isin([roi1, roi2])) &
        (grp_mod['Task'].isin(TASKS))
    ]
    mat = mat_for_plot(sub_grp, roi1, roi2)

    plt.figure(figsize=(5, 4))
    plt.plot(
        TASKS, mat[roi1], marker='o',
        label=ROI_LABELS.get(roi1, roi1),
    )
    plt.plot(
        TASKS, mat[roi2], marker='s',
        label=ROI_LABELS.get(roi2, roi2),
    )
    plt.title(f'{hemi} • {mod} • Encoding')
    plt.xlabel('Task')
    plt.ylabel('PSC (%)')

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, loc=LOC_LEG)

    ax.text(
        ANNO_X, ANNO_Y,
        rf"$r_{{rm}} = {r_val:.3f},\ p = {p_val:.3f}$",
        transform=ax.transAxes,
        va='top',
        bbox=dict(
            boxstyle="round,pad=0.3",
            fc="white", ec="gray", alpha=0.7,
        ),
    )

    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(
        out_dir,
        f"encoding_rmcorr_{indiv}_{N_ROIS}-rois_"
        f"{roi1}-{roi2}_{hemi}_{mod.lower()}.png"
    )
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    return fname


def render_p_summary_plots(
    base_out_folder: str,
    indiv_levels: List[str],
    p_unc_full: List[float],
    title: str,
    stem: str,
) -> Tuple[str, str, List[float], List[float]]:
    p_unc = [np.nan if v is None else v for v in p_unc_full]
    mask = ~np.isnan(p_unc)
    xs = np.arange(len(indiv_levels))[mask]
    labs = np.array(indiv_levels)[mask]
    y_unc = np.array(p_unc, float)[mask]

    # Holm–Bonferroni via Pingouin
    _, y_holm = pg.multicomp(y_unc.tolist(), method='holm')

    # Full-length Holm list with NaNs in missing positions
    y_holm_full = [np.nan] * len(indiv_levels)
    j = 0
    for i, m in enumerate(mask):
        if m:
            y_holm_full[i] = float(y_holm[j])
            j += 1

    # Uncorrected
    plt.figure(figsize=(6, 3))
    plt.plot(xs, y_unc, marker='o')
    plt.axhline(ALPHA, ls='--')
    plt.xticks(xs, labs, rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.ylabel('p (uncorrected)')
    plt.title(title + ' • p across individualization')
    plt.tight_layout()
    f_unc = os.path.join(base_out_folder, f"{stem}_p_unc.png")
    plt.savefig(f_unc, dpi=300, bbox_inches='tight')
    plt.close()

    # Holm–Bonferroni
    plt.figure(figsize=(6, 3))
    plt.plot(xs, y_holm, marker='o')
    plt.axhline(ALPHA, ls='--')
    plt.xticks(xs, labs, rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.ylabel('p (Holm-Bonf.)')
    plt.title(title + ' • Holm-adjusted p')
    plt.tight_layout()
    f_holm = os.path.join(base_out_folder, f"{stem}_p_holm.png")
    plt.savefig(f_holm, dpi=300, bbox_inches='tight')
    plt.close()

    return f_unc, f_holm, p_unc, y_holm_full


# =============================== MAIN =================================
def main() -> None:
    enc_dir = ensure_dirs_encoding(BASE_DIR)

    results: List[Dict[str, object]] = []

    # Collect stats across all individualization levels
    for indiv in INDIVID_LEVELS:
        df_dir = os.path.join(BASE_DIR, 'df_rois_volume')
        df_path = os.path.join(df_dir, f"dfrois_{indiv}_{N_ROIS}-rois.tsv")
        if not os.path.exists(df_path):
            print(f"[WARN] Missing file for {indiv}: {df_path}")
            continue

        df_all = pd.read_csv(
            df_path,
            sep='\t',
            dtype={
                'Subject': str,
                'Task': str,
                'ROI': str,
                'Hemisphere': str,
                'Category': str,
                'Modality': str,
                'PSC': float,
            }
        )
        df_all = df_all[df_all['Task'].isin(TASKS)]

        grp = (
            df_all
            .groupby(['Hemisphere', 'Modality', 'ROI', 'Task'])['PSC']
            .mean()
            .reset_index()
        )

        # ROI pairs excluding 'heschl' and 'occipital', alpha-ordered
        candidate_rois = sorted(
            [k for k in ROI_LABELS.keys() if k not in EXCLUDE_ROIS]
        )
        roi_pairs = unordered_pairs_alpha(candidate_rois)

        for hemi in HEMIS:
            for roi1, roi2 in roi_pairs:

                present = df_all[
                    (df_all['Hemisphere'] == hemi) &
                    (df_all['ROI'].isin([roi1, roi2]))
                ]['ROI'].unique()
                if not set([roi1, roi2]).issubset(set(present)):
                    continue

                for mod in MODALITIES:
                    if mod == 'Both':
                        df_mod = df_all[df_all['Hemisphere'] == hemi]
                    else:
                        df_mod = df_all[
                            (df_all['Hemisphere'] == hemi) &
                            (df_all['Modality'] == mod)
                        ]

                    stat = compute_rmcorr_for_setting(
                        df_all=df_all,
                        grp=grp,
                        hemi=hemi,
                        roi1=roi1,
                        roi2=roi2,
                        mod=mod,
                    )
                    row = {
                        'individualization': indiv,
                        'hemisphere': hemi,
                        'modality': mod,
                        'roi1': roi1,
                        'roi2': roi2,
                        **stat,
                    }
                    results.append(row)

    # Save summary TSV (remove bucket; enforce sort orders)
    out_dir = enc_dir
    os.makedirs(out_dir, exist_ok=True)
    if not results:
        print("[INFO] No results computed.")
        return

    df_res = pd.DataFrame(results)

    # Enforce categorical ordering for tidy sort
    df_res['hemisphere'] = pd.Categorical(
        df_res['hemisphere'], categories=HEMIS, ordered=True
    )
    df_res['modality'] = pd.Categorical(
        df_res['modality'], categories=MODALITIES, ordered=True
    )

    df_res = df_res.sort_values(
        by=['roi1', 'roi2', 'modality', 'hemisphere', 'individualization']
    )

    tsv_path = os.path.join(out_dir, 'summary_encoding.tsv')
    print(f"Saved summary table to {tsv_path}")
    df_res.to_csv(tsv_path, sep='\t', index=False)

    # Winners: for each (pair, hemi, modality), pick best individualization
    best_rows = []
    for (roi1, roi2, hemi, mod), g in df_res.groupby(
        ['roi1', 'roi2', 'hemisphere', 'modality'], sort=False
    ):
        g_sig = g[(~g['p'].isna()) & (g['p'] < ALPHA)]
        if g_sig.empty:
            continue
        min_p = g_sig['p'].min()
        g_min = g_sig[g_sig['p'] == min_p]
        if len(g_min) > 1:
            idx = (g_min['r_rm'].abs()).idxmax()
        else:
            idx = g_min.index[0]
        best_rows.append(df_res.loc[idx].to_dict())

    # Sort winners by pair α-order, then modality order, then hemisphere
    best_rows.sort(
        key=lambda d: (
            d['roi1'], d['roi2'],
            MOD_ORDER.get(d['modality'], 99),
            HEMI_ORDER.get(d['hemisphere'], 99)
        )
    )

    # Winners p-array TSV rows
    winners_rows: List[Dict[str, object]] = []

    # Render the winning panel and p-summary plots (unc & Holm)
    for best in best_rows:
        indiv_star = best['individualization']
        hemi = best['hemisphere']
        mod = best['modality']
        roi1 = best['roi1']
        roi2 = best['roi2']
        r_star = float(best['r_rm'])
        p_star = float(best['p'])

        # Load data for this indiv to render the panel figure
        df_path = os.path.join(
            BASE_DIR, 'df_rois_volume',
            f"dfrois_{indiv_star}_{N_ROIS}-rois.tsv"
        )
        if not os.path.exists(df_path):
            print(f"[WARN] Missing winner file {df_path}")
            continue

        df_all = pd.read_csv(
            df_path,
            sep='\t',
            dtype={
                'Subject': str, 'Task': str, 'ROI': str, 'Hemisphere': str,
                'Category': str, 'Modality': str, 'PSC': float,
            }
        )
        df_all = df_all[df_all['Task'].isin(TASKS)]
        grp = (
            df_all
            .groupby(['Hemisphere', 'Modality', 'ROI', 'Task'])['PSC']
            .mean()
            .reset_index()
        )

        parent = os.path.join(out_dir, bucket_for_pair(roi1, roi2))
        os.makedirs(parent, exist_ok=True)

        # Panel figure (single plot)
        _ = render_encoding_panel(
            df_all, grp, hemi, roi1, roi2, mod, parent, indiv_star,
            r_star, p_star
        )

        # Build p across all indiv for this (hemi, mod, roi1, roi2)
        sub = df_res[
            (df_res['hemisphere'] == hemi) &
            (df_res['modality'] == mod) &
            (df_res['roi1'] == roi1) &
            (df_res['roi2'] == roi2)
        ].set_index('individualization')

        p_list = [sub['p'].get(ind, np.nan) for ind in INDIVID_LEVELS]

        title = (f"{hemi} • {ROI_LABELS.get(roi1, roi1)} vs "
                 f"{ROI_LABELS.get(roi2, roi2)} • {mod} • Encoding")
        stem = (f"p_across_indiv_{roi1}-{roi2}_{hemi}_{mod.lower()}")

        _, _, p_unc_full, p_holm_full = render_p_summary_plots(
            parent, INDIVID_LEVELS, p_list, title, stem
        )

        # Print Holm array then the winner line (sorted print order)
        print(np.array(p_holm_full))
        print(f"Wrote winner + p-plots for {roi1}-{roi2} ({hemi}, {mod})")

        # Append tidy row to winners TSV (sorted later by same key)
        winners_rows.append({
            'roi1': roi1,
            'roi2': roi2,
            'hemisphere': hemi,
            'modality': mod,
            'p_unc': str(np.array(p_unc_full)),
            'p_holm': str(np.array(p_holm_full)),
        })

    # Write winners p-array log as TSV next to the summary table (sorted)
    if winners_rows:
        winners_df = pd.DataFrame(winners_rows)
        winners_df['hemisphere'] = pd.Categorical(
            winners_df['hemisphere'], categories=HEMIS, ordered=True
        )
        winners_df['modality'] = pd.Categorical(
            winners_df['modality'], categories=MODALITIES, ordered=True
        )
        winners_df = winners_df.sort_values(
            by=['roi1', 'roi2', 'modality', 'hemisphere']
        )
        winners_tsv = os.path.join(out_dir, 'winners_p_arrays.tsv')
        winners_df.to_csv(winners_tsv, sep='\t', index=False)
        print(f"Wrote winners p-array log to {winners_tsv}")


if __name__ == "__main__":
    main()