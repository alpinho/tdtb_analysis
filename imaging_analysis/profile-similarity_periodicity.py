"""
Profile similarity (within-subject rmcorr) by Category and Modality.

Pipeline
--------
For each individualization level, hemisphere, and unordered ROI pair:
- Compute repeated-measures correlation (r_rm) per Category
  (Beat/Interval) within each Modality (Auditory, Visual) and pooled.
- Test Beat vs Interval using subject-wise Pearson r across tasks,
  Fisher-z transform, and paired t-test (within-subject).
- Exclude ROI pairs that contain 'heschl' or 'occipital'.
- Log a TSV summary (adds 'direction' and 'bucket').
- For each ROI pair, pick the single "winning" setting by the smallest
  UNCORRECTED p_delta across all indiv×hemi×modality (must be p<ALPHA;
  ties broken by larger |Δr|), then:
    * save the repeated-measures panel for that setting;
    * in the same folder, save line plots of p across all
      individualization levels: uncorrected and Holm–Bonferroni.

Author: Ana Luisa Pinho
Last Update: 03 Oct 2025
Compat.: Python 3.10.16
"""

from __future__ import annotations

import os
import itertools
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ttest_rel
import pingouin as pg  # required

# ============================== INPUTS =================================
ANNO_X, ANNO_Y = 0.05, 0.15      # annotation box (axes-fraction)
LOC_LEG = 'best'                 # legend location, frameless
ALPHA = 0.05                     # significance threshold for Δ test

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
MODALITIES = ['Auditory', 'Visual', 'Both']
EXCLUDE_ROIS = {'heschl', 'occipital'}

# ============================ UTIL FUNCS ===============================
def fisher_z(r: float, eps: float = 1e-7) -> float:
    r = float(np.clip(r, -1 + eps, 1 - eps))
    return float(np.arctanh(r))

def fisher_r(z: float) -> float:
    return float(np.tanh(z))

def get_pcol(rmc_df: pd.DataFrame) -> str:
    for k in ('p-val', 'pval', 'p'):
        if k in rmc_df.columns:
            return k
    raise KeyError(f"No p column in rm_corr: {list(rmc_df.columns)}")

def per_subject_r(wide: pd.DataFrame, roi1: str, roi2: str) -> pd.Series:
    vals = {}
    for subj, subdf in wide.groupby('Subject', sort=False):
        if subdf[[roi1, roi2]].isna().any().any():
            continue
        if subdf.shape[0] < 2:
            continue
        r = np.corrcoef(subdf[roi1], subdf[roi2])[0, 1]
        vals[subj] = r
    return pd.Series(vals, name='r')

def unordered_pairs(keys: List[str]) -> List[Tuple[str, str]]:
    return list(itertools.combinations(keys, 2))

def make_wide_rmcorr(df: pd.DataFrame,
                     roi1: str,
                     roi2: str,
                     cat: str) -> pd.DataFrame:
    base = (
        df.query("ROI in [@roi1, @roi2] and Task in @TASKS and "
                 "Category == @cat")
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

def ensure_dirs(base_out: str) -> Dict[str, str]:
    beat_dir = os.path.join(base_out, 'beat_gt_interval')
    intv_dir = os.path.join(base_out, 'interval_gt_beat')
    for d in (beat_dir, intv_dir):
        os.makedirs(d, exist_ok=True)
    return {'beat': beat_dir, 'intv': intv_dir}

# ============================ PATHS/FILES ==============================
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL = 'rwls'          # 'rwls' or 'standard'
MASKING = 'wb'          # 'wb' or 'gm'
HRF = 'hrf128'          # 'hrf128' or 'hrf42'

BASE_DIR = os.path.join(
    WORKING_DIR,
    f"roi_analyses_{MODEL}_{HRF}_{MASKING}_puncorr_unsmoothed",
    'bothmod_allmain_tasks',
    'main_tasks',
)

# ============================== CORE RUN ==============================
def compute_stats_for_setting(
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

    panel_stats = {}
    for cat in CATS:
        wide = make_wide_rmcorr(df_mod, roi1, roi2, cat)
        if wide.empty:
            panel_stats[cat] = (np.nan, np.nan)
            continue
        rmc = pg.rm_corr(data=wide, x=roi1, y=roi2, subject='Subject')
        pcol = get_pcol(rmc)
        r_val = float(rmc['r'].iloc[0])
        p_val = float(rmc[pcol].iloc[0])
        panel_stats[cat] = (r_val, p_val)

    wide_b = make_wide_rmcorr(df_mod, roi1, roi2, 'Beat')
    wide_i = make_wide_rmcorr(df_mod, roi1, roi2, 'Interval')
    r_b = per_subject_r(wide_b, roi1, roi2)
    r_i = per_subject_r(wide_i, roi1, roi2)
    common = r_b.index.intersection(r_i.index)

    if len(common) >= 2:
        z_b = np.array([fisher_z(r) for r in r_b[common]])
        z_i = np.array([fisher_z(r) for r in r_i[common]])
        _, p_delta = ttest_rel(z_b, z_i)
        mean_r_b = fisher_r(float(np.mean(z_b)))
        mean_r_i = fisher_r(float(np.mean(z_i)))
        delta_r = mean_r_b - mean_r_i
        sig = bool(p_delta < ALPHA)
        if sig and delta_r > 0:
            direction = "Beat > Interval"
        elif sig and delta_r < 0:
            direction = "Interval > Beat"
        else:
            direction = "n.s."
        n_subj = int(len(common))
    else:
        p_delta, delta_r, sig, direction, n_subj = (
            np.nan, np.nan, False, "n.s.", 0
        )

    return {
        'r_rm_beat': panel_stats.get('Beat', (np.nan, np.nan))[0],
        'p_beat': panel_stats.get('Beat', (np.nan, np.nan))[1],
        'r_rm_interval': panel_stats.get('Interval', (np.nan, np.nan))[0],
        'p_interval': panel_stats.get('Interval', (np.nan, np.nan))[1],
        'delta_r': delta_r,
        'p_delta': p_delta,          # uncorrected p for winner selection
        'n_subjects': n_subj,
        'significant': sig,
        'direction': direction,
    }

def render_best_panel_plot(
    df_all: pd.DataFrame,
    grp: pd.DataFrame,
    hemi: str,
    roi1: str,
    roi2: str,
    mod: str,
    out_dir: str,
    indiv: str,
    delta_r: float,
    p_delta: float,
) -> str:
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

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, cat in zip(axes, CATS):
        sub_grp = grp_mod[
            (grp_mod['ROI'].isin([roi1, roi2])) &
            (grp_mod['Task'].isin(TASKS)) &
            (grp_mod['Category'] == cat)
        ]
        mat = mat_for_plot(sub_grp, roi1, roi2)

        wide = make_wide_rmcorr(df_mod, roi1, roi2, cat)
        rmc = pg.rm_corr(data=wide, x=roi1, y=roi2, subject='Subject')
        pcol = get_pcol(rmc)
        r_val = float(rmc['r'].iloc[0])
        p_val = float(rmc[pcol].iloc[0])

        ax.plot(
            TASKS, mat[roi1], marker='o',
            label=ROI_LABELS.get(roi1, roi1),
        )
        ax.plot(
            TASKS, mat[roi2], marker='s',
            label=ROI_LABELS.get(roi2, roi2),
        )
        ax.set_title(f'{hemi} • {mod} • {cat}')
        ax.set_xlabel('Task')
        ax.set_ylabel('PSC (%)')

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

    fig.suptitle(
        f"{hemi} • {ROI_LABELS.get(roi1, roi1)} vs "
        f"{ROI_LABELS.get(roi2, roi2)}"
    )
    fig.text(
        0.5, 0.01,
        f"Beat vs Interval (paired z): Δr={delta_r:.3f}, "
        f"p={p_delta:.3f}",
        ha='center', va='bottom'
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(
        out_dir,
        f"rmcorr_catdiff_{indiv}_{N_ROIS}-rois_"
        f"{roi1}-{roi2}_{hemi}_{mod.lower()}.png"
    )
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fname

def render_p_summary_plots(
    base_out_folder: str,
    indiv_levels: List[str],
    p_unc: List[float],
    title: str,
    stem: str,
) -> Tuple[str, str]:
    """Save uncorrected and Holm–Bonferroni p across individualization."""
    p_unc = [np.nan if v is None else v for v in p_unc]
    mask = ~np.isnan(p_unc)
    xs = np.arange(len(indiv_levels))[mask]
    labs = np.array(indiv_levels)[mask]
    y_unc = np.array(p_unc, float)[mask]

    # Holm–Bonferroni via Pingouin
    _, y_holm = pg.multicomp(y_unc.tolist(), method='holm')

    print(y_holm)

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
    plt.savefig(f_unc, dpi=300, bbox_inches='tight'); plt.close()

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
    plt.savefig(f_holm, dpi=300, bbox_inches='tight'); plt.close()

    return f_unc, f_holm

# =============================== MAIN =================================
def main() -> None:
    base_out = os.path.join(BASE_DIR, 'profile_similarity', 'periodicity')
    dirs = ensure_dirs(base_out)

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
            },
        )
        df_all = df_all[df_all['Task'].isin(TASKS)]
        grp = (
            df_all
            .groupby(
                ['Hemisphere', 'ROI', 'Task', 'Category', 'Modality']
            )['PSC']
            .mean()
            .reset_index()
        )

        candidate_rois = [k for k in ROI_LABELS.keys() if k not in EXCLUDE_ROIS]
        roi_pairs = unordered_pairs(candidate_rois)

        for hemi in HEMIS:
            for roi1, roi2 in roi_pairs:
                present = df_all[
                    (df_all['Hemisphere'] == hemi) &
                    (df_all['ROI'].isin([roi1, roi2]))
                ]['ROI'].unique()
                if not set([roi1, roi2]).issubset(set(present)):
                    continue
                for mod in MODALITIES:
                    stat = compute_stats_for_setting(
                        df_all, grp, hemi, roi1, roi2, mod
                    )
                    row = {
                        'individualization': indiv,
                        'hemisphere': hemi,
                        'modality': mod,
                        'roi1': roi1,
                        'roi2': roi2,
                        'bucket': bucket_for_pair(roi1, roi2),
                        **stat,
                    }
                    results.append(row)

    # Save summary TSV
    out_dir = base_out
    os.makedirs(out_dir, exist_ok=True)
    if not results:
        print("[INFO] No results computed.")
        return

    df_res = pd.DataFrame(results).sort_values(
        by=['individualization', 'hemisphere', 'modality', 'roi1', 'roi2']
    )
    tsv_path = os.path.join(out_dir, 'summary_periodicity.tsv')
    df_res.to_csv(tsv_path, sep='\t', index=False)
    print(f"Saved summary table to {tsv_path}")

    # For each ROI pair, choose the "winner" by smallest UNCORRECTED p
    # among significant rows; break ties by larger |delta_r|.
    best_rows = []
    for (roi1, roi2), g in df_res.groupby(['roi1', 'roi2']):
        g_sig = g[(~g['p_delta'].isna()) & (g['p_delta'] < ALPHA)]
        if g_sig.empty:
            continue
        # idx of min p; if multiple, pick max |delta_r|
        min_p = g_sig['p_delta'].min()
        g_min = g_sig[g_sig['p_delta'] == min_p]
        if len(g_min) > 1:
            idx = (g_min['delta_r'].abs()).idxmax()
        else:
            idx = g_min.index[0]
        best_rows.append(df_res.loc[idx].to_dict())

    # Render the winning panel and p-summary plots (unc & Holm)
    for best in best_rows:
        indiv_star = best['individualization']
        hemi = best['hemisphere']
        mod = best['modality']
        roi1 = best['roi1']
        roi2 = best['roi2']
        bucket = best['bucket']
        d_r = float(best['delta_r'])
        p_d = float(best['p_delta'])

        # Load data for this indiv to render the panel figure
        df_path = os.path.join(
            BASE_DIR, 'df_rois_volume',
            f"dfrois_{indiv_star}_{N_ROIS}-rois.tsv"
        )
        if not os.path.exists(df_path):
            continue
        df_all = pd.read_csv(
            df_path,
            sep='\t',
            dtype={
                'Subject': str, 'Task': str, 'ROI': str, 'Hemisphere': str,
                'Category': str, 'Modality': str, 'PSC': float,
            },
        )
        df_all = df_all[df_all['Task'].isin(TASKS)]
        grp = (
            df_all
            .groupby(
                ['Hemisphere', 'ROI', 'Task', 'Category', 'Modality']
            )['PSC']
            .mean()
            .reset_index()
        )

        # Decide parent folder based on direction/sign of delta
        if d_r > 0:
            parent = os.path.join(ensure_dirs(out_dir)['beat'], bucket)
        else:
            parent = os.path.join(ensure_dirs(out_dir)['intv'], bucket)
        os.makedirs(parent, exist_ok=True)

        # Panel figure for winning setting
        _ = render_best_panel_plot(
            df_all, grp, hemi, roi1, roi2, mod, parent, indiv_star, d_r, p_d
        )

        # Build p across all indiv for this (hemi, mod, roi1, roi2)
        sub = df_res[
            (df_res['hemisphere'] == hemi) &
            (df_res['modality'] == mod) &
            (df_res['roi1'] == roi1) &
            (df_res['roi2'] == roi2)
        ].set_index('individualization')

        p_list = [sub['p_delta'].get(ind, np.nan) for ind in INDIVID_LEVELS]

        # Save uncorrected and Holm-adjusted p plots (10 comparisons)
        title = (f"{hemi} • {ROI_LABELS.get(roi1, roi1)} vs "
                 f"{ROI_LABELS.get(roi2, roi2)} • {mod}")
        stem = (f"p_across_indiv_{roi1}-{roi2}_{hemi}_{mod.lower()}")
        render_p_summary_plots(parent, INDIVID_LEVELS, p_list, title, stem)

        print(f"Wrote winner + p-plots for {roi1}-{roi2} ({hemi}, {mod})")

if __name__ == "__main__":
    main()