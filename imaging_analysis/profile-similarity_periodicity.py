"""
Periodicity: within-subject rmcorr (Beat vs Interval) across Modality.

For each individualization level, hemisphere, and unordered ROI pair:
- Compute r_rm per Category (Beat, Interval) within each Modality
  (Both, Auditory, Visual).
- Paired test across tasks within subject:
  Beat vs Interval => Δr via Fisher-z per subject & paired t-test.
- Exclude ROI pairs containing 'heschl' or 'occipital'.
- Save a TSV summary over all settings (NO 'bucket' column).
- For each (roi1, roi2, hemisphere, modality), pick the single "winner"
  by the smallest UNCORRECTED p_delta across individualizations
  (p < ALPHA; tie by larger |Δr|), then:
    * save the two-panel figure (Beat & Interval) with modality in name;
    * save line plots of p_delta across individualizations:
      uncorrected and Holm–Bonferroni (Pingouin);
    * write winners_p_arrays.tsv sorted by pair, modality, hemisphere.

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
CATS = ['Beat', 'Interval']
MODALITIES = ['Both', 'Auditory', 'Visual']
EXCLUDE_ROIS = {'heschl', 'occipital'}

HEMI_ORDER = {h: i for i, h in enumerate(HEMIS)}
MOD_ORDER = {m: i for i, m in enumerate(MODALITIES)}


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


def unordered_pairs_alpha(keys: List[str]) -> List[Tuple[str, str]]:
    """All unordered, non-identical pairs from sorted ROI keys."""
    keys = sorted(keys)
    return list(itertools.combinations(keys, 2))


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


def make_wide_rmcorr(df: pd.DataFrame,
                     roi1: str,
                     roi2: str,
                     cat: str) -> pd.DataFrame:
    """Subject×Task wide table for rmcorr; averages duplicates if any."""
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


def periodicity_root(base_out: str) -> str:
    """Return (and create) the periodicity root folder only."""
    per_dir = os.path.join(base_out, 'profile_similarity', 'periodicity')
    os.makedirs(per_dir, exist_ok=True)
    return per_dir


def render_panel_beat_interval(
    df_mod: pd.DataFrame,
    grp_mod: pd.DataFrame,
    hemi: str,
    roi1: str,
    roi2: str,
    out_dir: str,
    indiv: str,
    mod: str,
    delta_r: float,
    p_delta: float,
) -> str:
    """Two subplots: Beat and Interval profiles with r_rm, and Δ line."""
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

        ax.plot(TASKS, mat[roi1], marker='o', label=ROI_LABELS.get(roi1, roi1))
        ax.plot(TASKS, mat[roi2], marker='s', label=ROI_LABELS.get(roi2, roi2))
        ax.set_title(f'{hemi} • {cat}')
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
            bbox=dict(boxstyle="round,pad=0.3",
                      fc="white", ec="gray", alpha=0.7),
        )

    fig.suptitle(f"{hemi} • {ROI_LABELS.get(roi1, roi1)} vs "
                 f"{ROI_LABELS.get(roi2, roi2)} • {mod}")
    fig.text(
        0.5, 0.01,
        f"Beat vs Interval (paired z): Δr={delta_r:.3f}, p={p_delta:.3f}",
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
    p_unc_full: List[float],
    title: str,
    stem: str,
) -> Tuple[str, str, List[float], List[float]]:
    """Save uncorrected and Holm p across individualization. Return both."""
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


# =============================== MAIN =================================
def main() -> None:
    per_root = periodicity_root(BASE_DIR)  # only create the root folder

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

        # Group means for plotting
        grp = (
            df_all
            .groupby(
                ['Hemisphere', 'Modality', 'ROI', 'Task', 'Category']
            )['PSC']
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
                    # subset for stats & plotting
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

                    # rmcorr per Category
                    panel_stats = {}
                    for cat in CATS:
                        wide = make_wide_rmcorr(df_mod, roi1, roi2, cat)
                        if wide.empty:
                            panel_stats[cat] = (np.nan, np.nan)
                            continue
                        rmc = pg.rm_corr(
                            data=wide, x=roi1, y=roi2, subject='Subject'
                        )
                        pcol = get_pcol(rmc)
                        r_val = float(rmc['r'].iloc[0])
                        p_val = float(rmc[pcol].iloc[0])
                        panel_stats[cat] = (r_val, p_val)

                    # Δ test Beat vs Interval
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
                        direction = ("Beat > Interval" if (sig and delta_r > 0)
                                     else "Interval > Beat" if sig else "n.s.")
                        n_subj = int(len(common))
                    else:
                        p_delta, delta_r, sig, direction, n_subj = (
                            np.nan, np.nan, False, "n.s.", 0
                        )

                    results.append({
                        'individualization': indiv,
                        'hemisphere': hemi,
                        'modality': mod,
                        'roi1': roi1,
                        'roi2': roi2,
                        'r_rm_beat': panel_stats.get('Beat', (np.nan,))[0],
                        'p_beat': panel_stats.get('Beat', (np.nan, np.nan))[1],
                        'r_rm_interval': panel_stats.get('Interval',
                                                         (np.nan,))[0],
                        'p_interval': panel_stats.get('Interval',
                                                      (np.nan, np.nan))[1],
                        'delta_r': delta_r,
                        'p_delta': p_delta,
                        'n_subjects': n_subj,
                        'significant': bool(sig),
                        'direction': direction,
                    })

    # Save summary TSV (no 'bucket' column)
    out_dir = per_root
    os.makedirs(out_dir, exist_ok=True)
    if not results:
        print("[INFO] No results computed.")
        return

    df_res = pd.DataFrame(results)
    df_res['hemisphere'] = pd.Categorical(df_res['hemisphere'],
                                          categories=HEMIS, ordered=True)
    df_res['modality'] = pd.Categorical(df_res['modality'],
                                        categories=MODALITIES, ordered=True)
    df_res = df_res.sort_values(
        by=['roi1', 'roi2', 'modality', 'hemisphere', 'individualization']
    )

    tsv_path = os.path.join(out_dir, 'summary_periodicity.tsv')
    print(f"Saved summary table to {tsv_path}")
    df_res.to_csv(tsv_path, sep='\t', index=False)

    # Winners: for each (pair, hemi, mod), pick best individualization
    best_rows = []
    for (roi1, roi2, hemi, mod), g in df_res.groupby(
        ['roi1', 'roi2', 'hemisphere', 'modality'], sort=False
    ):
        g_sig = g[(~g['p_delta'].isna()) & (g['p_delta'] < ALPHA)]
        if g_sig.empty:
            continue
        min_p = g_sig['p_delta'].min()
        g_min = g_sig[g_sig['p_delta'] == min_p]
        if len(g_min) > 1:
            idx = (g_min['delta_r'].abs()).idxmax()
        else:
            idx = g_min.index[0]
        best_rows.append(df_res.loc[idx].to_dict())

    # Sort winners by pair α-order, modality order, hemisphere order
    best_rows.sort(
        key=lambda d: (
            d['roi1'], d['roi2'],
            MOD_ORDER.get(d['modality'], 99),
            HEMI_ORDER.get(d['hemisphere'], 99)
        )
    )

    # Winners p-array TSV rows
    winners_rows: List[Dict[str, object]] = []

    # Render the winning panel & p-summary plots (unc & Holm)
    for best in best_rows:
        indiv_star = best['individualization']
        hemi = best['hemisphere']
        mod = best['modality']
        roi1 = best['roi1']
        roi2 = best['roi2']
        d_r = float(best['delta_r'])
        p_d = float(best['p_delta'])

        # Load data for the winner's individualization
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
            },
        )
        df_all = df_all[df_all['Task'].isin(TASKS)]

        # Subset modality for panel rendering
        if mod == 'Both':
            df_mod = df_all[df_all['Hemisphere'] == hemi]
            grp_mod = (
                df_all
                .groupby(
                    ['Hemisphere', 'Modality', 'ROI', 'Task', 'Category']
                )['PSC'].mean().reset_index()
            )
            grp_mod = grp_mod[grp_mod['Hemisphere'] == hemi]
        else:
            df_mod = df_all[
                (df_all['Hemisphere'] == hemi) &
                (df_all['Modality'] == mod)
            ]
            grp_mod = (
                df_all
                .groupby(
                    ['Hemisphere', 'Modality', 'ROI', 'Task', 'Category']
                )['PSC'].mean().reset_index()
            )
            grp_mod = grp_mod[
                (grp_mod['Hemisphere'] == hemi) &
                (grp_mod['Modality'] == mod)
            ]

        # Build parent path lazily (no empty folders)
        dir_name = 'beat_gt_interval' if d_r > 0 else 'interval_gt_beat'
        bucket = bucket_for_pair(roi1, roi2)
        parent = os.path.join(out_dir, dir_name, bucket)
        os.makedirs(parent, exist_ok=True)

        # Panel figure (two subplots) — include modality in filename
        _ = render_panel_beat_interval(
            df_mod, grp_mod, hemi, roi1, roi2, parent, indiv_star,
            mod, d_r, p_d
        )

        # Build p across all indiv for this (hemi, mod, roi1, roi2)
        sub = df_res[
            (df_res['hemisphere'] == hemi) &
            (df_res['modality'] == mod) &
            (df_res['roi1'] == roi1) &
            (df_res['roi2'] == roi2)
        ].set_index('individualization')

        p_list = [sub['p_delta'].get(ind, np.nan) for ind in INDIVID_LEVELS]

        title = (f"{hemi} • {ROI_LABELS.get(roi1, roi1)} vs "
                 f"{ROI_LABELS.get(roi2, roi2)} • {mod}")
        stem = (f"p_across_indiv_{roi1}-{roi2}_{hemi}_{mod.lower()}")

        _, _, p_unc_full, p_holm_full = render_p_summary_plots(
            parent, INDIVID_LEVELS, p_list, title, stem
        )

        # Clean console output: Holm array then the winner line
        print(np.array(p_holm_full))
        print(f"Wrote winner + p-plots for {roi1}-{roi2} ({hemi}, {mod})")

        winners_rows.append({
            'roi1': roi1,
            'roi2': roi2,
            'hemisphere': hemi,
            'modality': mod,
            'direction': ("Beat > Interval" if d_r > 0 else "Interval > Beat"),
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