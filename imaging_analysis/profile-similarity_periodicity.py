"""
Profile similarity (within-subject rmcorr) by Category and Modality.

For each individualization level, hemisphere, and unordered ROI pair:
- Compute repeated-measures correlation (r_rm) per Category
  (Beat/Interval) within each Modality (Auditory, Visual) and pooled.
- Test Beat vs Interval using subject-wise Pearson r across tasks,
  Fisher-z transform, and paired t-test (within-subject).
- Save a TSV summary (adds 'direction' column).
- Plot ONLY when Beat > Interval is significant (Δr > 0 and p < ALPHA).
- Outputs saved under .../profile_similarity/periodicity/.

Author: Ana Luisa Pinho
email: agrilopi@uwo.ca

Creation: 1st of October 2025
Last Update: October 2025

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

try:
    import pingouin as pg
except Exception:
    pg = None


# ============================== INPUTS ===============================
# Plot cosmetics
ANNO_X, ANNO_Y = 0.05, 0.15      # annotation box (axes-fraction)
LOC_LEG = 'best'                 # legend location, frameless

# Dataset selection
N_ROIS = 8
INDIVID_LEVELS = [
    'i', 'i9a', 'i8a', 'i7a', 'i6a',
    'a', 'a4g', 'a3g', 'a2g', 'a1g', 'g',
]
ALPHA = 0.05                     # significance threshold for Δ test

# Human-friendly labels (updated)
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

# Hemispheres, tasks, categories, modalities
HEMIS = ['bh', 'lh', 'rh']       # updated per request
TASKS = ['Production', 'Perception', 'NTFD']
CATS = ['Beat', 'Interval']
MODALITIES = ['Auditory', 'Visual', 'Both']  # 'Both' pools modalities


# ============================ UTIL FUNCS =============================
def fisher_z(r: float, eps: float = 1e-7) -> float:
    """Fisher-z transform with clipping to avoid infinities."""
    r = float(np.clip(r, -1 + eps, 1 - eps))
    return float(np.arctanh(r))


def fisher_r(z: float) -> float:
    """Inverse Fisher-z transform."""
    return float(np.tanh(z))


def get_pcol(rmc_df: pd.DataFrame) -> str:
    """Return the p-value column name from pingouin output."""
    for k in ('p-val', 'pval', 'p'):
        if k in rmc_df.columns:
            return k
    raise KeyError(f"No p column in rm_corr: {list(rmc_df.columns)}")


def per_subject_r(wide: pd.DataFrame, roi1: str, roi2: str) -> pd.Series:
    """Subject-wise Pearson r across tasks for two ROI columns."""
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
    """All unordered, non-identical pairs from a list of ROI keys."""
    return list(itertools.combinations(keys, 2))


def make_wide_rmcorr(df: pd.DataFrame,
                     roi1: str,
                     roi2: str,
                     cat: str) -> pd.DataFrame:
    """Build Subject×Task wide table for rmcorr, averaging duplicates.

    Averages over Modality when both are present so we end with a single
    PSC per Subject×Task×ROI cell.
    """
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
    """Task×ROI matrix for plotting; averages duplicates if any."""
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


def run_pair_one_mod(
    df_mod: pd.DataFrame,
    grp_mod: pd.DataFrame,
    hemi: str,
    roi1: str,
    roi2: str,
    mod: str,
    out_dir: str,
    indiv: str,
    do_plot: bool,
) -> Optional[Dict[str, object]]:
    """Compute stats and optionally plot for one ROI pair & modality.

    Returns a dict with summary stats if Beat–Interval Δ test ran,
    else None (e.g., insufficient subjects).
    """
    # Prepare figure with two panels
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    diff_text = ""
    panel_stats = {}

    for ax, cat in zip(axes, CATS):
        sub_grp = grp_mod[
            (grp_mod['ROI'].isin([roi1, roi2])) &
            (grp_mod['Task'].isin(TASKS)) &
            (grp_mod['Category'] == cat)
        ]
        mat = mat_for_plot(sub_grp, roi1, roi2)

        if mat[[roi1, roi2]].isna().any().any():
            ax.text(
                0.5, 0.5, "Missing data",
                transform=ax.transAxes,
                ha='center', va='center'
            )
            continue

        wide = make_wide_rmcorr(df_mod, roi1, roi2, cat)
        if wide.empty:
            ax.text(
                0.5, 0.5, "No rows",
                transform=ax.transAxes,
                ha='center', va='center'
            )
            continue

        rmc = pg.rm_corr(data=wide, x=roi1, y=roi2, subject='Subject')
        pcol = get_pcol(rmc)
        r_val = float(rmc['r'].iloc[0])
        p_val = float(rmc[pcol].iloc[0])
        panel_stats[cat] = (r_val, p_val)

        # Plot profiles
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

    # Beat vs Interval paired Fisher-z test
    wide_beat = make_wide_rmcorr(df_mod, roi1, roi2, 'Beat')
    wide_intv = make_wide_rmcorr(df_mod, roi1, roi2, 'Interval')

    r_b = per_subject_r(wide_beat, roi1, roi2)
    r_i = per_subject_r(wide_intv, roi1, roi2)

    common = r_b.index.intersection(r_i.index)
    if len(common) >= 2:
        z_b = np.array([fisher_z(r) for r in r_b[common]])
        z_i = np.array([fisher_z(r) for r in r_i[common]])
        tval, p_diff = ttest_rel(z_b, z_i)
        mean_r_b = fisher_r(float(np.mean(z_b)))
        mean_r_i = fisher_r(float(np.mean(z_i)))
        d_r = mean_r_b - mean_r_i
        sig = (p_diff < ALPHA)
        if sig and d_r > 0:
            direction = "Beat > Interval"
        elif sig and d_r < 0:
            direction = "Interval > Beat"
        else:
            direction = "n.s."
        diff_text = (
            f"Beat vs Interval (paired z): "
            f"Δr={d_r:.3f}, p={p_diff:.3f} (n={len(common)})"
        )
    else:
        p_diff, d_r, sig, direction = np.nan, np.nan, False, "n.s."
        diff_text = "Insufficient subjects for paired test."

    # Decide plotting: only if significant and Δr > 0 (Beat > Interval)
    if do_plot and sig and d_r > 0:
        fig.suptitle(
            f"{hemi} • {ROI_LABELS.get(roi1, roi1)} vs "
            f"{ROI_LABELS.get(roi2, roi2)}"
        )
        fig.text(0.5, 0.01, diff_text, ha='center', va='bottom')
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

        fname = os.path.join(
            out_dir,
            f"rmcorr_catdiff_{indiv}_{N_ROIS}-rois_"
            f"{roi1}-{roi2}_{hemi}_{mod.lower()}.png"
        )
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {fname}")
    plt.close(fig)

    # Build return dict (log row)
    if len(common) >= 2:
        ret: Dict[str, object] = {
            'individualization': indiv,
            'hemisphere': hemi,
            'modality': mod,
            'roi1': roi1,
            'roi2': roi2,
            'r_rm_beat': panel_stats.get('Beat', (np.nan, np.nan))[0],
            'p_beat': panel_stats.get('Beat', (np.nan, np.nan))[1],
            'r_rm_interval': panel_stats.get('Interval',
                                             (np.nan, np.nan))[0],
            'p_interval': panel_stats.get('Interval',
                                          (np.nan, np.nan))[1],
            'delta_r': d_r,
            'p_delta': p_diff,
            'n_subjects': int(len(common)),
            'significant': bool(sig),
            'direction': direction,
        }
        return ret
    return None


# ============================ PATHS/FILES ============================
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


# =============================== RUN =================================
def main() -> None:
    """Run rmcorr by Category×Modality across indiv levels and ROI pairs."""
    if pg is None:
        raise ImportError("pingouin is required for repeated-measures r.")

    results: List[Dict[str, object]] = []

    # Loop all individualization levels
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

        # Task-wise only
        df_all = df_all[df_all['Task'].isin(TASKS)]

        # Precompute group means for plotting
        grp = (
            df_all
            .groupby(
                ['Hemisphere', 'ROI', 'Task', 'Category', 'Modality']
            )['PSC']
            .mean()
            .reset_index()
        )

        roi_pairs = unordered_pairs(list(ROI_LABELS.keys()))

        for hemi in HEMIS:
            for roi1, roi2 in roi_pairs:

                present = df_all[
                    (df_all['Hemisphere'] == hemi) &
                    (df_all['ROI'].isin([roi1, roi2]))
                ]['ROI'].unique()
                if not set([roi1, roi2]).issubset(set(present)):
                    print(f"[INFO] Skip {indiv} {roi1}-{roi2} ({hemi}): "
                          f"no data.")
                    continue

                for mod in MODALITIES:

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

                    out_dir = os.path.join(
                        BASE_DIR, 'profile_similarity', 'periodicity'
                    )
                    os.makedirs(out_dir, exist_ok=True)

                    stat = run_pair_one_mod(
                        df_mod=df_mod,
                        grp_mod=grp_mod,
                        hemi=hemi,
                        roi1=roi1,
                        roi2=roi2,
                        mod=mod,
                        out_dir=out_dir,
                        indiv=indiv,
                        do_plot=True,   # plots only if sig and Δr > 0
                    )
                    if stat is not None:
                        results.append(stat)

    # Save summary table
    if results:
        df_res = pd.DataFrame(results)
        # Sort for readability
        df_res = df_res.sort_values(
            by=['individualization', 'hemisphere', 'modality',
                'roi1', 'roi2']
        )
        out_dir = os.path.join(BASE_DIR, 'profile_similarity', 'periodicity')
        os.makedirs(out_dir, exist_ok=True)
        tsv_path = os.path.join(out_dir, 'summary_periodicity.tsv')
        df_res.to_csv(tsv_path, sep='\t', index=False)
        print(f"Saved summary table to {tsv_path}")
    else:
        print("[INFO] No results to save (no valid paired tests).")


if __name__ == "__main__":
    main()