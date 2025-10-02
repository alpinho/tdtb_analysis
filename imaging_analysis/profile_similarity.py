"""
Compute profile similarity between two ROIs across tasks using either
repeated-measures correlation (Pearson) or cosine similarity.

- rmcorr: Pearson r within subjects (centering per subject), with p.
- cosine: cosine similarity of group-mean task profiles + perm p.

Author: Ana Luisa Pinho
Creation: 27 Jun 2025
Last Update: 01 Oct 2025

Compat.: Python 3.10.16
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import pingouin as pg
except Exception:
    pg = None

from scipy.spatial.distance import cosine
from itertools import permutations
from math import factorial
from numpy.random import default_rng


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity of two 1D vectors."""
    return 1.0 - float(cosine(v1, v2))


def cosine_p_value(v1: np.ndarray,
                   v2: np.ndarray,
                   two_sided: bool = True,
                   n_perm: int = 10000,
                   seed: int | None = 42) -> tuple[float, float]:
    """Permutation test for cosine similarity across tasks.

    Null: task pairing is arbitrary. We permute v2's task order.

    For small T (<=8), run exact test over all permutations. For larger
    T, use Monte-Carlo with n_perm samples.

    Returns
    -------
    cos_sim : float
        Observed cosine similarity.
    p_val : float
        Two- or one-sided p-value with add-one correction.
    """
    v1 = np.asarray(v1, float)
    v2 = np.asarray(v2, float)
    assert v1.shape == v2.shape and v1.ndim == 1

    obs = cosine_similarity(v1, v2)
    T = v1.size

    def stat(arr):
        return cosine_similarity(v1, arr)

    if T <= 8:
        sims = []
        for perm in permutations(range(T)):
            sims.append(stat(v2[list(perm)]))
        sims = np.asarray(sims, float)
        m = sims.size
    else:
        rng = default_rng(seed)
        sims = np.empty(n_perm, float)
        for i in range(n_perm):
            idx = rng.permutation(T)
            sims[i] = stat(v2[idx])
        m = n_perm

    if two_sided:
        extreme = np.sum(np.abs(sims) >= abs(obs))
    else:
        extreme = np.sum(sims >= obs)

    # add-one correction for valid p even when obs is most extreme
    p = (extreme + 1.0) / (m + 1.0)
    return obs, p


# ============================== INPUTS =================================
# Similarity: 'rmcorr' or 'cosine'
similarity = 'cosine'

# Plot cosmetics
anno_x, anno_y = 0.05, 0.10   # text-box (axes-fraction)
loc_leg = 'best'              # ROI legend location

# Dataset selection
n_rois = 8
individualization = 'i8a'     # e.g., 'i8a', 'g', etc.

# Pair of ROIs to compare
roi1, roi2 = 'dstr', 'sma'

# Human-friendly labels (extend as needed)
roi_labels = {
    'dstr': 'Dorsal Striatum',
    'sma': 'SMA',
    'cereb': 'Cerebellum',
    'pmv': 'PMV',
    'pmd': 'PMD',
    'presma': 'preSMA',
    'heschl': 'Heschl Gyrus',
    'occipital_lobe': 'Occipital Lobe',
}

# Hemispheres and task labels
hemis = ['bh']
tasks = ['Production', 'Perception', 'NTFD']

# ============================ PATHS/FILES ==============================
working_dir = os.path.dirname(os.path.abspath(__file__))
model = 'rwls'          # 'rwls' or 'standard'
masking = 'wb'          # 'wb' or 'gm'
hrf_cutoff = 'hrf128'   # 'hrf128' or 'hrf42'

rois_dir = os.path.join(
    working_dir,
    f"roi_analyses_{model}_{hrf_cutoff}_{masking}_puncorr_unsmoothed",
    'bothmod_allmain_tasks',
    'main_tasks',
)
df_dir = os.path.join(rois_dir, 'df_rois_volume')
df_path = os.path.join(
    df_dir, f"dfrois_{individualization}_{str(n_rois)}-rois.tsv"
)

# =============================== RUN ==================================
# Read and clean
df_all = pd.read_csv(
    df_path,
    sep='\t',
    dtype={
        'Subject': str,
        'Task': str,
        'ROI': str,
        'Hemisphere': str,
        'PSC': float,
    },
)
df_all = df_all[df_all['Task'] != 'All Tasks']

# Precompute group means for plotting and cosine
grp = (
    df_all
    .groupby(['Hemisphere', 'ROI', 'Task'])['PSC']
    .mean()
    .reset_index()
)

for hemi in hemis:
    # ---- group-mean matrix for plotting/cosine ----
    sub_grp = grp[
        (grp['Hemisphere'] == hemi) &
        (grp['ROI'].isin([roi1, roi2])) &
        (grp['Task'].isin(tasks))
    ]
    mat = (
        sub_grp
        .pivot(index='Task', columns='ROI', values='PSC')
        .reindex(index=tasks)
    )

    # Sanity check
    if mat[[roi1, roi2]].isna().any().any():
        print(f"[WARN] Missing data for {hemi}; skipping plot.")
        continue

    # ---- compute similarity ----
    stat_text = ""
    fname_prefix = ""

    if similarity == 'rmcorr':
        if pg is None:
            raise ImportError(
                "pingouin not available; install it or use 'cosine'."
            )
        # Wide table per Subject×Task for the two ROIs
        wide = (
            df_all
            .query(
                "Hemisphere == @hemi and ROI in [@roi1, @roi2] and "
                "Task in @tasks"
            )
            .pivot_table(
                index=['Subject', 'Task'],
                columns='ROI',
                values='PSC',
            )
            .reset_index()
            .dropna(subset=[roi1, roi2])
        )
        if wide.empty:
            print(f"[WARN] No rows for rmcorr in {hemi}; skipping.")
            continue

        rmc = pg.rm_corr(
            data=wide, x=roi1, y=roi2, subject='Subject'
        )
        r_val = float(rmc['r'].iloc[0])
        p_val = float(rmc['p-val'].iloc[0])

        stat_text = rf"$r_{{rm}} = {r_val:.3f},\ p = {p_val:.3f}$"
        fname_prefix = "pearson"

    elif similarity == 'cosine':
        # Cosine similarity with permutation test
        v1 = mat[roi1].to_numpy()
        v2 = mat[roi2].to_numpy()
        cos_sim, p_val = cosine_p_value(
            v1, v2, two_sided=True, n_perm=10000, seed=42
        )
        stat_text = rf"$\cos\theta = {cos_sim:.3f},\ p = {p_val:.3f}$"
        fname_prefix = "cosine"

    else:
        raise ValueError("similarity must be 'rmcorr' or 'cosine'.")

    # ---- plot ----
    plt.figure(figsize=(5, 4))
    plt.plot(
        tasks, mat[roi1], marker='o',
        label=roi_labels.get(roi1, roi1),
    )
    plt.plot(
        tasks, mat[roi2], marker='s',
        label=roi_labels.get(roi2, roi2),
    )
    plt.title(f'{hemi} ROI Profiles Comparison')
    plt.xlabel('Task')
    plt.ylabel('PSC (%)')

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, loc=loc_leg)

    ax.text(
        anno_x, anno_y, stat_text,
        transform=ax.transAxes,
        va='top',
        bbox=dict(
            boxstyle="round,pad=0.3",
            fc="white", ec="gray", alpha=0.7,
        ),
    )

    plt.tight_layout()

    # ---- save ----
    out_dir = os.path.join(rois_dir, 'profile_similarity')
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(
        out_dir,
        f"{fname_prefix}_{individualization}_{n_rois}-rois_"
        f"{roi1}-{roi2}_{hemi}.png",
    )
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {fname}")
    plt.close()