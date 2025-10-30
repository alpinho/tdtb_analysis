import os
from itertools import combinations

import numpy as np
import pandas as pd
import PcmPy as pcm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D


# ============================= FUNCTIONS =========================== #

def plot_mds_2d(coords, labels, explained_var, out_path, comps=(1, 2)):
    """
    Plot 2D MDS for chosen components (1-based indices).

    coords : (n, p) array of MDS scores
    labels : list-like of length n (e.g., ROI names)
    explained_var : (p,) eigenvalues or per-axis variance
    out_path : PNG output path
    comps : tuple of 2 components, 1-based (e.g., (1, 3))
    """
    c1, c2 = comps[0] - 1, comps[1] - 1

    var = np.clip(explained_var, 0, None)
    var = var / (var.sum() if var.sum() > 0 else 1.0)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    ax.scatter(coords[:, c1], coords[:, c2])
    for x, y, name in zip(coords[:, c1], coords[:, c2], labels):
        ax.text(x, y, f" {name}", va="center", ha="left")

    ax.set_xlabel(f"MDS{c1+1} ({var[c1]:.1%})")
    ax.set_ylabel(f"MDS{c2+1} ({var[c2]:.1%})")
    ax.axhline(0, lw=0.5)
    ax.axvline(0, lw=0.5)
    ax.set_title(
        f"Classical MDS - 2D (MDS{c1+1} vs MDS{c2+1})"
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_mds_3d(coords, labels, explained_var, out_path, comps=(1, 2, 3)):
    """
    Plot 3D MDS for chosen components (1-based indices).

    coords : (n, p) array of MDS scores
    labels : list-like of length n (e.g., ROI names)
    explained_var : (p,) eigenvalues or per-axis variance
    out_path : PNG output path
    comps : tuple of 3 components, 1-based (e.g., (1, 2, 4))
    """
    c1, c2, c3 = comps[0] - 1, comps[1] - 1, comps[2] - 1

    var = np.clip(explained_var, 0, None)
    var = var / (var.sum() if var.sum() > 0 else 1.0)

    fig = plt.figure(figsize=(6, 5), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(coords[:, c1], coords[:, c2], coords[:, c3])
    for x, y, z, name in zip(
        coords[:, c1], coords[:, c2], coords[:, c3], labels
    ):
        ax.text(x, y, z, f" {name}")

    ax.set_xlabel(f"MDS{c1+1} ({var[c1]:.1%})")
    ax.set_ylabel(f"MDS{c2+1} ({var[c2]:.1%})")
    ax.set_zlabel(f"MDS{c3+1} ({var[c3]:.1%})")
    ax.set_title(
        f"Classical MDS - 3D (MDS{c1+1}, {c2+1}, {c3+1})"
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

# ============================= CONFIG ============================== #

# Number of components to keep from the decomposition
N_COMPONENTS = 3

# Individualization level of input data
INDIVID_LEVEL = 'i'

# =============================== PATHS ============================= #

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL = 'rwls'
MASKING = 'wb'
HRF = 'hrf128'

BASE_ALL = os.path.join(
    WORKING_DIR,
    f"roi_analyses_{MODEL}_{HRF}_{MASKING}_puncorr_unsmoothed",
    'bothmod_allmain_tasks',
    'profile_similarity',
    'encoding_restrand',
    'i',
    'matrices',
)

MDS_OUTPUT_DIR = os.path.join(BASE_ALL, 'mds', INDIVID_LEVEL, 
                              f'{N_COMPONENTS}comps')

# ================================ RUN ============================== #

if __name__ == "__main__":

    # Load input matrix (ROI x ROI correlation or Gram matrix)
    mtx_path = os.path.join(
        BASE_ALL, 
        'matrix_r_' + INDIVID_LEVEL + '_Both_bh_8-rois_withrestrand.tsv'
    )
    df = pd.read_csv(mtx_path, sep="\t", index_col=0)
    mtx = df.to_numpy(dtype=float)

    # Normalize the correlation matrix by its rank
    rank = np.linalg.matrix_rank(mtx)
    mtx = mtx / rank

    # Classical MDS on the Gram/correlation matrix
    # scores are V * sqrt(eigenvalues); eigval are eigenvalues
    scores, eigval = pcm.util.classical_mds(mtx)

    # Keep the first N components
    coords = scores[:, :N_COMPONENTS]
    ev = eigval[:N_COMPONENTS]

    # Variance share over all positive eigenvalues
    ev_pos = np.clip(eigval, 0, None)
    denom = ev_pos.sum() if ev_pos.sum() > 0 else 1.0
    ev_ratio_full = ev_pos[:N_COMPONENTS] / denom

    os.makedirs(MDS_OUTPUT_DIR, exist_ok=True)

    # All 2D pairs
    axes_1based = list(range(1, coords.shape[1] + 1))
    for a, b in combinations(axes_1based, 2):
        out2d = os.path.join(
            MDS_OUTPUT_DIR, 
            f"{INDIVID_LEVEL}_ncomps-{N_COMPONENTS}_mds_pair_{a}_{b}.png"
        )
        plot_mds_2d(
            coords=coords,
            labels=df.index,
            explained_var=ev_ratio_full,
            out_path=out2d,
            comps=(a, b),
        )

    # All 3D triples
    if coords.shape[1] >= 3:
        for a, b, c in combinations(axes_1based, 3):
            out3d = os.path.join(
                MDS_OUTPUT_DIR, 
                f"{INDIVID_LEVEL}_ncomps-{N_COMPONENTS}_mds_triplet_"
                f"{a}_{b}_{c}.png"
            )
            plot_mds_3d(
                coords=coords,
                labels=df.index,
                explained_var=ev_ratio_full,
                out_path=out3d,
                comps=(a, b, c),
            )