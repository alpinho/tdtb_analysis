#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multidimensional scaling (MDS) plots for ROI similarity matrices

Author: Ana Luisa Pinho
email: agrilopi@uwo.ca

Created: 30th of October 2025
Last Update: January 2026

Compatibility: Python 3.10.16
"""

import os
from itertools import combinations

import numpy as np
import pandas as pd
import PcmPy as pcm

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D


# ============================= FUNCTIONS =========================== #

class SparseTickLabeler:
    """Callable tick labeler: label every `label_step`, keep others blank."""

    def __init__(
        self,
        label_step=0.1,
        decimals=1,
        zero_str="0.0",
        atol_mult=1e-8,
        atol_zero=1e-12,
    ):
        self.label_step = float(label_step)
        self.decimals = int(decimals)
        self.zero_str = str(zero_str)
        self.atol_mult = float(atol_mult)
        self.atol_zero = float(atol_zero)

    def __call__(self, val, _pos):
        q = val / self.label_step if self.label_step != 0 else val
        if np.isclose(q, np.round(q), atol=self.atol_mult):
            if np.isclose(val, 0.0, atol=self.atol_zero):
                return self.zero_str
            return f"{val:.{self.decimals}f}"
        return ""


def build_fixed_ticks(vmin, vmax, step):
    """Return fixed ticks spanning [vmin, vmax] with the given step."""
    step = float(step)
    if step <= 0:
        return np.array([vmin, vmax], dtype=float)

    lo = float(min(vmin, vmax))
    hi = float(max(vmin, vmax))

    start = np.floor(lo / step) * step
    stop = np.ceil(hi / step) * step

    return np.arange(start, stop + step / 2.0, step, dtype=float)


def nudge_texts_inside_axes(fig, ax, texts, pad_px=8, max_iter=30):
    """Nudge 2D text labels so their bboxes fit inside axes box.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure hosting the axes.
    ax : matplotlib.axes.Axes
        Target axes.
    texts : list[matplotlib.text.Text]
        Text artists to constrain.
    pad_px : int
        Pixel padding from axes box to avoid spines/ticks.
    max_iter : int
        Max iterations per label to reach containment.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_bbox = ax.get_window_extent(renderer=renderer)

    x0 = ax_bbox.x0 + pad_px
    x1 = ax_bbox.x1 - pad_px
    y0 = ax_bbox.y0 + pad_px
    y1 = ax_bbox.y1 - pad_px

    inv = ax.transData.inverted()

    for txt in texts:
        for _ in range(max_iter):
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bbox = txt.get_window_extent(renderer=renderer)

            shift_x = 0.0
            shift_y = 0.0

            if bbox.x0 < x0:
                shift_x += (x0 - bbox.x0)
            if bbox.x1 > x1:
                shift_x -= (bbox.x1 - x1)
            if bbox.y0 < y0:
                shift_y += (y0 - bbox.y0)
            if bbox.y1 > y1:
                shift_y -= (bbox.y1 - y1)

            if shift_x == 0.0 and shift_y == 0.0:
                break

            x_dat, y_dat = txt.get_position()
            p0 = ax.transData.transform((x_dat, y_dat))
            p1 = (p0[0] + shift_x, p0[1] + shift_y)
            x_new, y_new = inv.transform(p1)
            txt.set_position((x_new, y_new))


def plot_mds_2d(coords, labels, explained_var, out_path, comps=(1, 2)):
    """Plot 2D classical MDS and keep labels fully inside axes."""
    c1 = comps[0] - 1
    c2 = comps[1] - 1

    var = np.clip(explained_var, 0, None)
    denom = var.sum() if var.sum() > 0 else 1.0
    var = var / denom

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    ax.scatter(coords[:, c1], coords[:, c2])

    labels = [ROI_LABELS.get(str(lab), str(lab)) for lab in labels]

    x_rng = float(coords[:, c1].max() - coords[:, c1].min())
    y_rng = float(coords[:, c2].max() - coords[:, c2].min())
    dx = x_rng * 0.015 if x_rng > 0 else 0.01
    dy = y_rng * 0.015 if y_rng > 0 else 0.01

    offsets = [
        (dx, 0.0),
        (0.0, dy),
        (-dx, 0.0),
        (0.0, -dy),
        (dx, dy),
        (-dx, dy),
        (dx, -dy),
        (-dx, -dy),
    ]

    texts = []
    for k, (x_val, y_val, name) in enumerate(
        zip(coords[:, c1], coords[:, c2], labels)
    ):
        off_x, off_y = offsets[k % len(offsets)]
        txt = ax.text(
            x_val + off_x,
            y_val + off_y,
            name,
            va="center",
            ha="left",
            clip_on=True,
        )
        texts.append(txt)

    ax.set_xlabel(f"MDS{c1 + 1} ({var[c1]:.1%})")
    ax.set_ylabel(f"MDS{c2 + 1} ({var[c2]:.1%})")
    ax.axhline(0, lw=0.5)
    ax.axvline(0, lw=0.5)
    ax.set_title(f"Classical MDS - 2D (MDS{c1 + 1} vs MDS{c2 + 1})")

    nudge_texts_inside_axes(fig, ax, texts, pad_px=8, max_iter=30)

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

    ax.scatter(
        coords[:, c1],
        coords[:, c2],
        coords[:, c3],
        color="C0",
        alpha=0.8,
    )

    labels = [ROI_LABELS.get(str(lab), str(lab)) for lab in labels]

    x_rng = float(coords[:, c1].max() - coords[:, c1].min())
    y_rng = float(coords[:, c2].max() - coords[:, c2].min())
    z_rng = float(coords[:, c3].max() - coords[:, c3].min())

    dx = x_rng * 0.02 if x_rng > 0 else 0.01
    dy = y_rng * 0.02 if y_rng > 0 else 0.01
    dz = z_rng * 0.02 if z_rng > 0 else 0.01

    offsets = [
        (dx, 0.0, 0.0),
        (0.0, dy, 0.0),
        (0.0, 0.0, dz),
        (-dx, 0.0, 0.0),
        (0.0, -dy, 0.0),
        (0.0, 0.0, -dz),
        (dx, dy, 0.0),
        (-dx, -dy, 0.0),
    ]

    for k, (x, y, z, name) in enumerate(
        zip(coords[:, c1], coords[:, c2], coords[:, c3], labels)
    ):
        ox, oy, oz = offsets[k % len(offsets)]
        ha = "left"
        va = "center"

        if name == "Heschl Gyrus":
            # Keep inside axes; slightly closer to the marker.
            ox = -1.6 * dx
            oy = 2.6 * dy
            oz = 0.6 * dz
            ha = "right"

        elif name == "PreSMA":
            # Place label left and slightly below the marker.
            ox = -0.4 * dx
            oy = -1.4 * dy
            ha = "right"

        elif name == "Dorsal Striatum":
            # Lift label above the marker.
            oz = 2.5 * dz

        elif name == "Occipital Lobe":
            # Push label slightly forward and upward.
            oy = 1.8 * dy
            oz = 1.5 * dz

        elif name == "Cerebellum":
            # Pull label toward the cluster center.
            ox = 1.0 * dx
            oy = 0.6 * dy

        elif name == "PMV":
            # Pull label toward the cluster center.
            ox = 0.7 * dx
            oy = 1.0 * dy

        elif name == "PMD":
            # Pull label toward the cluster center.
            ox = 0.9 * dx
            oy = 0.4 * dy

        elif name == "SMA":
            # Toward center, but slightly higher than others.
            ox = 0.8 * dx
            oy = 0.7 * dy

        ax.text(
            x + ox,
            y + oy,
            z + oz,
            name,
            ha=ha,
            va=va,
            clip_on=True,
        )

    ax.set_xlabel(f"MDS{c1 + 1} ({var[c1]:.1%})")
    ax.set_ylabel(f"MDS{c2 + 1} ({var[c2]:.1%})")
    ax.set_zlabel(f"MDS{c3 + 1} ({var[c3]:.1%})")
    ax.set_title(f"Classical MDS - 3D (MDS{c1 + 1}, {c2 + 1}, {c3 + 1})")

    ax.view_init(elev=25, azim=-75)

    # MDS1: start at 0 and extend to the most negative value.
    x_min = float(np.min(coords[:, c1]))
    ax.set_xlim(x_min, 0.0)

    # Center MDS2 and MDS3 at 0 (symmetric limits).
    y_max = float(np.max(np.abs(coords[:, c2])))
    z_max = float(np.max(np.abs(coords[:, c3])))
    ax.set_ylim(-y_max, y_max)
    ax.set_zlim(-z_max, z_max)

    # Make panes transparent so lines keep uniform color.
    ax.xaxis.pane.set_alpha(0.0)
    ax.yaxis.pane.set_alpha(0.0)
    ax.zaxis.pane.set_alpha(0.0)

    # Ticks: keep 0.05 tick spacing on all axes, but label every 0.1.
    tick_step = 0.05
    label_step = 0.1

    labeler = SparseTickLabeler(
        label_step=label_step,
        decimals=1,
        zero_str="0.0",
        atol_mult=1e-8,
        atol_zero=1e-12,
    )
    formatter = FuncFormatter(labeler)

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    z0, z1 = ax.get_zlim()

    xt = build_fixed_ticks(x0, x1, tick_step)
    yt = build_fixed_ticks(y0, y1, tick_step)
    zt = build_fixed_ticks(z0, z1, tick_step)

    ax.xaxis.set_major_locator(FixedLocator(xt))
    ax.yaxis.set_major_locator(FixedLocator(yt))
    ax.zaxis.set_major_locator(FixedLocator(zt))

    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.zaxis.set_major_formatter(formatter)

    # Keep equal visual spacing per tick-step across all axes.
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    z0, z1 = ax.get_zlim()

    nx = abs(x1 - x0) / tick_step if tick_step > 0 else 1.0
    ny = abs(y1 - y0) / tick_step if tick_step > 0 else 1.0
    nz = abs(z1 - z0) / tick_step if tick_step > 0 else 1.0
    ax.set_box_aspect((nx, ny, nz))

    # Draw vertical lines after setting axis limits so they touch plane.
    z_bottom = float(ax.get_zlim()[0])
    for x, y, z in zip(coords[:, c1], coords[:, c2], coords[:, c3]):
        ax.plot(
            [x, x],
            [y, y],
            [z_bottom, z],
            color="C1",
            alpha=0.5,
            linewidth=0.8,
        )

    # Rotate tick labels to match the view angle (screen-space).
    ax.tick_params(axis="x", labelrotation=20)
    ax.tick_params(axis="y", labelrotation=-35)
    ax.tick_params(axis="z", labelrotation=90)

    # Padding between ticks and tick labels (points)
    ax.tick_params(axis="x", pad=-3)
    ax.tick_params(axis="y", pad=3)
    ax.tick_params(axis="z", pad=3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ============================= CONFIG ============================== #

# Number of components to keep from the decomposition
N_COMPONENTS = 3

# Individualization level of input data
INDIVID_LEVEL = "i"

ROI_LABELS = {
    "dstr": "Dorsal Striatum",
    "cereb": "Cerebellum",
    "pmv": "PMV",
    "pmd": "PMD",
    "presma": "PreSMA",
    "sma": "SMA",
    "heschl": "Heschl Gyrus",
    "occipital": "Occipital Lobe",
    "occipital_lobe": "Occipital Lobe",
}

# =============================== PATHS ============================= #

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL = "rwls"
MASKING = "wb"
HRF = "hrf128"

BASE_ALL = os.path.join(
    WORKING_DIR,
    f"roi_analyses_{MODEL}_{HRF}_{MASKING}_puncorr_unsmoothed",
    "bothmod_allmain_tasks",
    "profile_similarity",
    "encoding_restrand",
    "i",
    "matrices",
)

MDS_OUTPUT_DIR = os.path.join(
    BASE_ALL,
    "mds",
    INDIVID_LEVEL,
    f"{N_COMPONENTS}comps",
)

# ================================ RUN ============================== #

if __name__ == "__main__":

    # Load input matrix (ROI x ROI correlation or Gram matrix)
    mtx_path = os.path.join(
        BASE_ALL,
        "matrix_r_"
        + INDIVID_LEVEL
        + "_Both_bh_8-rois_withrestrand.tsv",
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
            f"{INDIVID_LEVEL}_ncomps-{N_COMPONENTS}_mds_pair_{a}_{b}.png",
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
                f"{a}_{b}_{c}.png",
            )
            plot_mds_3d(
                coords=coords,
                labels=df.index,
                explained_var=ev_ratio_full,
                out_path=out3d,
                comps=(a, b, c),
            )