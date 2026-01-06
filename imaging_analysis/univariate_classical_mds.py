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
from mpl_toolkits.mplot3d import proj3d


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
    """Nudge 2D text labels so their bboxes fit inside axes box."""
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

    ax.axhline(0, lw=0.5)
    ax.axvline(0, lw=0.5)
    ax.set_title(f"Classical MDS - 2D (MDS{c1 + 1} vs MDS{c2 + 1})")

    nudge_texts_inside_axes(fig, ax, texts, pad_px=8, max_iter=30)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _draw_custom_xticklabels_3d(
    fig,
    ax,
    fontsize=10,
    dx_px=0.0,
    dy_px=0.0,
    spread_x_px=12.0,
    spread_y_px=0.0,
):
    """
    Redraw x tick labels in 3D at projected tick positions, with 
    explicit screen-space (pixel) offsets and optional symmetric 
    spreading in x and y.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure containing the 3D axis.
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
        3D axis.
    fontsize : int
        Font size for tick labels.
    dx_px, dy_px : float
        Constant pixel shifts applied to all x tick labels 
        (right/up positive).
    spread_x_px, spread_y_px : float
        Additional symmetric "fan-out" applied based on tick value 
        relative to the center tick, in pixels. Positive values spread 
        labels away from the center tick in screen-space x and y, 
        respectively.
    """

    # Ensure final projection is available and tick Text artists exist.
    fig.canvas.draw()

    # Hide the default x tick labels (keep tick marks).
    for t in ax.get_xticklabels():
        t.set_visible(False)

    xticks = np.asarray(ax.get_xticks(), dtype=float)
    if xticks.size == 0:
        return

    # Center and half-range for normalized tick coordinate u in [-1, 1].
    x_center = 0.5 * (float(xticks.min()) + float(xticks.max()))
    x_half_range = 0.5 * (float(xticks.max()) - float(xticks.min()))
    if x_half_range <= 0:
        x_half_range = 1.0

    # Anchor for x-axis tick locations in mplot3d: (ymin, zmin).
    y0 = float(ax.get_ylim()[0])
    z0 = float(ax.get_zlim()[0])

    inv_fig = fig.transFigure.inverted()
    formatter = ax.xaxis.get_major_formatter()

    for xv in xticks:
        # Project 3D tick anchor to 2D.
        x2, y2, _ = proj3d.proj_transform(float(xv), y0, z0, ax.get_proj())
        x_disp, y_disp = ax.transData.transform((x2, y2))

        # Normalized position (center tick -> 0), used for symmetric spreading.
        u = (float(xv) - x_center) / x_half_range

        # Screen-space offsets (pixels): base shift + fan-out.
        x_disp += float(dx_px) + float(spread_x_px) * u
        y_disp += float(dy_px) + float(spread_y_px) * u

        # Convert display coords to figure fraction and draw label.
        x_fig, y_fig = inv_fig.transform((x_disp, y_disp))
        txt = fig.text(
            x_fig,
            y_fig,
            formatter(xv, None),
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )
        # Prevent layout engines from shrinking the axes due to these artists.
        txt.set_in_layout(False)


def _draw_custom_xlabel_3d(
    fig,
    ax,
    text,
    fontsize=12,
    dx_px=0.0,
    dy_px=0.0,
    rotation=0.0,
):
    fig.canvas.draw()

    ax.set_xlabel("")

    x0, x1 = ax.get_xlim()
    y0 = float(ax.get_ylim()[0])
    z0 = float(ax.get_zlim()[0])
    xmid = 0.5 * (float(x0) + float(x1))

    x2, y2, _ = proj3d.proj_transform(xmid, y0, z0, ax.get_proj())
    x_disp, y_disp = ax.transData.transform((x2, y2))

    x_disp += float(dx_px)
    y_disp += float(dy_px)

    x_fig, y_fig = fig.transFigure.inverted().transform((x_disp, y_disp))

    txt = fig.text(
        x_fig,
        y_fig,
        text,
        ha="center",
        va="top",
        fontsize=fontsize,
        rotation=float(rotation),
        rotation_mode="anchor",
    )
    txt.set_in_layout(False)


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
        color="mediumblue",
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
            ox = -1.6 * dx
            oy = 3.0 * dy
            oz = 0.4 * dz
            ha = "right"
        elif name == "PreSMA":
            oy = 1.3 * dy
            oz = -0.5 * dz
            ha = "right"
        elif name == "Dorsal Striatum":
            oy = 10.0 * dy
            oz = 4.5 * dz
        elif name == "Occipital\nLobe":
            oy = 0. * dy
            oz = 6.0 * dz
        elif name == "Cerebellum":
            oy = 16.0 * dy
            oz = 1.5 * dz
        elif name == "PMV":
            ox = 0.7 * dx
            oy = 6.7 * dy
            oz = 1.0 * dz
        elif name == "PMD":
            ox = 0.9 * dx
            oy = 7.8 * dy
        elif name == "SMA":
            oy = -1.3 * dy
            oz = 0.1 * dz

        if name == "Occipital\nLobe":
            ax.text(
                x + ox,
                y + oy,
                z + oz,
                name,
                ha="center",
                va="center",
                multialignment="center",
                clip_on=True,
            )
        else:
            ax.text(
                x + ox,
                y + oy,
                z + oz,
                name,
                ha=ha,
                va=va,
                clip_on=True,
            )

    xlabel_text = f"MDS{c1 + 1} ({var[c1]:.1%})"
    ax.set_xlabel("")  # hide default 3D xlabel (we redraw it below)

    ax.set_ylabel(f"MDS{c2 + 1} ({var[c2]:.1%})", labelpad=-1.)
    ax.set_zlabel(f"MDS{c3 + 1} ({var[c3]:.1%})", labelpad=1.)
    # ax.set_title(f"Classical MDS - 3D (MDS{c1 + 1}, {c2 + 1}, {c3 + 1})")

    ax.view_init(elev=15, azim=-10)

    # Fixed axis limits (explicit, not data-driven)
    ax.set_xlim(-0.35, 0.0)      # MDS1
    ax.set_ylim(0.35, -0.35)     # MDS2
    ax.set_zlim(-0.30, 0.30)     # MDS3

    # Make panes transparent (we will draw our own black box edges).
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

    # Draw vertical stems with an opacity gradient, without "dashed" artifacts.
    z_bottom = float(ax.get_zlim()[0])
    n_segments = 640  # dense enough to look continuous in raster output

    for x, y, z_top in zip(coords[:, c1], coords[:, c2], coords[:, c3]):
        zs = np.linspace(z_bottom, z_top, n_segments + 1)

        for i in range(n_segments):
            z0s = zs[i]
            z1s = zs[i + 1]
            alpha = 1.0 - (i / n_segments)

            ax.plot(
                [x, x],
                [y, y],
                [z0s, z1s],
                color="mediumblue",
                linewidth=1.5,
                alpha=alpha,
                linestyle="-",
                solid_capstyle="butt",
                solid_joinstyle="miter",
                antialiased=True,
                zorder=2,
            )

    # Tick label styling (keep your y/z behavior as-is).
    ax.tick_params(axis="x", labelrotation=0.0, labelsize=10.0, pad=0.)
    ax.tick_params(axis="y", labelrotation=5.0, labelsize=10.0, pad=-4.)
    ax.tick_params(axis="z", labelrotation=0.0, labelsize=10.0, pad=1.)

    # ---- X-axis-only fix: redraw x tick labels next to their ticks ----
    # Tune this if needed: typical working range is 6–12 px.
    X_TICKLABEL_DX_PX = -87.
    X_TICKLABEL_DY_PX = -48.
    _draw_custom_xticklabels_3d(
            fig=fig,
            ax=ax,
            fontsize=10,
            dy_px=X_TICKLABEL_DY_PX,
            dx_px=X_TICKLABEL_DX_PX,
            spread_x_px=5.,
            spread_y_px=-5.,
        )

    # ---- X-axis-only fix: redraw x-axis title with pixel offsets ----
    X_LABEL_DX_PX = -126.     # horizontal shift: +right / -left
    X_LABEL_DY_PX = -50.   # vertical shift: +up / -down (tune)

    _draw_custom_xlabel_3d(
        fig=fig,
        ax=ax,
        text=xlabel_text,
        fontsize=10,
        dx_px=X_LABEL_DX_PX,
        dy_px=X_LABEL_DY_PX,
        rotation=-73.
    )

    # Draw a black 3D bounding box (selected edges).
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    z0, z1 = ax.get_zlim()

    corners = [
        (x0, y0, z0),
        (x0, y0, z1),
        (x0, y1, z0),
        (x0, y1, z1),
        (x1, y0, z0),
        (x1, y0, z1),
        (x1, y1, z0),
        (x1, y1, z1),
    ]

    edges = [
        # Bottom rectangle (z = z0)
        (0, 2), (2, 6), (6, 4), (4, 0),

        # Vertical back edges
        (0, 1), (2, 3), (3, 7),

        # Vertical front edge (keep one depth cue)
        (6, 7),

        # Back top edge
        (1, 3),
    ]

    for i, j in edges:
        xi, yi, zi = corners[i]
        xj, yj, zj = corners[j]
        ax.plot(
            [xi, xj],
            [yi, yj],
            [zi, zj],
            color="black",
            linewidth=1.0,
            alpha=1.0,
        )

    # --- Remove tick marks (3D "spikes") but keep tick labels + grid ---
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["tick"]["inward_factor"] = 0.0
        axis._axinfo["tick"]["outward_factor"] = 0.0

    # Keep grid visible (uses locator positions, not tick mark length)
    ax.grid(True)

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
    "occipital": "Occipital\nLobe",
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