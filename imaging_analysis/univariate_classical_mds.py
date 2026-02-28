#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multidimensional scaling (MDS) plots for ROI similarity matrices

Author: Ana Luisa Pinho
email: agrilopi@uwo.ca

Created: 30th of October 2025
Last Update: February 2026

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

    # For the MDS1 vs MDS2 panel, reverse the view: MDS2 on x, MDS1 on y.
    if tuple(comps) == (1, 2):
        comps_plot = (2, 1)
    else:
        comps_plot = tuple(comps)

    c1 = comps_plot[0] - 1
    c2 = comps_plot[1] - 1

    var = np.clip(explained_var, 0, None)
    denom = var.sum() if var.sum() > 0 else 1.0
    var = var / denom

    axis_limits = {
        0: (-0.35, 0.05),   # MDS1
        1: (-0.35, 0.35),   # MDS2
        2: (-0.30, 0.30),   # MDS3
    }

    xlim = axis_limits.get(c1, (float(coords[:, c1].min()),
                                float(coords[:, c1].max())))
    ylim = axis_limits.get(c2, (float(coords[:, c2].min()),
                                float(coords[:, c2].max())))

    x_range = float(abs(xlim[1] - xlim[0]))
    y_range = float(abs(ylim[1] - ylim[0]))

    ref_figsize = (6.0, 5.0)
    ref_x_range = 0.70
    ref_y_range = 0.60

    k = min(ref_figsize[0] / ref_x_range, ref_figsize[1] / ref_y_range)
    pad = 1.08

    w = k * x_range * pad
    h = k * y_range * pad

    # Keep the same pixels-per-unit scale across all 2D panels by not
    # clamping the figure size (clamps change pixels per data unit).
    figsize = (w, h)

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    ax.scatter(coords[:, c1], coords[:, c2], color="mediumblue", alpha=0.8)

    labels_disp = [ROI_LABELS.get(str(lab), str(lab)) for lab in labels]

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
        zip(coords[:, c1], coords[:, c2], labels_disp)
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
    # Fixed axis limits to match the 3D plot.
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.set_aspect("equal", adjustable="box")

    # Ticks/grid: 0.05 tick spacing, label every 0.1 (as in 3D).
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

    xt = build_fixed_ticks(x0, x1, tick_step)
    yt = build_fixed_ticks(y0, y1, tick_step)

    ax.xaxis.set_major_locator(FixedLocator(xt))
    ax.yaxis.set_major_locator(FixedLocator(yt))
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    # Remove top/right spines and separate spines from the grid.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", 4))
    ax.spines["bottom"].set_position(("outward", 4))

    ax.grid(True, color="0.8", linewidth=0.8)
    ax.set_axisbelow(True)

    # Highlight zero axes in blue.
    ax.axhline(0, lw=0.9, color="mediumblue", zorder=1)
    ax.axvline(0, lw=0.9, color="mediumblue", zorder=1)

    xlabel = f"MDS{c1 + 1} ({var[c1]:.1%})"
    ylabel = f"MDS{c2 + 1} ({var[c2]:.1%})"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    nudge_texts_inside_axes(fig, ax, texts, pad_px=8, max_iter=30)

    # Lock margins so the axes box is identical 
    # (pixel-wise) across panels.
    fig.subplots_adjust(left=0.20, right=0.96, bottom=0.12, top=0.90)

    fig.savefig(out_path)
    plt.close(fig)

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
    """
    fig.canvas.draw()

    for t in ax.get_xticklabels():
        t.set_visible(False)

    xticks = np.asarray(ax.get_xticks(), dtype=float)
    if xticks.size == 0:
        return

    x_center = 0.5 * (float(xticks.min()) + float(xticks.max()))
    x_half_range = 0.5 * (float(xticks.max()) - float(xticks.min()))
    if x_half_range <= 0:
        x_half_range = 1.0

    y0 = float(ax.get_ylim()[1])
    z0 = float(ax.get_zlim()[0])

    inv_fig = fig.transFigure.inverted()
    formatter = ax.xaxis.get_major_formatter()

    for xv in xticks:
        x2, y2, _ = proj3d.proj_transform(float(xv), y0, z0, ax.get_proj())
        x_disp, y_disp = ax.transData.transform((x2, y2))

        u = (float(xv) - x_center) / x_half_range

        x_disp += float(dx_px) + float(spread_x_px) * u
        y_disp += float(dy_px) + float(spread_y_px) * u

        x_fig, y_fig = inv_fig.transform((x_disp, y_disp))
        txt = fig.text(
            x_fig,
            y_fig,
            formatter(xv, None),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            rotation=0.,             # ← rotate a little
            rotation_mode="anchor",  # important for correct pivot
        )
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
    y0 = float(ax.get_ylim()[1])
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

    labels_disp = [ROI_LABELS.get(str(lab), str(lab)) for lab in labels]

    xlabel_text = f"MDS{c1 + 1} ({var[c1]:.1%})"
    ax.set_xlabel("")
    ax.set_ylabel(f"MDS{c2 + 1} ({var[c2]:.1%})", labelpad=-1.0)
    ax.set_zlabel(f"MDS{c3 + 1} ({var[c3]:.1%})", labelpad=1.0, 
                  rotation=90.0,)
    ax.zaxis.set_rotate_label(False)

    # >>> UPDATED (as requested)
    ax.view_init(elev=15, azim=10)

    # Fixed axis limits (explicit, not data-driven)
    ax.set_xlim(-0.35, 0.0)      # MDS1
    ax.set_ylim(-0.35, 0.35)     # MDS2
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
    nx = abs(x1 - x0) / tick_step if tick_step > 0 else 1.0
    ny = abs(y1 - y0) / tick_step if tick_step > 0 else 1.0
    nz = abs(z1 - z0) / tick_step if tick_step > 0 else 1.0
    ax.set_box_aspect((nx, ny, nz))

    # Draw vertical stems with an opacity gradient, without "dashed" artifacts.
    z_bottom = float(ax.get_zlim()[0])
    n_segments = 640

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
    ax.tick_params(axis="x", labelrotation=0.0, labelsize=10.0, pad=0.0)
    ax.tick_params(axis="y", labelrotation=5.0, labelsize=10.0, pad=-4.0)
    ax.tick_params(axis="z", labelrotation=0.0, labelsize=10.0, pad=1.0)

    # ---- X-axis-only fix: redraw x tick labels next to their ticks ----
    # >>> UPDATED offsets for azim=10 (as requested)
    X_TICKLABEL_DX_PX = 62.0
    X_TICKLABEL_DY_PX = -50.
    _draw_custom_xticklabels_3d(
        fig=fig,
        ax=ax,
        fontsize=10,
        dy_px=X_TICKLABEL_DY_PX,
        dx_px=X_TICKLABEL_DX_PX,
        spread_x_px=-2.0,
        spread_y_px=-5.0,
    )

    # ---- X-axis-only fix: redraw x-axis title with pixel offsets ----
    # >>> UPDATED offsets for azim=10 (as requested)
    X_LABEL_DX_PX = 102.0
    X_LABEL_DY_PX = -45.0

    _draw_custom_xlabel_3d(
        fig=fig,
        ax=ax,
        text=xlabel_text,
        fontsize=10,
        dx_px=X_LABEL_DX_PX,
        dy_px=X_LABEL_DY_PX,
        rotation=73.0,
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
        (0, 2), (2, 6), (6, 4), (4, 0),
        (0, 1), (2, 3),
        # (3, 7),  # removed: top edge on right MDS1–MDS3 plane (y=y1)
        # (6, 7),  # removed: closer vertical edge on right MDS1–MDS3 plane (y=y1)
        (1, 3),
        (1, 5),    # added: top edge on left MDS1–MDS3 plane (y=y0)
    ]

    for i, j in edges:
        xi, yi, zi = corners[i]
        xj, yj, zj = corners[j]
        ax.plot([xi, xj], [yi, yj], [zi, zj], color="black", linewidth=1.0)

    # --- Remove tick marks (3D "spikes") but keep tick labels + grid ---
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["tick"]["inward_factor"] = 0.0
        axis._axinfo["tick"]["outward_factor"] = 0.0

    ax.grid(True)

    # =================== FIXED 3D LABEL POSITIONS (2D projected) =================== #
    fig.canvas.draw()
    inv_fig = fig.transFigure.inverted()

    # >>> slightly smaller labels (as requested)
    LABEL_FONTSIZE = 10

    # Per-label pixel offsets (tune once, then stable)
    LABEL_OFFSETS_PX = {
        "Heschl's Gyrus": (-27, -5),
        "Dorsal Striatum": (-80, 60),
        "Cerebellum": (-17, 33),
        "Occipital\nLobe": (25, 30),
        "Occipital Lobe": (8, 6),
        "PMD": (7, -5),
        "PMV": (9, 1),
        "PreSMA": (4, 6),
        "SMA": (-72, 10),
    }

    for x, y, z, name in zip(
        coords[:, c1], coords[:, c2], coords[:, c3], labels_disp
    ):
        x2, y2, _ = proj3d.proj_transform(float(x), float(y), float(z), ax.get_proj())
        x_px, y_px = ax.transData.transform((x2, y2))

        dx_px, dy_px = LABEL_OFFSETS_PX.get(name, (6, 6))
        x_px += float(dx_px)
        y_px += float(dy_px)

        x_fig, y_fig = inv_fig.transform((x_px, y_px))

        if name.startswith("Occipital"):
            t = fig.text(
                x_fig,
                y_fig,
                name,
                ha="center",
                va="center",
                multialignment="center",
                fontsize=LABEL_FONTSIZE,
            )
        else:
            t = fig.text(
                x_fig,
                y_fig,
                name,
                ha="left",
                va="center",
                fontsize=LABEL_FONTSIZE,
            )
        t.set_in_layout(False)
    # =============================================================================== #

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ============================= CONFIG ============================== #

N_COMPONENTS = 3
INDIVID_LEVEL = "i"

ROI_LABELS = {
    "dstr": "Dorsal Striatum",
    "cereb": "Cerebellum",
    "pmv": "PMV",
    "pmd": "PMD",
    "presma": "PreSMA",
    "sma": "SMA",
    "heschl": "Heschl's Gyrus",
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

    mtx_path = os.path.join(
        BASE_ALL,
        "matrix_r_"
        + INDIVID_LEVEL
        + "_Both_bh_8-rois_withrestrand.tsv",
    )
    df = pd.read_csv(mtx_path, sep="\t", index_col=0)
    mtx = df.to_numpy(dtype=float)

    rank = np.linalg.matrix_rank(mtx)
    mtx = mtx / rank

    scores, eigval = pcm.util.classical_mds(mtx)

    coords = scores[:, :N_COMPONENTS]

    ev_pos = np.clip(eigval, 0, None)
    denom = ev_pos.sum() if ev_pos.sum() > 0 else 1.0
    ev_ratio_full = ev_pos[:N_COMPONENTS] / denom

    os.makedirs(MDS_OUTPUT_DIR, exist_ok=True)

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