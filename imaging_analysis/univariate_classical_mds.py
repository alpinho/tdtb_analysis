#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multidimensional scaling (MDS) plots for ROI similarity matrices.

Author: Ana Luisa Pinho
email: agrilopi@uwo.ca

Created: 30th of October 2025
Last Update: April 2026

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


def get_spec(specs, key, default=None):
    """Return a spec value with a default fallback."""
    return specs.get(key, default)


def get_panel_spec(specs, group, panel, default=None):
    """Return a panel-specific spec with default fallback."""
    block = specs.get(group, {})
    return block.get(panel, block.get("default", default))


def mds_axis_label(dim_idx, var, show_variance):
    """Return MDS axis label with optional explained variance."""
    if show_variance:
        return f"MDS{dim_idx + 1} ({var[dim_idx]:.1%})"
    return f"MDS{dim_idx + 1}"


def plot_mds_2d(
    coords,
    labels,
    explained_var,
    out_path,
    comps=(1, 2),
    use_custom=False,
    specs=None,
):
    """
    Plot 2D classical MDS.

    If use_custom is True, apply panel-specific styling. If a spec is
    missing, internally generated defaults are used.
    """
    if specs is None:
        specs = {}

    swap_axes = get_spec(specs, "swap_axes", {})
    comps_plot = swap_axes.get(tuple(comps), tuple(comps))
    c1 = comps_plot[0] - 1
    c2 = comps_plot[1] - 1

    var = np.clip(explained_var, 0, None)
    denom = var.sum() if var.sum() > 0 else 1.0
    var = var / denom

    show_variance = get_spec(specs, "show_variance", True)
    xlabel = mds_axis_label(c1, var, show_variance)
    ylabel = mds_axis_label(c2, var, show_variance)

    labels_disp = [ROI_LABELS.get(str(lab), str(lab)) for lab in labels]

    point_color = get_spec(specs, "point_color", "mediumblue")
    point_alpha = get_spec(specs, "point_alpha", 0.8)
    zero_color = get_spec(specs, "zero_line_color", "mediumblue")

    if not use_custom:
        fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=150)
        ax.scatter(
            coords[:, c1],
            coords[:, c2],
            color=point_color,
            alpha=point_alpha,
        )

        for x_val, y_val, name in zip(
            coords[:, c1],
            coords[:, c2],
            labels_disp,
        ):
            ax.annotate(
                name,
                (float(x_val), float(y_val)),
                textcoords="offset points",
                xytext=(3, 3),
                ha="left",
                va="center",
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.axhline(0, lw=0.9, color=zero_color, zorder=1)
        ax.axvline(0, lw=0.9, color=zero_color, zorder=1)

        x_min = float(coords[:, c1].min())
        x_max = float(coords[:, c1].max())
        y_min = float(coords[:, c2].min())
        y_max = float(coords[:, c2].max())
        x_rng = x_max - x_min
        y_rng = y_max - y_min
        x_pad = 0.05 * x_rng if x_rng > 0 else 0.01
        y_pad = 0.05 * y_rng if y_rng > 0 else 0.01
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        return

    axis_limits = get_spec(specs, "axis_limits", {})
    xlim = axis_limits.get(
        comps_plot[0],
        (float(coords[:, c1].min()), float(coords[:, c1].max())),
    )
    ylim = axis_limits.get(
        comps_plot[1],
        (float(coords[:, c2].min()), float(coords[:, c2].max())),
    )

    fig_ref = get_spec(specs, "figsize_reference", {})
    ref_figsize = fig_ref.get("figsize", (6.0, 5.0))
    ref_x_range = fig_ref.get("x_range", 0.70)
    ref_y_range = fig_ref.get("y_range", 0.60)
    pad = fig_ref.get("pad", 1.08)

    x_range = float(abs(xlim[1] - xlim[0]))
    y_range = float(abs(ylim[1] - ylim[0]))

    k = min(ref_figsize[0] / ref_x_range, ref_figsize[1] / ref_y_range)
    figsize = (k * x_range * pad, k * y_range * pad)

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    ax.scatter(
        coords[:, c1],
        coords[:, c2],
        color=point_color,
        alpha=point_alpha,
    )

    remove_linebreak = get_spec(specs, "remove_linebreak_labels", {})
    labels_to_clean = remove_linebreak.get(comps_plot, [])
    labels_disp = [
        name.replace("\n", " ")
        if name in labels_to_clean
        else name
        for name in labels_disp
    ]

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

    label_offsets = get_spec(specs, "label_offsets", {})
    panel_offsets = label_offsets.get(comps_plot, {})

    leader_lines = get_spec(specs, "leader_lines", {})
    panel_leaders = leader_lines.get(comps_plot, {})

    texts = []
    for idx, (x_val, y_val, name) in enumerate(
        zip(coords[:, c1], coords[:, c2], labels_disp)
    ):
        if name in panel_offsets:
            off_x, off_y = panel_offsets[name]
        else:
            off_x, off_y = offsets[idx % len(offsets)]

        x_txt = float(x_val) + float(off_x)
        y_txt = float(y_val) + float(off_y)

        if name.startswith("Occipital"):
            txt = ax.text(
                x_txt,
                y_txt,
                name,
                va="center",
                ha="center",
                multialignment="center",
                clip_on=True,
            )
        else:
            txt = ax.text(
                x_txt,
                y_txt,
                name,
                va="center",
                ha="left",
                clip_on=True,
            )

        if name in panel_leaders:
            leader = panel_leaders[name]
            start_off = leader.get("start_offset", (0.0, 0.0))
            end_off = leader.get("end_offset", (0.0, 0.0))
            ax.plot(
                [float(x_val) + start_off[0], x_txt + end_off[0]],
                [float(y_val) + start_off[1], y_txt + end_off[1]],
                color=leader.get("color", "black"),
                linewidth=leader.get("linewidth", 0.8),
                zorder=leader.get("zorder", 5),
            )

        texts.append(txt)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")

    tick_step = get_spec(specs, "tick_step", 0.05)
    label_step = get_spec(specs, "label_step", 0.1)

    labeler = SparseTickLabeler(
        label_step=label_step,
        decimals=1,
        zero_str="0.0",
        atol_mult=1e-8,
        atol_zero=1e-12,
    )
    formatter = FuncFormatter(labeler)

    xt = build_fixed_ticks(xlim[0], xlim[1], tick_step)
    yt = build_fixed_ticks(ylim[0], ylim[1], tick_step)

    ax.xaxis.set_major_locator(FixedLocator(xt))
    ax.yaxis.set_major_locator(FixedLocator(yt))
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", 4))
    ax.spines["bottom"].set_position(("outward", 4))

    ax.grid(True, color="0.8", linewidth=0.8)
    ax.set_axisbelow(True)

    ax.axhline(0, lw=0.9, color=zero_color, zorder=1)
    ax.axvline(0, lw=0.9, color=zero_color, zorder=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    nudge_texts_inside_axes(fig, ax, texts, pad_px=8, max_iter=30)

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.08)
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
    Redraw x tick labels in 3D at projected tick positions.

    Offsets are in screen pixels after projection.
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
            rotation=0.0,
            rotation_mode="anchor",
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
    """Draw custom projected x-axis label for 3D plots."""
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


def _keep_interior(vals, vmin, vmax, atol=1e-12):
    """Return vals excluding points at vmin or vmax."""
    out = []
    for val in vals:
        if np.isclose(val, vmin, atol=atol):
            continue
        if np.isclose(val, vmax, atol=atol):
            continue
        out.append(val)
    return np.asarray(out, dtype=float)


def plot_mds_3d(
    coords,
    labels,
    explained_var,
    out_path,
    comps=(1, 2, 3),
    use_custom=False,
    specs=None,
):
    """
    Plot 3D MDS for chosen components.

    If use_custom is True, apply panel-specific styling. If a spec is
    missing, internally generated defaults are used.
    """
    if specs is None:
        specs = {}

    c1, c2, c3 = comps[0] - 1, comps[1] - 1, comps[2] - 1

    var = np.clip(explained_var, 0, None)
    var = var / (var.sum() if var.sum() > 0 else 1.0)

    show_variance = get_spec(specs, "show_variance", True)
    xlabel_text = mds_axis_label(c1, var, show_variance)
    ylabel = mds_axis_label(c2, var, show_variance)
    zlabel = mds_axis_label(c3, var, show_variance)

    labels_disp = [ROI_LABELS.get(str(lab), str(lab)) for lab in labels]

    figsize = get_panel_spec(specs, "figsize", tuple(comps), (6, 5))
    point_color = get_spec(specs, "point_color", "mediumblue")
    point_alpha = get_spec(specs, "point_alpha", 0.8)

    fig = plt.figure(figsize=figsize, dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        coords[:, c1],
        coords[:, c2],
        coords[:, c3],
        color=point_color,
        alpha=point_alpha,
    )

    if not use_custom:
        ax.set_xlabel(xlabel_text)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)

        def _lims(vals):
            vmin = float(vals.min())
            vmax = float(vals.max())
            rng = vmax - vmin
            pad = 0.05 * rng if rng > 0 else 0.01
            return vmin - pad, vmax + pad

        ax.set_xlim(*_lims(coords[:, c1]))
        ax.set_ylim(*_lims(coords[:, c2]))
        ax.set_zlim(*_lims(coords[:, c3]))

        for x_val, y_val, z_val, name in zip(
            coords[:, c1],
            coords[:, c2],
            coords[:, c3],
            labels_disp,
        ):
            ax.text(
                float(x_val),
                float(y_val),
                float(z_val),
                name,
            )

        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        return

    def _draw_minor_grid_planes(
        ax,
        xlim,
        ylim,
        zlim,
        step=0.05,
        color="0.7",
        lw=0.8,
    ):
        """Draw grid on the visible 3D planes."""
        x0, x1 = xlim
        y0, y1 = ylim
        z0, z1 = zlim

        xs = build_fixed_ticks(x0, x1, step)
        ys = build_fixed_ticks(y0, y1, step)
        zs = build_fixed_ticks(z0, z1, step)

        xs_in = _keep_interior(xs, x0, x1)
        ys_in = _keep_interior(ys, y0, y1)
        zs_in = _keep_interior(zs, z0, z1)

        for xv in xs_in:
            ax.plot(
                [xv, xv],
                [y0, y1],
                [z0, z0],
                color=color,
                linewidth=lw,
                zorder=0,
            )
        for yv in ys_in:
            ax.plot(
                [x0, x1],
                [yv, yv],
                [z0, z0],
                color=color,
                linewidth=lw,
                zorder=0,
            )

        for yv in ys_in:
            ax.plot(
                [x0, x0],
                [yv, yv],
                [z0, z1],
                color=color,
                linewidth=lw,
                zorder=0,
            )
        for zv in zs_in:
            ax.plot(
                [x0, x0],
                [y0, y1],
                [zv, zv],
                color=color,
                linewidth=lw,
                zorder=0,
            )

        for xv in xs_in:
            ax.plot(
                [xv, xv],
                [y0, y0],
                [z0, z1],
                color=color,
                linewidth=lw,
                zorder=0,
            )
        for zv in zs_in:
            ax.plot(
                [x0, x1],
                [y0, y0],
                [zv, zv],
                color=color,
                linewidth=lw,
                zorder=0,
            )

    ax.set_xlabel("")
    ax.set_ylabel(ylabel, labelpad=-1.0)
    ax.set_zlabel(
        zlabel,
        labelpad=1.0,
        rotation=90.0,
    )
    ax.zaxis.set_rotate_label(False)

    view = get_panel_spec(specs, "view", tuple(comps), {})
    ax.view_init(
        elev=view.get("elev", 15),
        azim=view.get("azim", 45),
    )

    axis_limits = get_spec(specs, "axis_limits", {})

    def _lims_for_dim(dim_num, vals):
        if dim_num in axis_limits:
            return axis_limits[dim_num]

        vmin = float(vals.min())
        vmax = float(vals.max())
        vrng = vmax - vmin
        pad = 0.05 * vrng if vrng > 0 else 0.01
        return vmin - pad, vmax + pad

    ax.set_xlim(*_lims_for_dim(comps[0], coords[:, c1]))
    ax.set_ylim(*_lims_for_dim(comps[1], coords[:, c2]))
    ax.set_zlim(*_lims_for_dim(comps[2], coords[:, c3]))

    ax.xaxis.pane.set_alpha(0.0)
    ax.yaxis.pane.set_alpha(0.0)
    ax.zaxis.pane.set_alpha(0.0)

    tick_step = get_spec(specs, "tick_step", 0.05)
    label_step = get_spec(specs, "label_step", 0.1)

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

    eps = 1e-12
    x_start = np.ceil((x0 - eps) / label_step) * label_step
    x_stop = np.floor((x1 + eps) / label_step) * label_step
    xt_major = np.arange(x_start, x_stop + eps, label_step)

    y_start = np.ceil((y0 - eps) / label_step) * label_step
    y_stop = np.floor((y1 + eps) / label_step) * label_step
    yt_major = np.arange(y_start, y_stop + eps, label_step)

    z_start = np.ceil((z0 - eps) / label_step) * label_step
    z_stop = np.floor((z1 + eps) / label_step) * label_step
    zt_major = np.arange(z_start, z_stop + eps, label_step)

    ax.xaxis.set_major_locator(FixedLocator(xt_major))
    ax.yaxis.set_major_locator(FixedLocator(yt_major))
    ax.zaxis.set_major_locator(FixedLocator(zt_major))

    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.zaxis.set_major_formatter(formatter)

    nx = abs(x1 - x0) / tick_step if tick_step > 0 else 1.0
    ny = abs(y1 - y0) / tick_step if tick_step > 0 else 1.0
    nz = abs(z1 - z0) / tick_step if tick_step > 0 else 1.0
    ax.set_box_aspect((nx, ny, nz))

    stem_specs = get_spec(specs, "stems", {})
    z_bottom = float(ax.get_zlim()[0])
    n_segments = int(stem_specs.get("n_segments", 640))
    stem_lw = stem_specs.get("linewidth", 1.5)

    for x_val, y_val, z_top in zip(
        coords[:, c1],
        coords[:, c2],
        coords[:, c3],
    ):
        zs = np.linspace(z_bottom, z_top, n_segments + 1)
        for idx in range(n_segments):
            z0s = zs[idx]
            z1s = zs[idx + 1]
            alpha = 1.0 - (idx / n_segments)
            ax.plot(
                [x_val, x_val],
                [y_val, y_val],
                [z0s, z1s],
                color=point_color,
                linewidth=stem_lw,
                alpha=alpha,
                linestyle="-",
                solid_capstyle="butt",
                solid_joinstyle="miter",
                antialiased=True,
                zorder=2,
            )

    ax.tick_params(axis="x", labelrotation=0.0, labelsize=10.0, pad=0.0)
    ax.tick_params(axis="y", labelrotation=5.0, labelsize=10.0, pad=-4.0)
    ax.tick_params(axis="z", labelrotation=0.0, labelsize=10.0, pad=1.0)

    xtick_specs = get_panel_spec(
        specs,
        "x_ticklabel_offsets",
        tuple(comps),
        {},
    )
    _draw_custom_xticklabels_3d(
        fig=fig,
        ax=ax,
        fontsize=10,
        dx_px=xtick_specs.get("dx_px", 0.0),
        dy_px=xtick_specs.get("dy_px", 0.0),
        spread_x_px=xtick_specs.get("spread_x_px", 12.0),
        spread_y_px=xtick_specs.get("spread_y_px", 0.0),
    )

    xlabel_specs = get_panel_spec(
        specs,
        "xlabel_offsets",
        tuple(comps),
        {},
    )
    _draw_custom_xlabel_3d(
        fig=fig,
        ax=ax,
        text=xlabel_text,
        fontsize=10,
        dx_px=xlabel_specs.get("dx_px", 0.0),
        dy_px=xlabel_specs.get("dy_px", 0.0),
        rotation=xlabel_specs.get("rotation", 0.0),
    )

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
        (0, 2),
        (2, 6),
        (6, 4),
        (4, 0),
        (0, 1),
        (2, 3),
        (1, 3),
        (1, 5),
        (5, 4),
    ]

    for idx_i, idx_j in edges:
        xi, yi, zi = corners[idx_i]
        xj, yj, zj = corners[idx_j]
        ax.plot(
            [xi, xj],
            [yi, yj],
            [zi, zj],
            color="black",
            linewidth=1.0,
            zorder=10,
        )

    def _draw_custom_major_ticks(ax, xlim, ylim, zlim, xt, yt, zt):
        """Draw custom major ticks on visible 3D spines."""
        x0, x1 = xlim
        y0, y1 = ylim
        z0, _z1 = zlim

        x_rng = float(abs(x1 - x0))
        y_rng = float(abs(y1 - y0))

        len_x = 0.010 * y_rng
        len_y = 0.040 * x_rng
        len_z = 0.040 * x_rng

        for xv in xt:
            ax.plot(
                [xv, xv],
                [y1 - len_x, y1 + len_x],
                [z0, z0],
                color="black",
                linewidth=1.0,
                zorder=30,
            )

        for yv in yt:
            ax.plot(
                [x1 - len_y, x1 + len_y],
                [yv, yv],
                [z0, z0],
                color="black",
                linewidth=1.0,
                zorder=30,
            )

        for zv in zt:
            ax.plot(
                [x1 - len_z, x1 + len_z],
                [y0, y0],
                [zv, zv],
                color="black",
                linewidth=1.2,
                zorder=10_000,
            )

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["tick"]["inward_factor"] = 0.0
        axis._axinfo["tick"]["outward_factor"] = 0.0
        axis._axinfo["tick"]["linewidth"] = {False: 0.0, True: 0.0}

    ax.grid(False)
    _draw_minor_grid_planes(
        ax=ax,
        xlim=(x0, x1),
        ylim=(y0, y1),
        zlim=(z0, z1),
        step=tick_step,
        color="0.7",
        lw=0.8,
    )

    _draw_custom_major_ticks(
        ax=ax,
        xlim=(x0, x1),
        ylim=(y0, y1),
        zlim=(z0, z1),
        xt=xt_major,
        yt=yt_major,
        zt=zt_major,
    )

    fig.canvas.draw()
    inv_fig = fig.transFigure.inverted()
    label_fontsize = 10

    label_offsets_px = get_panel_spec(
        specs,
        "label_offsets_px",
        tuple(comps),
        {},
    )

    for x_val, y_val, z_val, name in zip(
        coords[:, c1],
        coords[:, c2],
        coords[:, c3],
        labels_disp,
    ):
        x2, y2, _ = proj3d.proj_transform(
            float(x_val),
            float(y_val),
            float(z_val),
            ax.get_proj(),
        )
        x_px, y_px = ax.transData.transform((x2, y2))

        dx_px, dy_px = label_offsets_px.get(name, (6, 6))
        x_px += float(dx_px)
        y_px += float(dy_px)

        x_fig, y_fig = inv_fig.transform((x_px, y_px))

        if name.startswith("Occipital"):
            txt = fig.text(
                x_fig,
                y_fig,
                name,
                ha="center",
                va="center",
                multialignment="center",
                fontsize=label_fontsize,
            )
        else:
            txt = fig.text(
                x_fig,
                y_fig,
                name,
                ha="left",
                va="center",
                fontsize=label_fontsize,
            )
        txt.set_in_layout(False)

    adjust = get_panel_spec(specs, "subplots_adjust", tuple(comps), {})
    fig.subplots_adjust(
        left=adjust.get("left", 0.03),
        right=adjust.get("right", 0.97),
        bottom=adjust.get("bottom", 0.02),
        top=adjust.get("top", 0.98),
    )

    fig.savefig(out_path)
    plt.close(fig)


# ============================= CONFIG ============================== #

N_COMPONENTS = 3
INDIVID_LEVEL = "i"

USE_CUSTOM_2D_PLOTTING = (N_COMPONENTS == 3)
USE_CUSTOM_3D_PLOTTING = (N_COMPONENTS in (3, 4))

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

# MDS dimensions to flip manually, using 1-based indexing.
# Example: [3] flips MDS3. Use [] for no flips.
# FLIP_MDS_DIMS = []
FLIP_MDS_DIMS = [3]

# ======================= GLOBAL PLOT CONTROLS ====================== #

# Color of ROI points in all 2D and 3D MDS plots.
MDS_POINT_COLOR = "mediumblue"

# Transparency of ROI points. 1.0 is opaque; lower values are lighter.
MDS_POINT_ALPHA = 0.8

# Color of zero reference lines in 2D plots.
ZERO_LINE_COLOR = "mediumblue"


# ========================== PLOT SPECS ============================= #

MDS_2D_SPECS = {
    # Show explained variance in axis labels, e.g. MDS1 (58.2%).
    "show_variance": False,

    # Shared appearance for 2D point markers.
    "point_color": MDS_POINT_COLOR,
    "point_alpha": MDS_POINT_ALPHA,

    # Color of x=0 and y=0 reference lines in 2D plots.
    "zero_line_color": ZERO_LINE_COLOR,

    # Tick spacing. All ticks are drawn every 0.05.
    "tick_step": 0.05,

    # Tick labels are shown only every 0.1.
    "label_step": 0.1,

    # Optional axis swapping per panel. Here MDS1 vs MDS2 is shown as
    # MDS2 on x and MDS1 on y.
    "swap_axes": {
        (1, 2): (2, 1),
    },

    # Fixed axis limits by MDS dimension. If a dimension is missing,
    # limits are computed from the data.
    "axis_limits": {
        1: (-0.35, 0.05),
        2: (-0.35, 0.35),
        3: (-0.30, 0.30),
    },

    # Figure scaling reference used to preserve pixels per data unit.
    "figsize_reference": {
        "figsize": (6.0, 5.0),
        "x_range": 0.70,
        "y_range": 0.60,
        "pad": 1.08,
    },

    # Labels whose line breaks should be removed for specific panels.
    "remove_linebreak_labels": {
        (1, 3): ["Occipital\nLobe"],
        (2, 3): ["Occipital\nLobe"],
    },

    # Manual label offsets in data units, separated by 2D panel.
    # If a label is missing, an automatic small offset is used.
    "label_offsets": {
        (2, 1): {
            "Dorsal Striatum": (-0.175, 0.000),
            "Cerebellum": (-0.09, 0.035),
            "PreSMA": (-0.0825, -0.01),
            "SMA": (-0.055, 0.0005),
            "PMD": (0.01, -0.12),
            "PMV": (0.0125, 0.001),
            "Heschl's Gyrus": (0.015, -0.0015),
            "Occipital\nLobe": (0.0, 0.035),
        },
        (1, 3): {
            "Dorsal Striatum": (0.015, 0.000),
            "Cerebellum": (0.009, 0.0075),
            "PreSMA": (0.012, -0.0025),
            "SMA": (0.012, 0.000),
            "PMD": (0.01, -0.001),
            "PMV": (0.0125, -0.001),
            "Heschl's Gyrus": (-0.085, -0.025),
            "Occipital Lobe": (0.085, 0.0),
        },
        (2, 3): {
            "Dorsal Striatum": (-0.1725, 0.000),
            "Cerebellum": (-0.126, 0.0085),
            "PreSMA": (-0.087, -0.006),
            "SMA": (-0.057, 0.0001),
            "PMD": (0.015, -0.001),
            "PMV": (0.015, -0.001),
            "Heschl's Gyrus": (0.015, -0.001),
            "Occipital Lobe": (0.0, 0.02),
        },
    },

    # Optional leader lines from ROI point to label, in data units.
    "leader_lines": {
        (2, 1): {
            "Cerebellum": {
                "start_offset": (-0.0025, 0.009),
                "end_offset": (0.075, -0.01),
                "color": "black",
                "linewidth": 0.8,
                "zorder": 5,
            },
        },
    },
}

MDS_3D_SPECS = {
    # Show explained variance in axis labels, e.g. MDS1 (58.2%).
    "show_variance": False,

    # Shared appearance for 3D point markers and vertical stems.
    "point_color": MDS_POINT_COLOR,
    "point_alpha": MDS_POINT_ALPHA,

    # Tick spacing. All ticks are drawn every 0.05.
    "tick_step": 0.05,

    # Tick labels are shown only every 0.1.
    "label_step": 0.1,

    # Figure size by 3D component triplet. If missing, use default.
    "figsize": {
        (2, 3, 4): (8.5, 6.5),
        "default": (6, 5),
    },

    # View angle by 3D component triplet.
    "view": {
        (1, 2, 3): {
            "elev": 15,
            "azim": 10,
        },
        (2, 3, 4): {
            "elev": 10,
            "azim": 50,
        },
        "default": {
            "elev": 15,
            "azim": 45,
        },
    },

    # Fixed axis limits by MDS dimension. If missing, use data limits.
    "axis_limits": {
        1: (-0.35, 0.0),
        2: (-0.35, 0.35),
        3: (-0.30, 0.30),
        4: (-0.15, 0.15),
    },

    # Pixel offsets for projected x tick labels in 3D.
    "x_ticklabel_offsets": {
        (1, 2, 3): {
            "dx_px": 119.0,
            "dy_px": -78.0,
            "spread_x_px": -3.75,
            "spread_y_px": -15.0,
        },
        "default": {
            "dx_px": 0.0,
            "dy_px": 0.0,
            "spread_x_px": 12.0,
            "spread_y_px": 0.0,
        },
    },

    # Pixel offsets and rotation for projected x-axis label in 3D.
    "xlabel_offsets": {
        (1, 2, 3): {
            "dx_px": 158.0,
            "dy_px": -78.0,
            "rotation": 73.0,
        },
        "default": {
            "dx_px": 0.0,
            "dy_px": 0.0,
            "rotation": 0.0,
        },
    },

    # Manual projected label offsets in pixels, by 3D triplet.
    # If a label is missing, default offset (6, 6) is used.
    "label_offsets_px": {
        (1, 2, 3): {
            "Dorsal Striatum": (-98, 130),
            "Cerebellum": (-8, 44),
            "PreSMA": (-4, 32),
            "SMA": (-90, 38),
            "PMD": (7, 8),
            "PMV": (8, 18),
            "Heschl's Gyrus": (-85, 4),
            "Occipital\nLobe": (58, 40),
            "Occipital Lobe": (8, 6),
        },
        "default": {},
    },

    # Subplot margins by triplet. Useful when projected labels are cut.
    "subplots_adjust": {
        (1, 2, 3): {
            "left": 0.0,
            "right": 0.97,
            "bottom": -0.05,
            "top": 1.10,
        },
        (2, 3, 4): {
            "left": 0.03,
            "right": 0.98,
            "bottom": 0.02,
            "top": 0.98,
        },
        "default": {
            "left": 0.03,
            "right": 0.97,
            "bottom": 0.02,
            "top": 0.98,
        },
    },

    # Vertical stems from z-axis floor to each ROI point.
    "stems": {
        "n_segments": 640,
        "linewidth": 1.5,
    },
}


# =============================== PATHS ============================= #

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL = "rwls"
MASKING = "wb"
HRF = "hrf128"

# Use "encoding_restrand" for within-subject rm corr, and ...
# ... "subjectcorr_paired_restrand" for subject-wise corr.
# corrtype_folder = "encoding_restrand"
corrtype_folder = "subjectcorr_paired_restrand"

BASE_ALL = os.path.join(
    WORKING_DIR,
    f"roi_analyses_{MODEL}_{HRF}_{MASKING}_puncorr_unsmoothed",
    "bothmod_allmain_tasks",
    "profile_similarity",
    corrtype_folder,
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

    if corrtype_folder == "encoding_restrand":
        mtx_path = os.path.join(
            BASE_ALL,
            "matrix_r_"
            + INDIVID_LEVEL
            + "_Both_bh_8-rois_withrestrand.tsv",
        )
    else:
        assert corrtype_folder == "subjectcorr_paired_restrand"
        mtx_path = os.path.join(
            BASE_ALL,
            "matrix_mean_r_"
            + INDIVID_LEVEL
            + "_Both_bh_8-rois_withrestrand.tsv",
        )

    df = pd.read_csv(mtx_path, sep="\t", index_col=0)
    mtx = df.to_numpy(dtype=float)

    rank = np.linalg.matrix_rank(mtx)
    mtx = mtx / rank

    scores, eigval = pcm.util.classical_mds(mtx)

    for dim in FLIP_MDS_DIMS:
        idx = int(dim) - 1
        if idx < 0 or idx >= scores.shape[1]:
            raise ValueError(f"Invalid MDS dimension to flip: {dim}")
        scores[:, idx] *= -1

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
            use_custom=USE_CUSTOM_2D_PLOTTING,
            specs=MDS_2D_SPECS,
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
                use_custom=USE_CUSTOM_3D_PLOTTING,
                specs=MDS_3D_SPECS,
            )