#!/usr/bin/env python3
"""PSC boxplots by ROI for rand_ntfd_nonrandom.

Author: Ana Luisa Pinho
email: agrilopi@uwo.ca

Created: 15th of April 2026
Last update: April 2026

Compatibility: Python 3.10.14
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import numpy as np
import pandas as pd


# ============================ CONSTANTS ============================ #

TASK_NAME = "NTFD Random"
CATEGORIES = ["Non-Random", "Random"]
MODALITY_BLOCKS = ["Pooled", "Auditory", "Visual"]
MODALITY_BLOCKS_SENSORY = ["Auditory", "Visual"]

MOD_LABEL = {
    "Pooled": "Both\nModalities",
    "Auditory": "Auditory",
    "Visual": "Visual",
}

MOD_LABEL_CENTERED = {
    "Pooled": "Both\nModalities",
    "Auditory": "\nAuditory",
    "Visual": "\nVisual",
}

ROI_ORDER = [
    "dstr",
    "cereb",
    "presma",
    "sma",
    "pmd",
    "pmv",
    "heschl",
    "occipital",
]

ROI_PRETTY = {
    "dstr": "Dorsal\nStriatum",
    "cereb": "Cerebellum",
    "presma": "PreSMA",
    "sma": "SMA",
    "pmd": "PMD",
    "pmv": "PMV",
    "heschl": "Heschl’s Gyrus",
    "occipital": "Occipital Lobe",
}

MODALITY_COLORS = {
    "Pooled": {
        "Non-Random": "#B07A5E",   # lighter brown
        "Random": "#E6CFC3",
    },
    "Auditory": {
        "Non-Random": "#7B8EDB",   # soft periwinkle (lighter than blue)
        "Random": "#C9D2F3",
    },
    "Visual": {
        "Non-Random": "#E7549E",   # lighter magenta
        "Random": "#F7C6DC",
    },
}

YTICK_STEP = 0.20
Y_FORMATTER = FormatStrFormatter("%.1f")
INCHES_PER_STEP = 0.68
MIN_ROW_HEIGHT = 2.0
PAIR_POS = [1.03, 1.13]
PAIR_XLIM = (0.955, 1.205)
FIG_W_SCALE = 0.617
BOX_WIDTH = 0.075

# Match plot_anovas_all.py more closely.
TITLE_FS = 14
LEGEND_FS = 12
ROI_TITLE_FS = 20
AXIS_LABEL_FS = 12
YTICK_FS = 12

# Current layout params for this figure.
LEFT_MARGIN = 0.05
RIGHT_MARGIN = 0.98
TOP_MARGIN = 0.992
BOTTOM_MARGIN = 0.04
HSPACE = 0.40
WSPACE = 0.24

# Reference layout params from plot_anovas_all.py for y-step scaling.
REF_TOP = 0.965
REF_BOTTOM = 0.11
REF_HSPACE = 0.75


# ============================ UTILITIES ============================ #


def pval_label_converter(pvalues: Sequence[float]) -> List[str]:
    """Convert p-values to star labels."""
    out: List[str] = []
    for p in pvalues:
        if p <= 0.0001:
            out.append("****")
        elif p <= 0.001:
            out.append("***")
        elif p <= 0.01:
            out.append("**")
        elif p <= 0.05:
            out.append("*")
        else:
            out.append("ns")
    return out


def _roi_key(name: str) -> str | None:
    """Return canonical ROI key for matching."""
    tok = str(name).strip().lower()
    tok = tok.replace("’", "'")
    for ch in (" ", "_", "-"):
        tok = tok.replace(ch, "")

    if "presma" in tok or "presupplementary" in tok:
        return "presma"
    if tok == "sma" or tok.endswith("sma"):
        return "sma"
    if "dstr" in tok or "dorsalstriatum" in tok:
        return "dstr"
    if "cereb" in tok or "cerebell" in tok:
        return "cereb"
    if "pmd" in tok:
        return "pmd"
    if "pmv" in tok:
        return "pmv"
    if "heschl" in tok:
        return "heschl"
    if "occip" in tok:
        return "occipital"
    return None


def _matches_roi(resolved_roi: str, ann_roi: str) -> bool:
    """Match annotation ROI to resolved ROI using canonical keys."""
    k1 = _roi_key(resolved_roi)
    k2 = _roi_key(ann_roi)
    return (k1 is not None) and (k1 == k2)


def _resolve_roi(name: str, roi_values: Sequence[str]) -> str | None:
    """Resolve canonical ROI name to a value present in dataframe."""
    key = name.lower()
    roi_map = {str(r).lower(): r for r in roi_values}
    if key in roi_map:
        return roi_map[key]
    for k, v in roi_map.items():
        if key in k:
            return v
    return None


def bootstrap_median_ci(
    vals: np.ndarray,
    n_boot: int = 5000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the median."""
    x = np.asarray(vals, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        med = float(np.median(x)) if n > 0 else np.nan
        return med, med

    if rng is None:
        rng = np.random.default_rng()

    idx = rng.integers(0, n, size=(n_boot, n))
    meds = np.median(x[idx], axis=1)
    lo = float(np.quantile(meds, alpha / 2.0))
    hi = float(np.quantile(meds, 1.0 - alpha / 2.0))
    return lo, hi


def bootstrap_conf_intervals(
    data: List[np.ndarray],
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 12345,
) -> np.ndarray:
    """Compute bootstrap median CI for each box."""
    rng = np.random.default_rng(seed)
    cis = []
    for vals in data:
        lo, hi = bootstrap_median_ci(
            vals=np.asarray(vals, dtype=float),
            n_boot=n_boot,
            alpha=alpha,
            rng=rng,
        )
        cis.append((lo, hi))
    return np.asarray(cis, dtype=float)


def _poly_xspan_at_y(patch, y: float) -> tuple[float, float] | None:
    """Return x-span of box polygon at y in data coordinates."""
    verts = patch.get_path().vertices
    if verts.shape[0] < 3:
        return None

    xs = []
    n = verts.shape[0]
    for i in range(n):
        x0, y0 = verts[i]
        x1, y1 = verts[(i + 1) % n]
        if (y0 <= y <= y1) or (y1 <= y <= y0):
            dy = y1 - y0
            if abs(dy) < 1e-12:
                if abs(y - y0) < 1e-12:
                    xs.extend([float(x0), float(x1)])
                continue
            t = (y - y0) / dy
            xs.append(float(x0 + t * (x1 - x0)))

    if len(xs) < 2:
        return None
    return min(xs), max(xs)


def _subject_table(df: pd.DataFrame, roi: str, modality: str) -> pd.DataFrame:
    """Return subject x category PSC table for one ROI and modality block."""
    sub = df.loc[
        (df["ROI"] == roi) & (df["Task"] == TASK_NAME),
        ["Subject", "Category", "Modality", "PSC"],
    ].copy()

    if modality != "Pooled":
        sub = sub[sub["Modality"] == modality]

    if modality == "Pooled":
        sub = (
            sub.groupby(["Subject", "Category"], as_index=False)["PSC"]
            .mean()
        )

    wide = sub.pivot_table(
        index="Subject",
        columns="Category",
        values="PSC",
        aggfunc="mean",
    )
    wide = wide.dropna(subset=CATEGORIES, how="any")
    return wide[CATEGORIES]


def _ydata_to_yfig(fig: plt.Figure, ax: plt.Axes, y_data: float) -> float:
    """Convert y in axis data coords to figure fraction coords."""
    x0 = float(np.mean(ax.get_xlim()))
    x_disp, y_disp = ax.transData.transform((x0, y_data))
    _, y_fig = fig.transFigure.inverted().transform((x_disp, y_disp))
    return float(y_fig)


def span_annotation_datay_figspan(
    fig: plt.Figure,
    ax_left: plt.Axes,
    ax_right: plt.Axes,
    text: str,
    y_data: float,
    h_data: float,
    lw: float = 1.2,
    fs: float = 14.0,
) -> None:
    """Draw a bracket spanning ax_left -> ax_right.

    X uses figure coords. Y anchors to ax_left data coords.
    """
    if not getattr(fig, "_ann_canvas_drawn", False):
        fig.canvas.draw()
        fig._ann_canvas_drawn = True

    b1 = ax_left.get_position()
    b2 = ax_right.get_position()
    x1 = b1.x0 + 0.5 * b1.width
    x2 = b2.x0 + 0.5 * b2.width

    y0 = _ydata_to_yfig(fig, ax_left, y_data)
    y1 = _ydata_to_yfig(fig, ax_left, y_data + h_data)

    fig.add_artist(
        Line2D([x1, x1], [y0, y1], transform=fig.transFigure, lw=lw, c="k")
    )
    fig.add_artist(
        Line2D([x1, x2], [y1, y1], transform=fig.transFigure, lw=lw, c="k")
    )
    fig.add_artist(
        Line2D([x2, x2], [y1, y0], transform=fig.transFigure, lw=lw, c="k")
    )
    fig.text(
        (x1 + x2) / 2.0,
        y1,
        text,
        ha="center",
        va="bottom",
        fontsize=fs,
        color="k",
    )


def within_axis_annotation(
    ax: plt.Axes,
    x1: float,
    x2: float,
    text: str,
    y_data: float,
    h_data: float,
    lw: float = 1.2,
    fs: float = 14.0,
    text_dy_frac: float = 0.0,
) -> None:
    """Draw a bracket within a single axis (data coords).

    Parameters
    ----------
    text_dy_frac : float
        Manual vertical adjustment for the text position, expressed
        as a fraction of the axis y-range. Positive moves text up,
        negative moves it down.
    """
    y0 = y_data
    y1 = y_data + h_data
    yspan = ax.get_ylim()[1] - ax.get_ylim()[0]
    text_y = y1 + text_dy_frac * yspan

    ax.plot([x1, x1], [y0, y1], lw=lw, c="k", clip_on=True)
    ax.plot([x1, x2], [y1, y1], lw=lw, c="k", clip_on=True)
    ax.plot([x2, x2], [y1, y0], lw=lw, c="k", clip_on=True)
    ax.text(
        (x1 + x2) / 2.0,
        text_y,
        text,
        ha="center",
        va="bottom",
        fontsize=fs,
        color="k",
        clip_on=False,
    )


def _draw_boxplot(
    ax: plt.Axes,
    data: List[np.ndarray],
    modality: str,
) -> None:
    """Draw boxplots with bootstrap notches and mean line."""
    conf_intervals = bootstrap_conf_intervals(data)

    bp = ax.boxplot(
        data,
        positions=PAIR_POS,
        widths=BOX_WIDTH,
        notch=True,
        patch_artist=True,
        showfliers=False,
        showmeans=False,
        medianprops={"linewidth": 0, "color": "none"},
        whis=1.5,
        conf_intervals=conf_intervals,
    )

    for patch, cat in zip(bp["boxes"], CATEGORIES):
        patch.set_facecolor(MODALITY_COLORS[modality][cat])
        patch.set_edgecolor("0.2")
        patch.set_linewidth(0.8)

    for key in ("whiskers", "caps"):
        for artist in bp[key]:
            artist.set_color("0.2")
            artist.set_linewidth(0.8)

    for patch, vals in zip(bp["boxes"], data):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        mean_val = float(np.mean(vals))
        span = _poly_xspan_at_y(patch, mean_val)
        if span is None:
            x_left = float(patch.get_path().vertices[:, 0].min())
            x_right = float(patch.get_path().vertices[:, 0].max())
        else:
            x_left, x_right = span
        ax.plot(
            [x_left, x_right],
            [mean_val, mean_val],
            color="0.2",
            lw=0.8,
            zorder=3,
            solid_capstyle="butt",
        )


# ============================ PLOTTING ============================= #


def plot_psc_boxplots(
    df: pd.DataFrame,
    outpath: str | Path,
    figsize_scale: float = 1.0,
    y_limits: dict[str, tuple[float, float]] | None = None,
    show_yaxis: dict[str, bool] | None = None,
    modality_blocks: Sequence[str] = MODALITY_BLOCKS,
    center_singleline_xlabels: bool = True,
    xlabel_pad: float = 3.0,
    xlabel_pad_centered: float = -1.5,
) -> None:
    """Plot PSC boxplots by ROI and modality blocks."""
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df = df[df["Hemisphere"] == "bh"].copy()
    df = df[df["Task"] == TASK_NAME].copy()

    roi_values = list(pd.unique(df["ROI"]))
    rois = [_resolve_roi(r, roi_values) for r in ROI_ORDER]

    within_base_frac = 0.02
    within_step_frac = 0.07
    within_h_frac = 0.02
    within_headroom_frac = 0.012

    cross_gap_frac = 0.11
    cross_base_frac = 0.04
    cross_step_frac = 0.09
    cross_h_frac = 0.03
    cross_headroom_frac = 0.03

    specs = []
    for roi in rois:
        if roi is None:
            specs.append(
                {
                    "roi": None,
                    "y_lim": (0.0, 1.0),
                    "ticks": np.array([0.0, 1.0]),
                    "row_h": MIN_ROW_HEIGHT,
                    "eligible_within_by_mod": {},
                    "eligible_cross": [],
                    "y_max_data": 1.0,
                    "pad": 0.0,
                    "yr": 1.0,
                }
            )
            continue

        eligible_within_by_mod: Dict[str, List[dict]] = {
            m: [] for m in modality_blocks
        }
        for ann in WITHIN_ANNOTATIONS:
            if not _matches_roi(roi, ann.get("roi", "")):
                continue
            m = str(ann.get("modality", ""))
            if m in eligible_within_by_mod:
                eligible_within_by_mod[m].append(ann)

        for m in eligible_within_by_mod:
            eligible_within_by_mod[m].sort(key=lambda a: float(a["pvalue"]))

        eligible_cross: List[dict] = []
        if ("Auditory" in modality_blocks) and ("Visual" in modality_blocks):
            for ann in CROSS_AV_ANNOTATIONS:
                if not _matches_roi(roi, ann.get("roi", "")):
                    continue
                eligible_cross.append(dict(ann))
            eligible_cross.sort(key=lambda a: float(a["pvalue"]))

        vals = []
        for modality in modality_blocks:
            paired = _subject_table(df, roi, modality)
            for cat in CATEGORIES:
                x = paired[cat].to_numpy(dtype=float)
                x = x[np.isfinite(x)]
                if x.size:
                    vals.append(x)

        if vals:
            y_min = min(float(v.min()) for v in vals)
            y_max = max(float(v.max()) for v in vals)
        else:
            y_min, y_max = 0.0, 1.0

        yr = max(y_max - y_min, 0.1)
        pad = 0.08 * max(y_max - y_min, 0.1)

        max_stack_within = max(
            (len(v) for v in eligible_within_by_mod.values()),
            default=0,
        )
        top_needed_within = 0.0
        if max_stack_within > 0:
            top_needed_within = (
                within_base_frac
                + (max_stack_within - 1) * within_step_frac
                + within_h_frac
                + within_headroom_frac
            )

        top_needed_cross = 0.0
        if eligible_cross:
            top_needed_cross = (
                cross_gap_frac
                + cross_base_frac
                + (len(eligible_cross) - 1) * cross_step_frac
                + cross_h_frac
                + cross_headroom_frac
            )

        top_extra = max(top_needed_within, top_needed_cross) * yr

        rkey = _roi_key(roi)
        explicit = None if y_limits is None else y_limits.get(rkey)
        if explicit is not None:
            y0, y1 = float(explicit[0]), float(explicit[1])
        else:
            y0_raw = min(y_min - pad, 0.0)
            y1_raw = max(y_max + pad + top_extra, 0.0)
            y0 = np.floor(y0_raw / YTICK_STEP) * YTICK_STEP
            y1 = np.ceil(y1_raw / YTICK_STEP) * YTICK_STEP

        ticks = np.arange(y0, y1 + 0.5 * YTICK_STEP, YTICK_STEP)
        n_steps = max(int(np.ceil((y1 - y0) / YTICK_STEP)), 1)
        row_h = max(MIN_ROW_HEIGHT, n_steps * INCHES_PER_STEP)

        specs.append(
            {
                "roi": roi,
                "y_lim": (y0, y1),
                "ticks": ticks,
                "row_h": row_h,
                "eligible_within_by_mod": eligible_within_by_mod,
                "eligible_cross": eligible_cross,
                "y_max_data": y_max,
                "pad": pad,
                "yr": yr,
                "within_base_frac": within_base_frac,
                "within_step_frac": within_step_frac,
                "within_h_frac": within_h_frac,
                "cross_gap_frac": cross_gap_frac,
                "cross_base_frac": cross_base_frac,
                "cross_step_frac": cross_step_frac,
                "cross_h_frac": cross_h_frac,
            }
        )

    height_ratios = [s["row_h"] for s in specs]
    fig_w = 4.6 * FIG_W_SCALE * figsize_scale
    fig_w *= len(modality_blocks) / len(MODALITY_BLOCKS)

    n_rows = len(specs)

    ref_eff = (
        (REF_TOP - REF_BOTTOM) /
        (1.0 + REF_HSPACE * (n_rows - 1) / n_rows)
    )
    cur_eff = (
        (TOP_MARGIN - BOTTOM_MARGIN) /
        (1.0 + HSPACE * (n_rows - 1) / n_rows)
    )

    height_scale = ref_eff / cur_eff
    fig_h = float(sum(height_ratios)) * figsize_scale * height_scale

    fig, axes = plt.subplots(
        nrows=len(specs),
        ncols=len(modality_blocks),
        figsize=(fig_w, fig_h),
        gridspec_kw={"height_ratios": height_ratios},
        sharex=False,
        sharey=False,
    )

    fig.subplots_adjust(
        left=LEFT_MARGIN,
        right=RIGHT_MARGIN,
        top=TOP_MARGIN,
        bottom=BOTTOM_MARGIN,
        hspace=HSPACE,
        wspace=WSPACE,
    )

    if len(specs) == 1:
        axes = np.expand_dims(axes, axis=0)
    if len(modality_blocks) == 1:
        axes = np.expand_dims(axes, axis=1)

    xlabel_map = MOD_LABEL_CENTERED if center_singleline_xlabels else MOD_LABEL

    for row, spec in enumerate(specs):
        roi = spec["roi"]
        row_axes = []
        ax_lookup: Dict[str, plt.Axes] = {}

        for col, modality in enumerate(modality_blocks):
            ax = axes[row, col]
            row_axes.append(ax)
            ax_lookup[modality] = ax

            if roi is None:
                ax.axis("off")
                continue

            paired = _subject_table(df, roi, modality)
            data = [paired[cat].to_numpy(dtype=float) for cat in CATEGORIES]
            _draw_boxplot(ax, data, modality)

            ax.axhline(
                0,
                color="0.3",
                linestyle="--",
                linewidth=1.0,
                zorder=1,
            )
            ax.set_ylim(*spec["y_lim"])
            ax.set_xlim(*PAIR_XLIM)
            ax.yaxis.set_major_locator(MultipleLocator(YTICK_STEP))
            ax.yaxis.set_major_formatter(Y_FORMATTER)
            ax.set_yticks(spec["ticks"])
            ax.set_xticks([])
            ax.tick_params(axis="x", length=0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            pad = xlabel_pad
            if center_singleline_xlabels and modality != "Pooled":
                pad = xlabel_pad_centered

            ax.set_xlabel(
                xlabel_map[modality],
                fontsize=AXIS_LABEL_FS,
                labelpad=pad,
            )

            rkey = _roi_key(roi)
            display_axis = True
            if show_yaxis is not None:
                display_axis = show_yaxis.get(rkey, True)

            if col == 0 and display_axis:
                ax.set_ylabel("PSC (%)", fontsize=AXIS_LABEL_FS)
                ax.tick_params(axis="y", labelsize=YTICK_FS)
            else:
                ax.set_yticklabels([])
                ax.tick_params(axis="y", left=False)
                ax.spines["left"].set_visible(False)
                if not display_axis:
                    ax.set_ylabel("")

            within_anns = spec["eligible_within_by_mod"].get(modality, [])
            if within_anns:
                x1 = float(PAIR_POS[0])
                x2 = float(PAIR_POS[1])
                for level, ann in enumerate(within_anns):
                    text = pval_label_converter([float(ann["pvalue"])])[0]
                    y_top = ax.get_ylim()[1]
                    y_data = (
                        y_top
                        - (
                            spec["within_h_frac"]
                            + spec["within_base_frac"]
                            + within_headroom_frac
                            + level * spec["within_step_frac"]
                        ) * spec["yr"]
                    )
                    h_data = spec["within_h_frac"] * spec["yr"]

                    ann_key = (ann["roi"], ann["modality"], level)
                    text_dy_frac = WITHIN_TEXT_DY.get(ann_key, 0.0)

                    within_axis_annotation(
                        ax=ax,
                        x1=x1,
                        x2=x2,
                        text=text,
                        y_data=y_data,
                        h_data=h_data,
                        text_dy_frac=text_dy_frac,
                    )

        left_ax = row_axes[0]
        right_ax = row_axes[-1]
        x_center = 0.5 * (
            left_ax.get_position().x0 + right_ax.get_position().x1
        )
        y_top = max(ax.get_position().y1 for ax in row_axes) + 0.0020
        fig.text(
            x_center,
            y_top,
            ROI_PRETTY[_roi_key(roi)],
            ha="center",
            va="bottom",
            fontsize=ROI_TITLE_FS,
            fontweight="semibold",
        )

        if (
            roi is not None
            and ("Auditory" in ax_lookup)
            and ("Visual" in ax_lookup)
            and spec["eligible_cross"]
        ):
            ax_aud = ax_lookup["Auditory"]
            ax_vis = ax_lookup["Visual"]
            for level, ann in enumerate(spec["eligible_cross"]):
                text = pval_label_converter([float(ann["pvalue"])])[0]
                y_data = (
                    spec["y_max_data"]
                    + spec["pad"]
                    + (
                        spec["cross_gap_frac"]
                        + spec["cross_base_frac"]
                        + level * spec["cross_step_frac"]
                    ) * spec["yr"]
                )
                h_data = spec["cross_h_frac"] * spec["yr"]
                span_annotation_datay_figspan(
                    fig,
                    ax_left=ax_aud,
                    ax_right=ax_vis,
                    text=text,
                    y_data=y_data,
                    h_data=h_data,
                )

    top_axes = [axes[0, j] for j in range(len(modality_blocks))]
    x_right = max(ax.get_position().x1 for ax in top_axes)
    y_top_axes = max(ax.get_position().y1 for ax in top_axes)

    fig.text(
        x_right,
        y_top_axes + 0.055,
        "95% bootstrap CI for the Median of PSC",
        ha="right",
        va="top",
        fontsize=TITLE_FS,
        color="k",
    )

    legend_handles = [
        Patch(
            facecolor=MODALITY_COLORS["Pooled"]["Non-Random"],
            edgecolor="0.2",
        ),
        Patch(
            facecolor=MODALITY_COLORS["Pooled"]["Random"],
            edgecolor="0.2",
        ),
    ]
    legend_labels = ["Non-Random", "Random"]

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper right",
        bbox_to_anchor=(x_right, y_top_axes + 0.036),
        frameon=False,
        fontsize=LEGEND_FS,
        ncol=1,
        handlelength=1.0,
        handletextpad=0.35,
        labelspacing=0.12,
        borderaxespad=0.0,
        columnspacing=0.0,
    )

    fig.savefig(outpath, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


# ============================== I/O ================================ #


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--figscale",
        type=float,
        default=1.0,
        help="Scale factor for figure size.",
    )
    return parser.parse_args()


# =========================== INPUT PATHS =========================== #

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL = "rwls"
MASKING = "wb"
HRF_CUTOFF = "hrf128"
INDIVID_LEVEL = "i"
N_ROIS = 8

BASE_DIR = os.path.join(
    WORKING_DIR,
    f"roi_analyses_{MODEL}_{HRF_CUTOFF}_{MASKING}_puncorr_unsmoothed",
    "bothmod_allmain_tasks",
)

OUTPUT_DIR = os.path.join(BASE_DIR, "anova_plots_nonrand")

DATA_PATH = os.path.join(
    BASE_DIR,
    "rand_ntfd_nonrandom",
    "df_rois_volume",
    f"dfrois_{INDIVID_LEVEL}_{N_ROIS}-rois.tsv",
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "psc_boxplots_rand_ntfd_nonrandom.png",
)

OUTPUT_PATH_SENSORY = os.path.join(
    OUTPUT_DIR,
    "psc_boxplots_rand_ntfd_nonrandom_auditory_visual.png",
)


# ============= WITHIN-SUBPLOT ANNOTATIONS ========================= #

WITHIN_ANNOTATIONS: List[dict] = [
    dict(
        roi="cerebellum",
        modality="Pooled",
        pvalue=0.0000210418857266061,
    ),
    dict(
        roi="presma",
        modality="Pooled",
        pvalue=0.0000156873056962545,
    ),
    dict(
        roi="sma",
        modality="Pooled",
        pvalue=0.00000000670842733504835,
    ),
    dict(
        roi="pmd",
        modality="Pooled",
        pvalue=0.000000257147159216042,
    ),
    dict(
        roi="pmv",
        modality="Pooled",
        pvalue=0.0000794646286807802,
    ),
    dict(
        roi="heschl",
        modality="Pooled",
        pvalue=0.000031556368482059,
    ),
    dict(
        roi="heschl",
        modality="Auditory",
        pvalue=0.0000000559303322907296,
    ),
    dict(
        roi="occipital",
        modality="Pooled",
        pvalue=0.0000554500364608825,
    ),
    dict(
        roi="occipital",
        modality="Visual",
        pvalue=0.00000146503323769873,
    ),
]

WITHIN_TEXT_DY: Dict[Tuple[str, str, int], float] = {
    # Example:
    # ("occipital", "Visual", 0): -0.01,
}


# =================== CROSS-MODALITY (AUDIO ↔ VISUAL) ============== #

CROSS_AV_ANNOTATIONS: List[dict] = [
    # dict(
    #     roi="heschl",
    #     pvalue=0.0008,
    # ),
]


# ============================== RUN ================================ #

if __name__ == "__main__":
    args = parse_args()

    df_in = pd.read_csv(DATA_PATH, sep="\t")
    if "Task" in df_in.columns:
        df_in = df_in.copy()
        df_in["Task"] = df_in["Task"].replace(
            {"NTFD_Random": TASK_NAME, "NTFD-Random": TASK_NAME}
        )

    y_limits = {
        "heschl": (-0.8, 3.4),
        "occipital": (-0.8, 2.0),
        "dstr": (-0.6, 1.0),
        "cereb": (-0.6, 1.4),
        "presma": (-0.4, 1.6),
        "sma": (-0.4, 1.8),
        "pmd": (-0.6, 2.0),
        "pmv": (-0.2, 1.4),
    }

    show_yaxis = {
        "heschl": True,
        "occipital": True,
        "dstr": True,
        "cereb": True,
        "presma": True,
        "sma": True,
        "pmd": True,
        "pmv": True,
    }

    plot_psc_boxplots(
        df=df_in,
        outpath=OUTPUT_PATH,
        figsize_scale=args.figscale,
        y_limits=y_limits,
        show_yaxis=show_yaxis,
        modality_blocks=MODALITY_BLOCKS,
        center_singleline_xlabels=True,
        xlabel_pad=3.0,
        xlabel_pad_centered=-1.5,
    )

    plot_psc_boxplots(
        df=df_in,
        outpath=OUTPUT_PATH_SENSORY,
        figsize_scale=args.figscale,
        y_limits=y_limits,
        show_yaxis=show_yaxis,
        modality_blocks=MODALITY_BLOCKS_SENSORY,
        center_singleline_xlabels=False,
        xlabel_pad=3.0,
        xlabel_pad_centered=-1.5,
    )