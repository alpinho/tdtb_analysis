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
from typing import List, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import numpy as np
import pandas as pd


# ============================ CONSTANTS ============================ #

TASK_NAME = "NTFD Random"
CATEGORIES = ["Non-Random", "Random"]
MODALITY_BLOCKS = ["Pooled", "Auditory", "Visual"]

MOD_LABEL = {
    "Pooled": "Both\nModalities",
    "Auditory": "Auditory",
    "Visual": "Visual",
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

BOX_COLORS = {
    "Non-Random": "0.55",
    "Random": "0.80",
}

YTICK_STEP = 0.20
Y_FORMATTER = FormatStrFormatter("%.1f")
INCHES_PER_STEP = 0.68
MIN_ROW_HEIGHT = 2.0
FIG_W_SCALE = 0.74
PAIR_POS = [1.0, 1.16]
PAIR_XLIM = (0.90, 1.26)
BOX_WIDTH = 0.10

TITLE_FS = 10
LEGEND_FS = 9
ROI_TITLE_FS = 14
AXIS_LABEL_FS = 10
YTICK_FS = 8


# ============================ UTILITIES ============================ #


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


def _draw_boxplot(ax: plt.Axes, data: List[np.ndarray]) -> None:
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
        patch.set_facecolor(BOX_COLORS[cat])
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
) -> None:
    """Plot PSC boxplots by ROI and modality blocks."""
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df = df[df["Hemisphere"] == "bh"].copy()
    df = df[df["Task"] == TASK_NAME].copy()

    roi_values = list(pd.unique(df["ROI"]))
    rois = [_resolve_roi(r, roi_values) for r in ROI_ORDER]

    specs = []
    for roi in rois:
        if roi is None:
            specs.append(
                {
                    "roi": None,
                    "y_lim": (0.0, 1.0),
                    "ticks": np.array([0.0, 1.0]),
                    "row_h": MIN_ROW_HEIGHT,
                }
            )
            continue

        vals = []
        for modality in MODALITY_BLOCKS:
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

        rkey = _roi_key(roi)
        explicit = None if y_limits is None else y_limits.get(rkey)
        if explicit is not None:
            y0, y1 = float(explicit[0]), float(explicit[1])
        else:
            pad = 0.08 * max(y_max - y_min, 0.1)
            y0 = np.floor((min(y_min - pad, 0.0)) / YTICK_STEP) * YTICK_STEP
            y1 = np.ceil((max(y_max + pad, 0.0)) / YTICK_STEP) * YTICK_STEP

        ticks = np.arange(y0, y1 + 0.5 * YTICK_STEP, YTICK_STEP)
        n_steps = max(int(np.ceil((y1 - y0) / YTICK_STEP)), 1)
        row_h = max(MIN_ROW_HEIGHT, n_steps * INCHES_PER_STEP)

        specs.append(
            {
                "roi": roi,
                "y_lim": (y0, y1),
                "ticks": ticks,
                "row_h": row_h,
            }
        )

    height_ratios = [s["row_h"] for s in specs]
    fig_w = 4.6 * FIG_W_SCALE * figsize_scale

    n_rows = len(specs)

    # Match the effective vertical scaling of plot_anovas_all.py
    ref_top = 0.965
    ref_bottom = 0.11
    ref_hspace = 0.75

    cur_top = 0.992
    cur_bottom = 0.04
    cur_hspace = 0.18

    ref_eff = (
        (ref_top - ref_bottom) /
        (1.0 + ref_hspace * (n_rows - 1) / n_rows)
    )
    cur_eff = (
        (cur_top - cur_bottom) /
        (1.0 + cur_hspace * (n_rows - 1) / n_rows)
    )

    height_scale = ref_eff / cur_eff
    fig_h = float(sum(height_ratios)) * figsize_scale * height_scale

    fig, axes = plt.subplots(
        nrows=len(specs),
        ncols=len(MODALITY_BLOCKS),
        figsize=(fig_w, fig_h),
        gridspec_kw={"height_ratios": height_ratios},
        sharex=False,
        sharey=False,
    )

    fig.subplots_adjust(
        left=0.11,
        right=0.98,
        top=0.992,
        bottom=0.04,
        hspace=0.18,
        wspace=0.48,
    )

    if len(specs) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, spec in enumerate(specs):
        roi = spec["roi"]
        row_axes = []

        for col, modality in enumerate(MODALITY_BLOCKS):
            ax = axes[row, col]
            row_axes.append(ax)

            if roi is None:
                ax.axis("off")
                continue

            paired = _subject_table(df, roi, modality)
            data = [paired[cat].to_numpy(dtype=float) for cat in CATEGORIES]
            _draw_boxplot(ax, data)

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
            ax.set_xlabel(
                MOD_LABEL[modality],
                fontsize=AXIS_LABEL_FS,
                labelpad=2.,
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

        left_ax = row_axes[0]
        right_ax = row_axes[-1]
        x_center = 0.5 * (
            left_ax.get_position().x0 + right_ax.get_position().x1
        )
        y_top = max(ax.get_position().y1 for ax in row_axes) + 0.0015
        fig.text(
            x_center,
            y_top,
            ROI_PRETTY[_roi_key(roi)],
            ha="center",
            va="bottom",
            fontsize=ROI_TITLE_FS,
            fontweight="semibold",
        )

    top_axes = [axes[0, j] for j in range(len(MODALITY_BLOCKS))]
    x_right = max(ax.get_position().x1 for ax in top_axes)
    y_top_axes = max(ax.get_position().y1 for ax in top_axes)
    x_leg = x_right
    y_text = y_top_axes + 0.020
    y_leg = y_top_axes + 0.015

    fig.text(
        x_right,
        y_text,
        "95% bootstrap CI for the Median of PSC",
        ha="right",
        va="top",
        fontsize=TITLE_FS,
        color="k",
    )

    handles = [
        Patch(facecolor=BOX_COLORS["Non-Random"], edgecolor="0.2"),
        Patch(facecolor=BOX_COLORS["Random"], edgecolor="0.2"),
    ]
    labels = ["Non-Random", "Random"]

    fig.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(x_leg, y_leg),
        frameon=False,
        fontsize=LEGEND_FS,
        ncol=1,
        handlelength=1.0,
        handletextpad=0.35,
        labelspacing=0.12,
        borderaxespad=0.0,
        columnspacing=0.0,
    )

    fig.savefig(outpath, dpi=300, bbox_inches="tight", pad_inches=0.06)
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


# ============================== RUN ================================ #

if __name__ == "__main__":
    args = parse_args()

    df_in = pd.read_csv(DATA_PATH, sep="\t")
    if "Task" in df_in.columns:
        df_in = df_in.copy()
        df_in["Task"] = df_in["Task"].replace(
            {"NTFD_Random": TASK_NAME, "NTFD-Random": TASK_NAME}
        )

    plot_psc_boxplots(
        df=df_in,
        outpath=OUTPUT_PATH,
        figsize_scale=args.figscale,
        y_limits={
            "occipital": (-0.6, 2.2),
            "dstr": (-0.6, 1.0),
            "cereb": (-0.6, 2.2),
            "presma": (-0.2, 1.2),
            "sma": (-0.2, 1.2),
            "pmd": (-0.2, 1.2),
            "pmv": (-0.2, 1.2),
        },
        show_yaxis={
            "heschl": True,
            "occipital": True,
            "dstr": True,
            "cereb": True,
            "presma": True,
            "sma": True,
            "pmd": True,
            "pmv": True,
        },
    )