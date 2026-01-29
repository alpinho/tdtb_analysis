#!/usr/bin/env python3
"""
Plot PSC by ROI (rows) and modality/task blocks (columns).

Author: Ana Luisa Pinho
email: agrilopi@uwo.ca

Created: 28th of January, 2026
Last update: January 2026

Compatibility: Python 3.10.14
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.patches import Patch

import numpy as np
import pandas as pd


# =========================== ANOVAS ================================ #

def _subset(
    df: pd.DataFrame,
    roi: str,
    modality: str,
    task: str,
    category: str,
) -> np.ndarray:
    mask = (df["ROI"] == roi) & (df["Task"] == task) & (df["Category"] == category)
    if modality != "Pooled":
        mask &= df["Modality"] == modality
    return df.loc[mask, "PSC"].dropna().to_numpy()


def _paired_by_subject(
    df: pd.DataFrame,
    roi: str,
    modality: str,
    task: str,
) -> pd.DataFrame:
    cols = ["Subject", "PSC", "ROI", "Task", "Category", "Modality"]
    sub = df.loc[
        (df["ROI"] == roi)
        & (df["Task"] == task)
        & (df["Category"].isin(CATEGORIES)),
        cols,
    ].copy()

    if modality != "Pooled":
        sub = sub[sub["Modality"] == modality]

    # One value per subject per category (averaging duplicates if any).
    sub = (
        sub.groupby(["Subject", "Category"], sort=False)["PSC"]
        .mean()
        .unstack("Category")
    )

    # Keep only subjects with both Beat and Interval.
    sub = sub.dropna(subset=CATEGORIES, how="any")

    if sub.empty:
        return pd.DataFrame(columns=CATEGORIES)

    return sub[CATEGORIES]


def plot_psc_boxplots(
    df: pd.DataFrame,
    outpath: Path,
    figsize_scale: float = 1.0,
) -> None:
    outpath = Path(outpath)
    if outpath.suffix == "":
        raise ValueError("outpath must end with .png or .pdf")
    outpath.parent.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df = df[df["Hemisphere"] == "bh"]
    df = df[df["Task"].isin(TASKS)]
    df = df[df["Category"].isin(CATEGORIES)]
    df = df[df["Modality"].isin(["Auditory", "Visual"])]

    col_spec_block = [
        ("Pooled", "Production"),
        ("Pooled", "Perception"),
        ("Pooled", "NTFD"),
        ("SPACER", "SPACER"),
        ("Auditory", "Production"),
        ("Auditory", "Perception"),
        ("Auditory", "NTFD"),
        ("SPACER", "SPACER"),
        ("Visual", "Production"),
        ("Visual", "Perception"),
        ("Visual", "NTFD"),
    ]

    width_ratios_block = [1, 1, 1, 0.20, 1, 1, 1, 0.20, 1, 1, 1]
    n_cols_block = len(col_spec_block)

    panel_sep = 1.6
    col_spec = col_spec_block + [("ROI_GAP", "ROI_GAP")] + col_spec_block
    width_ratios = width_ratios_block + [panel_sep] + width_ratios_block

    roi_grid_left = ["dstr", "presma", "pmd", "Heschl"]
    roi_grid_right = ["cereb", "sma", "pmv", "Occipital"]
    n_rows = 4

    roi_map = {str(r).lower(): r for r in pd.unique(df["ROI"])}

    def _resolve_roi(name: str) -> str | None:
        key = name.lower()
        if key in roi_map:
            return roi_map[key]
        for k, v in roi_map.items():
            if key in k:
                return v
        return None

    rois_left = [_resolve_roi(r) for r in roi_grid_left]
    rois_right = [_resolve_roi(r) for r in roi_grid_right]

    # ---- pretty ROI labels ----
    roi_pretty = {
        "dstr": "Dorsal Striatum",
        "cereb": "Cerebellum",
        "presma": "PreSMA",
        "sma": "SMA",
        "pmd": "PMD",
        "pmv": "PMV",
        "heschl": "Heschl’s Gyrus",
        "occipital": "Occipital Lobe",
    }

    def _pretty_roi_label(resolved_roi: str) -> str:
        key = str(resolved_roi).strip().lower()
        for short, pretty in roi_pretty.items():
            if short in key:
                return pretty
        return str(resolved_roi)

    # ---- original width heuristic ----
    label_lines = []
    for mod, task in col_spec_block:
        if mod == "SPACER":
            continue
        if mod == "Pooled":
            label_lines.append(task)
        else:
            label_lines.extend([mod, task])

    max_line_len = max((len(s) for s in label_lines), default=10)
    per_col = 1.18 + 0.034 * max(0, max_line_len - 8)
    per_col = min(per_col, 1.55)

    fig_w = per_col * float(sum(width_ratios)) * figsize_scale
    fig_h = 4.2 * n_rows * figsize_scale

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=len(col_spec),
        figsize=(fig_w, fig_h),
        sharey=False,
        gridspec_kw={"width_ratios": width_ratios},
    )

    # Use same colors as plot_single_dissociation.py
    colors = {
        "Beat": "tab:blue",
        "Interval": "tab:orange",
    }

    # Transparency (boxes + legend)
    BOX_ALPHA = 0.6

    # Zero line style
    ZERO_LINE_COLOR = "0.35"   # dark grey
    ZERO_LINE_LS = "--"
    ZERO_LINE_LW = 1.5
    ZERO_LINE_ZORDER = 0

    # ---- box geometry ----
    box_w = 0.135
    delta = 0.155
    pos = [1.00, 1.00 + delta]
    x_pad = 0.135
    x_min = pos[0] - x_pad
    x_max = pos[1] + x_pad

    fig.subplots_adjust(
        left=0.055,
        right=0.995,
        top=0.93,
        bottom=0.14,
        wspace=0.16,
        hspace=0.18,
    )

    whis = 1.5
    ypad_frac = 0.02

    # ---- typography ----
    xlabel_fs = 14
    xlabel_pad = 4
    axis_label_fs = xlabel_fs
    ytick_fs = xlabel_fs
    legend_fs = axis_label_fs + 2

    # ---- tick normalization: force EXACT tick count everywhere ----
    n_yticks = 5
    y_formatter = FormatStrFormatter("%.2f")

    left_start = 0
    gap_col = n_cols_block
    right_start = n_cols_block + 1

    for r in range(n_rows):
        axes[r, gap_col].axis("off")

        for block_i, roi in enumerate([rois_left[r], rois_right[r]]):
            start = left_start if block_i == 0 else right_start
            y_col = start

            if roi is None:
                for j in range(n_cols_block):
                    axes[r, start + j].axis("off")
                continue

            # ---- ROI-wise y-lims from whiskers ----
            w_lows, w_highs = [], []
            for mod, task in col_spec_block:
                if mod == "SPACER":
                    continue
                paired = _paired_by_subject(df, roi=roi, modality=mod, task=task)
                for cat in CATEGORIES:
                    vals = paired[cat].dropna().to_numpy()
                    if vals.size < 3:
                        continue
                    stats = cbook.boxplot_stats(vals, whis=whis)[0]
                    w_lows.append(float(stats["whislo"]))
                    w_highs.append(float(stats["whishi"]))

            if w_lows:
                y_min, y_max = min(w_lows), max(w_highs)
            else:
                roi_vals = df.loc[df["ROI"] == roi, "PSC"].dropna().to_numpy()
                y_min, y_max = float(roi_vals.min()), float(roi_vals.max())

            yr = max(y_max - y_min, 0.1)
            pad = ypad_frac * yr
            y_lim = (y_min - pad, y_max + pad)

            # Precompute fixed ticks for this ROI panel
            y_ticks = np.linspace(y_lim[0], y_lim[1], n_yticks)

            # Pretty label for this ROI (same for left+right leaders)
            roi_label = _pretty_roi_label(roi)

            for j, (mod, task) in enumerate(col_spec_block):
                ax = axes[r, start + j]

                if mod == "SPACER":
                    ax.axis("off")
                    continue

                paired = _paired_by_subject(df, roi=roi, modality=mod, task=task)
                data = [
                    paired["Beat"].to_numpy(),
                    paired["Interval"].to_numpy(),
                ]

                bp = ax.boxplot(
                    data,
                    positions=pos,
                    widths=box_w,
                    notch=True,
                    patch_artist=True,
                    showfliers=False,
                    showmeans=True,
                    meanline=True,
                    whis=whis,
                    # remove median line
                    medianprops=dict(
                        linewidth=0,
                        color="none",
                    ),
                    meanprops=dict(
                        linestyle="--",
                        linewidth=2.2,
                        color="k",
                    ),
                )

                # Color + transparency for boxes
                for patch, cat in zip(bp["boxes"], CATEGORIES):
                    patch.set_facecolor(colors[cat])
                    patch.set_alpha(BOX_ALPHA)

                ax.set_xlim(x_min, x_max)
                ax.set_ylim(*y_lim)

                # PSC=0 reference line in every subplot
                ax.axhline(
                    0,
                    color=ZERO_LINE_COLOR,
                    linestyle=ZERO_LINE_LS,
                    linewidth=ZERO_LINE_LW,
                    zorder=ZERO_LINE_ZORDER,
                )

                ax.set_xticks([])

                xlabel = task if mod == "Pooled" else f"{mod}\n{task}"
                ax.set_xlabel(xlabel, fontsize=xlabel_fs, labelpad=xlabel_pad)

                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_visible(True)

                if (start + j) == y_col:
                    ax.spines["left"].set_visible(True)

                    # ROI on outer line, PSC on inner line
                    ax.set_ylabel(
                        f"{roi_label}\nPSC (%)",
                        fontsize=axis_label_fs,
                    )
                    ax.yaxis.label.set_linespacing(1.75)

                    ax.set_yticks(y_ticks)
                    ax.yaxis.set_major_formatter(y_formatter)
                    ax.tick_params(
                        axis="y",
                        left=True,
                        labelleft=True,
                        labelsize=ytick_fs,
                    )
                else:
                    ax.spines["left"].set_visible(False)
                    ax.tick_params(axis="y", left=False, labelleft=False)

    fig.legend(
        handles=[
            Patch(
                facecolor=colors["Beat"],
                edgecolor="none",
                alpha=BOX_ALPHA,
                label="Beat",
            ),
            Patch(
                facecolor=colors["Interval"],
                edgecolor="none",
                alpha=BOX_ALPHA,
                label="Interval",
            ),
            plt.Line2D(
                [0], [0],
                linestyle="--",
                linewidth=3.2,
                color="k",
                label="Mean",
            ),
        ],
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.985),
        fontsize=legend_fs,
        handlelength=4.6,
        handleheight=1.8,
        handletextpad=1.0,
        columnspacing=2.8,
        borderaxespad=0.2,
    )

    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tsv",
        type=Path,
        default=Path("/mnt/data/dfrois_i_8-rois.tsv"),
        help="Input TSV file.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("psc_boxplots_by_roi.png"),
        help="Output figure path (.png or .pdf).",
    )
    parser.add_argument(
        "--figscale",
        type=float,
        default=1.0,
        help="Scale factor for figure size.",
    )
    return parser.parse_args()


# =========================== INPUTS ================================ #

TASKS = ["Production", "Perception", "NTFD"]
CATEGORIES = ["Beat", "Interval"]
MOD_BLOCKS = ["Pooled", "Auditory", "Visual"]

working_dir = os.path.dirname(os.path.abspath(__file__))

model = 'rwls'
masking = 'wb'
hrf_cutoff = 'hrf128'

# Individualization level of input data
# INDIV_LEVELS = [
#     'i', 'i9a', 'i8a', 'i7a', 'i6a',
#     'a',
#     'a4g', 'a3g', 'a2g', 'a1g', 'g'
# ]
INDIVID_LEVEL = 'i'
N_ROIS = 8

roi_dir = os.path.join(
    working_dir,
    'roi_analyses_' + model + '_' + hrf_cutoff + '_' + masking +
    '_puncorr_unsmoothed',
    'bothmod_allmain_tasks',
    'main_tasks')

data_path = os.path.join(
    roi_dir,
    'df_rois_volume',
    'dfrois_' + INDIVID_LEVEL + '_' + str(N_ROIS) + '-rois.tsv'
)

output_path = os.path.join(working_dir, 'results', 'fig4', 
                           'psc_boxplots_by_roi.png')

# ============================= RUN ================================= #

if __name__ == "__main__":

    args = parse_args()
    df = pd.read_csv(data_path, sep="\t")
    plot_psc_boxplots(df=df, outpath=output_path, figsize_scale=args.figscale)