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
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import numpy as np
import pandas as pd


# =========================== FUNCTIONS ============================= #

def pval_label_converter(pvalues):
    """Convert p-values to star labels."""
    out = []
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


def _subset(
    df: pd.DataFrame,
    roi: str,
    modality: str,
    task: str,
    category: str,
) -> np.ndarray:
    mask = (
        (df["ROI"] == roi)
        & (df["Task"] == task)
        & (df["Category"] == category)
    )
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

    def _matches_roi(resolved_roi: str, roi_key: str) -> bool:
        return roi_key.lower() in str(resolved_roi).lower()

    def add_span_annotation_figcoords(
        fig,
        ax_left,
        ax_right,
        text: str,
        y_pad: float = 0.010,
        h: float = 0.006,
        lw: float = 1.4,
        fs: float = 14,
    ) -> None:
        """
        Draw a bracket spanning ax_left -> ax_right above the axes.

        Uses figure coordinates, allowing the bracket to span multiple axes
        without changing y-limits.
        """
        b1 = ax_left.get_position()
        b2 = ax_right.get_position()

        x1 = b1.x0 + 0.5 * b1.width
        x2 = b2.x0 + 0.5 * b2.width
        y = max(b1.y1, b2.y1) + y_pad

        fig.add_artist(
            Line2D(
                [x1, x1],
                [y, y + h],
                transform=fig.transFigure,
                lw=lw,
                color="k",
            )
        )
        fig.add_artist(
            Line2D(
                [x1, x2],
                [y + h, y + h],
                transform=fig.transFigure,
                lw=lw,
                color="k",
            )
        )
        fig.add_artist(
            Line2D(
                [x2, x2],
                [y + h, y],
                transform=fig.transFigure,
                lw=lw,
                color="k",
            )
        )

        fig.text(
            (x1 + x2) / 2,
            y + h,
            text,
            ha="center",
            va="bottom",
            fontsize=fs,
            color="k",
        )

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
    fig_h = 6. * n_rows * figsize_scale

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=len(col_spec),
        figsize=(fig_w, fig_h),
        sharey=False,
        gridspec_kw={"width_ratios": width_ratios},
    )

    colors = {
        "Beat": "tab:blue",
        "Interval": "tab:orange",
    }

    BOX_ALPHA = 0.6

    ZERO_LINE_COLOR = "0.35"
    ZERO_LINE_LS = "--"
    ZERO_LINE_LW = 1.5
    ZERO_LINE_ZORDER = 0

    # ---- annotation stacking (figure coords) ----
    annot_y_pad_base = 0.012
    annot_y_pad_step = 0.015
    annot_h = 0.006

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
        top=0.91,
        bottom=0.14,
        wspace=0.16,
        hspace=0.52,
    )

    whis = 1.5
    ypad_frac = 0.02

    xlabel_fs = 14
    xlabel_pad = 4
    axis_label_fs = xlabel_fs
    ytick_fs = xlabel_fs
    legend_fs = axis_label_fs + 2

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

            y_ticks = np.linspace(y_lim[0], y_lim[1], n_yticks)
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
                    medianprops={
                        "linewidth": 0,
                        "color": "none",
                    },
                    meanprops={
                        "linestyle": "--",
                        "linewidth": 2.2,
                        "color": "k",
                    },
                )

                for patch, cat in zip(bp["boxes"], CATEGORIES):
                    patch.set_facecolor(colors[cat])
                    patch.set_alpha(BOX_ALPHA)

                ax.set_xlim(x_min, x_max)
                ax.set_ylim(*y_lim)

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

            # ---- Add annotations for this ROI side (after axes exist) ----
            ax_lookup = {}
            for j2, (m2, t2) in enumerate(col_spec_block):
                if m2 == "SPACER":
                    continue
                ax_lookup[(m2, t2)] = axes[r, start + j2]

            eligible = []
            for ann in ANNOTATIONS:
                if not _matches_roi(roi, ann["roi"]):
                    continue

                m = ann["modality"]
                t_left, t_right = ann["task_pair"]

                if (m, t_left) not in ax_lookup:
                    continue
                if (m, t_right) not in ax_lookup:
                    continue

                eligible.append(ann)

            task_order = {"Production": 0, "Perception": 1, "NTFD": 2}

            def _ann_sort_key(ann):
                t1, t2 = ann["task_pair"]
                i1 = task_order[t1]
                i2 = task_order[t2]
                span = abs(i2 - i1)
                return (ann["modality"], span, min(i1, i2), max(i1, i2))

            eligible.sort(key=_ann_sort_key)

            for k, ann in enumerate(eligible):
                m = ann["modality"]
                t_left, t_right = ann["task_pair"]

                label = pval_label_converter([ann["pvalue"]])[0]
                ax_l = ax_lookup[(m, t_left)]
                ax_r = ax_lookup[(m, t_right)]

                y_pad = annot_y_pad_base + (k * annot_y_pad_step)
                add_span_annotation_figcoords(
                    fig,
                    ax_l,
                    ax_r,
                    label,
                    y_pad=y_pad,
                    h=annot_h,
                    lw=1.4,
                    fs=14,
                )

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
            Line2D(
                [0],
                [0],
                linestyle="--",
                linewidth=3.2,
                color="k",
                label="Mean",
            ),
        ],
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.015),
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

model = "rwls"
masking = "wb"
hrf_cutoff = "hrf128"

# Individualization level of input data
# INDIV_LEVELS = [
#     'i', 'i9a', 'i8a', 'i7a', 'i6a',
#     'a',
#     'a4g', 'a3g', 'a2g', 'a1g', 'g'
# ]
INDIVID_LEVEL = "i"
N_ROIS = 8

roi_dir = os.path.join(
    working_dir,
    "roi_analyses_" + model + "_" + hrf_cutoff + "_" + masking
    + "_puncorr_unsmoothed",
    "bothmod_allmain_tasks",
    "main_tasks",
)

data_path = os.path.join(
    roi_dir,
    "df_rois_volume",
    "dfrois_" + INDIVID_LEVEL + "_" + str(N_ROIS) + "-rois.tsv",
)

output_path = os.path.join(
    working_dir,
    "results",
    "fig4",
    "psc_boxplots_by_roi.png",
)

# ======================== ANNOTATIONS =========================== #
# Each entry defines ONE bracket spanning two task-panels within ONE modality
# block. "Pooled" refers to the left-most block (both modalities pooled).
#
# roi: short key ("dstr", "sma", "pmv", ...)
# modality: "Pooled" | "Auditory" | "Visual"
# task_pair: ("Production", "NTFD"), etc.
# pvalue: numeric p-value
ANNOTATIONS = [
    dict(
        roi="dstr",
        modality="Pooled",
        task_pair=("Production", "NTFD"),
        pvalue=0.000000747501935034951,
    ),
    dict(
        roi="dstr",
        modality="Pooled",
        task_pair=("Production", "Perception"),
        pvalue=0.000000183218571178129,
    ),
    dict(
        roi="cereb",
        modality="Pooled",
        task_pair=("Production", "NTFD"),
        pvalue=0.0038694053454941,
    ),
    dict(
        roi="presma",
        modality="Pooled",
        task_pair=("Perception", "NTFD"),
        pvalue=0.028120293825793,
    ),
    dict(
        roi="sma",
        modality="Pooled",
        task_pair=("Production", "Perception"),
        pvalue=0.0000130156708231288,
    ),
    dict(
        roi="sma",
        modality="Pooled",
        task_pair=("Perception", "NTFD"),
        pvalue=0.0003850973375033,
    ),
    dict(
        roi="heschl",
        modality="Pooled",
        task_pair=("Production", "NTFD"),
        pvalue=0.0000000326951551140279,
    ),
    dict(
        roi="heschl",
        modality="Pooled",
        task_pair=("Perception", "NTFD"),
        pvalue=0.0000000543008044450249,
    ),
    dict(
        roi="occipital",
        modality="Pooled",
        task_pair=("Production", "Perception"),
        pvalue=0.0022787859816777,
    ),
    dict(
        roi="occipital",
        modality="Pooled",
        task_pair=("Perception", "NTFD"),
        pvalue=0.0005348873092657,
    ),
]

# ============================= RUN ================================= #

if __name__ == "__main__":
    args = parse_args()
    df = pd.read_csv(data_path, sep="\t")
    plot_psc_boxplots(df=df, outpath=output_path, figsize_scale=args.figscale)