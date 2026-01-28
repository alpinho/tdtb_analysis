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


from matplotlib.ticker import MaxNLocator
import matplotlib.cbook as cbook


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

    roi_order = list(pd.unique(df["ROI"]))
    n_rois = len(roi_order)

    col_spec = [
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
    n_cols = len(col_spec)

    width_ratios = [1, 1, 1, 0.18, 1, 1, 1, 0.18, 1, 1, 1]

    label_lines = []
    for mod, task in col_spec:
        if mod == "SPACER":
            continue
        if mod == "Pooled":
            label_lines.append(task)
        else:
            label_lines.extend([mod, task])

    max_line_len = max((len(s) for s in label_lines), default=10)
    per_col = 0.70 + 0.020 * max(0, max_line_len - 8)
    per_col = min(max(per_col, 0.70), 0.88)

    fig_w = per_col * n_cols * figsize_scale

    # One more step up in per-row height.
    fig_h = 4.2 * max(n_rois, 1) * figsize_scale

    fig, axes = plt.subplots(
        nrows=n_rois,
        ncols=n_cols,
        figsize=(fig_w, fig_h),
        sharey="row",
        gridspec_kw={"width_ratios": width_ratios},
    )
    if n_rois == 1:
        axes = np.expand_dims(axes, axis=0)

    colors = {"Beat": "#E69F00", "Interval": "#009E73"}

    box_w = 0.055
    delta = 0.065
    pos0 = 1.00
    pos = [pos0, pos0 + delta]

    x_pad = 0.040
    x_min = pos[0] - x_pad
    x_max = pos[1] + x_pad

    # Keep extra height in axes, not in whitespace.
    fig.subplots_adjust(left=0.055, right=0.995, top=0.93, bottom=0.10)
    fig.subplots_adjust(wspace=0.10, hspace=0.18)

    y_col = 0
    whis = 1.5
    ypad_frac = 0.02
    nbins = 9

    for r, roi in enumerate(roi_order):
        roi_vals = df.loc[df["ROI"] == roi, "PSC"].dropna().to_numpy()
        if roi_vals.size == 0:
            continue

        w_lows = []
        w_highs = []

        for mod, task in col_spec:
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

        if w_lows and w_highs:
            y_min = float(np.min(w_lows))
            y_max = float(np.max(w_highs))
        else:
            y_min = float(np.min(roi_vals))
            y_max = float(np.max(roi_vals))

        yr = max(y_max - y_min, 0.1)
        pad = ypad_frac * yr
        y_lim = (y_min - pad, y_max + pad)

        for c, (mod, task) in enumerate(col_spec):
            ax = axes[r, c]

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
                meanprops={
                    "linestyle": "--",
                    "linewidth": 1.2,
                    "color": "k",
                },
            )

            for patch, cat in zip(bp["boxes"], CATEGORIES):
                patch.set_facecolor(colors[cat])

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(*y_lim)
            ax.set_xticks([])

            if mod == "Pooled":
                ax.set_xlabel(task)
            else:
                ax.set_xlabel(f"{mod}\n{task}")
            ax.xaxis.labelpad = 2

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(True)

            if c == y_col:
                ax.spines["left"].set_visible(True)
                ax.set_ylabel(f"{roi} PSC (%)")
                ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins))
                ax.tick_params(axis="y", left=True, labelleft=True)
            else:
                ax.spines["left"].set_visible(False)
                ax.set_ylabel("")
                ax.tick_params(axis="y", left=False, labelleft=False)

    fig.legend(
        handles=[
            plt.Line2D(
                [0], [0], marker="s", linestyle="",
                markerfacecolor=colors["Beat"],
                markeredgecolor="none", markersize=10,
                label="Beat",
            ),
            plt.Line2D(
                [0], [0], marker="s", linestyle="",
                markerfacecolor=colors["Interval"],
                markeredgecolor="none", markersize=10,
                label="Interval",
            ),
            plt.Line2D(
                [0], [0], linestyle="--", linewidth=1.2,
                color="k", label="Mean",
            ),
        ],
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.985),
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