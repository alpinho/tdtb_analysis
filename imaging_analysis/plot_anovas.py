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


def plot_psc_boxplots(
    df: pd.DataFrame,
    outpath: str,
    figsize_scale: float = 1.0,
) -> None:
    df = df.copy()

    df = df[df["Hemisphere"] == "bh"]
    df = df[df["Task"].isin(TASKS)]
    df = df[df["Category"].isin(CATEGORIES)]
    df = df[df["Modality"].isin(["Auditory", "Visual"])]

    roi_order = list(pd.unique(df["ROI"]))
    n_rois = len(roi_order)
    n_cols = len(MOD_BLOCKS) * len(TASKS)

    fig_w = 3.0 * n_cols * figsize_scale
    fig_h = 1.6 * max(n_rois, 1) * figsize_scale
    fig, axes = plt.subplots(
        nrows=n_rois,
        ncols=n_cols,
        figsize=(fig_w, fig_h),
        sharey="row",
    )
    if n_rois == 1:
        axes = np.expand_dims(axes, axis=0)

    # Okabe-Ito-like colors avoiding red/blue.
    colors = {
        "Beat": "#E69F00",      # orange
        "Interval": "#009E73",  # green
    }

    col_titles = []
    for mod in MOD_BLOCKS:
        for task in TASKS:
            col_titles.append(f"{mod} | {task}")

    for r, roi in enumerate(roi_order):
        for c, (mod, task) in enumerate(
            (pair for pair in [(m, t) for m in MOD_BLOCKS for t in TASKS])
        ):
            ax = axes[r, c]

            beat = _subset(df, roi=roi, modality=mod, task=task, category="Beat")
            interval = _subset(
                df, roi=roi, modality=mod, task=task, category="Interval"
            )

            data = [beat, interval]
            bp = ax.boxplot(
                data,
                labels=CATEGORIES,
                patch_artist=True,
                showfliers=False,
            )

            for patch, cat in zip(bp["boxes"], CATEGORIES):
                patch.set_facecolor(colors[cat])

            if r == 0:
                ax.set_title(col_titles[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(roi)

    fig.suptitle("PSC by ROI, modality block, task, and category", y=1.01)
    fig.tight_layout()
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