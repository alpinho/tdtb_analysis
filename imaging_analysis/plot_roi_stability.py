#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P-values profile of results obtained across different
 individualizations of ROIs

Author: Ana Luisa Pinho
email: agrilopi@uwo.ca

Created: 14th of March, 2025
Last update: January 2026

Compatibility: Python 3.10.14
"""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================== helpers ================================
def plot_pvals(weights: Iterable[float],
               pvals: Iterable[float],
               ylabel: str,
               y_min: float,
               y_max: float,
               y_nticks: int,
               out_png: str) -> None:
    """Plot p-values vs individualization weight."""
    plt.figure(figsize=(8, 5))
    plt.grid(True, linewidth=2.0, zorder=0)

    plt.scatter(weights,
                pvals,
                color="k",
                s=80,
                edgecolors="black",
                linewidth=1.5,
                clip_on=False,
                zorder=3)
    plt.plot(weights, pvals, linestyle="-", alpha=0.7, linewidth=3.0, c="k")

    plt.xlabel(r"$w_{i}$", labelpad=14, fontsize=24)
    plt.ylabel(ylabel, labelpad=14, fontsize=24)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_position(("outward", 10))
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_bounds(0, 1)
    ax.spines["bottom"].set_linewidth(2.0)
    ax.spines["left"].set_linewidth(2.0)

    plt.xlim(0.0, 1.0)
    plt.ylim(y_min, y_max)

    xt = np.linspace(0.0, 1.0, 11)
    plt.xticks(xt, fontsize=18)
    ax.set_xticklabels([str(t) if t in [0.0, 0.5, 1.0] else ""
                        for t in xt],
                       fontsize=18)

    yt = np.linspace(y_min, y_max, y_nticks)
    plt.yticks(yt, fontsize=18)
    ax.set_yticklabels([f"{tick:.3f}" for tick in ax.get_yticks()],
                       fontsize=20)

    plt.margins(x=0.05)
    plt.subplots_adjust(bottom=0.2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def holm_levels(pvals: Iterable[float]) -> np.ndarray:
    """Holm step-down across a 1D array (use uncorrected p-values)."""
    p = np.asarray(list(pvals), dtype=float)
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    raw = (m - np.arange(m)) * ranked
    adj_ranked = np.maximum.accumulate(raw)
    adj_ranked = np.minimum(adj_ranked, 1.0)
    out = np.empty_like(adj_ranked)
    out[order] = adj_ranked

    return out


def pick_row(df: pd.DataFrame) -> pd.Series:
    """Pick the exact target row from a Pingouin posthoc table."""
    q = (
        (df["Contrast"] == "ROI * Category") &
        (df["ROI"] == "dstr") &
        (df["A"] == "Beat") &
        (df["B"] == "Interval")
    )
    sub = df.loc[q]
    if sub.empty:
        raise ValueError("Row not found: ROI*Category, dstr, Beat vs Interval")
    return sub.iloc[0]


# ============================== inputs ================================
# Mask type: all, auditory, visual
ENC_MASK = "bothmod_allmain_tasks"

TASK_TYPE = "main_tasks"

# anova_type options:
# - "2way-anova_vol_cat2rois"
# - "2way-anova_vol_cat2rois_auditory"
# - "2way-anova_vol_cat2rois_visual"
ANOVA_TYPE = "2way-anova_vol_cat2rois_auditory"

TAGS = ["i", "i9a", "i8a", "i7a", "i6a",
        "a", "a4g", "a3g", "a2g", "a1g", "g"]
HEMI = "bh"
TASK = "allmain_tasks"

STEP = 0.005


# =============================== main =================================
if __name__ == "__main__":
    workdir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(workdir, "results", "pvalues_stability_plots")
    base_dir = os.path.join(
        workdir,
        "roi_analyses_rwls_hrf128_wb_puncorr_unsmoothed",
        ENC_MASK,
        TASK_TYPE,
        ANOVA_TYPE,
    )
    os.makedirs(out_dir, exist_ok=True)

    rev_tags = TAGS[::-1]
    posthoc_paths = [
        os.path.join(
            base_dir,
            f"{tag}_{HEMI}_2w-{TASK}_posthoc.tsv",
        )
        for tag in rev_tags
    ]

    df_list = [pd.read_csv(pth, sep="\t") for pth in posthoc_paths]
    rows = [pick_row(df) for df in df_list]

    # p_corr across ROIs computed by pingouin
    p_corr = np.array(
        [np.nan if pd.isna(r["p-corr"]) else round(r["p-corr"], 3)
         for r in rows]
    )
    p_unc = np.array([round(r["p-unc"], 3) for r in rows])

    # holm across individualization levels on **uncorrected** p-values
    p_holm = np.array([round(x, 6) for x in holm_levels(p_unc)])

    # x-axis weights
    w = np.round(np.arange(0.0, 1.1, 0.1), 1)

    # filenames
    if ANOVA_TYPE == "2way-anova_vol_cat2rois":
        suffix = "task-all"
    elif ANOVA_TYPE == "2way-anova_vol_cat2rois_auditory":
        suffix = "task-auditory"
    else:
        suffix = "task-visual"

    f_corr = os.path.join(
        out_dir, f"pcorrected_mask-{ENC_MASK}_{suffix}_plot.png"
    )
    f_unc = os.path.join(
        out_dir, f"puncorrected_mask-{ENC_MASK}_{suffix}_plot.png"
    )
    f_holm = os.path.join(
        out_dir, f"pcorrected-ilevels_mask-{ENC_MASK}_{suffix}_plot.png"
    )

    # y-lims
    if np.all(np.isnan(p_corr)):
        ycorr_min, ycorr_max = 0.0, 1.0
    else:
        ycorr_min = float(
            np.round(np.floor((np.nanmin(p_corr) - 1e-8) / STEP) * STEP, 3)
        )
        ycorr_max = float(
            np.round(np.ceil((np.nanmax(p_corr) + 1e-8) / STEP) * STEP, 3)
        )

    yunc_min = float(
        np.round(np.floor((np.amin(p_unc) - 1e-8) / STEP) * STEP, 3)
    )
    yunc_max = float(
        np.round(np.ceil((np.amax(p_unc) + 1e-8) / STEP) * STEP, 3)
    )

    yholm_min = float(
        np.round(np.floor((np.amin(p_holm) - 1e-8) / STEP) * STEP, 3)
    )
    yholm_max = float(
        np.round(np.ceil((np.amax(p_holm) + 1e-8) / STEP) * STEP, 3)
    )

    # labels
    lab_corr = r"$p_{\mathrm{FWE}}(w_{i})$"
    lab_unc = r"$p_{\mathrm{uncorr}}(w_{i})$"
    lab_holm = r"$p_{\mathrm{FWE}}(w_{i})$"

    # plots
    if not np.all(np.isnan(p_corr)):
        plot_pvals(w, p_corr, lab_corr, ycorr_min, ycorr_max, 6, f_corr)

    plot_pvals(w, p_unc, lab_unc, yunc_min, yunc_max, 4, f_unc)
    plot_pvals(w, p_holm, lab_holm, yholm_min, yholm_max, 6, f_holm)