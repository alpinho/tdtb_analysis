#!/usr/bin/env python3
"""PSC boxplots by ROI for rand_ntfd_nonrandom.

Author: Ana Luisa Pinho
email: agrilopi@uwo.ca

Created: April 2026
Last update: April 2026

Compatibility: Python 3.10.14
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import numpy as np
import pandas as pd


# ============================ CONSTANTS ============================ #

TASK_NAME = "NTFD Random"
CATEGORIES = ["Non-Random", "Random"]
X_LABELS = ["Non-\nRand", "Rand"]
MODALITY_BLOCKS = ["Pooled", "Auditory", "Visual"]

MOD_LABEL = {
    "Pooled": "Both Modalities",
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
Y_FORMATTER = FormatStrFormatter("%.2f")
INCHES_PER_STEP = 0.68
MIN_ROW_HEIGHT = 2.0
FIG_W_SCALE = 0.62
PAIR_POS = [1.0, 1.22]
PAIR_XLIM = (0.88, 1.34)
BOX_WIDTH = 0.10

XLABEL_FS = 12
AXIS_LABEL_FS = 12
YTICK_FS = 12
ROI_TITLE_FS = 20
SUPTITLE_FS = 12
MOD_LABEL_PAD = 14


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
    """Resolve short ROI name to dataframe ROI value."""
    wanted = name.lower()
    for roi in roi_values:
        if _roi_key(str(roi)) == wanted:
            return str(roi)
    return None


def bootstrap_median_ci(
    vals: np.ndarray,
    n_boot: int = 5000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the median."""
    x = np.asarray(vals, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size

    if n == 0:
        return (np.nan, np.nan)
    if n < 3:
        med = float(np.median(x))
        return (med, med)

    if rng is None:
        rng = np.random.default_rng()

    idx = rng.integers(0, n, size=(n_boot, n))
    meds = np.median(x[idx], axis=1)
    lo = float(np.quantile(meds, alpha / 2.0))
    hi = float(np.quantile(meds, 1.0 - alpha / 2.0))
    return (lo, hi)


def bootstrap_conf_intervals(
    data: List[np.ndarray],
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 12345,
) -> np.ndarray:
    """Return bootstrap median confidence intervals for boxplots."""
    rng = np.random.default_rng(seed)
    cis = []
    for vals in data:
        cis.append(bootstrap_median_ci(vals, n_boot, alpha, rng))
    return np.asarray(cis, dtype=float)


def _subject_table(df: pd.DataFrame, roi: str, modality: str) -> pd.DataFrame:
    """Return subject x category table for one ROI and modality block."""
    cols = ["Subject", "PSC", "ROI", "Task", "Category", "Modality"]
    sub = df.loc[
        (df["ROI"] == roi)
        & (df["Task"] == TASK_NAME)
        & (df["Category"].isin(CATEGORIES)),
        cols,
    ].copy()

    if modality != "Pooled":
        sub = sub[sub["Modality"] == modality]

    sub = (
        sub.groupby(["Subject", "Category"], sort=False)["PSC"]
        .mean()
        .unstack("Category")
    )
    sub = sub.dropna(subset=CATEGORIES, how="any")

    if sub.empty:
        return pd.DataFrame(columns=CATEGORIES)

    return sub[CATEGORIES]


def _box_y_limits(
    df: pd.DataFrame,
    roi: str,
    y_limits: dict[str, tuple[float, float]] | None = None,
) -> tuple[float, float, np.ndarray]:
    """Return rounded y-limits and ticks for one ROI."""
    rkey = _roi_key(roi)
    if y_limits is not None and rkey in y_limits:
        y0 = float(y_limits[rkey][0])
        y1 = float(y_limits[rkey][1])
        ticks = np.arange(y0, y1 + 0.5 * YTICK_STEP, YTICK_STEP)
        if ticks.size < 2:
            ticks = np.array([y0, y1])
        return (y0, y1, ticks)

    vals = df.loc[
        (df["ROI"] == roi)
        & (df["Task"] == TASK_NAME)
        & (df["Category"].isin(CATEGORIES)),
        "PSC",
    ].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        y0, y1 = (-0.2, 0.2)
    else:
        ymin = float(vals.min())
        ymax = float(vals.max())
        yrange = max(ymax - ymin, 0.1)
        pad = 0.10 * yrange
        y0 = ymin - pad
        y1 = ymax + pad
        y0 = min(y0, 0.0)
        y1 = max(y1, 0.0)
        eps = 1e-9
        y0 = np.floor((y0 + eps) / YTICK_STEP) * YTICK_STEP
        y1 = np.ceil((y1 - eps) / YTICK_STEP) * YTICK_STEP

    ticks = np.arange(y0, y1 + 0.5 * YTICK_STEP, YTICK_STEP)
    if ticks.size < 2:
        ticks = np.array([y0, y1])
    return (float(y0), float(y1), ticks)


def _draw_boxplot(ax: plt.Axes, data: List[np.ndarray]) -> None:
    """Draw one Non-Random vs Random boxplot pair."""
    conf_intervals = bootstrap_conf_intervals(data)

    bp = ax.boxplot(
        data,
        positions=PAIR_POS,
        widths=BOX_WIDTH,
        notch=True,
        patch_artist=True,
        showfliers=False,
        showmeans=False,
        whis=1.5,
        medianprops={"linewidth": 0, "color": "none"},
        conf_intervals=conf_intervals,
    )

    for patch, cat in zip(bp["boxes"], CATEGORIES):
        patch.set_facecolor(BOX_COLORS[cat])
        patch.set_edgecolor("0.2")

    for vals, patch in zip(data, bp["boxes"]):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        mean_val = float(np.mean(vals))
        verts = patch.get_path().vertices
        x_left = float(verts[:, 0].min())
        x_right = float(verts[:, 0].max())
        ax.plot(
            [x_left, x_right],
            [mean_val, mean_val],
            color="k",
            lw=patch.get_linewidth(),
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
    """Plot a single figure with pooled, auditory and visual panels."""
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df = df[df["Hemisphere"] == "bh"].copy()
    df = df[df["Task"] == TASK_NAME].copy()
    df = df[df["Category"].isin(CATEGORIES)].copy()

    roi_values = list(pd.unique(df["ROI"]))
    rois = [_resolve_roi(roi, roi_values) for roi in ROI_ORDER]

    roi_specs = []
    for roi in rois:
        if roi is None:
            roi_specs.append(
                {
                    "roi": None,
                    "y_lim": (-0.2, 0.2),
                    "ticks": np.array([-0.2, 0.0, 0.2]),
                    "row_h": MIN_ROW_HEIGHT,
                }
            )
            continue

        y0, y1, ticks = _box_y_limits(df, roi, y_limits=y_limits)
        n_steps = max(int(np.ceil((y1 - y0) / YTICK_STEP)), 1)
        row_h = max(MIN_ROW_HEIGHT, n_steps * INCHES_PER_STEP)
        roi_specs.append(
            {
                "roi": roi,
                "y_lim": (y0, y1),
                "ticks": ticks,
                "row_h": row_h,
            }
        )

    width_ratios = [1.0, 0.10, 1.0, 0.10, 1.0]
    fig_w = 5.4 * FIG_W_SCALE * figsize_scale
    fig_h = float(sum(spec["row_h"] for spec in roi_specs)) * figsize_scale

    fig, axes = plt.subplots(
        nrows=len(rois),
        ncols=len(width_ratios),
        figsize=(fig_w, fig_h),
        gridspec_kw={
            "width_ratios": width_ratios,
            "height_ratios": [spec["row_h"] for spec in roi_specs],
        },
        sharex=False,
        sharey=False,
    )

    if len(rois) == 1:
        axes = np.expand_dims(axes, axis=0)

    panel_cols = {"Pooled": 0, "Auditory": 2, "Visual": 4}
    spacer_cols = [1, 3]

    fig.subplots_adjust(
        top=0.968,
        left=0.10,
        right=0.98,
        bottom=0.03,
        hspace=1.05,
        wspace=0.04,
    )

    for r, spec in enumerate(roi_specs):
        roi = spec["roi"]
        if roi is None:
            for j in range(len(width_ratios)):
                axes[r, j].axis("off")
            continue

        for j in spacer_cols:
            axes[r, j].axis("off")

        row_axes = []
        for modality, col in panel_cols.items():
            ax = axes[r, col]
            row_axes.append(ax)
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
            ax.set_xticks(PAIR_POS)
            ax.set_xticklabels(X_LABELS, fontsize=XLABEL_FS)
            ax.tick_params(axis="x", length=0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xlabel(
                MOD_LABEL[modality],
                fontsize=AXIS_LABEL_FS,
                labelpad=MOD_LABEL_PAD,
            )

            rkey = _roi_key(roi)
            display_axis = True
            if show_yaxis is not None:
                display_axis = show_yaxis.get(rkey, True)

            if col == 0:
                ax.set_ylabel("PSC (%)", fontsize=AXIS_LABEL_FS)
                ax.tick_params(axis="y", labelsize=YTICK_FS)
            else:
                ax.set_yticklabels([])
                ax.tick_params(axis="y", left=False)
                ax.spines["left"].set_visible(False)

            if not display_axis:
                ax.tick_params(axis="y", left=False, labelleft=False)
                ax.set_yticklabels([])
                ax.spines["left"].set_visible(False)
                ax.set_ylabel("")
                ax.yaxis.label.set_visible(False)
                ax.yaxis.set_ticks_position("none")

        left_ax = row_axes[0]
        right_ax = row_axes[-1]
        x_center = 0.5 * (
            left_ax.get_position().x0 + right_ax.get_position().x1
        )
        y_top = max(ax.get_position().y1 for ax in row_axes) + 0.007
        fig.text(
            x_center,
            y_top,
            ROI_PRETTY[_roi_key(roi)],
            ha="center",
            va="bottom",
            fontsize=ROI_TITLE_FS,
            fontweight="semibold",
        )

    fig.suptitle(
        "95% bootstrap CI for the Median of PSC",
        fontsize=SUPTITLE_FS,
        y=0.992,
    )
    fig.savefig(outpath, dpi=300, bbox_inches="tight", pad_inches=0.10)
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
            "dstr": (-0.6, 2.2),
            "cereb": (-0.6, 2.2),
            "presma": (-0.2, 1.2),
            "sma": (-0.2, 1.2),
            "pmd": (-0.2, 1.2),
            "pmv": (-0.2, 1.2),
        },
        show_yaxis={
            "heschl": True,
            "occipital": False,
            "dstr": False,
            "cereb": False,
            "presma": True,
            "sma": False,
            "pmd": False,
            "pmv": False,
        },
    )