#!/usr/bin/env python3
"""PSC boxplots by ROI (single ROI column) and modality/task blocks.

Author: Ana Luisa Pinho
email: agrilopi@uwo.ca

Created: 28th of January, 2026
Last update: January 2026

Compatibility: Python 3.10.14
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter


# ============================ CONSTANTS ============================ #

TASKS_MAIN = ["Production", "Perception", "NTFD"]
TASK_NTFD_RANDOM = "NTFD Random"

CATS_MAIN = ["Beat", "Interval"]
CATS_NTFD_RANDOM = ["Beat", "Interval", "Random"]

MODALITIES = ["Auditory", "Visual"]
MOD_BLOCKS = ["Pooled", "Auditory", "Visual"]

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
    "dstr": "Dorsal Striatum",
    "cereb": "Cerebellum",
    "presma": "PreSMA",
    "sma": "SMA",
    "pmd": "PMD",
    "pmv": "PMV",
    "heschl": "Heschl’s Gyrus",
    "occipital": "Occipital Lobe",
}

# Geometry: make the 3-boxplot NTFD Random panels wider.
W_RATIO_STD = 1.0
W_RATIO_NTFD_RANDOM = 1.45
W_RATIO_SPACER = 0.20


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


def _roi_token(name: str) -> str:
    """Normalize ROI strings for matching."""
    s = str(name).strip().lower()
    s = s.replace("’", "'")
    for ch in (" ", "_", "-"):
        s = s.replace(ch, "")
    return s


def _roi_key(name: str) -> str | None:
    """Return a canonical ROI key for robust matching.

    Important: avoid substring collisions ("sma" vs "presma").
    """
    tok = _roi_token(name)

    if "presma" in tok or "presupplementary" in tok or "presupp" in tok:
        return "presma"
    if tok == "sma" or "supplementarymotor" in tok or tok.endswith("sma"):
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


def _pretty_roi_label(resolved_roi: str) -> str:
    """Convert resolved ROI name to a pretty label."""
    key = str(resolved_roi).strip().lower()
    for short, pretty in ROI_PRETTY.items():
        if short in key:
            return pretty
    return str(resolved_roi)


def _resolve_roi(name: str, roi_values: Sequence[str]) -> str | None:
    """Resolve a short ROI name (user order) to a value in df."""
    roi_map = {str(r).lower(): r for r in roi_values}
    key = name.lower()
    if key in roi_map:
        return roi_map[key]
    for k, v in roi_map.items():
        if key in k:
            return v
    return None


def _task_categories(task: str) -> List[str]:
    """Return categories to use for a task."""
    if task == TASK_NTFD_RANDOM:
        return list(CATS_NTFD_RANDOM)
    return list(CATS_MAIN)


def _paired_by_subject(
    df: pd.DataFrame,
    roi: str,
    modality: str,
    task: str,
    categories: Sequence[str],
) -> pd.DataFrame:
    """Subject-wise PSC table with one column per category."""
    cols = ["Subject", "PSC", "ROI", "Task", "Category", "Modality"]
    sub = df.loc[
        (df["ROI"] == roi)
        & (df["Task"] == task)
        & (df["Category"].isin(categories)),
        cols,
    ].copy()

    if modality != "Pooled":
        sub = sub[sub["Modality"] == modality]

    sub = (
        sub.groupby(["Subject", "Category"], sort=False)["PSC"]
        .mean()
        .unstack("Category")
    )
    sub = sub.dropna(subset=list(categories), how="any")

    if sub.empty:
        return pd.DataFrame(columns=list(categories))

    return sub[list(categories)]


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

    X uses figure coords (spans subplots). Y anchors to ax_left data coords.
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


# ============================ PLOTTING ============================= #


def plot_psc_boxplots(
    df: pd.DataFrame,
    outpath: str | Path,
    figsize_scale: float = 1.0,
    audivisual_only: bool = False,
    include_ntfd_random: bool = False,
) -> None:
    """Plot PSC boxplots by ROI and modality/task blocks.

    Adds support for an extra task (NTFD Random) shown as a fourth
    panel per modality block with 3 categories (Beat/Interval/Random).
    """
    outpath = Path(outpath)
    if outpath.suffix == "":
        raise ValueError("outpath must end with .png or .pdf")
    outpath.parent.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df = df[df["Hemisphere"] == "bh"].copy()

    if audivisual_only:
        df = df[df["Modality"].isin(MODALITIES)].copy()

    keep_tasks = set(TASKS_MAIN + [TASK_NTFD_RANDOM])
    df = df[df["Task"].isin(keep_tasks)].copy()

    # ------------------ column layout (tasks per block) ------------------
    tasks_per_block = list(TASKS_MAIN)
    if include_ntfd_random:
        tasks_per_block.append(TASK_NTFD_RANDOM)

    def _block(mod: str) -> List[Tuple[str, str]]:
        return [(mod, t) for t in tasks_per_block]

    def _with_spacer(
        parts: List[List[Tuple[str, str]]],
    ) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        for i, p in enumerate(parts):
            if i > 0:
                out.append(("SPACER", "SPACER"))
            out.extend(p)
        return out

    if audivisual_only:
        blocks = [_block("Auditory"), _block("Visual")]
    else:
        blocks = [_block("Pooled"), _block("Auditory"), _block("Visual")]

    col_spec_block = _with_spacer(blocks)

    # ------------------ width ratios (task-dependent) --------------------
    width_ratios: List[float] = []
    for mod, task in col_spec_block:
        if mod == "SPACER":
            width_ratios.append(W_RATIO_SPACER)
        elif task == TASK_NTFD_RANDOM:
            width_ratios.append(W_RATIO_NTFD_RANDOM)
        else:
            width_ratios.append(W_RATIO_STD)

    n_cols = len(col_spec_block)

    # ------------------------ ROI ordering -------------------------
    roi_values = list(pd.unique(df["ROI"]))
    rois = [_resolve_roi(r, roi_values) for r in ROI_ORDER]
    n_rows = len(rois)

    # ------------------------ style params -------------------------
    colors = {
        "Beat": "#6baed6",
        "Interval": "#fdae6b",
        "Random": "tab:pink",
    }
    box_alpha = 0.72
    whis = 1.5

    pos_2 = [1.0, 1.9]
    pos_3 = [1.0, 1.9, 2.8]

    xlim_2 = (0.55, 2.35)
    xlim_3 = (0.55, 3.25)

    box_w = 0.65

    xlabel_fs = 12
    xlabel_pad = 4
    axis_label_fs = xlabel_fs
    ytick_fs = xlabel_fs
    legend_fs = axis_label_fs + 2

    y_formatter = FormatStrFormatter("%.2f")

    ypad_frac = 0.06
    annot_y_frac_base = 0.04
    annot_y_frac_step = 0.09
    annot_h_frac = 0.03
    annot_headroom_frac = 0.01

    roi_annot_overrides = {
        "dstr": {"headroom_frac": 0.002},
        "cereb": {"headroom_frac": 0.002},
        "presma": {"headroom_frac": 0.002},
        "sma": {"headroom_frac": 0.002},
        "heschl": {
            "pad_frac": 0.022,
            "base_frac": 0.022,
            "headroom_frac": 0.003,
        },
        "occipital": {"base_frac": 0.020},
    }

    zero_line_color = "0.25"
    zero_line_ls = "--"
    zero_line_lw = 1.2
    zero_line_zorder = 1

    ytick_step = 0.20
    inches_per_step = 0.68
    min_row_height = 2.0

    task_order = {
        "Production": 0,
        "Perception": 1,
        "NTFD": 2,
        TASK_NTFD_RANDOM: 3,
    }

    def _ann_sort_key(ann: dict) -> Tuple[int, int, int]:
        t1, t2 = ann["task_pair"]
        i1 = task_order.get(t1, 99)
        i2 = task_order.get(t2, 99)
        span = abs(i2 - i1)
        return (span, min(i1, i2), max(i1, i2))

    # ---------------- PASS 1: per-ROI y-lims and row heights --------
    roi_specs: List[dict] = []

    eligible_template = (
        {"Auditory": [], "Visual": []}
        if audivisual_only
        else {"Pooled": [], "Auditory": [], "Visual": []}
    )

    ax_keys = [(m, t) for (m, t) in col_spec_block if m != "SPACER"]

    for roi in rois:
        if roi is None:
            roi_specs.append(
                {
                    "roi": None,
                    "roi_label": "",
                    "y_min": 0.0,
                    "y_max": 1.0,
                    "yr": 1.0,
                    "y_lim": (0.0, 1.0),
                    "y_ticks": np.array([0.0, 1.0]),
                    "eligible_by_mod": {k: [] for k in eligible_template},
                    "row_h": min_row_height,
                }
            )
            continue

        eligible_by_mod: Dict[str, List[dict]] = {
            k: [] for k in eligible_template
        }

        for ann in ANNOTATIONS:
            if not _matches_roi(roi, ann["roi"]):
                continue
            m = ann["modality"]
            t_left, t_right = ann["task_pair"]
            if (m, t_left) not in ax_keys or (m, t_right) not in ax_keys:
                continue
            if m in eligible_by_mod:
                eligible_by_mod[m].append(ann)

        for m in eligible_by_mod:
            eligible_by_mod[m].sort(key=_ann_sort_key)

        w_lows: List[float] = []
        w_highs: List[float] = []
        for mod, task in col_spec_block:
            if mod == "SPACER":
                continue
            cats = _task_categories(task)
            paired = _paired_by_subject(
                df,
                roi=roi,
                modality=mod,
                task=task,
                categories=cats,
            )
            for cat in cats:
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

        rkey = _roi_key(roi)
        ov = roi_annot_overrides.get(rkey or "", {})

        pad_frac_local = float(
            ov.get("pad_frac", (0.045 if rkey == "heschl" else ypad_frac))
        )
        pad = pad_frac_local * yr

        max_stack = max((len(v) for v in eligible_by_mod.values()), default=0)
        headroom_frac_local = float(
            ov.get(
                "headroom_frac",
                (0.006 if rkey == "heschl" else annot_headroom_frac),
            )
        )
        base_frac_local = float(ov.get("base_frac", annot_y_frac_base))

        if max_stack > 0:
            top_needed = (
                base_frac_local
                + (max_stack - 1) * annot_y_frac_step
                + annot_h_frac
                + headroom_frac_local
            )
            top_extra = top_needed * yr
        else:
            top_extra = 0.0

        y_lim_raw = (y_min - pad, y_max + pad + top_extra)

        eps = 1e-9
        y0 = float(np.floor((y_lim_raw[0] + eps) / ytick_step) * ytick_step)
        y1 = float(np.ceil((y_lim_raw[1] - eps) / ytick_step) * ytick_step)

        if (y1 - y_lim_raw[1]) < (0.22 * ytick_step):
            y1_candidate = y1 - ytick_step
            if y1_candidate >= (y_lim_raw[1] - 1e-12):
                y1 = y1_candidate

        y0 = min(y0, 0.0)
        y1 = max(y1, 0.0)

        y_ticks = np.arange(y0, y1 + 0.5 * ytick_step, ytick_step)
        if y_ticks.size < 2:
            y_ticks = np.array([y0, y1])

        n_steps = max(int(y_ticks.size - 1), 1)
        row_h = max(min_row_height, n_steps * inches_per_step)

        roi_specs.append(
            {
                "roi": roi,
                "roi_label": _pretty_roi_label(roi),
                "y_min": y_min,
                "y_max": y_max,
                "yr": yr,
                "pad": pad,
                "y_lim": (y0, y1),
                "y_ticks": y_ticks,
                "ann_base_frac": base_frac_local,
                "ann_step_frac": annot_y_frac_step,
                "ann_h_frac": annot_h_frac,
                "eligible_by_mod": eligible_by_mod,
                "row_h": row_h,
            }
        )

    height_ratios = [d["row_h"] for d in roi_specs]

    # -------------------------- figure size -------------------------
    label_lines: List[str] = []
    for mod, task in col_spec_block:
        if mod == "SPACER":
            continue
        label_lines.append(task if mod == "Pooled" else f"{mod}\n{task}")
    max_line_len = max(
        (
            max(len(s.split("\n")[0]), len(s.split("\n")[-1]))
            for s in label_lines
        ),
        default=8,
    )
    per_col = 1.18 + 0.034 * max(0, max_line_len - 8)
    per_col = min(per_col, 1.55)

    fig_w = per_col * float(sum(width_ratios)) * figsize_scale
    fig_h = float(sum(height_ratios)) * figsize_scale

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(fig_w, fig_h),
        sharey=False,
        gridspec_kw={
            "width_ratios": width_ratios,
            "height_ratios": height_ratios,
        },
    )
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    handles = [
        Patch(
            facecolor=colors["Beat"],
            edgecolor="none",
            alpha=box_alpha,
            label="Beat",
        ),
        Patch(
            facecolor=colors["Interval"],
            edgecolor="none",
            alpha=box_alpha,
            label="Interval",
        ),
    ]

    if include_ntfd_random:
        handles.insert(
            2,
            Patch(
                facecolor=colors["Random"],
                edgecolor="none",
                alpha=box_alpha,
                label="Random",
            ),
        )

    handles.append(
        Line2D(
            [0],
            [0],
            color="k",
            linestyle="--",
            linewidth=2.2,
            label="Mean",
        )
    )

    ncol = 4 if include_ntfd_random else 3
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=ncol,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        fontsize=legend_fs,
        handlelength=3.0,
        columnspacing=2.0,
    )

    fig.subplots_adjust(top=0.975, right=0.975, hspace=0.25, wspace=0.25)

    # --------------------- PASS 2: draw panels ----------------------
    for r, spec in enumerate(roi_specs):
        roi = spec["roi"]

        ax_lookup: Dict[Tuple[str, str], plt.Axes] = {}
        for j2, (m2, t2) in enumerate(col_spec_block):
            if m2 == "SPACER":
                continue
            ax_lookup[(m2, t2)] = axes[r, j2]

        for j, (mod, task) in enumerate(col_spec_block):
            ax = axes[r, j]

            if mod == "SPACER" or roi is None:
                ax.axis("off")
                continue

            cats = _task_categories(task)
            paired = _paired_by_subject(
                df,
                roi=roi,
                modality=mod,
                task=task,
                categories=cats,
            )

            data = [paired[c].to_numpy() for c in cats]

            if len(cats) == 2:
                positions = pos_2
                ax.set_xlim(*xlim_2)
            else:
                positions = pos_3
                ax.set_xlim(*xlim_3)

            bp = ax.boxplot(
                data,
                positions=positions,
                widths=box_w,
                notch=True,
                patch_artist=True,
                showfliers=False,
                showmeans=True,
                meanline=True,
                whis=whis,
                medianprops={"linewidth": 0, "color": "none"},
                meanprops={
                    "linestyle": "--",
                    "linewidth": 2.2,
                    "color": "k",
                },
            )

            for patch, cat in zip(bp["boxes"], cats):
                patch.set_facecolor(colors[cat])
                patch.set_alpha(box_alpha)

            ax.set_ylim(*spec["y_lim"])
            ax.set_yticks(spec["y_ticks"])
            ax.yaxis.set_major_formatter(y_formatter)

            ax.axhline(
                0,
                color=zero_line_color,
                linestyle=zero_line_ls,
                linewidth=zero_line_lw,
                zorder=zero_line_zorder,
            )

            ax.set_xticks([])
            xlabel = task if mod == "Pooled" else f"{mod}\n{task}"
            ax.set_xlabel(xlabel, fontsize=xlabel_fs, labelpad=xlabel_pad)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if j == 0:
                ax.set_ylabel(
                    f"{spec['roi_label']}\nPSC (%)",
                    fontsize=axis_label_fs,
                )
                ax.tick_params(axis="y", labelsize=ytick_fs)
                ax.spines["left"].set_visible(True)
            else:
                ax.set_yticklabels([])
                ax.tick_params(axis="y", left=False, length=0)
                ax.spines["left"].set_visible(False)

        if roi is None:
            continue

        for m, anns in spec["eligible_by_mod"].items():
            if not anns:
                continue
            for level, ann in enumerate(anns):
                t_left, t_right = ann["task_pair"]
                p = float(ann["pvalue"])
                text = pval_label_converter([p])[0]

                ax_left = ax_lookup.get((m, t_left))
                ax_right = ax_lookup.get((m, t_right))
                if ax_left is None or ax_right is None:
                    continue

                y_data = (spec["y_max"] + spec["pad"]) + (
                    spec.get("ann_base_frac", annot_y_frac_base)
                    + level * spec.get("ann_step_frac", annot_y_frac_step)
                ) * spec["yr"]
                h_data = annot_h_frac * spec["yr"]

                span_annotation_datay_figspan(
                    fig,
                    ax_left=ax_left,
                    ax_right=ax_right,
                    text=text,
                    y_data=y_data,
                    h_data=h_data,
                )

    fig.savefig(outpath, dpi=300, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)


# ============================== I/O ================================ #


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
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

ROI_DIR = os.path.join(
    WORKING_DIR,
    f"roi_analyses_{MODEL}_{HRF_CUTOFF}_{MASKING}_puncorr_unsmoothed",
    "bothmod_allmain_tasks",
)

DATA_PATH_MAIN = os.path.join(
    ROI_DIR,
    "main_tasks",
    "df_rois_volume",
    f"dfrois_{INDIVID_LEVEL}_{N_ROIS}-rois.tsv",
)

DATA_PATH_RAND = os.path.join(
    ROI_DIR,
    "rand_ntfd_pairs",
    "df_rois_volume",
    f"dfrois_{INDIVID_LEVEL}_{N_ROIS}-rois.tsv",
)

OUTPUT_PATH = os.path.join(
    WORKING_DIR,
    "roi_analyses_rwls_hrf128_wb_puncorr_unsmoothed",
    "bothmod_allmain_tasks",
    "main_tasks",
    "anova_plots",
    "psc_boxplots_by_roi.png",
)

# ============================ ANNOTATIONS =========================== #
# Each entry defines ONE bracket spanning two task-panels within ONE modality
# block.
#
# roi: short key ("dstr", "sma", "pmv", ...)
# modality: "Pooled" | "Auditory" | "Visual"
# task_pair: (task_left, task_right)
# pvalue: numeric p-value
ANNOTATIONS: List[dict] = [
    dict(
        roi="dstr",
        modality="Pooled",
        task_pair=("Production", "Perception"),
        pvalue=0.000000183218571178129,
    ),
    dict(
        roi="dstr",
        modality="Pooled",
        task_pair=("Production", "NTFD"),
        pvalue=0.000000747501935034951,
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
    dict(
        roi="dstr",
        modality="Auditory",
        task_pair=("Production", "Perception"),
        pvalue=0.0000223113707121616,
    ),
    dict(
        roi="dstr",
        modality="Auditory",
        task_pair=("Production", "NTFD"),
        pvalue=0.0000134800500526281,
    ),
    dict(
        roi="sma",
        modality="Auditory",
        task_pair=("Production", "Perception"),
        pvalue=0.0010309867929799,
    ),
    dict(
        roi="sma",
        modality="Auditory",
        task_pair=("Perception", "NTFD"),
        pvalue=0.0415934416483813,
    ),
    dict(
        roi="heschl",
        modality="Auditory",
        task_pair=("Production", "NTFD"),
        pvalue=0.00000276179676296787,
    ),
    dict(
        roi="heschl",
        modality="Auditory",
        task_pair=("Perception", "NTFD"),
        pvalue=0.00000046329180419678,
    ),
    dict(
        roi="occipital",
        modality="Auditory",
        task_pair=("Production", "Perception"),
        pvalue=0.0114282939876557,
    ),
    dict(
        roi="occipital",
        modality="Auditory",
        task_pair=("Perception", "NTFD"),
        pvalue=0.0002568484106432,
    ),
    dict(
        roi="dstr",
        modality="Visual",
        task_pair=("Production", "Perception"),
        pvalue=0.0000000147957389590551,
    ),
    dict(
        roi="dstr",
        modality="Visual",
        task_pair=("Production", "NTFD"),
        pvalue=0.0000183394276401402,
    ),
    dict(
        roi="cereb",
        modality="Visual",
        task_pair=("Production", "NTFD"),
        pvalue=0.0324187838480206,
    ),
    dict(
        roi="sma",
        modality="Visual",
        task_pair=("Production", "Perception"),
        pvalue=0.0000271705639262566,
    ),
    dict(
        roi="sma",
        modality="Visual",
        task_pair=("Perception", "NTFD"),
        pvalue=0.0160228828086688,
    ),
    dict(
        roi="occipital",
        modality="Visual",
        task_pair=("Production", "Perception"),
        pvalue=0.0134946966485245,
    ),
    dict(
        roi="occipital",
        modality="Visual",
        task_pair=("Production", "NTFD"),
        pvalue=0.02078514755058,
    ),
]


# ============================== RUN ================================ #

if __name__ == "__main__":
    args = parse_args()

    df_main = pd.read_csv(DATA_PATH_MAIN, sep="\t")
    df_rand = pd.read_csv(DATA_PATH_RAND, sep="\t")

    if "Task" in df_rand.columns:
        df_rand = df_rand.copy()
        df_rand["Task"] = df_rand["Task"].replace(
            {"NTFD_Random": TASK_NTFD_RANDOM, "NTFD-Random": TASK_NTFD_RANDOM}
        )

    df_in = pd.concat([df_main, df_rand], ignore_index=True, axis=0)

    # 1) Original figures (no NTFD Random panels)
    plot_psc_boxplots(
        df=df_in,
        outpath=OUTPUT_PATH,
        figsize_scale=args.figscale,
        audivisual_only=False,
        include_ntfd_random=False,
    )

    outpath_av = Path(OUTPUT_PATH)
    outpath_av = outpath_av.with_name(
        outpath_av.stem + "_audivisual_only" + outpath_av.suffix
    )
    plot_psc_boxplots(
        df=df_in,
        outpath=outpath_av,
        figsize_scale=args.figscale,
        audivisual_only=True,
        include_ntfd_random=False,
    )

    # 2) Extended figures (with NTFD Random panels)
    outpath_rand = Path(OUTPUT_PATH)
    outpath_rand = outpath_rand.with_name(
        outpath_rand.stem + "_ntfd_random" + outpath_rand.suffix
    )
    plot_psc_boxplots(
        df=df_in,
        outpath=outpath_rand,
        figsize_scale=args.figscale,
        audivisual_only=False,
        include_ntfd_random=True,
    )

    outpath_rand_av = Path(OUTPUT_PATH)
    outpath_rand_av = outpath_rand_av.with_name(
        outpath_rand_av.stem + "_ntfd_random_audivisual_only"
        + outpath_rand_av.suffix
    )
    plot_psc_boxplots(
        df=df_in,
        outpath=outpath_rand_av,
        figsize_scale=args.figscale,
        audivisual_only=True,
        include_ntfd_random=True,
    )