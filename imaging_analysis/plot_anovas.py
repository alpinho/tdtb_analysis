#!/usr/bin/env python3
"""PSC boxplots by ROI (single ROI column) and modality/task blocks.

Author: Ana Luisa Pinho
email: agrilopi@uwo.ca

Created: 28th of January, 2026
Last update: February 2026

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
#
# NOTE: Absolute panel width in inches is approximately:
#   panel_width ≈ per_col * W_RATIO_*
# because fig_w is computed as per_col * sum(width_ratios).
# To make each "pair-of-boxplots" panel narrower (without
# changing fontsize), reduce W_RATIO_STD (and scale
# W_RATIO_NTFD_RANDOM accordingly to preserve the relative width
# of the 3-box panels).

W_RATIO_STD = 0.95
W_RATIO_NTFD_RANDOM = 1.4455
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


def within_axis_annotation(
    ax: plt.Axes,
    x1: float,
    x2: float,
    text: str,
    y_data: float,
    h_data: float,
    lw: float = 1.2,
    fs: float = 14.0,
) -> None:
    """Draw a bracket within a single axis (data coords).

    Parameters
    ----------
    ax
        Target axis.
    x1, x2
        X positions in data coordinates (e.g., Beat vs Interval).
    text
        Annotation text (typically star label).
    y_data
        Baseline Y position in data coordinates.
    h_data
        Bracket height in data coordinates.
    """
    y0 = y_data
    y1 = y_data + h_data

    ax.plot([x1, x1], [y0, y1], lw=lw, c="k", clip_on=False)
    ax.plot([x1, x2], [y1, y1], lw=lw, c="k", clip_on=False)
    ax.plot([x2, x2], [y1, y0], lw=lw, c="k", clip_on=False)
    ax.text(
        (x1 + x2) / 2.0,
        y1,
        text,
        ha="center",
        va="bottom",
        fontsize=fs,
        color="k",
        clip_on=False,
    )



# ============================ PLOTTING ============================= #


def plot_psc_boxplots(
    df: pd.DataFrame,
    outpath: str | Path,
    figsize_scale: float = 1.0,
    audivisual_only: bool = False,
    pooled_only: bool = False,
    include_ntfd_random: bool = False,
) -> None:
    """Plot PSC boxplots by ROI and modality/task blocks.

    Adds support for an extra task (NTFD Random) shown as a fourth
    panel per modality block with 3 categories (Beat/Interval/Random).
    If pooled_only is True, only the pooled modality block is plotted.
    """
    outpath = Path(outpath)
    if outpath.suffix == "":
        raise ValueError("outpath must end with .png or .pdf")
    outpath.parent.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df = df[df["Hemisphere"] == "bh"].copy()

    if audivisual_only and pooled_only:
        raise ValueError(
            "audivisual_only and pooled_only are mutually exclusive"
        )

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

    if pooled_only:
        blocks = [_block("Pooled")]
    elif audivisual_only:
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

    box_w = 0.65

    # Make boxes touch (but avoid overlap due to linewidths/notches).
    gap = 1.02  # 1.00 = just-touch; >1 adds a small breathing room

    pos_2 = [1.0, 1.0 + gap * box_w]
    pos_3 = [1.0, 1.0 + gap * box_w, 1.0 + 2.0 * gap * box_w]

    xpad = 0.45
    xlim_2 = (pos_2[0] - xpad, pos_2[-1] + xpad)
    xlim_3 = (pos_3[0] - xpad, pos_3[-1] + xpad)

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
    annot_headroom_frac = 0.03
    within_annot_y_frac_base = 0.02
    within_annot_y_frac_step = 0.07
    within_annot_h_frac = 0.02
    within_annot_headroom_frac = 0.012

    within_to_span_gap_frac = 0.11

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

    
    # Slightly increase annotation stacking distance for pooled-only
    # figures to avoid rare bracket overlaps (e.g., Heschl's Gyrus).
    if pooled_only:
        annot_y_frac_step = 0.11
        roi_annot_overrides = dict(roi_annot_overrides)
        h_ov = dict(roi_annot_overrides.get("heschl", {}))
        h_ov["step_frac"] = 0.13
        roi_annot_overrides["heschl"] = h_ov

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

    if pooled_only:
        eligible_template = {"Pooled": []}
    elif audivisual_only:
        eligible_template = {"Auditory": [], "Visual": []}
    else:
        eligible_template = {"Pooled": [], "Auditory": [], "Visual": []}

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

        eligible_within_by_ax: Dict[Tuple[str, str], List[dict]] = {
            (m, t): []
            for (m, t) in ax_keys
            if (m in eligible_template) and (t != "SPACER")
        }

        eligible_cross: List[dict] = []
        
        for ann in ANNOTATIONS:
            if not _matches_roi(roi, ann["roi"]):
                continue
            m = ann["modality"]
            t_left, t_right = ann["task_pair"]
            if (m, t_left) not in ax_keys or (m, t_right) not in ax_keys:
                continue
            if m in eligible_by_mod:
                eligible_by_mod[m].append(ann)

        for ann in WITHIN_ANNOTATIONS:
            if not _matches_roi(roi, ann.get("roi", "")):
                continue
            m = str(ann.get("modality", ""))
            if m not in eligible_template:
                continue

            task_val = ann.get("task", None)
            if task_val is None:
                tasks = list(tasks_per_block)
            else:
                tasks = [str(task_val)]

            for t in tasks:
                key = (m, t)
                if key in eligible_within_by_ax:
                    eligible_within_by_ax[key].append(ann)

        for k in eligible_within_by_ax:
            eligible_within_by_ax[k].sort(key=lambda a: float(a["pvalue"]))

        # Cross-modality (Auditory ↔ Visual) annotations can specify a
        # single task ("task"), a task list ("tasks"), or omit tasks
        # to indicate all tasks in the current figure.
        if (
            ("Auditory" in eligible_template)
            and ("Visual" in eligible_template)
        ):
            for ann in CROSS_AV_ANNOTATIONS:
                if not _matches_roi(roi, ann.get("roi", "")):
                    continue

                tasks_val = ann.get("tasks", None)
                if tasks_val is None:
                    task_single = ann.get("task", None)
                    if task_single is None:
                        tasks = list(tasks_per_block)
                    else:
                        tasks = [str(task_single)]
                else:
                    tasks = [str(t) for t in tasks_val]

                tasks = [t for t in tasks if t in tasks_per_block]
                if not tasks:
                    continue

                if not all(
                    (("Auditory", t) in ax_keys) and (("Visual", t) in ax_keys)
                    for t in tasks
                ):
                    continue

                ann2 = dict(ann)
                ann2["tasks_resolved"] = tasks
                eligible_cross.append(ann2)

            def _cross_sort_key(a: dict) -> Tuple[int, float]:
                idx = [task_order.get(t, 99) for t in a["tasks_resolved"]]
                span = (max(idx) - min(idx)) if idx else 0
                return (-span, float(a["pvalue"]))

            eligible_cross.sort(key=_cross_sort_key)

        for m in eligible_by_mod:
            eligible_by_mod[m].sort(key=_ann_sort_key)

        within_stack_top_by_mod: Dict[str, float] = {
            k: 0.0 for k in eligible_template
        }
        for mm in within_stack_top_by_mod:
            n_within = max(
                (
                    len(v)
                    for (m_ax, _t_ax), v in eligible_within_by_ax.items()
                    if m_ax == mm
                ),
                default=0,
            )
            if n_within > 0:
                within_stack_top_by_mod[mm] = (
                    within_annot_y_frac_base
                    + (n_within - 1) * within_annot_y_frac_step
                    + within_annot_h_frac
                )

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

        max_stack_span = max(
            (len(v) for v in eligible_by_mod.values()),
            default=0,
        )
        max_stack_within = max(
            (len(v) for v in eligible_within_by_ax.values()),
            default=0,
        )
        headroom_frac_local = float(
            ov.get(
                "headroom_frac",
                (0.006 if rkey == "heschl" else annot_headroom_frac),
            )
        )
        base_frac_local = float(ov.get("base_frac", annot_y_frac_base))
        step_frac_local = float(ov.get("step_frac", annot_y_frac_step))
        top_needed_span = 0.0
        span_offset_max = max(
            (
                v
                + (within_to_span_gap_frac if v > 0.0 else 0.0)
                for v in within_stack_top_by_mod.values()
            ),
            default=0.0,
        )
        if max_stack_span > 0:
            top_needed_span = (
                span_offset_max
                + base_frac_local
                + (max_stack_span - 1) * step_frac_local
                + annot_h_frac
                + headroom_frac_local
            )

        # Reserve extra vertical room for cross-modality annotations
        # (Auditory ↔ Visual) drawn between the Auditory and Visual
        # blocks. These p-values typically correspond to ROI × Modality
        # pairwise tests (collapsed over Task).
        max_stack_cross = len(eligible_cross)
        cross_gap_frac = 0.12
        cross_start_frac = 0.0
        if max_stack_cross > 0:
            span_stack_top = span_offset_max
            if max_stack_span > 0:
                span_stack_top = (
                    span_offset_max
                    + base_frac_local
                    + (max_stack_span - 1) * step_frac_local
                    + annot_h_frac
                    + headroom_frac_local
                )
            cross_start_frac = span_stack_top + cross_gap_frac
            top_needed_cross = (
                cross_start_frac
                + base_frac_local
                + (max_stack_cross - 1) * step_frac_local
                + annot_h_frac
                + headroom_frac_local
            )
            top_needed_span = max(top_needed_span, top_needed_cross)

        top_needed_within = 0.0
        if max_stack_within > 0:
            top_needed_within = (
                within_annot_y_frac_base
                + (max_stack_within - 1) * within_annot_y_frac_step
                + within_annot_h_frac
                + within_annot_headroom_frac
            )

        top_extra = max(top_needed_span, top_needed_within) * yr

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
                "ann_step_frac": step_frac_local,
                "ann_h_frac": annot_h_frac,
                "eligible_by_mod": eligible_by_mod,
                "eligible_within_by_ax": eligible_within_by_ax,
                "eligible_cross": eligible_cross,
                "cross_start_frac": cross_start_frac,
                "within_base_frac": within_annot_y_frac_base,
                "within_step_frac": within_annot_y_frac_step,
                "within_h_frac": within_annot_h_frac,
                "within_stack_top_by_mod": within_stack_top_by_mod,
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

    ncol = 3 if include_ntfd_random else 2
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=ncol,
        frameon=False,
        bbox_to_anchor=(0.5, 0.985),
        fontsize=legend_fs,
        handlelength=3.0,
        columnspacing=2.0,
    )

    fig.subplots_adjust(top=0.945, right=0.975, hspace=0.45, wspace=0.1)

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
                showmeans=False,
                whis=whis,
                medianprops={"linewidth": 0, "color": "none"},
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
                    "PSC (%)",
                    fontsize=axis_label_fs,
                )

                ax.tick_params(axis="y", labelsize=ytick_fs)
                ax.spines["left"].set_visible(True)
            else:
                ax.set_yticklabels([])
                ax.tick_params(axis="y", left=False, length=0)
                ax.spines["left"].set_visible(False)
            within_anns = spec.get("eligible_within_by_ax", {}).get(
                (mod, task),
                [],
            )
            if within_anns and ("Beat" in cats) and ("Interval" in cats):
                x1 = float(positions[0])
                x2 = float(positions[1])
                for level, ann in enumerate(within_anns):
                    p = float(ann["pvalue"])
                    text = pval_label_converter([p])[0]
                    y_data = (
                        spec["y_max"]
                        + spec["pad"]
                        + (
                            spec.get(
                                "within_base_frac",
                                within_annot_y_frac_base,
                            )
                            + level
                            * spec.get(
                                "within_step_frac",
                                within_annot_y_frac_step,
                            )
                        )
                        * spec["yr"]
                    )
                    h_data = (
                        spec.get("within_h_frac", within_annot_h_frac)
                        * spec["yr"]
                    )
                    within_axis_annotation(
                        ax=ax,
                        x1=x1,
                        x2=x2,
                        text=text,
                        y_data=y_data,
                        h_data=h_data,
                    )


        # Row-level ROI title centered across all non-spacer panels.
        # Place it in figure coordinates so it is centered per row.
        row_axes = [
            axes[r, jj]
            for jj, (m_jj, _t_jj) in enumerate(col_spec_block)
            if m_jj != "SPACER"
        ]
        if row_axes:
            x0 = min(a.get_position().x0 for a in row_axes)
            x1 = max(a.get_position().x1 for a in row_axes)
            y1 = max(a.get_position().y1 for a in row_axes)

            fig.text(
                (x0 + x1) / 2.0,
                y1 + 0.010,
                spec["roi_label"],
                ha="center",
                va="bottom",
                fontsize=axis_label_fs + 6,
                color="k",
                fontweight="medium",
            )


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

                within_top = float(
                    spec.get("within_stack_top_by_mod", {}).get(m, 0.0)
                )
                gap_frac = within_to_span_gap_frac if within_top > 0.0 else 0.0

                y_data = (spec["y_max"] + spec["pad"]) + (
                    within_top
                    + gap_frac
                    + spec.get("ann_base_frac", annot_y_frac_base)
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

        # Cross-modality (Auditory ↔ Visual) annotations. Each entry in
        # CROSS_AV_ANNOTATIONS can specify a task group via "tasks", a
        # single task via "task", or omit tasks to span all tasks in the
        # current figure.
        eligible_cross = spec.get("eligible_cross", [])
        cross_start_frac = float(spec.get("cross_start_frac", 0.0))
        if eligible_cross and cross_start_frac > 0.0:
            for level, ann in enumerate(eligible_cross):
                tasks = list(ann.get("tasks_resolved", []))
                if not tasks:
                    continue

                tasks = sorted(tasks, key=lambda t: task_order.get(t, 99))
                t_left = tasks[0]
                t_right = tasks[-1]

                ax_aud = ax_lookup.get(("Auditory", t_left))
                ax_vis = ax_lookup.get(("Visual", t_right))
                if ax_aud is None or ax_vis is None:
                    continue

                p = float(ann["pvalue"])
                text = pval_label_converter([p])[0]

                y_data = (spec["y_max"] + spec["pad"]) + (
                    cross_start_frac
                    + spec.get("ann_base_frac", annot_y_frac_base)
                    + level * spec.get("ann_step_frac", annot_y_frac_step)
                ) * spec["yr"]

                h_data = annot_h_frac * spec["yr"]

                span_annotation_datay_figspan(
                    fig,
                    ax_left=ax_aud,
                    ax_right=ax_vis,
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
    "anova_plots",
    "psc_boxplots_by_roi.png",
)

# ===================== WITHIN-SUBPLOT ANNOTATIONS ===================== #
# Each entry defines ONE bracket between Beat and Interval WITHIN a single
# task panel (i.e., within a subplot).
#
# roi: short key ("dstr", "sma", "pmv", ...)
# modality: "Pooled" | "Auditory" | "Visual"
# task: optional; if omitted, applies to all tasks in that block
# pvalue: numeric p-value
WITHIN_ANNOTATIONS: List[dict] = [
#     dict(
#         roi="dstr",
#         modality="Pooled",
#         task="Production", # toy example
#         pvalue=0.0465950036732798,
#     ),
]

# =================== CROSS-MODALITY (AUDIO ↔ VISUAL) ================ #
# Each entry defines ONE bracket spanning Auditory and Visual panels
# for a given ROI. This is typically used for ROI × Modality pairwise
# tests from a 3-way RM ANOVA (collapsed over Task).
#
# roi: short key ("dstr", "sma", "pmv", ...)
# task: optional single task name
# tasks: optional list of tasks to span (group). If both task and
# tasks are omitted, the bracket spans all tasks in the current figure.
# pvalue: numeric p-value
CROSS_AV_ANNOTATIONS: List[dict] = [
    dict(
        roi="pmd",
        tasks=["Production", "Perception", "NTFD"],
        pvalue=0.00382583338647626,
    ),
    dict(
        roi="heschl",
        tasks=["Production", "Perception", "NTFD"],
        pvalue=0.000000000000019175900003915,
    ),
    dict(
        roi="occipital",
        tasks=["Production", "Perception", "NTFD"],
        pvalue=0.00000000290970950214818,
    ),
]

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
        pvalue=0.0000021248924487773,
    ),
    dict(
        roi="dstr",
        modality="Auditory",
        task_pair=("Production", "NTFD"),
        pvalue=0.00000183818864354019,
    ),
        dict(
        roi="cereb",
        modality="Auditory",
        task_pair=("Production", "Perception"),
        pvalue=0.0379419854399417,
    ),
    dict(
        roi="cereb",
        modality="Auditory",
        task_pair=("Production", "NTFD"),
        pvalue=0.0379419854399417,
    ),
    dict(
        roi="sma",
        modality="Auditory",
        task_pair=("Production", "Perception"),
        pvalue=0.000162787388365263,
    ),
    dict(
        roi="sma",
        modality="Auditory",
        task_pair=("Perception", "NTFD"),
        pvalue=0.00489334607628016,
    ),
    dict(
        roi="heschl",
        modality="Auditory",
        task_pair=("Production", "NTFD"),
        pvalue=0.000000240156240258075,
    ),
    dict(
        roi="heschl",
        modality="Auditory",
        task_pair=("Perception", "NTFD"),
        pvalue=0.0000000579114755245975,
    ),
    dict(
        roi="occipital",
        modality="Auditory",
        task_pair=("Production", "Perception"),
        pvalue=0.00126981044307286,
    ),
    dict(
        roi="occipital",
        modality="Auditory",
        task_pair=("Perception", "NTFD"),
        pvalue=0.0000385272615964822,
    ),
    dict(
        roi="dstr",
        modality="Visual",
        task_pair=("Production", "Perception"),
        pvalue=0.00000000184946736988189,
    ),
    dict(
        roi="dstr",
        modality="Visual",
        task_pair=("Production", "NTFD"),
        pvalue=0.00000159473283827306,
    ),
    dict(
        roi="cereb",
        modality="Visual",
        task_pair=("Production", "Perception"),
        pvalue=0.0403525496178749,
    ),
        dict(
        roi="cereb",
        modality="Visual",
        task_pair=("Production", "NTFD"),
        pvalue=0.00540313064133677,
    ),
        dict(
        roi="cereb",
        modality="Visual",
        task_pair=("Perception", "NTFD"),
        pvalue=0.048577005574705,
    ),
    dict(
        roi="sma",
        modality="Visual",
        task_pair=("Production", "Perception"),
        pvalue=0.00000370507689903499,
    ),
    dict(
        roi="sma",
        modality="Visual",
        task_pair=("Production", "NTFD"),
        pvalue=0.0237468150390989,
    ),
    dict(
        roi="sma",
        modality="Visual",
        task_pair=("Perception", "NTFD"),
        pvalue=0.00160228828086688,
    ),
    dict(
        roi="heschl",
        modality="Visual",
        task_pair=("Production", "NTFD"),
        pvalue=0.01626277489796,
    ),
    dict(
        roi="heschl",
        modality="Visual",
        task_pair=("Perception", "NTFD"),
        pvalue=0.0132859760754688,
    ),
    dict(
        roi="occipital",
        modality="Visual",
        task_pair=("Production", "Perception"),
        pvalue=0.00192781380693208,
    ),
    dict(
        roi="occipital",
        modality="Visual",
        task_pair=("Production", "NTFD"),
        pvalue=0.00218791026848211,
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

    # 1b) Pooled-only figure (pooled modality block only)
    outpath_pooled = Path(OUTPUT_PATH)
    outpath_pooled = outpath_pooled.with_name(
        outpath_pooled.stem + "_pooled_only" + outpath_pooled.suffix
    )
    plot_psc_boxplots(
        df=df_in,
        outpath=outpath_pooled,
        figsize_scale=args.figscale,
        pooled_only=True,
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

    # 2b) Pooled-only figure with NTFD Random panels
    outpath_pooled_rand = Path(OUTPUT_PATH)
    outpath_pooled_rand = outpath_pooled_rand.with_name(
        outpath_pooled_rand.stem + "_pooled_only_ntfd_random"
        + outpath_pooled_rand.suffix
    )
    plot_psc_boxplots(
        df=df_in,
        outpath=outpath_pooled_rand,
        figsize_scale=args.figscale,
        pooled_only=True,
        include_ntfd_random=True,
    )