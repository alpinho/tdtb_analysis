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
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, MultipleLocator


# ============================ CONSTANTS ============================ #

TASKS_MAIN = ["Production", "Perception", "NTFD"]
TASK_NTFD_RANDOM = "NTFD Random"

TASK_SHORT = {
    "Production": "Prod",
    "Perception": "Percep",
    "NTFD": "NTFD",
    "NTFD Random": "NTFD\nRandom",
}

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
W_RATIO_STD = 0.9
W_RATIO_NTFD_RANDOM = 1.3694
W_RATIO_SPACER = 0.30

# Scale overall figure width without changing internal layout.
FIG_W_SCALE = 0.50


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
    """Draw a bracket within a single axis (data coords)."""
    y0 = y_data
    y1 = y_data + h_data

    ax.plot([x1, x1], [y0, y1], lw=lw, c="k", clip_on=True)
    ax.plot([x1, x2], [y1, y1], lw=lw, c="k", clip_on=True)
    ax.plot([x2, x2], [y1, y0], lw=lw, c="k", clip_on=True)
    ax.text(
        (x1 + x2) / 2.0,
        y1,
        text,
        ha="center",
        va="bottom",
        fontsize=fs,
        color="k",
        clip_on=True,
    )


# ============================ PLOTTING ============================= #


def plot_psc_boxplots(
    df: pd.DataFrame,
    outpath: str | Path,
    figsize_scale: float = 1.0,
    audivisual_only: bool = False,
    pooled_only: bool = False,
    include_ntfd_random: bool = False,
    y_limits: dict[str, tuple[float, float]] | None = None,
) -> None:
    """Plot PSC boxplots by ROI and modality/task blocks."""
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
    POOLED_BASE = ["steelblue", "darkturquoise", "turquoise", "cyan"]
    AUD_BASE = ["saddlebrown", "chocolate", "orange", "gold"]
    VIS_BASE = ["rebeccapurple", "darkorchid", "violet", "pink"]

    def _shades(base: list[str], n: int) -> list[tuple[float, float, float]]:
        if n <= 0:
            return []
        return [to_rgb(base[i % len(base)]) for i in range(n)]

    def _lighten(
        rgb: tuple[float, float, float],
        amount: float = 0.78,
    ) -> tuple[float, float, float]:
        # Blend toward white by "amount" (0..1). Larger -> lighter.
        r, g, b = rgb
        return (
            min(1.0, r + (1.0 - r) * amount),
            min(1.0, g + (1.0 - g) * amount),
            min(1.0, b + (1.0 - b) * amount),
        )

    n_task_shades = len(tasks_per_block)
    pooled_task_shades = _shades(POOLED_BASE, n_task_shades)
    aud_task_shades = _shades(AUD_BASE, n_task_shades)
    vis_task_shades = _shades(VIS_BASE, n_task_shades)

    task_shade_by_mod: dict[str, dict[str, tuple[float, float, float]]] = {
        "Pooled": dict(zip(tasks_per_block, pooled_task_shades)),
        "Auditory": dict(zip(tasks_per_block, aud_task_shades)),
        "Visual": dict(zip(tasks_per_block, vis_task_shades)),
    }

    def _cat_color(
        mod: str,
        task: str,
        cat: str,
    ) -> tuple[float, float, float]:
        if cat == "Random":
            return (0.70, 0.70, 0.70)

        base = task_shade_by_mod.get(mod, {}).get(task)
        if base is None:
            base = to_rgb("0.5")

        if cat == "Beat":
            return base
        return _lighten(base)

    box_alpha = 1.0
    whis = 1.5

    box_w = 0.48
    center_step = box_w

    pos_2 = [1.0, 1.0 + center_step]
    pos_3 = [1.0, 1.0 + center_step, 1.0 + 2.0 * center_step]

    xpad = 0.4
    xlim_2 = (pos_2[0] - xpad, pos_2[-1] + xpad)
    xlim_3 = (pos_3[0] - xpad, pos_3[-1] + xpad)

    xlabel_fs = 12
    xlabel_pad = 4
    xlabel_rotation = 0

    axis_label_fs = xlabel_fs
    ytick_fs = xlabel_fs

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
        # Explicit params for ALL ROIs (current values).
        #
        # Keys used by the code:
        # - pad_frac (fallback is ypad_frac, 
        #             except heschl fallback 0.045)
        # - base_frac (fallback annot_y_frac_base)
        # - step_frac (fallback annot_y_frac_step)
        # - headroom_frac 
        #   (fallback annot_headroom_frac, 
        #    except heschl fallback 0.006)
        # - step_min_frac / step_max_frac (fallback step_frac)
        # - step_scale_per_layer (currently present in your overrides)

        "dstr": {
            "pad_frac": 0.06,
            "base_frac": 0.04,
            "step_frac": 0.20,
            "step_min_frac": 0.20,
            "step_max_frac": 0.20,
            "step_scale_per_layer": 0.0,
            "headroom_frac": 0.0,
        },
        "cereb": {
            "pad_frac": 0.06,
            "base_frac": 0.04,
            "step_frac": 0.20,
            "step_min_frac": 0.20,
            "step_max_frac": 0.20,
            "step_scale_per_layer": 0.0,
            "headroom_frac": 0.0,
        },
        # "presma": {
        #     "pad_frac": 0.06,
        #     "base_frac": 0.04,
        #     "step_frac": 0.09,
        #     "step_min_frac": 0.09,
        #     "step_max_frac": 0.09,
        #     "step_scale_per_layer": 0.0,
        #     "headroom_frac": 0.0,
        # },
        "sma": {
            "pad_frac": 0.06,
            "base_frac": 0.04,
            "step_frac": 0.20,
            "step_min_frac": 0.20,
            "step_max_frac": 0.20,
            "step_scale_per_layer": 0.0,
            "headroom_frac": 0.0,
        },
        "pmd": {
            "pad_frac": 0.06,
            "base_frac": 0.04,
            "step_frac": 0.09,
            "step_min_frac": 0.09,
            "step_max_frac": 0.09,
            "step_scale_per_layer": 0.0,
            "headroom_frac": 0.0,
        },
        "pmv": {
            "pad_frac": 0.06,
            "base_frac": 0.04,
            "step_frac": 0.09,
            "step_min_frac": 0.09,
            "step_max_frac": 0.09,
            "step_scale_per_layer": 0.0,
            "headroom_frac": 0.0,
        },
        "heschl": {
            "pad_frac": 0.020,
            "base_frac": 0.020,
            "step_frac": 0.05,
            "step_min_frac": 0.10,
            "step_max_frac": 0.10,
            "step_scale_per_layer": 0.25,
            "headroom_frac": 0.0,
        },
        "occipital": {
            "pad_frac": 0.06,
            "base_frac": 0.020,
            "step_frac": 0.16,
            "step_min_frac": 0.16,
            "step_max_frac": 0.16,
            "step_scale_per_layer": 0.0,
            "headroom_frac": 0.03,
        },
    }

    # ------------------------------------------------------------------
    # Legacy tuning block (kept for reference; disabled by request)
    # ------------------------------------------------------------------
    # roi_annot_overrides.update({
    #     "heschl": {
    #         "pad_frac": 0.020,
    #         "base_frac": 0.020,
    #         "step_frac": 0.05,               # default
    #         "step_min_frac": 0.10,           # compact when few layers
    #         "step_max_frac": 0.1,            # avoid huge gaps
    #         "step_scale_per_layer": 0.25,    # grow when many layers
    #         "headroom_frac": 0.,
    #     },
    #     "occipital": {
    #         "base_frac": 0.020,
    #         "step_frac": 0.16,
    #         "step_min_frac": 0.16,
    #         "step_max_frac": 0.16,
    #         "step_scale_per_layer": 0.0,
    #     },
    # })


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

        if ("Auditory" in eligible_template) and (
            "Visual" in eligible_template
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
                v + (within_to_span_gap_frac if v > 0.0 else 0.0)
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

        explicit_ylim = None
        if (y_limits is not None) and (rkey is not None):
            explicit_ylim = y_limits.get(rkey)

        if explicit_ylim is not None:
            y0, y1 = float(explicit_ylim[0]), float(explicit_ylim[1])
            y_ticks = np.arange(y0, y1 + 0.5 * ytick_step, ytick_step)
            if y_ticks.size < 2:
                y_ticks = np.array([y0, y1])
        else:
            eps = 1e-9
            y0 = float(
                np.floor((y_lim_raw[0] + eps) / ytick_step) * ytick_step
            )
            y1 = float(
                np.ceil((y_lim_raw[1] - eps) / ytick_step) * ytick_step
            )

            if (y1 - y_lim_raw[1]) < (0.22 * ytick_step):
                y1_candidate = y1 - ytick_step
                if y1_candidate >= (y_lim_raw[1] - 1e-12):
                    y1 = y1_candidate

            y0 = min(y0, 0.0)
            y1 = max(y1, 0.0)

            y_ticks = np.arange(y0, y1 + 0.5 * ytick_step, ytick_step)
            if y_ticks.size < 2:
                y_ticks = np.array([y0, y1])

        n_steps = max(int(np.ceil((y1 - y0) / ytick_step)), 1)
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
                "ann_step_min_frac": float(
                    ov.get("step_min_frac", step_frac_local)
                ),
                "ann_step_max_frac": float(
                    ov.get("step_max_frac", step_frac_local)
                ),
                "ann_step_scale_per_layer": float(
                    ov.get("step_scale_per_level", 0.0)
                ),
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

    fig_w = per_col * float(sum(width_ratios)) * figsize_scale * FIG_W_SCALE
    fig_h = float(sum(height_ratios)) * figsize_scale


    fig, axes = plt.subplots(
        nrows=len(roi_specs),
        ncols=n_cols,
        figsize=(fig_w, fig_h),
        gridspec_kw={
            "width_ratios": width_ratios,
            "height_ratios": height_ratios,
        },
        sharex=False,
        sharey=False,
    )
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    mods_present: List[str] = []
    cols_by_mod: Dict[str, List[int]] = {}
    for j, (mod, _task) in enumerate(col_spec_block):
        if mod == "SPACER":
            continue
        if mod not in mods_present:
            mods_present.append(mod)
        cols_by_mod.setdefault(mod, []).append(j)

    fig.subplots_adjust(
        top=0.965,
        left=0.015,
        right=0.990,
        hspace=0.75,
        wspace=0.22,
    )


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
                patch.set_facecolor(_cat_color(mod, task, cat))
                patch.set_alpha(box_alpha)

            ax.set_ylim(*spec["y_lim"])
            ax.yaxis.set_major_locator(MultipleLocator(ytick_step))
            ax.spines["left"].set_bounds(*spec["y_lim"])
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
            xlabel = TASK_SHORT.get(task, task)
            ax.set_xlabel(
                xlabel,
                fontsize=xlabel_fs,
                labelpad=xlabel_pad,
                rotation=xlabel_rotation,
                ha="center",
                rotation_mode="anchor",
            )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if j == 0:
                ax.set_ylabel("PSC (%)", fontsize=axis_label_fs)
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

        # Modality block labels under each block, below the task xlabels.
        if not getattr(fig, "_modlabel_canvas_drawn", False):
            fig.canvas.draw()
            fig._modlabel_canvas_drawn = True

        renderer = fig.canvas.get_renderer()
        gap_fig = 0.006

        for mod in mods_present:
            cols = cols_by_mod.get(mod, [])
            if not cols:
                continue

            row_axes_mod = [axes[r, jj] for jj in cols]
            x0m = min(a.get_position().x0 for a in row_axes_mod)
            x1m = max(a.get_position().x1 for a in row_axes_mod)
            x_center = (x0m + x1m) / 2.0

            ax_ref = row_axes_mod[0]
            bb = ax_ref.xaxis.get_label().get_window_extent(renderer=renderer)
            label_h = float(bb.height) / float(fig.bbox.height)
            y = float(ax_ref.get_position().y0) - label_h - gap_fig

            fig.text(
                x_center,
                y,
                mod,
                ha="center",
                va="top",
                fontsize=axis_label_fs + 2,
                color="k",
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

                # Adaptive vertical spacing: ...
                # ... tighter when few spans, larger when many.
                n_layers = len(anns)
                step_default = spec.get("ann_step_frac", annot_y_frac_step)
                step_min = float(spec.get("ann_step_min_frac", step_default))
                step_max = float(spec.get("ann_step_max_frac", step_default))
                step_scale = float(spec.get("ann_step_scale_per_layer", 0.0))

                step_frac = step_default * (
                    1.0 + step_scale * max(0, n_layers - 1)
                )
                step_frac = max(step_min, min(step_max, step_frac))

                y_data = (spec["y_max"] + spec["pad"]) + (
                    within_top
                    + gap_frac
                    + spec.get("ann_base_frac", annot_y_frac_base)
                    + level * step_frac
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

                n_layers = len(eligible_cross)
                step_default = spec.get("ann_step_frac", annot_y_frac_step)
                step_min = float(spec.get("ann_step_min_frac", step_default))
                step_max = float(spec.get("ann_step_max_frac", step_default))
                step_scale = float(spec.get("ann_step_scale_per_layer", 0.0))

                step_frac = step_default * (
                    1.0 + step_scale * max(0, n_layers - 1)
                )
                step_frac = max(step_min, min(step_max, step_frac))

                y_data = (spec["y_max"] + spec["pad"]) + (
                    cross_start_frac
                    + spec.get("ann_base_frac", annot_y_frac_base)
                    + level * step_frac
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

# ============= WITHIN-SUBPLOT ANNOTATIONS (BEAT ↔ INTERVAL) ======== #

WITHIN_ANNOTATIONS: List[dict] = [
    # dict(
    #     roi="dstr",
    #     modality="Pooled",
    #     task="Production",
    #     pvalue=0.0465950036732798,
    # ),
]

# =================== CROSS-MODALITY (AUDIO ↔ VISUAL) =============== #

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

# ========================== CROSS-TASKS ============================ #

# Pooled refer three-way ANOVA for ROI and Task,...
# ... collapsing across modalities
# For separate modalities, we also did 2way RM ANOVA pairwise comparisons

# Reporting uncorrected pvalues that survived to ...
# ... pHolm-Bonferroni correction across the 3way ANOVA
ANNOTATIONS: List[dict] = [
    dict(
        roi="dstr",
        modality="Pooled",
        task_pair=("Production", "Perception"),
        pvalue=0.00000000832811687173313,
    ),
    dict(
        roi="dstr",
        modality="Pooled",
        task_pair=("Production", "NTFD"),
        pvalue=0.0000000355953302397596,
    ),
    dict(
        roi="cereb",
        modality="Pooled",
        task_pair=("Production", "NTFD"),
        pvalue=0.000241837834093381,
    ),
    dict(
        roi="presma",
        modality="Pooled",
        task_pair=("Perception", "NTFD"),
        pvalue=0.00187468625505287,
    ),
    dict(
        roi="sma",
        modality="Pooled",
        task_pair=("Production", "Perception"),
        pvalue=0.000000650783541156439,
    ),
    dict(
        roi="sma",
        modality="Pooled",
        task_pair=("Perception", "NTFD"),
        pvalue=0.0000202682809212273,
    ),
    dict(
        roi="heschl",
        modality="Pooled",
        task_pair=("Production", "NTFD"),
        pvalue=0.00000000136229812975116,
    ),
    dict(
        roi="heschl",
        modality="Pooled",
        task_pair=("Perception", "NTFD"),
        pvalue=0.00000000236090454108804,
    ),
    dict(
        roi="occipital",
        modality="Pooled",
        task_pair=("Production", "Perception"),
        pvalue=0.000134046234216341,
    ),
    dict(
        roi="occipital",
        modality="Pooled",
        task_pair=("Perception", "NTFD"),
        pvalue=0.000029715961625874,
    ),
    dict(
        roi="dstr",
        modality="Auditory",
        task_pair=("Production", "Perception"),
        pvalue=0.00000106244622438865,
    ),
    dict(
        roi="dstr",
        modality="Auditory",
        task_pair=("Production", "NTFD"),
        pvalue=0.000000612729547846731,
    ),
    dict(
        roi="cereb",
        modality="Auditory",
        task_pair=("Production", "Perception"),
        pvalue=0.0142145961772677,
    ),
    dict(
        roi="cereb",
        modality="Auditory",
        task_pair=("Production", "NTFD"),
        pvalue=0.0126473284799806,
    ),
    dict(
        roi="sma",
        modality="Auditory",
        task_pair=("Production", "Perception"),
        pvalue=0.000054262462788421,
    ),
    dict( # significant only in the 3way ANOVA
        roi="sma",
        modality="Auditory",
        task_pair=("Production", "NTFD"),
        pvalue=0.0374092539730202,
    ),
    dict(
        roi="sma",
        modality="Auditory",
        task_pair=("Perception", "NTFD"),
        pvalue=0.00244667303814008,
    ),
    dict(
        roi="heschl",
        modality="Auditory",
        task_pair=("Production", "NTFD"),
        pvalue=0.000000120078120129038,
    ),
    dict(
        roi="heschl",
        modality="Auditory",
        task_pair=("Perception", "NTFD"),
        pvalue=0.0000000193038251748658,
    ),
    dict(
        roi="occipital",
        modality="Auditory",
        task_pair=("Production", "Perception"),
        pvalue=0.000634905221536432,
    ),
    dict(
        roi="occipital",
        modality="Auditory",
        task_pair=("Perception", "NTFD"),
        pvalue=0.0000128424205321607,
    ),
    dict(
        roi="dstr",
        modality="Visual",
        task_pair=("Production", "Perception"),
        pvalue=0.000000000616489123293964,
    ),
    dict(
        roi="dstr",
        modality="Visual",
        task_pair=("Production", "NTFD"),
        pvalue=0.00000079736641913653,
    ),
    dict(
        roi="cereb",
        modality="Visual",
        task_pair=("Production", "Perception"),
        pvalue=0.0201762748089375,
    ),
    dict(
        roi="cereb",
        modality="Visual",
        task_pair=("Production", "NTFD"),
        pvalue=0.00180104354711226,
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
        pvalue=0.00000123502563301166,
    ),
    dict( # significant only in the 3way ANOVA
        roi="sma",
        modality="Visual",
        task_pair=("Production", "NTFD"),
        pvalue=0.0237468150390989,
    ),
    dict(
        roi="sma",
        modality="Visual",
        task_pair=("Perception", "NTFD"),
        pvalue=0.000801144140433441,
    ),
    dict(
        roi="heschl",
        modality="Visual",
        task_pair=("Production", "NTFD"),
        pvalue=0.00813138744897999,
    ),
    dict(
        roi="heschl",
        modality="Visual",
        task_pair=("Perception", "NTFD"),
        pvalue=0.00442865869182293,
    ),
    dict(
        roi="occipital",
        modality="Visual",
        task_pair=("Production", "Perception"),
        pvalue=0.000642604602310693,
    ),
    dict(
        roi="occipital",
        modality="Visual",
        task_pair=("Production", "NTFD"),
        pvalue=0.00109395513424105,
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

    # # 1) Original figures (no NTFD Random panels)
    # plot_psc_boxplots(
    #     df=df_in,
    #     outpath=OUTPUT_PATH,
    #     figsize_scale=args.figscale,
    #     audivisual_only=False,
    #     include_ntfd_random=False,
    # )

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
        y_limits={
            "occipital": (-0.6, 2.2),
            "cereb": (-0.4, 1.4),
            "presma": (-0.4, 1.2),
            "sma": (-0.4, 1.2),
            "pmv": (-0., 1.),
        },
    )

    # # 1b) Pooled-only figure (pooled modality block only)
    # outpath_pooled = Path(OUTPUT_PATH)
    # outpath_pooled = outpath_pooled.with_name(
    #     outpath_pooled.stem + "_pooled_only" + outpath_pooled.suffix
    # )
    # plot_psc_boxplots(
    #     df=df_in,
    #     outpath=outpath_pooled,
    #     figsize_scale=args.figscale,
    #     pooled_only=True,
    #     include_ntfd_random=False,
    # )

    # # 2) Extended figures (with NTFD Random panels)
    # outpath_rand = Path(OUTPUT_PATH)
    # outpath_rand = outpath_rand.with_name(
    #     outpath_rand.stem + "_ntfd_random" + outpath_rand.suffix
    # )
    # plot_psc_boxplots(
    #     df=df_in,
    #     outpath=outpath_rand,
    #     figsize_scale=args.figscale,
    #     audivisual_only=False,
    #     include_ntfd_random=True,
    # )

    # outpath_rand_av = Path(OUTPUT_PATH)
    # outpath_rand_av = outpath_rand_av.with_name(
    #     outpath_rand_av.stem + "_ntfd_random_audivisual_only"
    #     + outpath_rand_av.suffix
    # )
    # plot_psc_boxplots(
    #     df=df_in,
    #     outpath=outpath_rand_av,
    #     figsize_scale=args.figscale,
    #     audivisual_only=True,
    #     include_ntfd_random=True,
    # )

    # # 2b) Pooled-only figure with NTFD Random panels
    # outpath_pooled_rand = Path(OUTPUT_PATH)
    # outpath_pooled_rand = outpath_pooled_rand.with_name(
    #     outpath_pooled_rand.stem + "_pooled_only_ntfd_random"
    #     + outpath_pooled_rand.suffix
    # )
    # plot_psc_boxplots(
    #     df=df_in,
    #     outpath=outpath_pooled_rand,
    #     figsize_scale=args.figscale,
    #     pooled_only=True,
    #     include_ntfd_random=True,
    # )