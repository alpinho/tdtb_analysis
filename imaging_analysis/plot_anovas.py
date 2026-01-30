#!/usr/bin/env python3
"""
PSC boxplots by ROI (single ROI column) and 
modality/task blocks (columns).

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
from typing import Dict, List, Tuple

import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter


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

    sub = (
        sub.groupby(["Subject", "Category"], sort=False)["PSC"]
        .mean()
        .unstack("Category")
    )

    sub = sub.dropna(subset=CATEGORIES, how="any")

    if sub.empty:
        return pd.DataFrame(columns=CATEGORIES)

    return sub[CATEGORIES]


def plot_psc_boxplots(
    df: pd.DataFrame,
    outpath: str | Path,
    figsize_scale: float = 1.0,
    audivisual_only: bool = False,
) -> None:
    """Plot PSC by ROI (single column) with constant y-step and constant pixels per step.

    Requested changes:
      1) Single ROI column ordered: dstr, cerebellum, presma, sma, pmd, pmv, heschl, occipital.
      2) Same y tick step size across plots AND same pixel distance per step across plots
         by scaling each row's height proportional to its y-range (after padding/headroom).
      3) Generous pixel spacing per step for readability.
    """
    outpath = Path(outpath)
    if outpath.suffix == "":
        raise ValueError("outpath must end with .png or .pdf")
    outpath.parent.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df = df[df["Hemisphere"] == "bh"]
    df = df[df["Task"].isin(TASKS)]
    df = df[df["Category"].isin(CATEGORIES)]
    # If requested, restrict to Auditory+Visual only (no pooled panels).
    # Note: "Pooled" panels are conceptual (they aggregate across modalities)
    # and do not rely on a "Pooled" value in df["Modality"].
    if audivisual_only:
        df = df[df["Modality"].isin(["Auditory", "Visual"])]    # --- column spec ---
    # By default we plot: Pooled + Auditory + Visual.
    # If audivisual_only=True, we plot: Auditory + Visual only.
    if audivisual_only:
        col_spec_block = [
            ("Auditory", "Production"),
            ("Auditory", "Perception"),
            ("Auditory", "NTFD"),
            ("SPACER", "SPACER"),
            ("Visual", "Production"),
            ("Visual", "Perception"),
            ("Visual", "NTFD"),
        ]
        width_ratios = [1, 1, 1, 0.20, 1, 1, 1]
    else:
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
        width_ratios = [1, 1, 1, 0.20, 1, 1, 1, 0.20, 1, 1, 1]
    n_cols = len(col_spec_block)

    # --- ROI order (single column) ---
    roi_order_user = ["dstr", "cereb", "presma", "sma", "pmd", "pmv", "heschl", "occipital"]
    roi_map = {str(r).lower(): r for r in pd.unique(df["ROI"])}

    def _resolve_roi(name: str) -> str | None:
        key = name.lower()
        if key in roi_map:
            return roi_map[key]
        for k, v in roi_map.items():
            if key in k:
                return v
        return None

    rois = [_resolve_roi(r) for r in roi_order_user]
    n_rows = len(rois)

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

    
    def _roi_token(name: str) -> str:
        """Normalize ROI strings for matching."""
        s = str(name).strip().lower()
        s = s.replace("’", "'")
        s = s.replace(" ", "")
        s = s.replace("_", "")
        s = s.replace("-", "")
        return s

    def _roi_key(name: str) -> str | None:
        """
        Canonical ROI key for robust matching.

        IMPORTANT: 'presma' must NOT be matched by 'sma' (substring issue),
        so we check 'presma' before 'sma'.
        """
        tok = _roi_token(name)

        # Order matters (avoid substring collisions)
        if "presma" in tok or "presupplementary" in tok or "pre-sma" in tok:
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

    def _ydata_to_yfig(fig, ax, y_data: float) -> float:
        """Convert y in ax data coords to figure fraction coords."""
        x0 = float(np.mean(ax.get_xlim()))
        x_disp, y_disp = ax.transData.transform((x0, y_data))
        _, y_fig = fig.transFigure.inverted().transform((x_disp, y_disp))
        return float(y_fig)

    def span_annotation_datay_figspan(
        fig,
        ax_left,
        ax_right,
        text: str,
        y_data: float,
        h_data: float,
        lw: float = 1.4,
        fs: float = 14,
        ) -> None:
        """
        Draw a bracket spanning ax_left -> ax_right.

        X is in figure coords (so it spans subplots).
        Y is anchored to ax_left data coords (so it stays attached).
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

    # --- plotting params ---
    colors = {
        "Beat": "#6baed6",
        "Interval": "#fdae6b",
    }
    BOX_ALPHA = 0.72

    pos = [1.0, 1.9]
    x_min, x_max = 0.55, 2.35
    box_w = 0.65
    whis = 1.5

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

    # ROI-specific overrides to control vertical whitespace around annotations.
    # Keys refer to canonical ROI keys returned by _roi_key().
    roi_annot_overrides = {
        # Tighten the headroom above the TOP annotation.
        "dstr": {"headroom_frac": 0.002},
        "cereb": {"headroom_frac": 0.002},
        "presma": {"headroom_frac": 0.002},
        "sma": {"headroom_frac": 0.002},
        # Heschl: tighten both headroom and the gap between whisker and lower annotation.
        "heschl": {"pad_frac": 0.022, "base_frac": 0.022, "headroom_frac": 0.003},
        # Occipital: tighten the gap between whisker and lower annotation.
        "occipital": {"base_frac": 0.020},
    }

    ZERO_LINE_COLOR = "0.25"
    ZERO_LINE_LS = "--"
    ZERO_LINE_LW = 1.2
    ZERO_LINE_ZORDER = 1

    # ---------- GLOBAL Y SCALE CONTROL ----------
    ytick_step = 0.20
    inches_per_step = 0.68
    min_row_height = 2.0

    task_order = {"Production": 0, "Perception": 1, "NTFD": 2}

    def _ann_sort_key(ann):
        t1, t2 = ann["task_pair"]
        i1 = task_order[t1]
        i2 = task_order[t2]
        span = abs(i2 - i1)
        return (span, min(i1, i2), max(i1, i2))

    # ---------- PASS 1: compute per-ROI y-lims + ticks + row heights ----------
    roi_specs: List[dict] = []

    for roi in rois:
        if roi is None:
            roi_specs.append({
                "roi": None,
                "roi_label": "",
                "y_min": 0.0,
                "y_max": 1.0,
                "yr": 1.0,
                "y_lim": (0.0, 1.0),
                "y_ticks": np.array([0.0, 1.0]),
                "eligible_by_mod": ({"Auditory": [], "Visual": []} if audivisual_only else {"Pooled": [], "Auditory": [], "Visual": []}),
                "row_h": min_row_height,
            })
            continue

        ax_keys = [(m, t) for (m, t) in col_spec_block if m != "SPACER"]

        eligible_by_mod: Dict[str, List[dict]] = (
            {"Auditory": [], "Visual": []}
            if audivisual_only
            else {"Pooled": [], "Auditory": [], "Visual": []}
        )
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

        rkey = _roi_key(roi)
        ov = roi_annot_overrides.get(rkey or "", {})

        pad_frac_local = float(ov.get("pad_frac", (0.045 if rkey == "heschl" else ypad_frac)))
        pad = pad_frac_local * yr

        max_stack = max((len(v) for v in eligible_by_mod.values()), default=0)
        headroom_frac_local = float(ov.get(
            "headroom_frac",
            (0.006 if rkey == "heschl" else annot_headroom_frac),
        ))
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

        # Avoid an extra full tick-step of empty headroom caused by ceil() when the
        # required upper bound is very close to the next tick.
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

        roi_specs.append({
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
        })

    height_ratios = [d["row_h"] for d in roi_specs]

    # ---------- figure size ----------
    label_lines = []
    for mod, task in col_spec_block:
        if mod == "SPACER":
            continue
        label_lines.append(task if mod == "Pooled" else f"{mod}\n{task}")
    max_line_len = max((max(len(s.split("\n")[0]), len(s.split("\n")[-1])) for s in label_lines), default=8)
    per_col = 1.18 + 0.034 * max(0, max_line_len - 8)
    per_col = min(per_col, 1.55)

    fig_w = per_col * float(sum(width_ratios)) * figsize_scale
    fig_h = float(sum(height_ratios)) * figsize_scale

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(fig_w, fig_h),
        sharey=False,
        gridspec_kw={"width_ratios": width_ratios, "height_ratios": height_ratios},
    )
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    handles = [
        Patch(facecolor=colors["Beat"], edgecolor="none", alpha=BOX_ALPHA, label="Beat"),
        Patch(facecolor=colors["Interval"], edgecolor="none", alpha=BOX_ALPHA, label="Interval"),
        Line2D([0], [0], color="k", linestyle="--", linewidth=2.2, label="Mean"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        fontsize=legend_fs,
        handlelength=3.0,
        columnspacing=2.0,
    )
    # Row heights vary; keep a small but non-zero hspace to prevent overlap.
    # (0.10–0.15 is typically safe with the generous inches_per_step setting.)
    fig.subplots_adjust(top=0.975, hspace=0.25, wspace=0.25)

    # ---------- PASS 2: draw ----------
    for r, spec in enumerate(roi_specs):
        roi = spec["roi"]

        ax_lookup: Dict[Tuple[str, str], plt.Axes] = {}
        for j2, (m2, t2) in enumerate(col_spec_block):
            if m2 == "SPACER":
                continue
            ax_lookup[(m2, t2)] = axes[r, j2]

        for j, (mod, task) in enumerate(col_spec_block):
            ax = axes[r, j]

            if mod == "SPACER":
                ax.axis("off")
                continue

            if roi is None:
                ax.axis("off")
                continue

            paired = _paired_by_subject(df, roi=roi, modality=mod, task=task)
            data = [paired["Beat"].to_numpy(), paired["Interval"].to_numpy()]

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
                medianprops={"linewidth": 0, "color": "none"},
                meanprops={"linestyle": "--", "linewidth": 2.2, "color": "k"},
            )

            for patch, cat in zip(bp["boxes"], CATEGORIES):
                patch.set_facecolor(colors[cat])
                patch.set_alpha(BOX_ALPHA)

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(*spec["y_lim"])
            ax.set_yticks(spec["y_ticks"])
            ax.yaxis.set_major_formatter(y_formatter)

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

            if j == 0:
                ax.set_ylabel(f"{spec['roi_label']}\nPSC (%)", fontsize=axis_label_fs)
                ax.tick_params(axis="y", labelsize=ytick_fs)
                ax.spines['left'].set_visible(True)
            else:
                ax.set_yticklabels([])
                ax.tick_params(axis='y', left=False, length=0)
                ax.spines['left'].set_visible(False)

        if roi is None:
            continue

        for m, anns in spec["eligible_by_mod"].items():
            if not anns:
                continue

            for level, ann in enumerate(anns):
                t_left, t_right = ann["task_pair"]
                p = ann["pvalue"]
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
                    ax_left,
                    ax_right,
                    text=text,
                    y_data=y_data,
                    h_data=h_data,
                    lw=1.2,
                    fs=14,
                )

    fig.savefig(outpath, dpi=200, bbox_inches="tight")
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

# output_path = os.path.join(
#     working_dir,
#     "results",
#     "fig4",
#     "psc_boxplots_by_roi.png",
# )

output_path = os.path.join(
    working_dir,
    "roi_analyses_rwls_hrf128_wb_puncorr_unsmoothed",
    "bothmod_allmain_tasks",
    "main_tasks",
    "anova_plots",
    "psc_boxplots_by_roi.png"
)

# ======================== ANNOTATIONS =========================== #
# Each entry defines ONE bracket spanning two task-panels within ONE modality
# block. "Pooled" refers to the left-most block (both modalities pooled).
#
# roi: short key ("dstr", "sma", "pmv", ...)
# modality: "Pooled" | "Auditory" | "Visual"
# task_pair: ("Production", "Perception", "NTFD")
# pvalue: numeric p-value
ANNOTATIONS = [
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

# ============================= RUN ================================= #

if __name__ == "__main__":
    args = parse_args()
    df_in = pd.read_csv(data_path, sep="\t")
    plot_psc_boxplots(
        df=df_in,
        outpath=output_path,
        figsize_scale=args.figscale,
        audivisual_only=False,
    )

    # Also export an Auditory+Visual-only version (no pooled panels).
    outpath2 = Path(output_path)
    outpath2 = outpath2.with_name(
        outpath2.stem + "_audivisual_only" + outpath2.suffix)
    plot_psc_boxplots(
        df=df_in,
        outpath=outpath2,
        figsize_scale=args.figscale,
        audivisual_only=True,
    )