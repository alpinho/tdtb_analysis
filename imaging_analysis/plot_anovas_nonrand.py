#!/usr/bin/env python3
"""PSC boxplots by ROI for rand_ntfd_nonrandom.

Author: Ana Luisa Pinho
email: agrilopi@uwo.ca

Created: 15th of April 2026
Last update: July 2026

Compatibility: Python 3.10.14

Notes
-----
This script mirrors ``plot_anovas_all.py`` in two respects requested for
consistency between the two figures:

  * The geometry of the significance brackets (within-axis Non-Random vs
    Random, and cross-modality Auditory vs Visual) uses the same tick-based,
    ylim-independent model as ``plot_anovas_all.py`` (a bracket cap is a
    constant number of y-tick units, hence a constant size on the page).
  * The legend at the top uses the same point-based vertical-spacing model and
    layout as ``plot_anovas_all.py`` (caption + two swatch rows + labels).

The boxplot palette (``MODALITY_COLORS``), the variables (Non-Random / Random),
the bootstrap notches and the mean-line styling are kept unchanged from the
original ``plot_anovas_nonrand.py``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import numpy as np
import pandas as pd


# ============================ CONSTANTS ============================ #

TASK_NAME = "NTFD Random"
CATEGORIES = ["Non-Random", "Random"]
MODALITY_BLOCKS = ["Pooled", "Auditory", "Visual"]
MODALITY_BLOCKS_SENSORY = ["Auditory", "Visual"]

MOD_LABEL = {
    "Pooled": "Both\nModalities",
    "Auditory": "Auditory",
    "Visual": "Visual",
}

# When a row mixes the 2-line "Both Modalities" with single-line labels, the
# single-line labels are pushed down one line so all baselines align.
MOD_LABEL_CENTERED = {
    "Pooled": "Both\nModalities",
    "Auditory": "\nAuditory",
    "Visual": "\nVisual",
}

ROI_ORDER = [
    "dstr",
    "cereb",
    "presma",
    "sma",
    "pmd",
    "pmv",
    "auditory_cortex",
    "visual_cortex",
]

ROI_PRETTY = {
    "dstr": "Dorsal\nStriatum",
    "cereb": "Cerebellum",
    "presma": "PreSMA",
    "sma": "SMA",
    "pmd": "PMD",
    "pmv": "PMV",
    "auditory_cortex": "Auditory Cortex",
    "visual_cortex": "Visual Cortex",
}

# ---- boxplot palette ----
# Single desaturated family, so the flatmap stays the only vivid element.
# Modality = muted hue direction (Auditory warm, Visual cool, Both violet);
# Non-Random vs Random = lightness (dark = Non-Random, light = Random). Auditory
# vs Visual (the only within-panel pair) sits on the warm/cool (blue-yellow)
# axis, preserved under red-green colour blindness; Both takes the violet corner
# so it is distinct from both rather than sitting neutral between them.
MODALITY_COLORS = {
    "Pooled":   {"Non-Random": "#CE8D1D", "Random": "#EAD2A0"},  # ochre (Both)
    "Auditory": {"Non-Random": "#1BB18E", "Random": "#C9F3E9"},  # teal
    "Visual":   {"Non-Random": "#9A85F6", "Random": "#E1DBFF"},  # purple
}

# ---- box geometry (UNCHANGED: do not alter the boxplots themselves) ----
PAIR_POS = [1.03, 1.13]
PAIR_XLIM = (0.955, 1.205)
BOX_WIDTH = 0.075

# ---- y-axis / row sizing ----
YTICK_STEP = 0.20
Y_FORMATTER = FormatStrFormatter("%.1f")

# Tick-to-inch mapping matches plot_anovas_all.py so the panels share the same
# physical PSC scale and the brackets share the same physical proportions.
INCHES_PER_STEP = 0.4
MIN_ROW_HEIGHT = 2.0

# Overall figure-width scale (kept from the original nonrand layout).
FIG_W_SCALE = 0.617

# ---- fonts (match plot_anovas_all.py) ----
AXIS_LABEL_FS = 12          # ticks, axis labels, modality x-labels
YTICK_FS = 12
ROI_TITLE_FS = AXIS_LABEL_FS + 8     # 20
LEGEND_FS = AXIS_LABEL_FS            # swatch row font
LEGEND_CAPTION_FS = AXIS_LABEL_FS + 2

# ===================================================================== #
# VERTICAL-SPACING MODEL (ported verbatim from plot_anovas_all.py).
#
#   * Annotation distances are in MULTIPLES OF ONE Y-TICK INTERVAL.  One tick
#     interval equals INCHES_PER_STEP inches of row height, so a step of e.g.
#     0.7 ticks is the same number of millimetres on paper regardless of the
#     panel's y-range.  They are converted to data coordinates on the fly by
#     multiplying by YTICK_STEP.
#   * Legend / title / label distances are in ABSOLUTE POINTS, converted to
#     figure fractions at draw time via _pts_to_figfrac_y().
# ===================================================================== #

# -- In-panel significance brackets (units: multiples of one y-tick) --
ANNOT_ANCHOR_PAD_TICKS = 0.25     # gap from data top to where stacks start
ANNOT_CLEARANCE_TICKS = 0.35      # extra gap before the first cross span
ANNOT_LAYER_TICKS = 0.9           # vertical step between stacked brackets
ANNOT_CAP_TICKS = 0.22            # end-cap height of cross brackets (y-ticks)
ANNOT_HEADROOM_TICKS = 0.45       # blank space kept above the topmost bracket

WITHIN_CLEARANCE_TICKS = 0.30     # gap from data top to within-axis bracket
WITHIN_LAYER_TICKS = 0.55         # step between stacked within-axis brackets
WITHIN_CAP_TICKS = 0.22           # within-axis bracket cap height

CROSS_GAP_TICKS = 0.9             # gap before the cross-modality stack begins

# Honor enforced y-limits exactly (see plot_anovas_all.py for the rationale).
EXPAND_TOP_FOR_ANNOTATIONS = False

# -- Legend / title spacing (units: absolute points) --
LEGEND_CI_GAP_PT = 16.0       # CI caption above the Non-Random swatch row
LEGEND_ROW_GAP_PT = 15.0      # gap between the Non-Random and Random rows
LEGEND_TITLE_CLEAR_PT = 16.0  # min clearance between legend block and 1st title
TITLE_GAP_PT = 14.0           # ROI title above its row of panels
MODLABEL_GAP_PT = 6.0         # extra gap below the modality x-labels

# Vertical breathing room (inches) added into every inter-row gap.
ROW_BREATHING_IN = 0.18

# ----- assemble_panel (grid panel) spacing, in inches -----
PANEL_ROW_GAP_IN = 0.30
PANEL_TITLE_GAP_IN = -0.25


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
    s = s.replace("\u2019", "'")
    for ch in (" ", "_", "-"):
        s = s.replace(ch, "")
    return s


def _roi_key(name: str) -> str | None:
    """Return canonical ROI key for matching (avoids sma/presma collisions)."""
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
    if "auditorycortex" in tok or "auditoryctx" in tok:
        return "auditory_cortex"
    if "visualcortex" in tok or "visualctx" in tok:
        return "visual_cortex"
    return None


def _matches_roi(resolved_roi: str, ann_roi: str) -> bool:
    """Match annotation ROI to resolved ROI using canonical keys."""
    k1 = _roi_key(resolved_roi)
    k2 = _roi_key(ann_roi)
    return (k1 is not None) and (k1 == k2)


def _pretty_roi_label(resolved_roi: str) -> str:
    """Convert resolved ROI name to a pretty label."""
    key = _roi_key(resolved_roi)
    if key is not None and key in ROI_PRETTY:
        return ROI_PRETTY[key]
    return str(resolved_roi)


def _resolve_roi(name: str, roi_values: Sequence[str]) -> str | None:
    """Resolve a canonical ROI name to a value present in the dataframe."""
    key = name.lower()
    roi_map = {str(r).lower(): r for r in roi_values}
    if key in roi_map:
        return roi_map[key]
    # match by canonical key
    want = _roi_key(name)
    if want is not None:
        for v in roi_values:
            if _roi_key(v) == want:
                return v
    for k, v in roi_map.items():
        if key in k:
            return v
    return None


def _pts_to_figfrac_y(fig: plt.Figure, pts: float) -> float:
    """Convert a vertical distance in points to a figure-height fraction."""
    return float(pts) / (fig.get_figheight() * 72.0)


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
    The end-cap height is ``h_data`` in data units (driven by ANNOT_CAP_TICKS,
    so it stays a constant size on the page).
    """
    if not getattr(fig, "_ann_canvas_drawn", False):
        fig.canvas.draw()
        fig._ann_canvas_drawn = True

    b1 = ax_left.get_position()
    b2 = ax_right.get_position()
    x1 = b1.x0 + 0.5 * b1.width
    x2 = b2.x0 + 0.5 * b2.width

    y0 = _ydata_to_yfig(fig, ax_left, y_data)
    y1 = _ydata_to_yfig(fig, ax_left, y_data + float(h_data))

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

    clip_on is False so a bracket sitting above the enforced y-limit is drawn
    in the margin rather than being chopped off at the top spine.
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


def bootstrap_median_ci(
    vals: np.ndarray,
    n_boot: int = 5000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the median (subjects resampled)."""
    x = np.asarray(vals, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        med = float(np.median(x)) if n > 0 else np.nan
        return med, med

    if rng is None:
        rng = np.random.default_rng()

    idx = rng.integers(0, n, size=(n_boot, n))
    meds = np.median(x[idx], axis=1)
    lo = float(np.quantile(meds, alpha / 2.0))
    hi = float(np.quantile(meds, 1.0 - alpha / 2.0))
    return lo, hi


def bootstrap_conf_intervals(
    data: List[np.ndarray],
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 12345,
) -> np.ndarray:
    """Compute bootstrap median CI for each box."""
    rng = np.random.default_rng(seed)
    cis = []
    for vals in data:
        lo, hi = bootstrap_median_ci(
            vals=np.asarray(vals, dtype=float),
            n_boot=n_boot,
            alpha=alpha,
            rng=rng,
        )
        cis.append((lo, hi))
    return np.asarray(cis, dtype=float)


def _poly_xspan_at_y(patch, y: float) -> tuple[float, float] | None:
    """Return x-span of box polygon at y in data coordinates."""
    verts = patch.get_path().vertices
    if verts.shape[0] < 3:
        return None

    xs = []
    n = verts.shape[0]
    for i in range(n):
        x0, y0 = verts[i]
        x1, y1 = verts[(i + 1) % n]
        if (y0 <= y <= y1) or (y1 <= y <= y0):
            dy = y1 - y0
            if abs(dy) < 1e-12:
                if abs(y - y0) < 1e-12:
                    xs.extend([float(x0), float(x1)])
                continue
            t = (y - y0) / dy
            xs.append(float(x0 + t * (x1 - x0)))

    if len(xs) < 2:
        return None
    return min(xs), max(xs)


def _subject_table(df: pd.DataFrame, roi: str, modality: str) -> pd.DataFrame:
    """Return subject x category PSC table for one ROI and modality block."""
    sub = df.loc[
        (df["ROI"] == roi) & (df["Task"] == TASK_NAME),
        ["Subject", "Category", "Modality", "PSC"],
    ].copy()

    if modality != "Pooled":
        sub = sub[sub["Modality"] == modality]

    if modality == "Pooled":
        sub = (
            sub.groupby(["Subject", "Category"], as_index=False)["PSC"]
            .mean()
        )

    wide = sub.pivot_table(
        index="Subject",
        columns="Category",
        values="PSC",
        aggfunc="mean",
    )
    for cat in CATEGORIES:
        if cat not in wide.columns:
            wide[cat] = np.nan
    wide = wide.dropna(subset=CATEGORIES, how="any")
    if wide.empty:
        return pd.DataFrame(columns=CATEGORIES)
    return wide[CATEGORIES]


# ---- boxplot drawing (UNCHANGED palette / styling) ----


def _draw_boxplot(
    ax: plt.Axes,
    data: List[np.ndarray],
    modality: str,
) -> None:
    """Draw boxplots with bootstrap notches and mean line (unchanged style)."""
    conf_intervals = bootstrap_conf_intervals(data)

    bp = ax.boxplot(
        data,
        positions=PAIR_POS,
        widths=BOX_WIDTH,
        notch=True,
        patch_artist=True,
        showfliers=False,
        showmeans=False,
        medianprops={"linewidth": 0, "color": "none"},
        whis=1.5,
        conf_intervals=conf_intervals,
    )

    for patch, cat in zip(bp["boxes"], CATEGORIES):
        patch.set_facecolor(MODALITY_COLORS[modality][cat])
        patch.set_edgecolor("0.2")
        patch.set_linewidth(0.8)

    for key in ("whiskers", "caps"):
        for artist in bp[key]:
            artist.set_color("0.2")
            artist.set_linewidth(0.8)

    for patch, vals in zip(bp["boxes"], data):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        mean_val = float(np.mean(vals))
        span = _poly_xspan_at_y(patch, mean_val)
        if span is None:
            x_left = float(patch.get_path().vertices[:, 0].min())
            x_right = float(patch.get_path().vertices[:, 0].max())
        else:
            x_left, x_right = span
        ax.plot(
            [x_left, x_right],
            [mean_val, mean_val],
            color="0.2",
            lw=0.8,
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
    modality_blocks: Sequence[str] = MODALITY_BLOCKS,
    center_singleline_xlabels: bool = True,
    xlabel_pad: float = 3.0,
    xlabel_pad_centered: float = -1.5,
    roi_subset: Sequence[str] | None = None,
    draw_legend: bool = True,
    draw_title: bool = True,
    x_leg_dx: float = -0.02,
    tag_dx_nonrandom: float = 0.0,
    tag_dx_random: float = 0.0,
    tag_dy: float = 0.0,
    save_tight: bool = True,
    left_margin: float = 0.05,
    extra_left_canvas_in: float = 0.0,
) -> None:
    """Plot PSC boxplots by ROI and modality blocks.

    The annotation-bracket geometry and the top legend follow
    ``plot_anovas_all.py``.  Box colors / variables are unchanged.

    roi_subset:  if given, only these ROI keys are drawn, in order.
    draw_legend: set False to omit the top legend (used by assemble_panel).
    draw_title:  set False to omit the per-row ROI title (used by panel).
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df = df[df["Hemisphere"] == "bh"].copy()
    df = df[df["Task"] == TASK_NAME].copy()

    modality_blocks = list(modality_blocks)
    n_cols = len(modality_blocks)

    roi_values = list(pd.unique(df["ROI"]))
    roi_source = list(roi_subset) if roi_subset is not None else list(ROI_ORDER)
    rois = [_resolve_roi(r, roi_values) for r in roi_source]

    whis = 1.5
    ypad_frac = 0.08

    # ---------------- PASS 1: per-ROI y-lims and row heights ----------------
    specs: List[dict] = []
    for roi in rois:
        if roi is None:
            specs.append(
                {
                    "roi": None,
                    "roi_label": "",
                    "y_min": 0.0,
                    "y_max": 1.0,
                    "yr": 1.0,
                    "pad": 0.0,
                    "y_lim": (0.0, 1.0),
                    "y_ticks": np.array([0.0, 1.0]),
                    "overflow_in": 0.0,
                    "tick": YTICK_STEP,
                    "eligible_within_by_mod": {m: [] for m in modality_blocks},
                    "eligible_cross": [],
                    "within_stack_top_by_mod": {m: 0.0 for m in modality_blocks},
                    "cross_start_t": 0.0,
                    "row_h": MIN_ROW_HEIGHT,
                }
            )
            continue

        # ---- gather eligible annotations ----
        eligible_within_by_mod: Dict[str, List[dict]] = {
            m: [] for m in modality_blocks
        }
        for ann in WITHIN_ANNOTATIONS:
            if not _matches_roi(roi, ann.get("roi", "")):
                continue
            m = str(ann.get("modality", ""))
            if m in eligible_within_by_mod:
                eligible_within_by_mod[m].append(ann)
        for m in eligible_within_by_mod:
            eligible_within_by_mod[m].sort(key=lambda a: float(a["pvalue"]))

        eligible_cross: List[dict] = []
        if ("Auditory" in modality_blocks) and ("Visual" in modality_blocks):
            for ann in CROSS_AV_ANNOTATIONS:
                if not _matches_roi(roi, ann.get("roi", "")):
                    continue
                eligible_cross.append(dict(ann))
            eligible_cross.sort(key=lambda a: float(a["pvalue"]))

        # ---- annotation spacing in TICK units ----
        anchor_pad_t = ANNOT_ANCHOR_PAD_TICKS
        clearance_t = ANNOT_CLEARANCE_TICKS
        layer_t = ANNOT_LAYER_TICKS
        cap_t = ANNOT_CAP_TICKS
        headroom_t = ANNOT_HEADROOM_TICKS
        within_clear_t = WITHIN_CLEARANCE_TICKS
        within_layer_t = WITHIN_LAYER_TICKS
        within_cap_t = WITHIN_CAP_TICKS
        cross_gap_t = CROSS_GAP_TICKS

        within_stack_top_by_mod: Dict[str, float] = {
            m: 0.0 for m in modality_blocks
        }
        for mm in within_stack_top_by_mod:
            n_within = len(eligible_within_by_mod[mm])
            if n_within > 0:
                within_stack_top_by_mod[mm] = (
                    within_clear_t
                    + (n_within - 1) * within_layer_t
                    + within_cap_t
                )

        # ---- data extent (whiskers) ----
        w_lows: List[float] = []
        w_highs: List[float] = []
        for modality in modality_blocks:
            paired = _subject_table(df, roi, modality)
            for cat in CATEGORIES:
                vals = paired[cat].dropna().to_numpy() if not paired.empty \
                    else np.array([])
                if vals.size < 3:
                    continue
                stats = cbook.boxplot_stats(vals, whis=whis)[0]
                w_lows.append(float(stats["whislo"]))
                w_highs.append(float(stats["whishi"]))

        if w_lows:
            y_min, y_max = min(w_lows), max(w_highs)
        else:
            roi_vals = df.loc[df["ROI"] == roi, "PSC"].dropna().to_numpy()
            if roi_vals.size:
                y_min, y_max = float(roi_vals.min()), float(roi_vals.max())
            else:
                y_min, y_max = 0.0, 1.0

        yr = max(y_max - y_min, 0.1)
        pad = ypad_frac * yr

        max_stack_within = max(
            (len(v) for v in eligible_within_by_mod.values()), default=0
        )
        max_stack_cross = len(eligible_cross)

        within_offset_t = max(
            (v + (clearance_t if v > 0.0 else 0.0)
             for v in within_stack_top_by_mod.values()),
            default=0.0,
        )

        cross_start_t = 0.0
        cross_top_t = 0.0
        if max_stack_cross > 0:
            cross_start_t = within_offset_t + cross_gap_t
            cross_top_t = (
                cross_start_t
                + clearance_t
                + (max_stack_cross - 1) * layer_t
                + cap_t
            )

        within_only_top_t = 0.0
        if max_stack_within > 0:
            within_only_top_t = (
                within_clear_t
                + (max_stack_within - 1) * within_layer_t
                + within_cap_t
            )

        needed_tops = [y_min]
        if max_stack_within > 0:
            needed_tops.append(
                y_max
                + (anchor_pad_t + within_only_top_t + headroom_t) * YTICK_STEP
            )
        if max_stack_cross > 0:
            needed_tops.append(
                y_max
                + (anchor_pad_t + cross_top_t + headroom_t) * YTICK_STEP
            )
        top_needed = max(needed_tops)
        y_lim_raw = (y_min - pad, max(y_max, top_needed))

        # ---- resolve y-limits / ticks ----
        rkey = _roi_key(roi)
        explicit_ylim = None
        if (y_limits is not None) and (rkey is not None):
            explicit_ylim = y_limits.get(rkey)

        if explicit_ylim is not None:
            y0, y1 = float(explicit_ylim[0]), float(explicit_ylim[1])
            if EXPAND_TOP_FOR_ANNOTATIONS and (y1 < y_lim_raw[1] - 1e-9):
                y1 = float(np.ceil((y_lim_raw[1] - 1e-9) / YTICK_STEP)
                           * YTICK_STEP)
            y_ticks = np.arange(y0, y1 + 0.5 * YTICK_STEP, YTICK_STEP)
            if y_ticks.size < 2:
                y_ticks = np.array([y0, y1])
        else:
            eps = 1e-9
            y0 = float(np.floor((y_lim_raw[0] + eps) / YTICK_STEP) * YTICK_STEP)
            y1 = float(np.ceil((y_lim_raw[1] - eps) / YTICK_STEP) * YTICK_STEP)
            y0 = min(y0, 0.0)
            y1 = max(y1, 0.0)
            y_ticks = np.arange(y0, y1 + 0.5 * YTICK_STEP, YTICK_STEP)
            if y_ticks.size < 2:
                y_ticks = np.array([y0, y1])

        n_steps = max(int(round((y1 - y0) / YTICK_STEP)), 1)
        row_h = max(MIN_ROW_HEIGHT, n_steps * INCHES_PER_STEP)

        overflow_data = max(0.0, top_needed - y1)
        overflow_in = (overflow_data / YTICK_STEP) * INCHES_PER_STEP

        specs.append(
            {
                "roi": roi,
                "roi_label": _pretty_roi_label(roi),
                "y_min": y_min,
                "y_max": y_max,
                "yr": yr,
                "pad": pad,
                "y_lim": (y0, y1),
                "y_ticks": y_ticks,
                "overflow_in": overflow_in,
                "tick": YTICK_STEP,
                "anchor_pad_t": anchor_pad_t,
                "clearance_t": clearance_t,
                "layer_t": layer_t,
                "cap_t": cap_t,
                "within_clear_t": within_clear_t,
                "within_layer_t": within_layer_t,
                "within_cap_t": within_cap_t,
                "cross_start_t": cross_start_t,
                "eligible_within_by_mod": eligible_within_by_mod,
                "eligible_cross": eligible_cross,
                "within_stack_top_by_mod": within_stack_top_by_mod,
                "row_h": row_h,
            }
        )

    n_rows = len(specs)
    height_ratios = [s["row_h"] for s in specs]

    # -------------------------- figure width -------------------------
    # Keep the original nonrand width scaling (horizontal geometry unchanged).
    base_fig_w = 4.6 * FIG_W_SCALE * figsize_scale
    base_fig_w *= len(modality_blocks) / len(MODALITY_BLOCKS)
    fig_w = base_fig_w + float(extra_left_canvas_in)

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(fig_w, float(sum(height_ratios)) * figsize_scale),
        gridspec_kw={"height_ratios": height_ratios},
        sharex=False,
        sharey=False,
    )
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    # ================== manual vertical layout (ported from all) ==========
    PT = 1.0 / 72.0

    def _title_in(spec: dict) -> float:
        # When panel cells are generated with draw_title=False, do not reserve
        # hidden title space. This keeps the temporary cell canvas compact while
        # preserving identical data-axis geometry across cells in the same row.
        if (spec["roi"] is None) or (not draw_title):
            return 0.0
        n_lines = spec["roi_label"].count("\n") + 1
        return (n_lines * ROI_TITLE_FS * 1.30 + TITLE_GAP_PT) * PT

    # Space below a row = modality x-labels (1 or 2 lines) + gap.
    xlabel_map = MOD_LABEL_CENTERED if center_singleline_xlabels else MOD_LABEL
    max_xlabel_lines = max(
        (xlabel_map.get(m, m).count("\n") + 1 for m in modality_blocks),
        default=1,
    )
    modlabel_in = (
        max_xlabel_lines * AXIS_LABEL_FS * 1.30 + xlabel_pad + MODLABEL_GAP_PT
        + 6.0
    ) * PT
    breathing_in = ROW_BREATHING_IN
    legend_in = (
        LEGEND_TITLE_CLEAR_PT + LEGEND_ROW_GAP_PT + LEGEND_CI_GAP_PT
        + (AXIS_LABEL_FS + 2) * 1.30 + 18.0
    ) * PT

    row_h_in = [s["row_h"] * figsize_scale for s in specs]
    ovf_in = [float(s.get("overflow_in", 0.0)) for s in specs]
    ttl_in = [_title_in(s) for s in specs]

    top_margin = ovf_in[0] + ttl_in[0] + (
        legend_in if draw_legend else 0.0
    ) + breathing_in
    bottom_margin = modlabel_in + breathing_in
    gaps = [
        modlabel_in + ovf_in[r + 1] + ttl_in[r + 1] + breathing_in
        for r in range(n_rows - 1)
    ]

    fig_h = top_margin + sum(row_h_in) + sum(gaps) + bottom_margin
    fig.set_figheight(fig_h)

    left_frac = (
        float(extra_left_canvas_in) + float(left_margin) * base_fig_w
    ) / fig_w
    right_frac = (float(extra_left_canvas_in) + 0.98 * base_fig_w) / fig_w
    fig.subplots_adjust(
        left=left_frac, right=right_frac, top=0.999, bottom=0.001,
        wspace=0.24
    )

    y_top_in = fig_h - top_margin
    for r in range(n_rows):
        h = row_h_in[r]
        y0f = (y_top_in - h) / fig_h
        hf = h / fig_h
        for j in range(n_cols):
            ax = axes[r, j]
            pos = ax.get_position()
            ax.set_position([pos.x0, y0f, pos.width, hf])
        y_top_in -= h + (gaps[r] if r < n_rows - 1 else 0.0)

    ovf_frac = [o / fig_h for o in ovf_in]

    if not getattr(fig, "_top_canvas_drawn", False):
        fig.canvas.draw()
        fig._top_canvas_drawn = True

    # --------------------- top legend (ported from all) -------------------
    ax_last = axes[0, n_cols - 1]
    x_right = float(ax_last.get_position().x1)
    y_top_axes = float(max(axes[0, j].get_position().y1 for j in range(n_cols)))

    if draw_legend:
        first_label = specs[0]["roi_label"] if specs else ""
        n_title_lines = first_label.count("\n") + 1
        title_h_pt = n_title_lines * (AXIS_LABEL_FS + 8) * 1.25

        y_random = y_top_axes + ovf_frac[0] + _pts_to_figfrac_y(
            fig, TITLE_GAP_PT + title_h_pt + LEGEND_TITLE_CLEAR_PT
        )
        y_nonrandom = y_random + _pts_to_figfrac_y(fig, LEGEND_ROW_GAP_PT)
        y_ci = y_nonrandom + _pts_to_figfrac_y(fig, LEGEND_CI_GAP_PT)

        fig.text(
            x_right,
            y_ci,
            "95% bootstrap CI for the Median of PSC",
            ha="right",
            va="top",
            fontsize=LEGEND_CAPTION_FS,
            color="k",
        )

        legend_mods = list(modality_blocks)

        # Swatch + label geometry follows plot_anovas_all.py's two-row legend,
        # but the horizontal placement is measured from the rendered label width
        # (as in _panel_legend) so the long "Non-Random"/"Random" labels never
        # overlap the swatches regardless of figure width.  The labels are
        # left-aligned with each other, their right edge reaching x_right, and
        # the per-modality swatches sit to their left.
        fw_in = fig.get_figwidth()
        fh_in = fig.get_figheight()
        swatch_w_in, swatch_h_in = 0.22, 0.20
        col_gap_in, label_pad_in = 0.07, 0.10
        sw = swatch_w_in / fw_in
        sh = swatch_h_in / fh_in
        cg = col_gap_in / fw_in
        pad = label_pad_in / fw_in

        n_sw = len(legend_mods)
        total_w = n_sw * sw + (n_sw - 1) * cg

        renderer = fig.canvas.get_renderer()
        _tmp = [fig.text(0, 0, lab, fontsize=LEGEND_CAPTION_FS)
                for lab in ("Non-Random", "Random")]
        label_w = max(t.get_window_extent(renderer=renderer).width
                      for t in _tmp) / float(fig.bbox.width)
        for t in _tmp:
            t.remove()

        # Optional manual nudge of the whole legend block (x_leg_dx) and of
        # each label baseline (tag_dx_*, tag_dy) is still honoured.
        x_right_block = x_right + float(x_leg_dx)
        x_label_left = x_right_block - label_w
        x_swatch_right = x_label_left - pad
        x0 = x_swatch_right - total_w

        for cat, y_top, tag_dx in (
            ("Non-Random", y_nonrandom, tag_dx_nonrandom),
            ("Random", y_random, tag_dx_random),
        ):
            for ci, m in enumerate(legend_mods):
                x = x0 + ci * (sw + cg)
                fig.add_artist(Rectangle(
                    (x, y_top - sh), sw, sh,
                    transform=fig.transFigure,
                    facecolor=MODALITY_COLORS[m][cat],
                    edgecolor="0.2", lw=0.8,
                ))
            fig.text(
                x_label_left + float(tag_dx),
                y_top - sh / 2.0 + float(tag_dy),
                cat,
                ha="left",
                va="center",
                fontsize=LEGEND_CAPTION_FS,
                color="k",
            )

    # --------------------- PASS 2: draw panels ----------------------
    for r, spec in enumerate(specs):
        roi = spec["roi"]
        ax_lookup: Dict[str, plt.Axes] = {}
        row_axes: List[plt.Axes] = []

        for col, modality in enumerate(modality_blocks):
            ax = axes[r, col]
            ax_lookup[modality] = ax
            row_axes.append(ax)

            if roi is None:
                ax.axis("off")
                continue

            paired = _subject_table(df, roi, modality)
            data = [
                (paired[cat].to_numpy(dtype=float) if not paired.empty
                 else np.array([]))
                for cat in CATEGORIES
            ]
            _draw_boxplot(ax, data, modality)

            ax.axhline(0, color="0.3", linestyle="--", linewidth=1.0, zorder=1)
            ax.set_ylim(*spec["y_lim"])
            ax.set_xlim(*PAIR_XLIM)
            ax.yaxis.set_major_locator(MultipleLocator(YTICK_STEP))
            ax.yaxis.set_major_formatter(Y_FORMATTER)
            ax.set_yticks(spec["y_ticks"])
            ax.set_xticks([])
            ax.tick_params(axis="x", length=0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_bounds(*spec["y_lim"])

            pad = xlabel_pad
            if center_singleline_xlabels and modality != "Pooled":
                pad = xlabel_pad_centered
            ax.set_xlabel(
                xlabel_map.get(modality, modality),
                fontsize=AXIS_LABEL_FS,
                labelpad=pad,
            )

            rkey = _roi_key(roi)
            display_axis = True
            if show_yaxis is not None:
                display_axis = show_yaxis.get(rkey, True)

            if col == 0 and display_axis:
                ax.set_ylabel("PSC (%)", fontsize=AXIS_LABEL_FS)
                ax.tick_params(axis="y", labelsize=YTICK_FS)
                ax.spines["left"].set_visible(True)
            else:
                ax.set_yticklabels([])
                ax.tick_params(axis="y", left=False, length=0)
                ax.spines["left"].set_visible(False)
                if not display_axis:
                    ax.set_ylabel("")

            # ---- within-axis bracket (Non-Random vs Random) ----
            within_anns = spec["eligible_within_by_mod"].get(modality, [])
            if within_anns:
                x1 = float(PAIR_POS[0])
                x2 = float(PAIR_POS[1])
                tick = spec["tick"]
                anchor = spec["y_max"] + spec["anchor_pad_t"] * tick
                for level, ann in enumerate(within_anns):
                    text = pval_label_converter([float(ann["pvalue"])])[0]
                    y_data = anchor + (
                        spec["within_clear_t"]
                        + level * spec["within_layer_t"]
                    ) * tick
                    h_data = spec["within_cap_t"] * tick
                    within_axis_annotation(
                        ax=ax, x1=x1, x2=x2, text=text,
                        y_data=y_data, h_data=h_data,
                    )

        # ---- ROI title centered across the row ----
        if roi is not None and draw_title:
            x0 = min(a.get_position().x0 for a in row_axes)
            x1 = max(a.get_position().x1 for a in row_axes)
            y1 = max(a.get_position().y1 for a in row_axes)
            fig.text(
                (x0 + x1) / 2.0,
                y1 + ovf_frac[r] + _pts_to_figfrac_y(fig, TITLE_GAP_PT),
                spec["roi_label"],
                ha="center",
                va="bottom",
                fontsize=ROI_TITLE_FS,
                fontweight="semibold",
                color="k",
            )

        if roi is None:
            continue

        # ---- cross-modality bracket (Auditory vs Visual) ----
        eligible_cross = spec.get("eligible_cross", [])
        cross_start_t = float(spec.get("cross_start_t", 0.0))
        if (
            eligible_cross
            and ("Auditory" in ax_lookup)
            and ("Visual" in ax_lookup)
            and cross_start_t > 0.0
        ):
            ax_aud = ax_lookup["Auditory"]
            ax_vis = ax_lookup["Visual"]
            tick = spec["tick"]
            anchor = spec["y_max"] + spec["anchor_pad_t"] * tick
            for level, ann in enumerate(eligible_cross):
                text = pval_label_converter([float(ann["pvalue"])])[0]
                y_data = anchor + (
                    cross_start_t + level * spec["layer_t"]
                ) * tick
                h_data = spec["cap_t"] * tick
                span_annotation_datay_figspan(
                    fig,
                    ax_left=ax_aud,
                    ax_right=ax_vis,
                    text=text,
                    y_data=y_data,
                    h_data=h_data,
                )

    if save_tight:
        fig.savefig(outpath, dpi=300, bbox_inches="tight", pad_inches=0.08)
    else:
        fig.savefig(outpath, dpi=300)
    plt.close(fig)


# ============================ PANEL ASSEMBLY ======================= #


def _content_right_frac(png_path) -> float:
    """Fraction across an image at which the right-most non-white pixel sits."""
    from PIL import Image
    arr = np.asarray(Image.open(png_path).convert("L"))
    cols = np.where((arr < 250).any(axis=0))[0]
    if cols.size == 0:
        return 1.0
    return float(cols.max() + 1) / float(arr.shape[1])


def _panel_legend(fig, x_right_fig, y_top_fig, mods, axis_label_fs,
                  swatch_w_in, swatch_h_in, row_gap_in, col_gap_in,
                  label_pad_in):
    """Draw the Non-Random/Random colour legend with right edge at x_right_fig.

    The category labels are LEFT-aligned with each other; the longer label
    ("Non-Random") reaches x_right_fig and the swatches sit to its left.
    One swatch per modality, per category row.  Swatch sizes/gaps in inches.
    """
    fw = fig.get_figwidth()
    fh = fig.get_figheight()
    sw = swatch_w_in / fw
    sh = swatch_h_in / fh
    cg = col_gap_in / fw
    rg = row_gap_in / fh
    pad = label_pad_in / fw

    n = len(mods)
    total_w = n * sw + (n - 1) * cg

    renderer = fig.canvas.get_renderer()
    tmp = [fig.text(0, 0, lab, fontsize=axis_label_fs + 2)
           for lab in ("Non-Random", "Random")]
    label_w = max(t.get_window_extent(renderer=renderer).width
                  for t in tmp) / float(fig.bbox.width)
    for t in tmp:
        t.remove()

    x_label_left = x_right_fig - label_w
    x_swatch_right = x_label_left - pad
    x0 = x_swatch_right - total_w

    for ri, cat in enumerate(("Non-Random", "Random")):
        y = y_top_fig - ri * (sh + rg)
        for ci, m in enumerate(mods):
            x = x0 + ci * (sw + cg)
            fig.add_artist(plt.matplotlib.patches.Rectangle(
                (x, y - sh), sw, sh, transform=fig.transFigure,
                facecolor=MODALITY_COLORS[m][cat], edgecolor="0.2", lw=0.8))
        fig.text(x_label_left, y - sh / 2.0, cat, ha="left",
                 va="center", fontsize=axis_label_fs + 2, color="k")


# Named legend presets: each value is the list of modalities to show.
PANEL_LEGENDS = {
    "full": ["Pooled", "Auditory", "Visual"],
    "audivisual": ["Auditory", "Visual"],
    "pooled": ["Pooled"],
}

# Map a panel-cell "mode" to the modality blocks it should display.
_MODE_TO_BLOCKS = {
    "pooled": ["Pooled"],
    "audivisual": ["Auditory", "Visual"],
    "full": ["Pooled", "Auditory", "Visual"],
}


def assemble_panel(
    df: pd.DataFrame,
    outpath: str | Path,
    rows: Sequence[Sequence[Tuple[str, str]]],
    row_ylims: Sequence[Tuple[float, float]],
    *,
    legend: str | None = "full",
    col_gap_in: float = 0.30,
    row_gap_in: float = PANEL_ROW_GAP_IN,
    title_gap_in: float = PANEL_TITLE_GAP_IN,
    side_pad_in: float = 0.20,
    top_pad_in: float = 0.20,
    bottom_pad_in: float = 0.06,
    title_fontsize: int = 14,
    dpi: int = 300,
    cell_kwargs: dict | None = None,
) -> None:
    """Compose single-ROI plots into a centered multi-row panel (as in all).

    rows :
        One list per panel row; each entry is ``(roi_key, mode)`` where mode is
        ``"audivisual"`` (Auditory + Visual) or ``"pooled"`` (Both Modalities).
    row_ylims :
        One ``(lo, hi)`` per row, applied to every cell in that row so the
        PSC = 0 lines and the y-scale line up within a row.
    legend :
        Which legend preset to draw at the top-right (see PANEL_LEGENDS) or None.
    """
    import tempfile
    import shutil
    from PIL import Image

    if len(rows) != len(row_ylims):
        raise ValueError("row_ylims must have one (lo, hi) per row")
    if legend is not None and legend not in PANEL_LEGENDS:
        raise ValueError(f"unknown legend preset: {legend!r}")

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    base_kwargs = dict(cell_kwargs or {})

    tmpdir = Path(tempfile.mkdtemp(prefix="psc_nonrand_panel_"))
    try:
        cell_w: List[List[float]] = []
        cell_h: List[List[float]] = []
        cell_png: List[List[Path]] = []
        for ri, row in enumerate(rows):
            ylim = tuple(row_ylims[ri])
            ws, hs, ps = [], [], []
            for ci, (roi_key, mode) in enumerate(row):
                png = tmpdir / f"cell_{ri}_{ci}.png"
                plot_psc_boxplots(
                    df=df, outpath=png,
                    modality_blocks=_MODE_TO_BLOCKS[mode],
                    roi_subset=[roi_key],
                    y_limits={roi_key: ylim},
                    show_yaxis={roi_key: ci == 0},
                    draw_legend=False, draw_title=False,
                    save_tight=False,
                    left_margin=0.05,
                    extra_left_canvas_in=0.65 if ci == 0 else 0.0,
                    **base_kwargs,
                )
                with Image.open(png) as im:
                    w_px, h_px = im.size
                ws.append(w_px / dpi)
                hs.append(h_px / dpi)
                ps.append(png)
            cell_w.append(ws)
            cell_h.append(hs)
            cell_png.append(ps)

        roi_values = list(pd.unique(df["ROI"]))

        def _label_lines(roi_key: str) -> int:
            rv = _resolve_roi(roi_key, roi_values) or roi_key
            return _pretty_roi_label(rv).count("\n") + 1

        line_h_in = title_fontsize * 1.15 / 72.0
        title_band = [
            line_h_in * max(_label_lines(rk) for rk, _ in row) + title_gap_in
            for row in rows
        ]

        row_w = [sum(cell_w[ri]) + col_gap_in * (len(cell_w[ri]) - 1)
                 for ri in range(len(rows))]
        row_h = [max(cell_h[ri]) for ri in range(len(rows))]
        content_w = max(row_w)

        legend_band = 0.76 if legend is not None else 0.0

        fig_w = content_w + 2 * side_pad_in
        fig_h = (top_pad_in + bottom_pad_in + legend_band
                 + sum(title_band[ri] + row_h[ri] for ri in range(len(rows)))
                 + row_gap_in * (len(rows) - 1))

        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

        y_top = fig_h - top_pad_in - legend_band
        for ri, row in enumerate(rows):
            cell_top = y_top - title_band[ri]
            cell_bottom = cell_top - row_h[ri]
            x = side_pad_in + (content_w - row_w[ri]) / 2.0
            title_y = y_top / fig_h
            for ci, (roi_key, mode) in enumerate(row):
                w, h = cell_w[ri][ci], cell_h[ri][ci]
                ax = fig.add_axes([x / fig_w, cell_bottom / fig_h,
                                   w / fig_w, h / fig_h])
                with Image.open(cell_png[ri][ci]) as im:
                    ax.imshow(np.asarray(im), aspect="auto",
                              interpolation="lanczos")
                ax.axis("off")
                rv = _resolve_roi(roi_key, roi_values) or roi_key
                label = _pretty_roi_label(rv)
                fig.text((x + w / 2.0) / fig_w, title_y, label,
                         ha="center", va="top", fontsize=title_fontsize,
                         fontweight="semibold", color="k")
                x += w + col_gap_in
            y_top = cell_bottom - row_gap_in

        if legend is not None:
            mods = PANEL_LEGENDS[legend]
            ri_max = max(range(len(rows)), key=lambda r: row_w[r])
            w_last = cell_w[ri_max][-1]
            last_left_in = side_pad_in + content_w - w_last
            right_in = last_left_in + w_last * _content_right_frac(
                cell_png[ri_max][-1])
            _panel_legend(
                fig,
                x_right_fig=right_in / fig_w,
                y_top_fig=(fig_h - top_pad_in) / fig_h,
                mods=mods, axis_label_fs=12,
                swatch_w_in=0.16, swatch_h_in=0.16,
                row_gap_in=0.06, col_gap_in=0.045, label_pad_in=0.06,
            )

        fig.savefig(outpath, dpi=dpi)
        plt.close(fig)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


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

OUTPUT_PATH_SENSORY = os.path.join(
    OUTPUT_DIR,
    "psc_boxplots_rand_ntfd_nonrandom_auditory_visual.png",
)

OUTPUT_PATH_PANEL = os.path.join(
    OUTPUT_DIR,
    "psc_boxplots_rand_ntfd_nonrandom_panel.png",
)

OUTPUT_PATH_PANEL_SINGLE_ROW = os.path.join(
    OUTPUT_DIR,
    "psc_boxplots_rand_ntfd_nonrandom_panel_single_row.png",
)


# ============= WITHIN-SUBPLOT ANNOTATIONS (Non-Random vs Random) ===== #
# p-values UNCHANGED (update later as needed); stars derive from them.

WITHIN_ANNOTATIONS: List[dict] = [
    dict(
        roi="cerebellum",
        modality="Pooled",
        pvalue=0.0143899704097716,
    ),
    dict(
        roi="presma",
        modality="Pooled",
        pvalue=0.00122916990122567,
    ),
    dict(
        roi="pmd",
        modality="Pooled",
        pvalue=0.000399959965496988,
    ),
    dict(
        roi="pmv",
        modality="Pooled",
        pvalue=0.000480355630824121,
    ),
    dict(
        roi="auditory_cortex",
        modality="Auditory",
        pvalue=0.00000261961870073949,
    ),
]


# =================== CROSS-MODALITY (AUDIO vs VISUAL) =============== #
# p-values UNCHANGED.

CROSS_AV_ANNOTATIONS: List[dict] = [
    dict(
        roi="auditory_cortex",
        pvalue=0.00000000000000168588784426346,
    ),
    dict(
        roi="visual_cortex",
        pvalue=0.00000000296509869947305,
    ),
]


# ============================== RUN ================================ #

if __name__ == "__main__":
    args = parse_args()

    df_in = pd.read_csv(DATA_PATH, sep="\t")
    if "Task" in df_in.columns:
        df_in = df_in.copy()
        df_in["Task"] = df_in["Task"].replace(
            {"NTFD_Random": TASK_NAME, "NTFD-Random": TASK_NAME}
        )

    y_limits = {
        "auditory_cortex": (-0.6, 1.8),
        "visual_cortex": (-0.4, 1.2),
        "dstr": (-0.4, 0.6),
        "cereb": (-0.4, 0.8),
        "presma": (-0.2, 0.8),
        "sma": (-0.2, 1.0),
        "pmd": (-0.4, 1.0),
        "pmv": (-0.2, 0.8),
    }

    show_yaxis = {
        "auditory_cortex": True,
        "visual_cortex": True,
        "dstr": True,
        "cereb": True,
        "presma": True,
        "sma": True,
        "pmd": True,
        "pmv": True,
    }

    # 1) Full figure (Pooled + Auditory + Visual blocks)
    plot_psc_boxplots(
        df=df_in,
        outpath=OUTPUT_PATH,
        figsize_scale=args.figscale,
        y_limits=y_limits,
        show_yaxis=show_yaxis,
        modality_blocks=MODALITY_BLOCKS,
        center_singleline_xlabels=True,
        xlabel_pad=3.0,
        xlabel_pad_centered=-1.5,
    )

    # 2) Sensory-only figure (Auditory + Visual blocks)
    plot_psc_boxplots(
        df=df_in,
        outpath=OUTPUT_PATH_SENSORY,
        figsize_scale=args.figscale,
        y_limits=y_limits,
        show_yaxis=show_yaxis,
        modality_blocks=MODALITY_BLOCKS_SENSORY,
        center_singleline_xlabels=False,
        xlabel_pad=3.0,
        xlabel_pad_centered=-1.5,
    )

    # 3) Combined panel (same arrangement as plot_anovas_all.py).
    #    First row keeps the same regions as in `all` (sensory cortices + the
    #    first basal-ganglia region); second row keeps the rest.  Only the
    #    Auditory and Visual cortices show the Auditory + Visual blocks; every
    #    other region shows the Both-Modalities (pooled) block only.
    panel_rows = [
        [("auditory_cortex", "audivisual"),
         ("visual_cortex", "audivisual"),
         ("dstr", "pooled")],
        [("cereb", "pooled"),
         ("presma", "pooled"),
         ("sma", "pooled"),
         ("pmd", "pooled"),
         ("pmv", "pooled")],
    ]
    # One uniform y-range per row (encompassing the regions in that row).
    panel_row_ylims = [(-0.6, 1.6), (-0.2, 1.0)]
    assemble_panel(
        df=df_in,
        outpath=OUTPUT_PATH_PANEL,
        rows=panel_rows,
        row_ylims=panel_row_ylims,
        legend="full",
        cell_kwargs=dict(center_singleline_xlabels=True),
    )

    # 4) Combined panel in one single row.  The same broader y-range used for
    #    the first row of the two-row panel is applied to every ROI.
    panel_rows_single = [
        [("auditory_cortex", "audivisual"),
         ("visual_cortex", "audivisual"),
         ("dstr", "pooled"),
         ("cereb", "pooled"),
         ("presma", "pooled"),
         ("sma", "pooled"),
         ("pmd", "pooled"),
         ("pmv", "pooled")],
    ]
    panel_row_ylims_single = [panel_row_ylims[0]]
    assemble_panel(
        df=df_in,
        outpath=OUTPUT_PATH_PANEL_SINGLE_ROW,
        rows=panel_rows_single,
        row_ylims=panel_row_ylims_single,
        legend="full",
        cell_kwargs=dict(center_singleline_xlabels=True),
    )