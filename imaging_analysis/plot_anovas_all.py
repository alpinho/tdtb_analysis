#!/usr/bin/env python3
"""PSC boxplots by ROI (single ROI column) and modality/task blocks.

Author: Ana Luisa Pinho
email: agrilopi@uwo.ca

Created: 28th of January, 2026
Last update: June 2026

Compatibility: Python 3.10.14
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from matplotlib.patches import Patch


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

# Geometry: make the 3-boxplot NTFD Random panels wider.
W_RATIO_STD = 0.9
W_RATIO_NTFD_RANDOM = 1.3694
W_RATIO_SPACER = 0.30

# Scale overall figure width without changing internal layout.
FIG_W_SCALE = 0.50

# Annotation vertical pin heights in figure-fraction units.
WITHIN_MODALITY_BRACKET_HEIGHT_FIG = 0.0015
BETWEEN_MODALITY_BRACKET_HEIGHT_FIG = 0.0025

# ===================================================================== #
# VERTICAL-SPACING MODEL (the single source of truth for all gaps).
#
# The whole point of this block is modularity: change a y-limit anywhere
# and NOTHING here needs to be re-tuned, because every distance is stored
# in a unit that maps to a CONSTANT distance on the printed page.
#
#   * Annotation distances are in MULTIPLES OF ONE Y-TICK INTERVAL.
#     One tick interval always equals INCHES_PER_STEP inches of row height
#     (see _RowGeometry below), so a step of e.g. 0.7 ticks is the same
#     number of millimetres on paper whether the panel spans (-0.4, 1.0)
#     or (-0.2, 2.0). They are converted to data coordinates on the fly by
#     multiplying by the panel's ytick_step.
#
#   * Legend / title / label distances are in ABSOLUTE POINTS, converted
#     to figure fractions at draw time via _pts_to_figfrac_y(). A point is
#     a point regardless of how tall the figure ends up being, so legend
#     rows never overlap and titles never collide with the row above.
# ===================================================================== #

# -- In-panel significance brackets (units: multiples of one y-tick) --
ANNOT_ANCHOR_PAD_TICKS = 0.25     # gap from data top to where stacks start
ANNOT_CLEARANCE_TICKS = 0.35      # extra gap before the first across-task span
ANNOT_LAYER_TICKS = 0.9           # vertical step between stacked span brackets
ANNOT_CAP_TICKS = 0.3             # end-cap height of across-task & cross
                                  # brackets, in y-tick units (responsive;
                                  # one tick == inches_per_step inches)
ANNOT_HEADROOM_TICKS = 0.45       # blank space kept above the topmost bracket

WITHIN_CLEARANCE_TICKS = 0.30     # gap from data top to first within-axis span
WITHIN_LAYER_TICKS = 0.55         # step between stacked within-axis brackets
WITHIN_CAP_TICKS = 0.10           # within-axis bracket cap height

CROSS_GAP_TICKS = 0.9            # gap before the cross-modality stack begins

# How enforced y-limits interact with the annotation stack:
#   False (default) -> your y_limits are honored EXACTLY. The data box is
#       always the range you set; significance brackets that don't fit are
#       drawn in the margin ABOVE the top spine, and the row above is given
#       just enough extra room so brackets never touch the title.
#   True -> the top is instead raised to the next tick to swallow the stack
#       (axes get taller; your enforced top is treated as a floor).
EXPAND_TOP_FOR_ANNOTATIONS = False


# -- Legend / title spacing (units: absolute points) --
LEGEND_CI_GAP_PT = 16.0       # CI caption above the Beat swatch row
LEGEND_ROW_GAP_PT = 15.0      # gap between the Beat and Interval swatch rows
LEGEND_TOP_PAD_PT = 30.0      # gap from the top panel up to the Beat row
LEGEND_TITLE_CLEAR_PT = 16.0  # min clearance between legend block and 1st title
TITLE_GAP_PT = 14.0           # ROI title above its row of panels
MODLABEL_GAP_PT = 28.0        # modality label below the task x-labels

# Vertical breathing room (inches) added into every inter-row gap on top of
# the space already reserved for that row's title, bracket overflow, and the
# modality labels of the row above. Bumping this loosens the whole figure
# uniformly; it does NOT need changing when ylims change.
ROW_BREATHING_IN = 0.18


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


def _to_bold_mathtext(label: str) -> str:
    """Wrap a label as bold Computer Modern mathtext, one span per line.

    Spaces are escaped (\\ ) so multi-word names stay on one line; this lets
    the ROI title use the traditional LaTeX font while ticks/axis labels keep
    the matplotlib defaults.
    """
    out: List[str] = []
    for line in str(label).split("\n"):
        esc = line.replace(" ", r"\ ")
        out.append(rf"$\mathbf{{{esc}}}$")
    return "\n".join(out)


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
    if "auditorycortex" in tok or "heschl" in tok:
        return "auditory_cortex"
    if "visualcortex" in tok or "occip" in tok:
        return "visual_cortex"

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


def _pts_to_figfrac_y(fig: plt.Figure, pts: float) -> float:
    """Convert a vertical distance in points to a figure-height fraction.

    This is the key to ylim-independent legend/title spacing: a point is a
    fixed physical distance, so the converted fraction automatically shrinks
    as the figure grows taller, keeping the on-page gap constant.
    """
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
    bracket_height_fig: float = 0.0,  # deprecated/ignored (kept for compat)
    lw: float = 1.2,
    fs: float = 14.0,
) -> None:
    """Draw a bracket spanning ax_left -> ax_right.

    X uses figure coords (spans subplots). Y anchors to ax_left data coords.
    The vertical end-cap height is `h_data` in data units, so it is driven by
    ANNOT_CAP_TICKS and stays a constant size on the page (one tick always maps
    to inches_per_step inches) regardless of the figure's height.
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


def span_annotation_figx_figspan(
    fig: plt.Figure,
    ax_ref: plt.Axes,
    x1: float,
    x2: float,
    text: str,
    y_data: float,
    h_data: float,
    bracket_height_fig: float = 0.0,  # deprecated/ignored (kept for compat)
    lw: float = 1.2,
    fs: float = 14.0,
) -> None:
    """Draw a bracket using figure x-coordinates.

    Y anchors to ax_ref data coords; the cap height is `h_data` in data units
    (driven by ANNOT_CAP_TICKS, constant on the page).
    """
    if not getattr(fig, "_ann_canvas_drawn", False):
        fig.canvas.draw()
        fig._ann_canvas_drawn = True

    y0 = _ydata_to_yfig(fig, ax_ref, y_data)
    y1 = _ydata_to_yfig(fig, ax_ref, y_data + float(h_data))

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
        return (med, med)
    if rng is None:
        rng = np.random.default_rng()
    idx = rng.integers(0, n, size=(n_boot, n))
    meds = np.median(x[idx], axis=1)
    lo = float(np.quantile(meds, alpha / 2.0))
    hi = float(np.quantile(meds, 1.0 - alpha / 2.0))
    return (lo, hi)


def bootstrap_conf_intervals(
    data: list[np.ndarray],
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 12345,
) -> np.ndarray:
    """Compute median CIs for each box in data."""
    rng = np.random.default_rng(seed)
    cis: list[tuple[float, float]] = []
    for vals in data:
        lo, hi = bootstrap_median_ci(
            vals=np.asarray(vals, dtype=float),
            n_boot=n_boot,
            alpha=alpha,
            rng=rng,
        )
        cis.append((lo, hi))
    return np.asarray(cis, dtype=float)


def _poly_xspan_at_y(
    patch,
    y: float,
) -> tuple[float, float] | None:
    """Return x-span of a box polygon at a given y (data coords).

    Computes intersections of the polygon edges with the horizontal
    line y = y. Returns (xmin, xmax) or None if it cannot be computed.
    """
    verts = patch.get_path().vertices
    if verts.shape[0] < 3:
        return None

    xs: list[float] = []
    n = verts.shape[0]

    for i in range(n):
        x0, y0 = verts[i]
        x1, y1 = verts[(i + 1) % n]

        # Edge crosses y (including endpoints).
        if (y0 <= y <= y1) or (y1 <= y <= y0):
            dy = y1 - y0
            if abs(dy) < 1e-12:
                # Horizontal edge lying on y: include both endpoints.
                if abs(y - y0) < 1e-12:
                    xs.extend([float(x0), float(x1)])
                continue

            t = (y - y0) / dy
            x = x0 + t * (x1 - x0)
            xs.append(float(x))

    if len(xs) < 2:
        return None

    return (min(xs), max(xs))



# ===================== BOX COLOR PALETTE (module scope) ============= #

COLOR_MAP = {
    "Pooled": {
        # Turquoise panel adjusted lighter
        # Beat slightly lighter for black mean-line visibility
        # Intervals rebalanced accordingly

        "Production": {
            "Beat": "#1ead9a",
            "Interval": "#a6e5d3",
        },

        "Perception": {
            "Beat": "#01e1e1",
            "Interval": "#BBFCFA",
        },

        "NTFD": {
            "Beat": "#16ace7",
            "Interval": "#91d2e1",
        },

        "NTFD Random": {
            "Beat": "#6fa9b6",
            "Interval": "#cce1e4",
        },
    },

    "Auditory": {
        "Production": {
            "Beat": "#d49a00",
            "Interval": "#f6d56f",
        },

        "Perception": {
            "Beat": "#ffb300",
            "Interval": "#fffb8a",
        },

        "NTFD": {
            "Beat": "#bfba33",
            "Interval": "#efec9f",
        },

        "NTFD Random": {
            "Beat": "#b5b18f",
            "Interval": "#fffbe2",
        },
    },

    "Visual": {

        # Visual panel only

        "Production": {
            "Beat": "#8682e7",
            "Interval": "#c2cbf0",   # darker
        },

        "Perception": {
            "Beat": "#b06ff1",
            "Interval": "#d8b4f5",   # mid-light
        },

        "NTFD": {
            "Beat": "#d44ce6",
            "Interval": "#fabaf5",   # lighter
        },

        "NTFD Random": {
            "Beat": "#cc9cc2",
            "Interval": "#f2e2f8",
        },
    },
}

def cat_color(mod: str, task: str, cat: str) -> str:
    """Return explicit box color."""
    if cat == "Random":
        return "#bdbdbd"

    return COLOR_MAP.get(mod, {}).get(task, {}).get(cat, "#808080")


# ============================ PLOTTING ============================= #


def plot_psc_boxplots(
    df: pd.DataFrame,
    outpath: str | Path,
    figsize_scale: float = 1.0,
    audivisual_only: bool = False,
    pooled_only: bool = False,
    include_ntfd_random: bool = False,
    y_limits: dict[str, tuple[float, float]] | None = None,
    show_yaxis: dict[str, bool] | None = None,
    x_leg_dx: float = -0.15,
    tag_dx_beat: float = -0.0885,
    tag_dx_interval: float = -0.0515,
    tag_dy: float = -0.001,
    row_gap: float = 0.005,
    roi_subset: Sequence[str] | None = None,
    draw_legend: bool = True,
    draw_title: bool = True,
    title_mathtext: bool = False,
    within_modality_bracket_height_fig: float = (
        WITHIN_MODALITY_BRACKET_HEIGHT_FIG
    ),
    between_modality_bracket_height_fig: float = (
        BETWEEN_MODALITY_BRACKET_HEIGHT_FIG
    ),
) -> None:
    """Plot PSC boxplots by ROI and modality/task blocks.

    roi_subset:    if given, only these ROI short-keys are drawn, in order
                   (otherwise the full ROI_ORDER is used).
    draw_legend:   set False to omit the Beat/Interval legend at the top.
    title_mathtext: render the ROI title in bold Computer Modern (mathtext)
                   while leaving ticks and axis labels at matplotlib defaults.
    """
    outpath = Path(outpath)
    if outpath.suffix == "":
        raise ValueError("outpath must end with .png or .pdf")
    outpath.parent.mkdir(parents=True, exist_ok=True)

    # Render the title in Computer Modern (mathtext) without disturbing the
    # default font used for ticks/axis labels. Restored at the end.
    _saved_fontset = plt.rcParams["mathtext.fontset"]
    if title_mathtext:
        plt.rcParams["mathtext.fontset"] = "cm"

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
    roi_source = list(roi_subset) if roi_subset is not None else list(ROI_ORDER)
    rois = [_resolve_roi(r, roi_values) for r in roi_source]
    n_rows = len(rois)

    # ------------------------ style params -------------------------
    # Box colors live at module scope (see COLOR_MAP / cat_color).
    _cat_color = cat_color

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

    # ------------------------------------------------------------------
    # Bottom y-padding below the data (cosmetic only). Expressed as a
    # fraction of the whisker range so a panel never clips its lowest box.
    # ------------------------------------------------------------------
    ypad_frac = 0.06

    # Optional, normally-empty per-ROI nudges. With the tick-step model
    # below these are NO LONGER NEEDED for correct spacing; they exist only
    # if you ever want to hand-tweak a single ROI. Keys (all in tick units):
    #   anchor_pad_ticks, clearance_ticks, layer_ticks, headroom_ticks,
    #   within_clearance_ticks, within_layer_ticks, cross_gap_ticks
    roi_annot_overrides: dict[str, dict[str, float]] = {}

    zero_line_color = "0.25"
    zero_line_ls = "--"
    zero_line_lw = 1.2
    zero_line_zorder = 1

    ytick_step = 0.20

    inches_per_step = 0.4
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
                    "overflow_in": 0.0,
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

        # ---- per-ROI annotation spacing, in TICK units (ylim-independent) ----
        rkey = _roi_key(roi)
        ov = roi_annot_overrides.get(rkey or "", {})

        anchor_pad_t = float(ov.get("anchor_pad_ticks", ANNOT_ANCHOR_PAD_TICKS))
        clearance_t = float(ov.get("clearance_ticks", ANNOT_CLEARANCE_TICKS))
        layer_t = float(ov.get("layer_ticks", ANNOT_LAYER_TICKS))
        cap_t = float(ov.get("cap_ticks", ANNOT_CAP_TICKS))
        headroom_t = float(ov.get("headroom_ticks", ANNOT_HEADROOM_TICKS))
        within_clear_t = float(
            ov.get("within_clearance_ticks", WITHIN_CLEARANCE_TICKS)
        )
        within_layer_t = float(
            ov.get("within_layer_ticks", WITHIN_LAYER_TICKS)
        )
        within_cap_t = float(ov.get("within_cap_ticks", WITHIN_CAP_TICKS))
        cross_gap_t = float(ov.get("cross_gap_ticks", CROSS_GAP_TICKS))

        # Height (in ticks) of the within-axis bracket stack, per modality.
        # Measured from the data top up to the top cap of the last bracket.
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
                    within_clear_t
                    + (n_within - 1) * within_layer_t
                    + within_cap_t
                )

        # ---- data extent (whiskers), per axis and overall ----
        axis_top_by_key: Dict[Tuple[str, str], float] = {}
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
            axis_hi: float | None = None
            for cat in cats:
                vals = paired[cat].dropna().to_numpy()
                if vals.size < 3:
                    continue
                stats = cbook.boxplot_stats(vals, whis=whis)[0]
                w_lows.append(float(stats["whislo"]))
                w_highs.append(float(stats["whishi"]))
                hi = float(stats["whishi"])
                axis_hi = hi if axis_hi is None else max(axis_hi, hi)
            if axis_hi is not None:
                axis_top_by_key[(mod, task)] = axis_hi

        if w_lows:
            y_min, y_max = min(w_lows), max(w_highs)
        else:
            roi_vals = df.loc[df["ROI"] == roi, "PSC"].dropna().to_numpy()
            y_min, y_max = float(roi_vals.min()), float(roi_vals.max())

        yr = max(y_max - y_min, 0.1)
        pad = float(ov.get("pad_frac", ypad_frac)) * yr

        def _axis_top(mod: str, task: str) -> float:
            return axis_top_by_key.get((mod, task), y_max)

        # Per-modality anchor for across-task spans = tallest box in that
        # modality block (so a span hugs only the data it actually covers).
        mod_anchor: Dict[str, float] = {}
        for (mm, tt) in ax_keys:
            mod_anchor[mm] = max(mod_anchor.get(mm, -np.inf), _axis_top(mm, tt))
        # Cross-modality anchor = tallest box across Auditory+Visual blocks.
        cross_anchor = max(
            (_axis_top(mm, tt) for (mm, tt) in ax_keys
             if mm in ("Auditory", "Visual")),
            default=y_max,
        )

        # ---- how tall is each annotation stack, in TICKS, above its anchor --
        max_stack_span = max(
            (len(v) for v in eligible_by_mod.values()), default=0
        )
        max_stack_within = max(
            (len(v) for v in eligible_within_by_ax.values()), default=0
        )
        max_stack_cross = len(eligible_cross)

        # Span brackets sit above any within-axis stack in the same modality.
        within_offset_t = max(
            (v + (clearance_t if v > 0.0 else 0.0)
             for v in within_stack_top_by_mod.values()),
            default=0.0,
        )
        span_top_t = 0.0
        if max_stack_span > 0:
            span_top_t = (
                within_offset_t
                + clearance_t
                + (max_stack_span - 1) * layer_t
                + cap_t
            )

        # Cross-modality brackets sit above the span stack.
        cross_start_t = 0.0
        cross_top_t = 0.0
        if max_stack_cross > 0:
            base_for_cross = span_top_t if max_stack_span > 0 else within_offset_t
            cross_start_t = base_for_cross + cross_gap_t
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

        # Reserve head space using each family's OWN anchor (not the global
        # top), so a low-block span does not inflate the whole panel.
        span_anchor = max(
            (mod_anchor.get(m, y_max) for m, v in eligible_by_mod.items() if v),
            default=y_min,
        )
        needed_tops = [y_min]
        if max_stack_span > 0:
            needed_tops.append(
                span_anchor + (anchor_pad_t + span_top_t + headroom_t) * ytick_step
            )
        if max_stack_cross > 0:
            needed_tops.append(
                cross_anchor
                + (anchor_pad_t + cross_top_t + headroom_t) * ytick_step
            )
        if max_stack_within > 0:
            win_anchor = max(
                (_axis_top(m, t)
                 for (m, t), v in eligible_within_by_ax.items() if v),
                default=y_max,
            )
            needed_tops.append(
                win_anchor
                + (anchor_pad_t + within_only_top_t + headroom_t) * ytick_step
            )
        top_needed = max(needed_tops)
        y_lim_raw = (y_min - pad, max(y_max, top_needed))

        # ---- resolve the y-limits / ticks ----
        explicit_ylim = None
        if (y_limits is not None) and (rkey is not None):
            explicit_ylim = y_limits.get(rkey)

        if explicit_ylim is not None:
            y0, y1 = float(explicit_ylim[0]), float(explicit_ylim[1])
            # Modularity guarantee: if the requested top cannot hold the
            # annotation stack, lift it to the next tick (never overlap).
            if EXPAND_TOP_FOR_ANNOTATIONS and (y1 < y_lim_raw[1] - 1e-9):
                y1 = float(np.ceil((y_lim_raw[1] - 1e-9) / ytick_step)
                           * ytick_step)
            y_ticks = np.arange(y0, y1 + 0.5 * ytick_step, ytick_step)
            if y_ticks.size < 2:
                y_ticks = np.array([y0, y1])
        else:
            eps = 1e-9
            y0 = float(np.floor((y_lim_raw[0] + eps) / ytick_step) * ytick_step)
            y1 = float(np.ceil((y_lim_raw[1] - eps) / ytick_step) * ytick_step)
            y0 = min(y0, 0.0)
            y1 = max(y1, 0.0)
            y_ticks = np.arange(y0, y1 + 0.5 * ytick_step, ytick_step)
            if y_ticks.size < 2:
                y_ticks = np.array([y0, y1])

        n_steps = max(int(round((y1 - y0) / ytick_step)), 1)
        row_h = max(min_row_height, n_steps * inches_per_step)

        # How far the bracket stack reaches ABOVE the enforced top spine,
        # converted to inches (one tick == inches_per_step inches). This is
        # the room the row above must reserve so brackets never hit a title.
        overflow_data = max(0.0, top_needed - y1)
        overflow_in = (overflow_data / ytick_step) * inches_per_step
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
                "overflow_in": overflow_in,
                # tick-unit annotation parameters (consumed in PASS 2)
                "tick": ytick_step,
                "anchor_pad_t": anchor_pad_t,
                "clearance_t": clearance_t,
                "layer_t": layer_t,
                "cap_t": cap_t,
                "within_clear_t": within_clear_t,
                "within_layer_t": within_layer_t,
                "within_cap_t": within_cap_t,
                "cross_start_t": cross_start_t,
                "eligible_by_mod": eligible_by_mod,
                "eligible_within_by_ax": eligible_within_by_ax,
                "eligible_cross": eligible_cross,
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
    fig_h_axes = float(sum(d["row_h"] for d in roi_specs)) * figsize_scale

    fig, axes = plt.subplots(
        nrows=len(roi_specs),
        ncols=n_cols,
        figsize=(fig_w, fig_h_axes),
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

    # ================== manual vertical layout ==================
    # Axes keep EXACTLY the y-limits requested. Each inter-row gap is sized to
    # hold: the modality labels under the row above + the bracket overflow of
    # the row below (brackets drawn in the margin) + that row's title. Rows
    # without brackets stay tight; rows with tall stacks get just what they
    # need. Nothing here depends on the ylim VALUES, only on the (already
    # computed) overflow, so changing ylims can never cause a collision.
    PT = 1.0 / 72.0

    def _title_in(spec: dict) -> float:
        if spec["roi"] is None:
            return 0.0
        n_lines = spec["roi_label"].count("\n") + 1
        return (n_lines * (axis_label_fs + 8) * 1.30 + TITLE_GAP_PT) * PT

    # Space below a row: x-tick labels + gap + modality block label.
    modlabel_in = (
        xlabel_fs * 1.30 + xlabel_pad + MODLABEL_GAP_PT
        + (axis_label_fs + 2) * 1.30 + 6.0
    ) * PT
    breathing_in = ROW_BREATHING_IN
    # Legend block sitting above the first ROI title.
    legend_in = (
        LEGEND_TITLE_CLEAR_PT + LEGEND_ROW_GAP_PT + LEGEND_CI_GAP_PT
        + (axis_label_fs + 2) * 1.30 + 18.0
    ) * PT

    row_h_in = [d["row_h"] * figsize_scale for d in roi_specs]
    ovf_in = [float(d.get("overflow_in", 0.0)) for d in roi_specs]
    ttl_in = [_title_in(d) for d in roi_specs]

    top_margin = ovf_in[0] + ttl_in[0] + legend_in + breathing_in
    bottom_margin = modlabel_in + breathing_in
    gaps = [
        modlabel_in + ovf_in[r + 1] + ttl_in[r + 1] + breathing_in
        for r in range(len(roi_specs) - 1)
    ]

    fig_h = top_margin + sum(row_h_in) + sum(gaps) + bottom_margin
    fig.set_figheight(fig_h)

    # x-layout comes from the gridspec (width_ratios / wspace); y is overwritten.
    fig.subplots_adjust(
        left=0.015, right=0.990, top=0.999, bottom=0.001, wspace=0.22
    )

    y_top_in = fig_h - top_margin  # top edge (inches) of the current row
    for r in range(len(roi_specs)):
        h = row_h_in[r]
        y0f = (y_top_in - h) / fig_h
        hf = h / fig_h
        for j, (_m, _t) in enumerate(col_spec_block):
            ax = axes[r, j]
            pos = ax.get_position()
            ax.set_position([pos.x0, y0f, pos.width, hf])
        y_top_in -= h + (gaps[r] if r < len(roi_specs) - 1 else 0.0)

    # Per-row overflow in figure-fraction (used to lift titles above brackets).
    ovf_frac = [o / fig_h for o in ovf_in]

    # --- Condition legend: 2 rows × N cols ---
    if not getattr(fig, "_top_canvas_drawn", False):
        fig.canvas.draw()
        fig._top_canvas_drawn = True

    cols_nonspacer = [
        j for j, (m, _t) in enumerate(col_spec_block)
        if m != "SPACER"
    ]
    ax_last = axes[0, cols_nonspacer[-1]]

    x_right = float(ax_last.get_position().x1)
    y_top_axes = float(
        max(axes[0, j].get_position().y1 for j in cols_nonspacer)
    )

    if draw_legend:
        # Legend vertical layout in ABSOLUTE POINTS (constant on the page).
        # The bottom swatch row is lifted to clear the first ROI title, so the
        # legend never lands on top of "Dorsal Striatum" regardless of fig height.
        first_label = roi_specs[0]["roi_label"] if roi_specs else ""
        n_title_lines = first_label.count("\n") + 1
        title_h_pt = n_title_lines * (axis_label_fs + 8) * 1.25

        y_int = y_top_axes + ovf_frac[0] + _pts_to_figfrac_y(
            fig, TITLE_GAP_PT + title_h_pt + LEGEND_TITLE_CLEAR_PT
        )
        y_beat = y_int + _pts_to_figfrac_y(fig, LEGEND_ROW_GAP_PT)
        y_ci = y_beat + _pts_to_figfrac_y(fig, LEGEND_CI_GAP_PT)

        fig.text(
            x_right,
            y_ci,
            "95% bootstrap CI for the Median of PSC",
            ha="right",
            va="top",
            fontsize=axis_label_fs + 2,
            color="k",
        )

        legend_mods: list[str]
        if pooled_only:
            legend_mods = ["Pooled"]
        elif audivisual_only:
            legend_mods = ["Auditory", "Visual"]
        else:
            legend_mods = ["Auditory", "Visual", "Pooled"]

        beat_colors: list[tuple[float, float, float]] = []
        interval_colors: list[tuple[float, float, float]] = []

        for mod in legend_mods:
            for task in tasks_per_block:
                beat_colors.append(_cat_color(mod, task, "Beat"))
                interval_colors.append(_cat_color(mod, task, "Interval"))

        beat_handles = [
            Patch(facecolor=c, edgecolor="0.2")
            for c in beat_colors
        ]
        int_handles = [
            Patch(facecolor=c, edgecolor="0.2")
            for c in interval_colors
        ]

        x_leg = x_right + float(x_leg_dx)

        fig.legend(
            handles=beat_handles,
            labels=[""] * len(beat_handles),
            loc="upper right",
            bbox_to_anchor=(x_leg, y_beat),
            ncol=len(beat_handles),
            frameon=False,
            fontsize=axis_label_fs,
            handlelength=0.95,
            handletextpad=0.0,
            columnspacing=0.45,
            borderaxespad=0.0,
        )

        fig.legend(
            handles=int_handles,
            labels=[""] * len(int_handles),
            loc="upper right",
            bbox_to_anchor=(x_leg, y_int),
            ncol=len(int_handles),
            frameon=False,
            fontsize=axis_label_fs,
            handlelength=0.95,
            handletextpad=0.0,
            columnspacing=0.45,
            borderaxespad=0.0,
        )

        fig.text(
            x_right + float(tag_dx_beat),
            y_beat + float(tag_dy),
            "Beat",
            ha="right",
            va="top",
            fontsize=axis_label_fs + 2,
            color="k",
        )

        fig.text(
            x_right + float(tag_dx_interval),
            y_int + float(tag_dy),
            "Interval",
            ha="right",
            va="top",
            fontsize=axis_label_fs + 2,
            color="k",
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

            # --- Bootstrap notch confidence intervals ---
            use_bootstrap_notches = True
            boot_n = 5000
            boot_alpha = 0.05
            boot_seed = 12345

            conf_intervals = None
            if use_bootstrap_notches:
                conf_intervals = bootstrap_conf_intervals(
                    data=data,
                    n_boot=boot_n,
                    alpha=boot_alpha,
                    seed=boot_seed,
                )

            box_lw = 1.2
            bp = ax.boxplot(
                data,
                positions=positions,
                widths=box_w,
                notch=True,
                patch_artist=True,
                showfliers=False,
                showmeans=False,
                meanline=False,
                meanprops=dict(
                    color="k",
                    linestyle="-",
                    linewidth=box_lw,),
                whis=whis,
                medianprops={"linewidth": 0, "color": "none"},
                conf_intervals=conf_intervals,
            )

            for patch, cat in zip(bp["boxes"], cats):
                patch.set_facecolor(_cat_color(mod, task, cat))
                patch.set_alpha(box_alpha)

            for patch, values in zip(bp["boxes"], data):
                vals = np.asarray(values, dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue

                mean_val = float(np.mean(vals))

                span = _poly_xspan_at_y(patch, mean_val)
                if span is None:
                    # Fallback: box full width (should be rare).
                    verts = patch.get_path().vertices
                    x_left = float(verts[:, 0].min())
                    x_right = float(verts[:, 0].max())
                else:
                    x_left, x_right = span

                ax.plot(
                    [x_left, x_right],
                    [mean_val, mean_val],
                    color="k",
                    lw=patch.get_linewidth(),
                    zorder=3,
                    solid_capstyle="butt",
                )

            ax.set_ylim(*spec["y_lim"])
            ax.yaxis.set_major_locator(MultipleLocator(ytick_step))
            ax.spines["left"].set_bounds(*spec["y_lim"])
            ax.set_yticks(spec["y_ticks"])
            ax.yaxis.set_major_formatter(y_formatter)

            # --- Optional per-ROI y-axis visibility 
            # (computed here, applied later)
            rkey = _roi_key(str(spec.get("roi", ""))) or ""
            display_axis = True
            if show_yaxis is not None:
                display_axis = show_yaxis.get(rkey, True)

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

            # --- Apply per-ROI y-axis visibility 
            # (after column-specific formatting)
            if not display_axis:
                ax.tick_params(axis="y", left=False, labelleft=False)
                ax.set_yticklabels([])
                ax.spines["left"].set_visible(False)
                ax.set_ylabel("")
                ax.yaxis.label.set_visible(False)
                ax.yaxis.set_ticks_position("none")

            within_anns = spec.get("eligible_within_by_ax", {}).get(
                (mod, task),
                [],
            )
            if within_anns and ("Beat" in cats) and ("Interval" in cats):
                x1 = float(positions[0])
                x2 = float(positions[1])
                tick = spec["tick"]
                anchor = spec["y_max"] + spec["anchor_pad_t"] * tick
                for level, ann in enumerate(within_anns):
                    p = float(ann["pvalue"])
                    text = pval_label_converter([p])[0]
                    y_data = anchor + (
                        spec["within_clear_t"]
                        + level * spec["within_layer_t"]
                    ) * tick
                    h_data = spec["within_cap_t"] * tick
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
        if row_axes and draw_title:
            x0 = min(a.get_position().x0 for a in row_axes)
            x1 = max(a.get_position().x1 for a in row_axes)
            y1 = max(a.get_position().y1 for a in row_axes)

            # String values for fonweight: 
            # 'ultralight', 'light', 'normal', 'regular', 'book', 
            # 'medium', 'roman', 'semibold', 'demibold', 'demi', 
            # 'bold', 'heavy', 'extra bold', 'black'
            # Relative values like 'lighter' and 'bolder' are also 
            # available.
            _title_txt = spec["roi_label"]
            _title_kw = dict(fontweight="semibold")
            if title_mathtext:
                # Bold Computer Modern via mathtext; ticks/labels stay default.
                _title_txt = _to_bold_mathtext(spec["roi_label"])
                _title_kw = {}
            fig.text(
                (x0 + x1) / 2.0,
                y1 + ovf_frac[r] + _pts_to_figfrac_y(fig, TITLE_GAP_PT),
                _title_txt,
                ha="center",
                va="bottom",
                fontsize=axis_label_fs + 8,
                color="k",
                **_title_kw,
            )

        # Modality block labels under each block, below the task xlabels.
        if not getattr(fig, "_modlabel_canvas_drawn", False):
            fig.canvas.draw()
            fig._modlabel_canvas_drawn = True

        renderer = fig.canvas.get_renderer()
        gap_fig = _pts_to_figfrac_y(fig, MODLABEL_GAP_PT)

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
                MOD_LABEL.get(mod, mod),
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
                tick = spec["tick"]
                anchor = spec["y_max"] + spec["anchor_pad_t"] * tick
                # within-axis stack (if any) plus a clearance gap sits below.
                within_offset_t = (
                    within_top + spec["clearance_t"] if within_top > 0.0 else 0.0
                )
                y_data = anchor + (
                    within_offset_t
                    + spec["clearance_t"]
                    + level * spec["layer_t"]
                ) * tick
                h_data = spec["cap_t"] * tick

                span_annotation_datay_figspan(
                    fig,
                    ax_left=ax_left,
                    ax_right=ax_right,
                    text=text,
                    y_data=y_data,
                    h_data=h_data,
                )

        eligible_cross = spec.get("eligible_cross", [])
        cross_start_t = float(spec.get("cross_start_t", 0.0))
        if eligible_cross and cross_start_t > 0.0:
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

                tick = spec["tick"]
                anchor = spec["y_max"] + spec["anchor_pad_t"] * tick
                y_data = anchor + (
                    cross_start_t + level * spec["layer_t"]
                ) * tick
                h_data = spec["cap_t"] * tick

                use_lower_lines = (
                    len(tasks) == len(TASKS_MAIN)
                    and set(tasks) == set(TASKS_MAIN)
                )

                if use_lower_lines:
                    ax_aud_left = ax_lookup.get(("Auditory", t_left))
                    ax_aud_right = ax_lookup.get(("Auditory", t_right))
                    ax_vis_left = ax_lookup.get(("Visual", t_left))
                    ax_vis_right = ax_lookup.get(("Visual", t_right))
                    if (
                        (ax_aud_left is None)
                        or (ax_aud_right is None)
                        or (ax_vis_left is None)
                        or (ax_vis_right is None)
                    ):
                        continue

                    b_al = ax_aud_left.get_position()
                    b_ar = ax_aud_right.get_position()
                    b_vl = ax_vis_left.get_position()
                    b_vr = ax_vis_right.get_position()

                    x_aud_left = b_al.x0
                    x_aud_right = b_ar.x1
                    x_vis_left = b_vl.x0
                    x_vis_right = b_vr.x1

                    y_base = _ydata_to_yfig(fig, ax_aud, y_data)

                    line_margin = 0.035
                    fig.add_artist(
                        Line2D(
                            [x_aud_left + line_margin,
                             x_aud_right - line_margin],
                            [y_base, y_base],
                            transform=fig.transFigure,
                            lw=1.2,
                            c="k",
                        )
                    )
                    fig.add_artist(
                        Line2D(
                            [x_vis_left + line_margin,
                             x_vis_right - line_margin],
                            [y_base, y_base],
                            transform=fig.transFigure,
                            lw=1.2,
                            c="k",
                        )
                    )

                    x_mid_aud = (x_aud_left + x_aud_right) / 2.0
                    x_mid_vis = (x_vis_left + x_vis_right) / 2.0

                    span_annotation_figx_figspan(
                        fig,
                        ax_ref=ax_aud,
                        x1=x_mid_aud,
                        x2=x_mid_vis,
                        text=text,
                        y_data=y_data,
                        h_data=h_data,
                    )
                else:
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
    plt.rcParams["mathtext.fontset"] = _saved_fontset


# ============================== I/O ================================ #


def _panel_legend(fig, x_right_fig, y_top_fig, mods, tasks, axis_label_fs,
                  swatch_w_in, swatch_h_in, row_gap_in, col_gap_in,
                  label_pad_in):
    """Draw the Beat/Interval colour legend at a fixed top-right anchor.

    Swatch sizes/gaps are in inches so the legend looks identical on any
    panel. (x_right_fig, y_top_fig) is the top-right corner in figure coords.
    """
    fw = fig.get_figwidth()
    fh = fig.get_figheight()
    sw = swatch_w_in / fw
    sh = swatch_h_in / fh
    cg = col_gap_in / fw
    rg = row_gap_in / fh
    pad = label_pad_in / fw

    pairs = [(m, t) for m in mods for t in tasks]
    n = len(pairs)
    total_w = n * sw + (n - 1) * cg
    x0 = x_right_fig - total_w

    # Beat row (top), Interval row (below it).
    for ri, cat in enumerate(("Beat", "Interval")):
        y = y_top_fig - ri * (sh + rg)
        for ci, (m, t) in enumerate(pairs):
            x = x0 + ci * (sw + cg)
            fig.add_artist(plt.matplotlib.patches.Rectangle(
                (x, y - sh), sw, sh, transform=fig.transFigure,
                facecolor=cat_color(m, t, cat), edgecolor="0.2", lw=0.8))
        fig.text(x_right_fig + pad, y - sh / 2.0, cat, ha="left",
                 va="center", fontsize=axis_label_fs + 2, color="k")


# Named legend presets (extensible for future use).
PANEL_LEGENDS = {
    "full": (["Auditory", "Visual", "Pooled"], list(TASKS_MAIN)),
    "audivisual": (["Auditory", "Visual"], list(TASKS_MAIN)),
    "pooled": (["Pooled"], list(TASKS_MAIN)),
}


def assemble_panel(
    df: pd.DataFrame,
    outpath: str | Path,
    rows: Sequence[Sequence[Tuple[str, str]]],
    row_ylims: Sequence[Tuple[float, float]],
    *,
    include_ntfd_random: bool = False,
    legend: str | None = "full",
    col_gap_in: float = 0.30,
    row_gap_in: float = 0.55,
    title_gap_in: float = 0.10,
    title_band_in: float = 0.42,
    side_pad_in: float = 0.20,
    top_pad_in: float = 0.20,
    bottom_pad_in: float = 0.20,
    legend_pad_in: float = 0.25,
    title_fontsize: int = 20,
    dpi: int = 300,
    cell_kwargs: dict | None = None,
) -> None:
    """Compose existing single-ROI plots into a centered multi-row panel.

    Each cell is rendered by ``plot_psc_boxplots`` (so it is exactly that
    function's output) with its title and legend suppressed, then tiled with
    Matplotlib. Within a row cells are bottom-aligned (PSC = 0 lines line up)
    and the row is centred; ROI titles are drawn at a common height per row in
    the default font; only the left-most cell of a row keeps its y-axis.

    Parameters
    ----------
    rows :
        One list per panel row; each entry is ``(roi_key, mode)`` where mode is
        ``"audivisual"`` or ``"pooled"``. Edit this to re-arrange / aggregate.
    row_ylims :
        One ``(lo, hi)`` per row, applied to every cell in that row.
    legend :
        Which legend preset to draw at the top-right (see PANEL_LEGENDS), or
        None for no legend. Kept as a parameter for future legend types.
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

    tmpdir = Path(tempfile.mkdtemp(prefix="psc_panel_"))
    try:
        # 1) render each cell (no title, no legend) and record its inch size
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
                    audivisual_only=(mode == "audivisual"),
                    pooled_only=(mode == "pooled"),
                    include_ntfd_random=include_ntfd_random,
                    roi_subset=[roi_key],
                    y_limits={roi_key: ylim},
                    show_yaxis={roi_key: ci == 0},
                    draw_legend=False, draw_title=False,
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

        # 2) geometry (inches)
        roi_values = list(pd.unique(df["ROI"]))

        def _label_lines(roi_key: str) -> int:
            rv = _resolve_roi(roi_key, roi_values) or roi_key
            return _pretty_roi_label(rv).count("\n") + 1

        line_h_in = title_fontsize * 1.35 / 72.0
        # title band per row scales with the tallest title in that row
        title_band = [
            line_h_in * max(_label_lines(rk) for rk, _ in row) + title_gap_in
            for row in rows
        ]

        row_w = [sum(cell_w[ri]) + col_gap_in * (len(cell_w[ri]) - 1)
                 for ri in range(len(rows))]
        row_h = [max(cell_h[ri]) for ri in range(len(rows))]
        content_w = max(row_w)

        legend_band = 0.0
        legend_label_w = 0.0
        if legend is not None:
            legend_band = 0.70  # vertical room reserved above the first row
            legend_label_w = 0.75  # right-hand room for "Beat"/"Interval"

        fig_w = content_w + 2 * side_pad_in + legend_label_w
        fig_h = (top_pad_in + bottom_pad_in + legend_band
                 + sum(title_band[ri] + row_h[ri] for ri in range(len(rows)))
                 + row_gap_in * (len(rows) - 1))

        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

        # 3) place cells + titles, top to bottom (titles TOP-aligned per row)
        y_top = fig_h - top_pad_in - legend_band
        for ri, row in enumerate(rows):
            cell_top = y_top - title_band[ri]
            cell_bottom = cell_top - row_h[ri]
            x = side_pad_in + (content_w - row_w[ri]) / 2.0  # centre row
            title_y = y_top / fig_h  # common top for every title in the row
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

        # 4) legend at top-right (swatches end at the content edge; labels in
        #    the reserved right margin so nothing is clipped)
        if legend is not None:
            mods, tasks = PANEL_LEGENDS[legend]
            _panel_legend(
                fig,
                x_right_fig=(side_pad_in + content_w) / fig_w,
                y_top_fig=(fig_h - top_pad_in) / fig_h,
                mods=mods, tasks=tasks, axis_label_fs=12,
                swatch_w_in=0.16, swatch_h_in=0.16,
                row_gap_in=0.06, col_gap_in=0.045, label_pad_in=0.06,
            )

        fig.savefig(outpath, dpi=dpi)
        plt.close(fig)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)



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

# Reporting uncorrected pvalues that survived to ...
# ... pHolm-Bonferroni correction across the 2way ANOVAs
# ... per ROI for Task and Modality as within-subject factors
# Pairwise comparisons only for the Modality factor, across all tasks
CROSS_AV_ANNOTATIONS: List[dict] = [
    dict(
        roi="auditory_cortex",
        tasks=["Production", "Perception", "NTFD"],
        pvalue=0.000000000000000013809316592507
    ),
    dict(
        roi="visual_cortex",
        tasks=["Production", "Perception", "NTFD"],
        pvalue=0.00000000363753126470459,
    ),
    dict(
        roi="pmd",
        tasks=["Production", "Perception", "NTFD"],
        pvalue=0.00366481397614986,
    ),
]

# ========================== CROSS-TASKS ============================ #

# Reporting uncorrected pvalues that survived to ...
# ... pHolm-Bonferroni correction across the 2way ANOVAs
# ... per ROI for Task and Modality as within-subject factors
# Pairwise comparisons only for the Task factor, within each modality
ANNOTATIONS: List[dict] = [
    dict(
        roi="auditory_cortex",
        modality="Auditory",
        task_pair=("Production", "NTFD"),
        pvalue=0.00000744213825471056,
    ),
    dict(
        roi="auditory_cortex",
        modality="Auditory",
        task_pair=("Perception", "NTFD"),
        pvalue=0.00000145219763263706,
    ),
    dict(
        roi="visual_cortex",
        modality="Auditory",
        task_pair=("Production", "Perception"),
        pvalue=0.000793532154268527,
    ),
    dict(
        roi="visual_cortex",
        modality="Auditory",
        task_pair=("Perception", "NTFD"),
        pvalue=0.0000151274029394139,
    ),
    dict(
        roi="visual_cortex",
        modality="Visual",
        task_pair=("Production", "Perception"),
        pvalue=0.000675594786736658,
    ),
    dict(
        roi="visual_cortex",
        modality="Visual",
        task_pair=("Production", "NTFD"),
        pvalue=0.000929304603146975,
    ),
    dict(
        roi="dorsal striatum",
        modality="Pooled",
        task_pair=("Production", "Perception"),
        pvalue=0.0000000207249593036707,
    ),
    dict(
        roi="dorsal striatum",
        modality="Pooled",
        task_pair=("Production", "NTFD"),
        pvalue=0.0000000737910684521213,
    ),
    dict(
        roi="cerebellum",
        modality="Pooled",
        task_pair=("Production", "Perception"),
        pvalue=0.00371185077635477,
    ),
    dict(
        roi="cerebellum",
        modality="Pooled",
        task_pair=("Production", "NTFD"),
        pvalue=0.000227317411495172,
    ),
    dict(
        roi="presma",
        modality="Pooled",
        task_pair=("Perception", "NTFD"),
        pvalue=0.00187469080530894,
    ),
    dict(
        roi="sma",
        modality="Pooled",
        task_pair=("Production", "Perception"),
        pvalue=0.000000650783541156441,
    ),
    dict(
        roi="sma",
        modality="Pooled",
        task_pair=("Production", "NTFD"),
        pvalue=0.0102669303444073,
    ),
    dict(
        roi="sma",
        modality="Pooled",
        task_pair=("Perception", "NTFD"),
        pvalue=0.0000202681809469692,
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
        y_limits={
            "auditory_cortex": (-0.4, 1.6),
            "visual_cortex": (-0.4, 1.6),
            "dstr": (-0.4, 1.6),
            "presma": (-0.2, 1.2),
            "sma": (-0.2, 1.2),
            "pmd": (-0.2, 1.2),
            "pmv": (-0.2, 1.2),
            "cereb": (-0.2, 1.2),
        },
        show_yaxis={
            "auditory_cortex": True,
            "visual_cortex": False,
            "dstr": False,
            "cereb": True,
            "presma": False,
            "sma": False,
            "pmd": False,
            "pmv": False,
        },
        row_gap=0.004,
        tag_dx_beat=-0.056,
        tag_dx_interval=-0.001,
        within_modality_bracket_height_fig=(
            WITHIN_MODALITY_BRACKET_HEIGHT_FIG
        ),
        between_modality_bracket_height_fig=(
            BETWEEN_MODALITY_BRACKET_HEIGHT_FIG
        ),
    )

    # Only with audio and visual plots
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
            "auditory_cortex": (-0.4, 1.6),
            "visual_cortex": (-0.4, 1.6),
            "dstr": (-0.4, 1.6),
            "presma": (-0.2, 1.2),
            "sma": (-0.2, 1.2),
            "pmd": (-0.2, 1.2),
            "pmv": (-0.2, 1.2),
            "cereb": (-0.2, 1.2),
        },
        show_yaxis={
            "auditory_cortex": True,
            "visual_cortex": False,
            "dstr": False,
            "cereb": True,
            "presma": False,
            "sma": False,
            "pmd": False,
            "pmv": False,
        },
        x_leg_dx=-0.175,
        tag_dx_beat=-0.056,
        tag_dx_interval=-0.001,
        tag_dy=-0.00075,
        row_gap=0.005,
        within_modality_bracket_height_fig=(
            WITHIN_MODALITY_BRACKET_HEIGHT_FIG
        ),
        between_modality_bracket_height_fig=(
            BETWEEN_MODALITY_BRACKET_HEIGHT_FIG
        ),
    )

    # 1c) Combined panel: re-arrange the individual ROI plots into rows.
    #     Each cell is the exact plot plot_psc_boxplots() produces; cells are
    #     bottom-aligned within a row so the PSC = 0 lines align, and only the
    #     left-most cell of each row shows the y-axis. Edit `panel_rows` /
    #     `panel_row_ylims` to re-arrange or to aggregate other plots.
    outpath_panel = Path(OUTPUT_PATH)
    outpath_panel = outpath_panel.with_name(
        outpath_panel.stem + "_panel" + outpath_panel.suffix
    )
    panel_rows = [
        [("auditory_cortex", "audivisual"),
         ("visual_cortex", "audivisual"),
         ("dstr", "pooled")],
        [("cereb", "pooled"),
         ("presma", "pooled"),
         ("sma", "pooled"),
         ("pmd", "audivisual"),
         ("pmv", "pooled")],
    ]
    panel_row_ylims = [(-0.4, 1.6), (-0.2, 1.2)]
    assemble_panel(
        df=df_in,
        outpath=outpath_panel,
        rows=panel_rows,
        row_ylims=panel_row_ylims,
        include_ntfd_random=False,
        legend="full",      # full Beat/Interval colour key at the top-right
        row_gap_in=0.30,    # vertical space between rows
        cell_kwargs=dict(tag_dx_beat=-0.056, tag_dx_interval=-0.001),
    )

    # 1b) Pooled-only figure (pooled modality block only)
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
    #     x_leg_dx=-0.15,
    #     tag_dy=-0.001,
    #     within_modality_bracket_height_fig=(
    #         WITHIN_MODALITY_BRACKET_HEIGHT_FIG
    #     ),
    #     between_modality_bracket_height_fig=(
    #         BETWEEN_MODALITY_BRACKET_HEIGHT_FIG
    #     ),
    # )

    # 2) Extended figures (with NTFD Random panels)
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
    #     within_modality_bracket_height_fig=(
    #         WITHIN_MODALITY_BRACKET_HEIGHT_FIG
    #     ),
    #     between_modality_bracket_height_fig=(
    #         BETWEEN_MODALITY_BRACKET_HEIGHT_FIG
    #     ),
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
    #     within_modality_bracket_height_fig=(
    #         WITHIN_MODALITY_BRACKET_HEIGHT_FIG
    #     ),
    #     between_modality_bracket_height_fig=(
    #         BETWEEN_MODALITY_BRACKET_HEIGHT_FIG
    #     ),
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
    #     within_modality_bracket_height_fig=(
    #         WITHIN_MODALITY_BRACKET_HEIGHT_FIG
    #     ),
    #     between_modality_bracket_height_fig=(
    #         BETWEEN_MODALITY_BRACKET_HEIGHT_FIG
    #     ),
    # )