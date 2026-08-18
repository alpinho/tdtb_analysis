#!/usr/bin/env python3
"""Estimation plot for the cross-cohort behavioural comparison.

Replaces Supplementary Table S2 (``etab:cohort_consistency``). For each
task and modality the panel shows three quantities on one axis:

    First cohort    Interval - Beat effect, 95% CI
    Second cohort   Interval - Beat effect, 95% CI
    Cohort x Structure contrast (first - second), 95% CI, Welch df

The contrast is the quantity the Welch test evaluates; it is drawn
against a zero line so that "no interaction" can be read directly off
the figure. BF01 is printed in-panel.

Inputs are the two TSVs written by ``cross_cohort_behaviour.py``:

``interaction_summary.tsv``
    supplies both cohorts' effects and CIs (eff0/eff1) together with
    welch_t, welch_df, p and bf01. Taking the effects from here rather
    than from the per-cohort file guarantees that the intervals drawn
    and the statistics printed come from one computation.
``per_cohort_effects.tsv``
    supplies the per-cohort sample sizes only.

There are no hard-coded results, so the figure cannot silently
reproduce superseded numbers. Each input's modification time is printed
on every run, and ``--stamp`` writes it into the figure footer while
drafting.

The contrast CI is built from the two cohort CIs: each half-width gives
that cohort's standard error, the contrast SE is the root sum of
squares, and the interval uses the reported Welch df. The implied Welch
t is compared against the reported one, and a warning is issued if they
disagree beyond ``TOLERANCE`` -- a consistency check on the inputs, not
a re-analysis.

Usage::

    python3 cohort_estimation_plot.py --results cross_cohort_results \\
        --out figures/panel_cohort-consistency [--stamp]

author: Ana Luisa Pinho
e-mail: agrilopi@uwo.ca

Created: August 17, 2026
Last update: August 2026

Compatibility: Python 3.10.14
"""

import argparse
import datetime as dt
import os
import sys

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

RESULTS_DIR = 'cross_cohort_results'

EFFECTS_FILE = 'per_cohort_effects.tsv'
INTERACTION_FILE = 'interaction_summary.tsv'

#: Columns of interaction_summary.tsv. Suffix 0 is the first cohort
#: (imaging), suffix 1 the second.
INTERACTION_COLUMNS = {
    'task': 'task',
    'modality': 'modality',
    'first_est': 'eff0',
    'first_lo': 'ci0_low',
    'first_hi': 'ci0_high',
    'second_est': 'eff1',
    'second_lo': 'ci1_low',
    'second_hi': 'ci1_high',
    'rep_t': 'welch_t',
    'rep_df': 'welch_df',
    'p': 'p',
    'bf01': 'bf01',
}

#: Columns of per_cohort_effects.tsv, used only for the sample sizes.
EFFECTS_COLUMNS = {
    'task': 'task',
    'modality': 'modality',
    'cohort': 'cohort',
    'grouping': 'grouping',
    'n': 'n',
}

#: Cohort labels in per_cohort_effects.tsv, in the order matching the
#: 0/1 suffixes of interaction_summary.tsv. Left as None, the labels are
#: detected and reported; set them explicitly once you know them.
COHORT_KEYS = None

COHORT_LABELS = ['First cohort', 'Second cohort']

#: Task display order, axis label, and the factor applied to the stored
#: values. Production and Perception are proportions of the Standard;
#: x100 makes them percentage points.
TASKS = [
    ('Production', 'Mean SA (% of Standard)', 100.0),
    ('Perception', 'DL (% of Standard)', 100.0),
    ('NTFD', 'RT (ms)', 1.0),
]

MODALITIES = [
    ('Auditory', 'Auditory  \u2014  PsychoPy in second cohort'),
    ('Visual', 'Visual  \u2014  same-software control'),
]

COLOURS = {
    'first': '#5B2C82',
    'second': '#C89B2C',
    'contrast': '#2F2F2F',
}

FIGSIZE = (13.0, 9.5)
FONT_BASE = 15
MARKER_SIZE = 9
LINE_WIDTH = 2.6

#: Warn if the implied Welch t drifts past this bound.
TOLERANCE_T = 0.10


def fail(message):
    """Abort with a message on stderr."""
    print('error: {}'.format(message), file=sys.stderr)
    raise SystemExit(1)


def read_tsv(path, mapping):
    """Read a TSV and rename its columns, or abort if any are absent."""
    if not os.path.exists(path):
        fail('{} not found. Run cross_cohort_behaviour.py first, or '
             'point --results at the directory holding it.'.format(path))

    frame = pd.read_csv(path, sep='\t')
    missing = [name for name in mapping.values()
               if name not in frame.columns]
    if missing:
        fail('{} lacks the column(s) {}. Found: {}. Edit the column '
             'maps at the top of this script.'.format(
                 path, ', '.join(missing), ', '.join(frame.columns)))

    stamp = dt.datetime.fromtimestamp(os.path.getmtime(path))
    print('reading {} (modified {})'.format(
        os.path.basename(path), stamp.strftime('%Y-%m-%d %H:%M')))

    return frame.rename(columns={v: k for k, v in mapping.items()})


def check_unique(frame, path):
    """Abort if a modality/task/cohort combination appears twice."""
    keys = ['modality', 'task', 'cohort']
    counts = frame.groupby(keys).size()
    repeated = counts[counts > 1]
    if not repeated.empty:
        fail('{} has more than one row for: {}. Restrict to a single '
             'set with --grouping LEVEL.'.format(
                 path,
                 '; '.join('/'.join(map(str, key))
                           for key in repeated.index)))


def select_grouping(frame, requested, path):
    """Optionally restrict to one level of the grouping column."""
    levels = sorted(frame['grouping'].dropna().astype(str).unique())

    if requested is not None:
        if requested not in levels:
            fail('grouping {!r} not present in {}. Available: {}.'.format(
                requested, path, ', '.join(levels)))
        frame = frame[frame['grouping'].astype(str) == requested]

    check_unique(frame, path)
    return frame


def report_grouping(frame):
    """Print the grouping level attached to each cohort."""
    pairs = frame[['cohort', 'grouping']].drop_duplicates()
    summary = '; '.join(
        '{} = {}'.format(cohort, grouping)
        for cohort, grouping in pairs.to_numpy())
    print('grouping: {}'.format(summary))


def resolve_cohorts(frame):
    """Return the two cohort labels, in eff0 / eff1 order."""
    labels = list(pd.unique(frame['cohort']))

    if COHORT_KEYS is not None:
        missing = [key for key in COHORT_KEYS if key not in labels]
        if missing:
            fail('cohort label(s) {} absent. Found: {}.'.format(
                ', '.join(map(str, missing)), ', '.join(map(str, labels))))
        return list(COHORT_KEYS)

    if len(labels) != 2:
        fail('expected two cohort labels, found {}: {}.'.format(
            len(labels), ', '.join(map(str, labels))))

    print('cohort labels: {!r} -> {}, {!r} -> {}  '
          '(set COHORT_KEYS or pass --swap-cohorts if reversed)'.format(
              labels[0], COHORT_LABELS[0],
              labels[1], COHORT_LABELS[1]))
    return labels


def load_results(results_dir, grouping, swap):
    """Return the assembled frame and a provenance list."""
    int_path = os.path.join(results_dir, INTERACTION_FILE)
    eff_path = os.path.join(results_dir, EFFECTS_FILE)

    inter = read_tsv(int_path, INTERACTION_COLUMNS)
    eff = read_tsv(eff_path, EFFECTS_COLUMNS)

    eff = select_grouping(eff, grouping, eff_path)
    report_grouping(eff)

    cohorts = resolve_cohorts(eff)
    if swap:
        cohorts = cohorts[::-1]

    sizes = eff.groupby(
        eff['cohort'].astype(str).str.strip().str.lower())['n'].max()
    inter = inter.copy()
    inter['first_n'] = int(sizes[str(cohorts[0]).strip().lower()])
    inter['second_n'] = int(sizes[str(cohorts[1]).strip().lower()])

    sources = []
    for path in (int_path, eff_path):
        stamp = dt.datetime.fromtimestamp(os.path.getmtime(path))
        sources.append('{} ({})'.format(
            os.path.basename(path), stamp.strftime('%Y-%m-%d %H:%M')))

    return inter, sources


def add_contrast(frame):
    """Derive the contrast and check it against the reported Welch t."""
    records = []

    for _, row in frame.iterrows():
        se_first = (row.first_hi - row.first_lo) / 2.0 / stats.t.ppf(
            0.975, row.first_n - 1)
        se_second = (row.second_hi - row.second_lo) / 2.0 / stats.t.ppf(
            0.975, row.second_n - 1)

        diff = row.first_est - row.second_est
        se = np.hypot(se_first, se_second)
        crit = stats.t.ppf(0.975, row.rep_df)
        t_hat = diff / se

        if np.isfinite(row.rep_t) and abs(t_hat - row.rep_t) > TOLERANCE_T:
            print('  warning: {}/{} implied t={:+.2f} vs reported '
                  '{:+.2f}'.format(row.task, row.modality, t_hat,
                                   row.rep_t), file=sys.stderr)

        records.append({
            'contrast_est': diff,
            'contrast_lo': diff - crit * se,
            'contrast_hi': diff + crit * se,
        })

    return pd.concat(
        [frame.reset_index(drop=True), pd.DataFrame(records)], axis=1)


def panel_entries(row):
    """Return the three (key, estimate, lo, hi, label) rows of a panel."""
    return [
        ('first', row.first_est, row.first_lo, row.first_hi,
         '{} (N\u2009=\u2009{})'.format(
             COHORT_LABELS[0], int(row.first_n))),
        ('second', row.second_est, row.second_lo, row.second_hi,
         '{} (N\u2009=\u2009{})'.format(
             COHORT_LABELS[1], int(row.second_n))),
        ('contrast', row.contrast_est, row.contrast_lo, row.contrast_hi,
         'Cohort \u00d7 Structure'),
    ]


def draw_panel(ax, row, scale, xlabel, show_labels):
    """Draw one task-by-modality panel."""
    ypos = {'first': 2.0, 'second': 1.0, 'contrast': -0.25}

    ax.axhspan(-0.95, 0.35, color='0.93', zorder=0, lw=0)
    ax.axvline(0.0, color='0.45', lw=1.2, ls=(0, (5, 4)), zorder=1)

    entries = panel_entries(row)
    ticks = []
    labels = []

    for key, est, lo, hi, label in entries:
        y = ypos[key]
        marker = 'D' if key == 'contrast' else 'o'
        ax.plot([lo * scale, hi * scale], [y, y],
                color=COLOURS[key], lw=LINE_WIDTH,
                solid_capstyle='round', zorder=3)
        ax.plot([est * scale], [y], marker,
                color=COLOURS[key], ms=MARKER_SIZE,
                mec='white', mew=1.4, zorder=4)
        ticks.append(y)
        labels.append(label)

    lo_all = min(entry[2] for entry in entries) * scale
    hi_all = max(entry[3] for entry in entries) * scale
    pad = 0.09 * (hi_all - lo_all)
    ax.set_xlim(min(lo_all, 0.0) - pad, max(hi_all, 0.0) + pad)

    ax.set_yticks(ticks)
    ax.set_yticklabels(labels if show_labels else [],
                       fontsize=FONT_BASE - 2)
    ax.set_ylim(-1.05, 2.75)
    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='x', labelsize=FONT_BASE - 3)
    ax.xaxis.set_major_locator(
        MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)

    ax.set_xlabel('Interval \u2212 Beat,  {}'.format(xlabel),
                  fontsize=FONT_BASE - 2)
    ax.text(0.985, 0.955,
            '$p$ = {:.2f}    BF$_{{01}}$ = {:.1f}'.format(row.p, row.bf01),
            transform=ax.transAxes, ha='right', va='top',
            fontsize=FONT_BASE - 3, color='0.25')


def select_row(frame, task, modality):
    """Return the row for one task and modality, matching case-blind."""
    match = frame[
        (frame.task.astype(str).str.strip().str.lower() == task.lower())
        & (frame.modality.astype(str).str.strip().str.lower()
           == modality.lower())
    ]
    return match


def make_figure(frame, outstem, sources=None):
    """Render the six panels and write PDF and PNG."""
    plt.rcParams.update({
        'font.size': FONT_BASE,
        'font.family': 'sans-serif',
        'axes.linewidth': 1.2,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    fig, axes = plt.subplots(
        len(TASKS), len(MODALITIES), figsize=FIGSIZE,
        gridspec_kw={
            'hspace': 0.62,
            'wspace': 0.10,
            'left': 0.235,
            'right': 0.975,
            'top': 0.855,
            'bottom': 0.075,
        },
    )

    for i, (task, xlabel, scale) in enumerate(TASKS):
        for j, (mod, mod_title) in enumerate(MODALITIES):
            ax = axes[i, j]
            match = select_row(frame, task, mod)
            if match.empty:
                print('  warning: no row for {}/{}'.format(task, mod),
                      file=sys.stderr)
                ax.set_visible(False)
                continue

            draw_panel(ax, match.iloc[0], scale, xlabel,
                       show_labels=j == 0)
            if i == 0:
                ax.set_title(mod_title, fontsize=FONT_BASE, pad=18,
                             color='0.25')

    for i, (task, _, _) in enumerate(TASKS):
        box = axes[i, 0].get_position()
        fig.text(0.022, (box.y0 + box.y1) / 2.0, task,
                 rotation=90, ha='center', va='center',
                 fontsize=FONT_BASE + 4, fontweight='bold', color='0.15')

    fig.text(0.5, 0.975, 'Behavioural consistency across cohorts',
             ha='center', va='top',
             fontsize=FONT_BASE + 7, fontweight='bold')
    fig.text(0.5, 0.933,
             'Diamonds give the between-cohort contrast; overlap with '
             'zero indicates no change in the '
             'Interval\u2009\u2212\u2009Beat effect.',
             ha='center', va='top', fontsize=FONT_BASE - 2, color='0.35')

    if sources:
        fig.text(0.5, 0.012, 'source: ' + '; '.join(sources),
                 ha='center', va='bottom', fontsize=FONT_BASE - 6,
                 color='0.55')

    for ext in ('pdf', 'png'):
        path = '{}.{}'.format(outstem, ext)
        fig.savefig(path, dpi=300 if ext == 'png' else None,
                    facecolor='white')
        print('wrote {}'.format(path))

    plt.close(fig)


def parse_args():
    """Return the parsed command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Estimation plot of the cross-cohort comparison.')
    parser.add_argument(
        '--results', default=RESULTS_DIR,
        help='directory holding the two TSVs (default: %(default)s)')
    parser.add_argument(
        '--out', default='cohort_consistency',
        help='output stem, without extension (default: %(default)s)')
    parser.add_argument(
        '--grouping', default=None,
        help='level of the grouping column to take sample sizes from')
    parser.add_argument(
        '--swap-cohorts', action='store_true',
        help='use if the detected cohort order is reversed')
    parser.add_argument(
        '--stamp', action='store_true',
        help='print the input files and modification times in the '
             'figure footer; use while drafting')
    return parser.parse_args()


def main():
    """Load the results, derive the contrasts, and write the figure."""
    args = parse_args()

    outdir = os.path.dirname(args.out)
    if outdir and not os.path.isdir(outdir):
        fail('output directory {!r} does not exist'.format(outdir))

    frame, sources = load_results(
        args.results, args.grouping, args.swap_cohorts)
    frame = add_contrast(frame)
    make_figure(frame, args.out, sources if args.stamp else None)


if __name__ == '__main__':
    main()