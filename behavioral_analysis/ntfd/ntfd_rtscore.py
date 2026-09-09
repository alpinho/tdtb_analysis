"""
Analysis of behavioral data for the NTFD Tasks of the TDTB project

Plots reaction times and success rates for the beat, interval and (when
present) random conditions, separately for the auditory and the visual
modality, at two levels of analysis.

Group level, in <results>/rt_and_success/group/
    One value per subject and condition, obtained by averaging over
    trials and sessions. Conditions are compared with a paired t-test
    across subjects, and displayed as boxplots. This is the level that
    carries the inference.

Individual level, in <results>/rt_and_success/individual/sub-<nn>/
    Single trials of one subject, treated as independent samples within
    that subject, so the conditions are compared with unpaired tests:
    Welch's t-test for reaction times, displayed as boxplots. Success
    rates are shown as proportions with 95% Wilson score intervals and
    no test: a boxplot of binary trials would collapse onto 0 and 1,
    and near ceiling the conditions differ by so few errors that an
    exact test can only return a couple of distinct p-values.

    These figures are descriptive. With a few tens of trials per
    condition the individual tests have low power, so a non-significant
    annotation is weak evidence rather than evidence of absence, and
    the inference should be read off the group level.

No trials are excluded other than by the dropna on 'reaction_time',
which removes non-responses. No reaction-time window and no trimming
are applied at either level, so the group and the individual figures
rest on the same trials.

Reaction times are always latency-corrected, by subtracting the device
latency and the button-press latency of the corresponding batch.

author: Ana Luisa Pinho
e-mail: agrilopi@uwo.ca

Created: May 5, 2024
Last update: September 2026

Compatibility: Python 3.10.14
"""

import os
import itertools
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import seaborn as sns

from scipy import stats
from matplotlib import pyplot as plt
from statannotations.Annotator import Annotator


# %%
# ======================== MAIN FUNCTIONS ==============================

def ffx_dvar(df):
    # Fixed Effects within subjects, averaged across subjects
    df_ffx = df.drop(['session'], axis=1)
    df_ffx = df_ffx.groupby([
        'condition', 'modality', 'subject']).mean(
            numeric_only=True).reset_index()

    return df_ffx


def get_ylim(data, pad_ratio=.15, default=(0., 1.), bounds=None):
    """Return padded y-limits from finite data values."""
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]

    if data.size == 0:
        return default

    ymin = np.min(data)
    ymax = np.max(data)
    pad = (ymax - ymin) * pad_ratio if ymax != ymin else 1.

    ylim_b = ymin - pad
    ylim_t = ymax + pad

    if bounds is not None:
        lower, upper = bounds
        if lower is not None:
            ylim_b = max(lower, ylim_b)
        if upper is not None:
            ylim_t = min(upper, ylim_t)

    return ylim_b, ylim_t


def plot_box_comparison(arrays_audio, arrays_visual, condition_labels,
                        y, title, output_dir, fname,
                        pvalues_audio, pvalues_visual,
                        loc='inside',
                        annotation_text_format='star',
                        hide_non_significant=True,
                        y_bounds=None,
                        show_title=False,
                        notch=True):
    """Boxplots of one dependent variable, auditory and visual side by side.

    ``arrays_audio`` and ``arrays_visual`` are lists of 1-D arrays, one per
    condition, in the same order as ``condition_labels``. The arrays may
    have different lengths.

    ``pvalues_audio`` and ``pvalues_visual`` must follow the order given by
    ``itertools.combinations(condition_labels, 2)``, i.e. for three
    conditions: (Beat, Interval), (Beat, Random), (Interval, Random).
    """

    allowed_formats = ['star', 'simple', 'full']
    if annotation_text_format not in allowed_formats:
        raise ValueError(
            'annotation_text_format must be one of: ' +
            ', '.join(allowed_formats))

    pairs = list(itertools.combinations(condition_labels, 2))

    if len(pvalues_audio) != len(pairs) or len(pvalues_visual) != len(pairs):
        raise ValueError(
            'Number of p-values does not match the number of condition '
            'pairs (%d expected).' % len(pairs))

    modalities = ['audio', 'visual']
    fig, ax = plt.subplots(1, len(modalities))

    plt.subplots_adjust(left=.15, bottom=.15, wspace=.25, top=.8)

    x = 'Conditions'

    all_values = np.concatenate(
        [np.asarray(a, dtype=float) for a in arrays_audio + arrays_visual])
    ylim_b, ylim_t = get_ylim(all_values, bounds=y_bounds)

    for m, modality in enumerate(modalities):
        if modality == 'audio':
            arrays = arrays_audio
            pvalue = list(pvalues_audio)
        else:
            assert modality == 'visual'
            arrays = arrays_visual
            pvalue = list(pvalues_visual)

        conditions = []
        values = []
        for label, array in zip(condition_labels, arrays):
            array = np.asarray(array, dtype=float)
            conditions += [label] * array.size
            values += array.tolist()

        d = {x: conditions, y: values}
        df = pd.DataFrame(data=d)

        colors = CONDITION_COLORS[:len(condition_labels)]

        sns.boxplot(
            ax=ax[m],
            x=x,
            y=y,
            data=df,
            order=condition_labels,
            palette=colors,
            medianprops={"color": "k", "linewidth": 0.},
            notch=notch,
            meanline=True,
            showmeans=True,
            meanprops=dict(color="tab:brown", linewidth=1.5),
            **{'boxprops': {'alpha': 0.5, 'edgecolor': 'black'}})

        annotator = Annotator(ax[m], pairs, data=df, x=x, y=y,
                              order=condition_labels)
        annotator.configure(test=None,
                            text_format=annotation_text_format,
                            fontsize=10.,
                            hide_non_significant=hide_non_significant,
                            loc=loc,
                            line_offset_to_group=1.,
                            line_height=0.05)

        annotator.set_pvalues(pvalue)
        annotator.annotate()

        if annotation_text_format == 'full':
            for text in ax[m].texts:
                current = text.get_text()
                if current.startswith('None '):
                    text.set_text(current.replace('None ', ''))

        ax[m].set_ylim(bottom=ylim_b, top=ylim_t)

        if m == 1:
            ax[m].axes.get_yaxis().set_visible(False)
            ax[m].spines['left'].set_visible(False)
            ax[m].set_xlabel('Visual Conditions', fontweight='semibold',
                             labelpad=14, fontsize=12)
        else:
            assert m == 0
            ax[m].set_xlabel('Auditory Conditions', fontweight='semibold',
                             labelpad=14, fontsize=12)
            ax[m].set_ylabel(y, fontsize=13, labelpad=10)
            ax[m].tick_params(axis='y', labelsize=11)

        ax[m].tick_params(axis='x', labelsize=11)

        ax[m].spines['right'].set_visible(False)
        ax[m].spines['top'].set_visible(False)

    if show_title and title is not None:
        fig.suptitle(title, fontsize=11)

    plt.savefig(
        os.path.join(output_dir, fname + '.png'),
        dpi=300,
        bbox_inches='tight')
    plt.close(fig)


def ecdf(values):
    """Return the empirical cumulative distribution of a 1-D array.

    The step function is returned as the sorted values and the
    corresponding cumulative proportions (i + 1) / n, to be drawn with
    ``drawstyle='steps-post'``.
    """
    values = np.sort(np.asarray(values, dtype=float))
    proportions = np.arange(1, values.size + 1) / values.size

    return values, proportions


def format_pvalue(pvalue, text_format='star', test_label=None):
    """Format a p-value following the statannotations conventions."""
    if not np.isfinite(pvalue):
        return 'n/a'

    if text_format == 'star':
        if pvalue <= 1e-4:
            return '****'
        if pvalue <= 1e-3:
            return '***'
        if pvalue <= 1e-2:
            return '**'
        if pvalue <= 5e-2:
            return '*'
        return 'ns'

    if text_format == 'simple':
        return 'p = %.3f' % pvalue

    assert text_format == 'full'
    if test_label is None:
        return 'p = %.2e' % pvalue
    return '%s p = %.2e' % (test_label, pvalue)


def plot_ecdf_comparison(arrays_audio, arrays_visual, condition_labels,
                         x_label, title, output_dir, fname,
                         pvalues_audio, pvalues_visual,
                         annotation_text_format='star',
                         hide_non_significant=True,
                         test_label=None,
                         show_title=False,
                         show_mean=True,
                         alpha=.05):
    """Empirical cumulative distributions, auditory and visual.

    Every trial contributes one step, so the whole distribution is shown
    rather than a five-number summary: a subject alternating between two
    response strategies appears as a double step, and the tail is drawn
    instead of being reduced to outlier markers.

    The condition mean is marked with a vertical dashed line, because it
    is the statistic the unpaired t-test compares.

    P-value ordering follows ``itertools.combinations`` as elsewhere.
    """

    pairs_idx = list(
        itertools.combinations(range(len(condition_labels)), 2))

    modalities = ['audio', 'visual']
    fig, ax = plt.subplots(1, len(modalities))

    plt.subplots_adjust(left=.15, bottom=.15, wspace=.25, top=.85)

    colors = CONDITION_COLORS[:len(condition_labels)]

    all_values = np.concatenate(
        [np.asarray(a, dtype=float) for a in arrays_audio + arrays_visual])
    all_values = all_values[np.isfinite(all_values)]

    if all_values.size == 0:
        plt.close(fig)
        return

    xlim_b, xlim_t = get_ylim(all_values, pad_ratio=.05)

    for m, modality in enumerate(modalities):
        if modality == 'audio':
            arrays = arrays_audio
            pvalues = pvalues_audio
        else:
            assert modality == 'visual'
            arrays = arrays_visual
            pvalues = pvalues_visual

        for label, array, color in zip(condition_labels, arrays, colors):
            array = np.asarray(array, dtype=float)
            array = array[np.isfinite(array)]
            if array.size == 0:
                continue

            values, proportions = ecdf(array)
            ax[m].plot(
                np.concatenate([[xlim_b], values]),
                np.concatenate([[0.], proportions]),
                drawstyle='steps-post',
                color=color, linewidth=1.5, label=label)

            if show_mean:
                ax[m].axvline(
                    np.mean(array), color=color, linewidth=1.,
                    linestyle='--', alpha=.7)

        annotations = []
        for (i, j), pvalue in zip(pairs_idx, pvalues):
            if not np.isfinite(pvalue):
                continue
            if hide_non_significant and pvalue >= alpha:
                continue
            annotations.append(
                '%s vs %s: %s'
                % (condition_labels[i], condition_labels[j],
                   format_pvalue(pvalue, annotation_text_format, test_label)))

        if annotations:
            ax[m].text(
                .5, 1.01, '\n'.join(annotations),
                transform=ax[m].transAxes,
                ha='center', va='bottom', fontsize=8.)

        ax[m].set_xlim(left=xlim_b, right=xlim_t)
        ax[m].set_ylim(bottom=0., top=1.02)
        ax[m].set_xlabel(x_label, fontsize=11, labelpad=8)

        title_pad = 14. + 11. * len(annotations)

        if m == 1:
            ax[m].axes.get_yaxis().set_visible(False)
            ax[m].spines['left'].set_visible(False)
            ax[m].set_title('Visual', fontweight='semibold', fontsize=12,
                            pad=title_pad)
        else:
            assert m == 0
            ax[m].set_title('Auditory', fontweight='semibold', fontsize=12,
                            pad=title_pad)
            ax[m].set_ylabel('Proportion of trials', fontsize=13,
                             labelpad=10)
            ax[m].tick_params(axis='y', labelsize=11)
            ax[m].legend(fontsize=9, loc='upper left', frameon=False)

        ax[m].tick_params(axis='x', labelsize=10)

        ax[m].spines['right'].set_visible(False)
        ax[m].spines['top'].set_visible(False)

    if show_title and title is not None:
        fig.suptitle(title, fontsize=11)

    plt.savefig(
        os.path.join(output_dir, fname + '.png'),
        dpi=300,
        bbox_inches='tight')
    plt.close(fig)


def wilson_interval(count, nobs, alpha=.05):
    """Wilson score interval for a binomial proportion."""
    if nobs == 0:
        return np.nan, np.nan

    z = stats.norm.ppf(1. - alpha / 2.)
    p = count / nobs
    denom = 1. + z ** 2 / nobs
    centre = (p + z ** 2 / (2. * nobs)) / denom
    half = z * np.sqrt(p * (1. - p) / nobs +
                       z ** 2 / (4. * nobs ** 2)) / denom

    return centre - half, centre + half


def plot_proportions(counts_audio, counts_visual, condition_labels,
                     y, title, output_dir, fname,
                     show_title=False,
                     alpha=.05):
    """Proportions in %, with Wilson intervals, auditory and visual.

    Used instead of boxplots because the success rate is binary at the
    trial level, so a boxplot would collapse onto 0 and 1. The Wilson
    interval is used rather than the Wald interval because it does not
    overshoot 100% and keeps its coverage near ceiling.

    No test is annotated. Near ceiling the conditions differ by a
    handful of errors, and an exact test on so few events can only
    return a couple of distinct values: with 30 trials per condition
    and 3 errors in total, the only possible two-sided p-values are
    0.24 and 1, and below 5 total errors no p-value under 0.05 exists.
    These panels are therefore descriptive.

    ``counts_audio`` and ``counts_visual`` are lists of ``(successes, n)``
    tuples, one per condition, in the same order as ``condition_labels``.
    """

    modalities = ['audio', 'visual']
    fig, ax = plt.subplots(1, len(modalities))

    plt.subplots_adjust(left=.15, bottom=.15, wspace=.25, top=.8)

    colors = CONDITION_COLORS[:len(condition_labels)]

    stats_per_modality = {}
    lows, highs = [], []
    for modality, counts in zip(modalities, [counts_audio, counts_visual]):
        props, ci_lo, ci_hi = [], [], []
        for successes, nobs in counts:
            prop = 100. * successes / nobs if nobs else np.nan
            lo, hi = wilson_interval(successes, nobs, alpha=alpha)
            props.append(prop)
            ci_lo.append(100. * lo)
            ci_hi.append(100. * hi)
        stats_per_modality[modality] = (props, ci_lo, ci_hi)
        lows += ci_lo
        highs += ci_hi

    lows = np.asarray(lows, dtype=float)
    highs = np.asarray(highs, dtype=float)
    lows = lows[np.isfinite(lows)]
    highs = highs[np.isfinite(highs)]

    if lows.size == 0 or highs.size == 0:
        plt.close(fig)
        return

    span = np.max(highs) - np.min(lows)
    if span <= 0:
        span = 1.
    step = .1 * span

    ylim_b = max(0., np.min(lows) - .5 * step)
    ylim_t = np.max(highs) + .5 * step

    for m, modality in enumerate(modalities):
        props, ci_lo, ci_hi = stats_per_modality[modality]

        positions = np.arange(len(condition_labels))
        # At 0% or 100% the Wilson bound equals the proportion exactly,
        # so the subtraction can return a tiny negative value, which
        # errorbar rejects. Clip it, leaving NaN untouched.
        lower_err = np.clip(
            np.asarray(props) - np.asarray(ci_lo), 0., None)
        upper_err = np.clip(
            np.asarray(ci_hi) - np.asarray(props), 0., None)

        for pos, prop, color in zip(positions, props, colors):
            ax[m].errorbar(
                pos, prop,
                yerr=np.array([[lower_err[pos]], [upper_err[pos]]]),
                fmt='o', markersize=7,
                color=color, ecolor=color,
                elinewidth=1.5, capsize=5., capthick=1.5)

        ax[m].set_xticks(positions)
        ax[m].set_xticklabels(condition_labels)
        ax[m].set_xlim(-.6, len(condition_labels) - .4)
        ax[m].set_ylim(bottom=ylim_b, top=ylim_t)

        if m == 1:
            ax[m].axes.get_yaxis().set_visible(False)
            ax[m].spines['left'].set_visible(False)
            ax[m].set_xlabel('Visual Conditions', fontweight='semibold',
                             labelpad=14, fontsize=12)
            ax[m].text(
                1., 1.02,
                'bars: %d%% Wilson interval' % round(100. * (1. - alpha)),
                transform=ax[m].transAxes,
                ha='right', va='bottom', fontsize=9.)
        else:
            assert m == 0
            ax[m].set_xlabel('Auditory Conditions', fontweight='semibold',
                             labelpad=14, fontsize=12)
            ax[m].set_ylabel(y, fontsize=13, labelpad=10)
            ax[m].tick_params(axis='y', labelsize=11)

        ax[m].tick_params(axis='x', labelsize=11)

        ax[m].spines['right'].set_visible(False)
        ax[m].spines['top'].set_visible(False)

    if show_title and title is not None:
        fig.suptitle(title, fontsize=11)

    plt.savefig(
        os.path.join(output_dir, fname + '.png'),
        dpi=300,
        bbox_inches='tight')
    plt.close(fig)


def condition_arrays(df, modality, conditions, value_col):
    """Return subject-aligned arrays for one modality and value column."""
    sub_df = df[
        (df['modality'] == modality) &
        (df['condition'].isin(conditions))
    ].copy()

    wide = sub_df.pivot(
        index='subject',
        columns='condition',
        values=value_col,
    )

    available = [cond for cond in conditions if cond in wide.columns]
    if len(available) != len(conditions):
        return [np.array([]) for _ in conditions]

    wide = wide.dropna(subset=conditions).sort_index()
    return [wide[cond].to_numpy() for cond in conditions]


def trial_arrays(df, modality, conditions, value_col):
    """Return single-trial arrays for one modality and value column.

    Unlike ``condition_arrays``, no pairing is imposed: each condition
    contributes its own trials and the arrays may differ in length.
    """
    arrays = []
    for condition in conditions:
        values = df.loc[
            (df['modality'] == modality) &
            (df['condition'] == condition),
            value_col].to_numpy(dtype=float)
        arrays.append(values[np.isfinite(values)])

    return arrays


def paired_pvalue(values_a, values_b):
    """Return paired t-test p-value, or NaN when n < 2.

    Used at the group level, where the two samples are the same subjects
    measured in two conditions.
    """
    if len(values_a) < 2 or len(values_b) < 2:
        return np.nan

    _, p_value = stats.ttest_rel(
        values_a,
        values_b,
        alternative='two-sided',
    )
    return p_value


def welch_pvalue(values_a, values_b):
    """Return Welch's two-sample t-test p-value, or NaN when n < 2.

    Used at the individual level, where the two samples are different
    trials of the same subject and therefore unpaired. Welch rather than
    Student because the trial counts and the variances differ across
    conditions.
    """
    if len(values_a) < 2 or len(values_b) < 2:
        return np.nan

    _, p_value = stats.ttest_ind(
        values_a,
        values_b,
        equal_var=False,
        alternative='two-sided',
    )
    return p_value


def mannwhitney_pvalue(values_a, values_b):
    """Return Mann-Whitney U p-value, or NaN when either sample is < 2."""
    if len(values_a) < 2 or len(values_b) < 2:
        return np.nan

    _, p_value = stats.mannwhitneyu(
        values_a,
        values_b,
        alternative='two-sided',
    )
    return p_value


def pairwise_pvalues(arrays, test_fn):
    """Apply ``test_fn`` to every condition pair, in combinations order."""
    return [
        test_fn(arrays[i], arrays[j])
        for i, j in itertools.combinations(range(len(arrays)), 2)]


def has_random_condition(df):
    """Return True when random trials exist for both modalities."""
    random_df = df[df['condition'] == 'random']
    modalities = set(random_df['modality'].unique())
    return {'auditory', 'visual'}.issubset(modalities)


def is_binary(series):
    """Return True when a column only takes the values 0 and 1."""
    values = pd.unique(pd.Series(series).dropna())
    return values.size > 0 and bool(np.all(np.isin(values, [0., 1.])))


def subject_label(subject):
    """Return the BIDS-style folder name for one subject."""
    return 'sub-%02d' % int(subject)


def make_group_plots(df_trials, key, value, output_dir,
                     annotation_text_format, hide_non_significant):
    """Group-level figures: subject means, paired tests across subjects."""

    df_ffx = ffx_dvar(df_trials)
    has_random = has_random_condition(df_ffx)

    conditions = ['beat', 'interval']
    if has_random:
        conditions.append('random')
    labels = [CONDITION_LABELS[cond] for cond in conditions]

    rt_audio_arrays = condition_arrays(
        df_ffx, 'auditory', conditions, 'reaction_time')
    rt_visual_arrays = condition_arrays(
        df_ffx, 'visual', conditions, 'reaction_time')
    score_audio_arrays = condition_arrays(
        df_ffx, 'auditory', conditions, 'score')
    score_visual_arrays = condition_arrays(
        df_ffx, 'visual', conditions, 'score')

    n_audio = len(rt_audio_arrays[0])
    n_visual = len(rt_visual_arrays[0])
    if n_audio < 2 or n_visual < 2:
        print(
            '  Skipping group statistics and plotting for ' +
            key + ': at least 2 complete subjects are required; '
            f'found auditory={n_audio}, visual={n_visual}.')
        return

    pvals_rt_audio = pairwise_pvalues(rt_audio_arrays, paired_pvalue)
    pvals_rt_visual = pairwise_pvalues(rt_visual_arrays, paired_pvalue)
    pvals_score_audio = pairwise_pvalues(score_audio_arrays, paired_pvalue)
    pvals_score_visual = pairwise_pvalues(score_visual_arrays, paired_pvalue)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    rt_title = (
        'Group Mean of Reaction Time for the NTFD tasks: ' + value)
    score_title = (
        'Group Mean of the Success Rate for the NTFD tasks: ' + value)

    plot_box_comparison(
        rt_audio_arrays,
        rt_visual_arrays,
        labels,
        'Reaction Time (ms)',
        rt_title, output_dir, 'group_rt_ntfd_' + key,
        pvals_rt_audio,
        pvals_rt_visual,
        loc='inside',
        annotation_text_format=annotation_text_format,
        hide_non_significant=hide_non_significant,
        show_title=SHOW_FIGURE_TITLE)

    plot_box_comparison(
        [array * 100. for array in score_audio_arrays],
        [array * 100. for array in score_visual_arrays],
        labels,
        'Group Mean Score (%)',
        score_title, output_dir, 'group_scores_ntfd_' + key,
        pvals_score_audio,
        pvals_score_visual,
        loc='outside',
        annotation_text_format=annotation_text_format,
        hide_non_significant=hide_non_significant,
        y_bounds=(70, 100) if has_random else (0, 100),
        show_title=SHOW_FIGURE_TITLE)


def make_individual_plots(df_trials, key, value, output_root,
                          annotation_text_format, hide_non_significant):
    """Per-subject figures: single trials, unpaired tests across trials."""

    score_binary = is_binary(df_trials['score'])
    if not score_binary:
        print('  Success rate is not binary at the trial level; '
              'falling back to boxplots for the individual scores.')

    for subject in sorted(df_trials['subject'].unique()):
        df_sub = df_trials[df_trials['subject'] == subject]

        has_random = has_random_condition(df_sub)
        conditions = ['beat', 'interval']
        if has_random:
            conditions.append('random')
        labels = [CONDITION_LABELS[cond] for cond in conditions]

        rt_audio_arrays = trial_arrays(
            df_sub, 'auditory', conditions, 'reaction_time')
        rt_visual_arrays = trial_arrays(
            df_sub, 'visual', conditions, 'reaction_time')
        score_audio_arrays = trial_arrays(
            df_sub, 'auditory', conditions, 'score')
        score_visual_arrays = trial_arrays(
            df_sub, 'visual', conditions, 'score')

        n_trials = [array.size
                    for array in rt_audio_arrays + rt_visual_arrays]
        if min(n_trials) < MIN_TRIALS_PER_CONDITION:
            print(
                '  ' + subject_label(subject) +
                ': skipped, fewer than %d trials in at least one '
                'condition (counts: %s).'
                % (MIN_TRIALS_PER_CONDITION, n_trials))
            continue

        output_dir = os.path.join(output_root, subject_label(subject))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        rt_title = (
            'Reaction Time for the NTFD tasks, ' +
            subject_label(subject) + ': ' + value)
        score_title = (
            'Success Rate for the NTFD tasks, ' +
            subject_label(subject) + ': ' + value)

        if INDIVIDUAL_RT_PLOT == 'box':
            plot_box_comparison(
                rt_audio_arrays,
                rt_visual_arrays,
                labels,
                'Reaction Time (ms)',
                rt_title, output_dir,
                'rt_ntfd_' + key + '_' + subject_label(subject),
                pairwise_pvalues(rt_audio_arrays, INDIVIDUAL_RT_TEST),
                pairwise_pvalues(rt_visual_arrays, INDIVIDUAL_RT_TEST),
                loc='inside',
                annotation_text_format=annotation_text_format,
                hide_non_significant=hide_non_significant,
                show_title=SHOW_FIGURE_TITLE)
        else:
            assert INDIVIDUAL_RT_PLOT == 'ecdf'
            plot_ecdf_comparison(
                rt_audio_arrays,
                rt_visual_arrays,
                labels,
                'Reaction Time (ms)',
                rt_title, output_dir,
                'rt_ntfd_' + key + '_' + subject_label(subject),
                pairwise_pvalues(rt_audio_arrays, INDIVIDUAL_RT_TEST),
                pairwise_pvalues(rt_visual_arrays, INDIVIDUAL_RT_TEST),
                annotation_text_format=annotation_text_format,
                hide_non_significant=hide_non_significant,
                test_label=INDIVIDUAL_RT_TEST_LABEL,
                show_title=SHOW_FIGURE_TITLE)

        score_fname = 'scores_ntfd_' + key + '_' + subject_label(subject)

        if score_binary:
            counts_audio = [
                (int(np.sum(array)), int(array.size))
                for array in score_audio_arrays]
            counts_visual = [
                (int(np.sum(array)), int(array.size))
                for array in score_visual_arrays]

            plot_proportions(
                counts_audio,
                counts_visual,
                labels,
                'Success Rate (%)',
                score_title, output_dir, score_fname,
                show_title=SHOW_FIGURE_TITLE)
        else:
            plot_box_comparison(
                [array * 100. for array in score_audio_arrays],
                [array * 100. for array in score_visual_arrays],
                labels,
                'Score (%)',
                score_title, output_dir, score_fname,
                pairwise_pvalues(score_audio_arrays, INDIVIDUAL_RT_TEST),
                pairwise_pvalues(score_visual_arrays, INDIVIDUAL_RT_TEST),
                loc='outside',
                annotation_text_format=annotation_text_format,
                hide_non_significant=hide_non_significant,
                y_bounds=(0, 100),
                show_title=SHOW_FIGURE_TITLE)


# %%
# ========================== INPUTS ====================================

# *********************** First Batch *********************************
# Expyriment / Implicit

# All good subjects including img pilot (sub-04)
GOOD_SUBJECTS = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 32,
                 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

# Excludes subjects 4, 5, and 9 from GOOD_SUBJECTS.
RAND_SUBJECTS = [3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 32, 34,
                 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

# Imaging subjects only, without the pilot subject.
IMG_SUBJECTS = [3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21,
                22, 23, 26, 28, 29, 32, 34, 35, 38, 39, 40, 41,
                42, 43, 44, 45, 46, 47]

# Subjects who completed behavioral sessions with the random condition.
BEHAV_RAND_SUBJECTS = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                       26, 27, 28, 29, 32, 34, 35, 38, 39, 40,
                       41, 42, 43, 44, 45, 46, 47]

# *********************** Second Batch ********************************
# Psychopy / Implicit

# All subjects
ALL_SB_SUBJECTS = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
                   62, 63]

# First Session completed
GOOD_SB_SUBJECTS = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61,
                    62, 63]

# Second Session completed
SB2_SUBJECTS = [48, 50, 51, 52, 53, 54, 55, 56, 57, 59, 61, 62, 63]

# *********************** Third Batch *********************************
# Psychopy / Explicit

# All subjects
ALL_TB_SUBJECTS = [64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75]

# First Session completed
GOOD_TB_SUBJECTS = [64, 65, 66, 67, 68, 70, 71, 72, 73, 74, 75]

# Second Session completed
TB2_SUBJECTS = [65, 66, 68, 72, 73, 75]

# #####################################################################

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))

CONDITION_LABELS = {
    'beat': 'Beat',
    'interval': 'Interval',
    'random': 'Random',
}

CONDITION_COLORS = ['tab:blue', 'tab:orange', 'tab:pink']

# #### First Batch ####

fb_sessions_dic = {
    'allses': 'All Sessions',
    'behavses': 'Behavioral Sessions',
    'imgses': 'Imaging Sessions',
    'ses-01': 'Session 1',
    'ses-02': 'Session 2',
    'ses-03': 'Session 3',
    'ses-04': 'Session 4',
    'ses-05': 'Session 5',
    'behav12': 'Behavioral Sessions 1 and 2',
    'behav13': 'Behavioral Sessions 1 and 3',
    'behav23': 'Behavioral Sessions 2 and 3',
}

fb_subjects_dic = {
    'allses': RAND_SUBJECTS,
    'behavses': BEHAV_RAND_SUBJECTS,
    'imgses': IMG_SUBJECTS,
    'ses-01': BEHAV_RAND_SUBJECTS,
    'ses-02': BEHAV_RAND_SUBJECTS,
    'ses-03': BEHAV_RAND_SUBJECTS,
    'ses-04': IMG_SUBJECTS,
    'ses-05': IMG_SUBJECTS,
    'behav12': BEHAV_RAND_SUBJECTS,
    'behav13': BEHAV_RAND_SUBJECTS,
    'behav23': BEHAV_RAND_SUBJECTS,
}

# #### Second Batch ####

sb_sessions_dic = {
    'behavses': 'Behavioral Sessions',
    'ses-01': 'Session 1',
    'ses-02': 'Session 2',
}

sb_subjects_dic = {
    'behavses': GOOD_SB_SUBJECTS,
    'ses-01': GOOD_SB_SUBJECTS,
    'ses-02': SB2_SUBJECTS,
}

# #### Third Batch ####

tb_sessions_dic = {
    'behavses': 'Behavioral Sessions',
    'ses-01': 'Session 1',
    'ses-02': 'Session 2',
}

tb_subjects_dic = {
    'behavses': GOOD_TB_SUBJECTS,
    'ses-01':   GOOD_TB_SUBJECTS,
    'ses-02':   TB2_SUBJECTS,
}

# #### Map tag -> integer session list ####

sessions_list_dic = {
    'allses': [1, 2, 3, 4, 5],
    'behavses': [1, 2, 3],
    'imgses': [4, 5],
    'ses-01': [1],
    'ses-02': [2],
    'ses-03': [3],
    'ses-04': [4],
    'ses-05': [5],
    'behav12': [1, 2],
    'behav13': [1, 3],
    'behav23': [2, 3],
}

# #### Batch configurations ####

# Reaction times are always latency-corrected in this script.
# First batch: Expyriment auditory latency.
# Second batch: PsychoPy auditory latency.
# Third batch: PsychoPy auditory latency.

expy_audio_latency = 133
psychopy_audio_latency = 63
expy_visual_latency = 35
button_press = 20

# Keep this list explicit so each batch can be run one at a time
# by commenting out any entry if needed.
BATCHES_TO_RUN = ['first', 'second', 'third']
# BATCHES_TO_RUN = ['second', 'third']
# BATCHES_TO_RUN = ['second']
# BATCHES_TO_RUN = ['third']

batch_dic = {
    'first': {
        'sessions': fb_sessions_dic,
        'subjects': fb_subjects_dic,
        'audio_latency': expy_audio_latency,
        'visual_latency': expy_visual_latency,
        'button_press': button_press,
        'results_folder': os.path.join(MAIN_DIR, 'ntfd_results_first_batch'),
    },
    'second': {
        'sessions': sb_sessions_dic,
        'subjects': sb_subjects_dic,
        'audio_latency': psychopy_audio_latency,
        'visual_latency': expy_visual_latency,
        'button_press': button_press,
        'results_folder': os.path.join(MAIN_DIR, 'ntfd_results_second_batch'),
    },
    'third': {
        'sessions': tb_sessions_dic,
        'subjects': tb_subjects_dic,
        'audio_latency': psychopy_audio_latency,
        'visual_latency': expy_visual_latency,
        'button_press': button_press,
        'results_folder': os.path.join(MAIN_DIR, 'ntfd_results_third_batch'),
    },
}

# Plot annotation options.
# Use 'star' for significance stars, 'simple' for p-values, or
# 'full' for the verbose statannotations format.
ANNOTATION_TEXT_FORMAT = 'full'
HIDE_NON_SIGNIFICANT = False

# Draw the figure title above each panel pair. Kept False to reproduce
# the previous output, where the title was computed but never rendered.
SHOW_FIGURE_TITLE = False

# #### Individual-level options ####

# Trials are treated as independent within a subject, so the conditions
# are independent samples and the test is unpaired. Welch's t-test keeps
# the comparison on means, consistent with the group figures. Replace it
# with mannwhitney_pvalue, and the label with 'Mann-Whitney', to test
# stochastic dominance instead.
INDIVIDUAL_RT_TEST = welch_pvalue
INDIVIDUAL_RT_TEST_LABEL = 'Welch'

# Display of the individual reaction times: 'box' for boxplots, matching
# the group figures, or 'ecdf' for empirical cumulative distributions,
# which draw every trial. The test is the same either way; only the
# placement of the annotations differs, inside the axes for boxplots and
# above them for the ECDFs.
INDIVIDUAL_RT_PLOT = 'box'

# Minimum number of trials per condition and modality required for a
# subject to be plotted. Two is the lower bound for a t-test; raise it
# to drop subjects whose individual figures would be uninformative.
MIN_TRIALS_PER_CONDITION = 2


# %%
# ============================ RUN =====================================

if __name__ == "__main__":

    for batch_tag in BATCHES_TO_RUN:
        batch_info = batch_dic[batch_tag]
        sessions_dic = batch_info['sessions']
        subjects_dic = batch_info['subjects']
        audio_latency = batch_info['audio_latency']
        visual_latency = batch_info['visual_latency']
        button_press = batch_info['button_press']
        results_folder = batch_info['results_folder']
        dataframes_folder = os.path.join(results_folder, 'dataframes')
        plots_folder = os.path.join(results_folder, 'rt_and_success')
        group_folder = os.path.join(plots_folder, 'group')
        individual_folder = os.path.join(plots_folder, 'individual')

        print('\n' + '=' * 60)
        print(f'Batch: {batch_tag}')
        print(
            'Latencies: '
            f'audio={audio_latency}, visual={visual_latency}, '
            f'button={button_press}')
        print(f'Results folder: {results_folder}')
        print('=' * 60)

        for key, value in sessions_dic.items():
            sessions_list = sessions_list_dic[key]

            print(f'\nSession tag: {key}  |  {value}')

            db_path = os.path.join(dataframes_folder, f'df_ntfd_{key}.tsv')
            if not os.path.exists(db_path):
                raise FileNotFoundError(
                    'Could not find NTFD dataframe for tag ' + key +
                    ' in ' + dataframes_folder)

            db = pd.read_csv(db_path, sep='\t')

            df_subfiltered = db[db['subject'].isin(subjects_dic[key])]
            df = df_subfiltered[
                df_subfiltered['session'].isin(sessions_list)]

            df = df.dropna(subset=['reaction_time'])
            df = df.drop(columns=['answer'])

            df['reaction_time'] = df['reaction_time'].astype(float)
            df.loc[df['modality'] == 'auditory', 'reaction_time'] -= (
                audio_latency + button_press)
            df.loc[df['modality'] == 'visual', 'reaction_time'] -= (
                visual_latency + button_press)

            print(' Group plots')
            make_group_plots(
                df, key, value, group_folder,
                ANNOTATION_TEXT_FORMAT, HIDE_NON_SIGNIFICANT)

            print(' Individual plots')
            make_individual_plots(
                df, key, value, individual_folder,
                ANNOTATION_TEXT_FORMAT, HIDE_NON_SIGNIFICANT)