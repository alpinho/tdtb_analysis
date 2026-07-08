"""
Analysis of behavioral data for the Perception Tasks of 
the TDTB project

author: Ana Luisa Pinho
e-mail: agrilopi@uwo.ca

Created: February, 2023
Last update: July 2026

Compatibility: Python 3.10.14
"""

import os
import warnings

import numpy as np
import pandas as pd

import pingouin as pg
from scipy import stats, optimize, special
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from statsmodels.stats.anova import AnovaRM

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# %%
# ======================== MAIN FUNCTIONS ==============================

def perception_data(data):
    trials = []
    for dt, datum in enumerate(data):
        if datum[5] == 'interval_1':
            condition = datum[4]
            theoretical_isi1 = datum[8]
            theoretical_isi5 = data[dt + 8][8]
            if data[dt + 10][5] == 'feedback' and \
               data[dt + 10][11] in ['o', 'p', 'b', 'y']:
                # rt = int(data[dt+9][7]) + int(data[dt+10][10])
                answer = data[dt + 10][11]
            elif data[dt + 10][5] == 'feedback' and \
                 data[dt + 10][11] == 'None':
                answer = np.nan
            else:
                raise ValueError('No feedback entry!')
            trials.append([condition, theoretical_isi1, theoretical_isi5,
                           answer])

    return trials


def filter_trialtype(trs, category):
    beat = [tr[1:] for tr in trs if tr[0][:4] == 'beat']
    interval = [tr[1:] for tr in trs if tr[0][:8] == 'interval']
    random = [tr[1:] for tr in trs if tr[0][:6] == 'random']

    if category in ['production', 'ntfd']:
        beat = [list(map(int, b)) if ~np.any(np.isnan(b)) else b
                for b in beat]
        interval = [list(map(int, i)) if ~np.any(np.isnan(i)) else i
                    for i in interval]
        if random:
            random = [list(map(int, r)) if ~np.any(np.isnan(r)) else r
                      for r in random]
    else:
        assert category == 'perception'
        beat = [[int(b[0]), int(b[1]), b[2]] for b in beat]
        interval = [[int(i[0]), int(i[1]), i[2]] for i in interval]

    return beat, interval, random


def perception_frequencies(beat_trials, interval_trials):
    """Count shorter/longer responses for matched beat/interval grids."""
    if not beat_trials or not interval_trials:
        raise ValueError('Beat and interval trials must both be non-empty.')

    beat_standards = np.sort(np.unique(np.array(beat_trials)[:, 0]))
    interval_standards = np.sort(np.unique(np.array(interval_trials)[:, 0]))
    beat_comparisons = np.sort(np.unique([
        round((bt[1] - bt[0]) / bt[0], 2) for bt in beat_trials
    ]))
    interval_comparisons = np.sort(np.unique([
        round((it[1] - it[0]) / it[0], 2) for it in interval_trials
    ]))

    if not np.array_equal(beat_standards, interval_standards):
        raise ValueError('Beat and interval standards do not match.')
    if not np.array_equal(beat_comparisons, interval_comparisons):
        raise ValueError('Beat and interval comparisons do not match.')

    standards = np.array(beat_standards, dtype='int')
    comparisons = beat_comparisons
    n1_beat = []
    n2_beat = []
    n1_interval = []
    n2_interval = []
    for standard in standards:
        diff_standard_beat_shorter = \
            [round((bt[1] - bt[0]) / bt[0], 2)
             for bt in beat_trials
             if bt[0] == standard and bt[2] in ['p', 'y']]
        diff_standard_beat_longer = \
            [round((bt[1] - bt[0]) / bt[0], 2)
             for bt in beat_trials
             if bt[0] == standard and bt[2] in ['o', 'b']]
        diff_standard_interval_shorter = \
            [round((it[1] - it[0]) / it[0], 2)
             for it in interval_trials
             if it[0] == standard and it[2] in ['p', 'y']]
        diff_standard_interval_longer = \
            [round((it[1] - it[0]) / it[0], 2)
             for it in interval_trials
             if it[0] == standard and it[2] in ['o', 'b']]
        n1_comp_beat = []
        n2_comp_beat = []
        n1_comp_interval = []
        n2_comp_interval = []
        for comparison in comparisons:
            comparisons_beat_shorter = \
                [cbs for cbs in diff_standard_beat_shorter
                 if cbs == comparison]
            comparisons_beat_longer = \
                [cbl for cbl in diff_standard_beat_longer
                 if cbl == comparison]
            comparisons_interval_shorter = \
                [cis for cis in diff_standard_interval_shorter
                 if cis == comparison]
            comparisons_interval_longer = \
                [cil for cil in diff_standard_interval_longer
                 if cil == comparison]
            n1_comp_beat.append(len(comparisons_beat_shorter))
            n2_comp_beat.append(len(comparisons_beat_longer))
            n1_comp_interval.append(len(comparisons_interval_shorter))
            n2_comp_interval.append(len(comparisons_interval_longer))
            del comparisons_beat_shorter
            del comparisons_beat_longer
            del comparisons_interval_shorter
            del comparisons_interval_longer
        n1_beat.append(n1_comp_beat)
        n2_beat.append(n2_comp_beat)
        n1_interval.append(n1_comp_interval)
        n2_interval.append(n2_comp_interval)
        del diff_standard_beat_shorter
        del diff_standard_beat_longer
        del diff_standard_interval_shorter
        del diff_standard_interval_longer

    return standards, comparisons, n1_beat, n2_beat, n1_interval, n2_interval


def loglik_cdf(par_vec, y, n2, n1):
    """Return the negative log-likelihood for a normal-CDF fit."""
    if par_vec[1] <= 0:
        return 1e8

    y = np.asarray(y, dtype=float)
    n1 = np.asarray(n1, dtype=float)
    n2 = np.asarray(n2, dtype=float)
    valid = np.isfinite(y) & np.isfinite(n1) & np.isfinite(n2)
    if not np.any(valid):
        return 1e8

    lik = stats.norm.cdf(y[valid], loc=par_vec[0], scale=par_vec[1])
    lik = np.clip(lik, np.finfo(float).eps, 1 - np.finfo(float).eps)

    return -(
        np.sum(n2[valid] * np.log(lik)) +
        np.sum(n1[valid] * np.log(1 - lik))
    )


def loglik_expit(par_vec, y, n2, n1):
    """Return the negative log-likelihood for a logistic-sigmoid fit."""
    if par_vec[1] <= 0:
        return 1e8

    y = np.asarray(y, dtype=float)
    n1 = np.asarray(n1, dtype=float)
    n2 = np.asarray(n2, dtype=float)
    valid = np.isfinite(y) & np.isfinite(n1) & np.isfinite(n2)
    if not np.any(valid):
        return 1e8

    z = (y[valid] - par_vec[0]) / par_vec[1]
    lik = np.clip(special.expit(z),
                  np.finfo(float).eps,
                  1 - np.finfo(float).eps)
    minus_lik = np.clip(special.expit(-z),
                         np.finfo(float).eps,
                         1 - np.finfo(float).eps)

    return -(
        np.sum(n2[valid] * np.log(lik)) +
        np.sum(n1[valid] * np.log(minus_lik))
    )


def errFit(hess_inv, resVariance):
    """Error of the fit parameters"""
    return np.sqrt(np.diag(hess_inv * resVariance))


def outliers(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 4:
        return np.inf, -np.inf

    q75, q25 = np.percentile(arr, [75, 25])
    iqr = q75 - q25
    high_thresh = q75 + 1.7 * iqr
    low_thresh = q25 - 1.7 * iqr

    return high_thresh, low_thresh


def fit_is_valid(opt_res, pse, dl, fit_max_abs_pse, fit_max_dl,
                 fit_min_dl):
    """Return True for finite, converged, and plausible fits."""
    return (
        opt_res.success and
        np.isfinite(opt_res.fun) and
        np.isfinite(pse) and
        np.isfinite(dl) and
        np.abs(pse) <= fit_max_abs_pse and
        fit_min_dl < dl <= fit_max_dl
    )


def _relative_frequencies(n_shorter, n_longer):
    """Convert shorter/longer counts to response probabilities."""
    rf_shorter = []
    rf_longer = []
    for short_row, long_row in zip(n_shorter, n_longer):
        short_row = np.asarray(short_row, dtype=float)
        long_row = np.asarray(long_row, dtype=float)
        total = short_row + long_row
        with np.errstate(divide='ignore', invalid='ignore'):
            rf_shorter.append(np.divide(short_row, total).tolist())
            rf_longer.append(np.divide(long_row, total).tolist())

    return rf_shorter, rf_longer


def individual_perception(
        subjects, this_dir, output_dir, condition,
        sessions, sesstag, session_label, fit_max_abs_pse,
        fit_max_dl, fit_min_dl, estimator='mle_expit',
        modalities=None, raw_df_dir=None):
    if modalities is None:
        modalities = ['auditory', 'visual']

    if raw_df_dir is None:
        raw_df_dir = os.path.join(
            os.path.abspath(output_dir), 'raw_dataframes')

    df_path = os.path.join(raw_df_dir, 'df_perception_' + sesstag + '.tsv')

    if not os.path.exists(df_path):
        raise FileNotFoundError(
            'Raw perception dataframe not found: ' + df_path)

    df = pd.read_csv(df_path, sep='\t')

    # Filter Dataframe according to list of sessions
    df = df[df['session'].isin(sessions)]

    all_rf1_audio = []
    all_rf2_audio = []
    all_rf1_visual = []
    all_rf2_visual = []
    all_pse_audio = []
    all_dl_audio = []
    all_pse_visual = []
    all_dl_visual = []

    fig = None

    for s, subject in enumerate(subjects):
        for m, modality in enumerate(modalities):
            if modality not in ['auditory', 'visual']:
                raise NameError('Modality not valid!')

            beat_trials = df[
                (df['subject'] == subject) &
                (df['modality'] == modality) &
                (df['condition'] == 'beat')][[
                    'standard', 'comparison', 'answer']].values.tolist()

            interval_trials = df[
                (df['subject'] == subject) &
                (df['modality'] == modality) &
                (df['condition'] == 'interval')][[
                    'standard', 'comparison', 'answer']].values.tolist()

            if condition == 'beat':
                standards, comparisons, n1_beat, n2_beat, _, _ = \
                    perception_frequencies(beat_trials, interval_trials)
                fit_n1, fit_n2 = n1_beat, n2_beat
                rf1, rf2 = _relative_frequencies(n1_beat, n2_beat)
            else:
                assert condition == 'interval'
                standards, comparisons, _, _, n1_interval, n2_interval = \
                    perception_frequencies(beat_trials, interval_trials)
                fit_n1, fit_n2 = n1_interval, n2_interval
                rf1, rf2 = _relative_frequencies(n1_interval, n2_interval)

            if modality == 'auditory':
                all_rf1_audio.append(rf1)
                all_rf2_audio.append(rf2)
            else:
                assert modality == 'visual'
                all_rf1_visual.append(rf1)
                all_rf2_visual.append(rf2)

            colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
                      'tab:purple']

            if fig is None:
                fig = plt.figure(figsize=(16, 100))

            ax = plt.axes([.1 + m * .46, .9685 - s * .023, .428, .0125])

            std_pse_audio = []
            std_dl_audio = []
            std_pse_visual = []
            std_dl_visual = []

            for i, st in enumerate(standards):
                if estimator == 'mle_cdf':
                    func = loglik_cdf
                    constant = stats.norm.ppf(0.75)
                else:
                    assert estimator == 'mle_expit'
                    func = loglik_expit
                    constant = np.log(3)

                opt_res = optimize.minimize(
                    fun=func,
                    x0=[np.mean(comparisons), .1],
                    args=(comparisons, fit_n2[i], fit_n1[i]))

                pse = opt_res.x[0]
                dl = opt_res.x[1] * constant
                is_valid = fit_is_valid(opt_res, pse, dl,
                                        fit_max_abs_pse,
                                        fit_max_dl, fit_min_dl)

                if is_valid:
                    se_pse = np.sqrt(np.diag(opt_res.hess_inv))[0]
                    se_dl = np.sqrt(np.diag(opt_res.hess_inv))[1] * constant
                    ci95_pse_val = se_pse * 1.96
                    ci95_dl_val = se_dl * 1.96
                else:
                    pse = np.nan
                    dl = np.nan

                if modality == 'auditory':
                    std_pse_audio.append(pse)
                    std_dl_audio.append(dl)
                else:
                    assert modality == 'visual'
                    std_pse_visual.append(pse)
                    std_dl_visual.append(dl)

                x = np.linspace(np.amin(comparisons), np.amax(comparisons),
                                100)

                if is_valid and estimator == 'mle_cdf':
                    ax.plot(x, stats.norm(pse, opt_res.x[1]).cdf(x),
                            color=colors[i],
                            label='Standard = ' + str(st) + 'ms')
                elif is_valid:
                    ax.plot(x, special.expit((x - pse) / opt_res.x[1]),
                            color=colors[i],
                            label='Standard = ' + str(st) + 'ms')

                ax.axhline(.5, linestyle='--', color='silver', linewidth=1)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)

                x_values = np.insert(comparisons, 3, 0)
                x_labels = [str(int(xl * 100)) + '%' for xl in x_values]
                ax.set_xticks(x_values, x_labels)

                ax.text(-.21, 1.53, 'For 95% CI,', fontsize=7.5)
                if is_valid:
                    ax.text(-.21, 1.41 - i * .098,
                            'PSE=%.02f' % (pse * 100) +
                            '\u00B1%.02f' % (ci95_pse_val * 100) + '%; ' +
                            'DL=%.02f' % (dl * 100) +
                            '\u00B1%.02f' % (ci95_dl_val * 100) + '%',
                            fontsize=7.5, color=colors[i])

                ax.set_ylim([-.1, 1.1])

            if s == 0:
                if m == 0:
                    ax.legend(loc='lower right', frameon=False,
                              prop={'size': 6})
                    ax.set_title('Auditory Perception', weight='bold',
                                 pad=60, fontsize=16)
                else:
                    assert m == 1
                    ax.set_title('Visual Perception', weight='bold',
                                 pad=60, fontsize=16)

            fig.text(.495, .0025, 'Comparisons (%)', fontsize=14)
            fig.text(.062, .46, 'Relative Frequency of "longer" responses',
                     fontsize=14, rotation=90)

            if modality == 'auditory':
                all_pse_audio.append(std_pse_audio)
                all_dl_audio.append(std_dl_audio)
            else:
                assert modality == 'visual'
                all_pse_visual.append(std_pse_visual)
                all_dl_visual.append(std_dl_visual)

        fig.text(.03, .9765 - s * .023, 'Subject %d' % subject,
                 ha='center', fontsize=10, weight='bold')

    if estimator == 'mle_cdf':
        suffix = '(Estimator: MLE of Norm CDF)'
    else:
        assert estimator == 'mle_expit'
        suffix = '(Estimator: MLE of Logistic-Sigmoid Function)'

    plt.suptitle(
        'Individual Relative Frequencies for the ' + condition.capitalize() +
        ' condition of the Perception Tasks: ' + session_label + ' ' +
        suffix, x=.5, y=.9975, size=16, linespacing=.75)

    output_folder = os.path.join(output_dir, 'individual_psychometric')
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    plt.savefig(
        os.path.join(
            this_dir, output_folder,
            'individual_psychometric_' + condition + '_' + estimator + '_' +
            sesstag + '.png'),
        dpi=150,
        bbox_inches='tight')

    plt.close('all')

    return (all_rf1_audio, all_rf2_audio, all_rf1_visual, all_rf2_visual,
            standards, comparisons, all_pse_audio, all_dl_audio,
            all_pse_visual, all_dl_visual)


def _fit_perception_estimates(beat_trials, interval_trials, condition,
                              fit_max_abs_pse, fit_max_dl, fit_min_dl,
                              estimator='mle_expit'):
    """Fit PSE/DL estimates for one subject/modality/condition."""
    if condition == 'beat':
        standards, comparisons, n1, n2, _, _ = perception_frequencies(
            beat_trials, interval_trials)
    else:
        assert condition == 'interval'
        standards, comparisons, _, _, n1, n2 = perception_frequencies(
            beat_trials, interval_trials)

    if estimator == 'mle_cdf':
        func = loglik_cdf
        constant = stats.norm.ppf(0.75)
    else:
        assert estimator == 'mle_expit'
        func = loglik_expit
        constant = np.log(3)

    pse_values = []
    dl_values = []
    for i, _ in enumerate(standards):
        opt_res = optimize.minimize(
            fun=func,
            x0=[np.mean(comparisons), .1],
            args=(comparisons, n2[i], n1[i]),
            method='BFGS')

        pse = opt_res.x[0]
        dl = opt_res.x[1] * constant
        is_valid = fit_is_valid(opt_res, pse, dl,
                                fit_max_abs_pse,
                                fit_max_dl, fit_min_dl)
        if not is_valid:
            pse = np.nan
            dl = np.nan

        pse_values.append(pse)
        dl_values.append(dl)

    return standards, comparisons, pse_values, dl_values


def _has_trial_pair(beat_trials, interval_trials):
    """Return True when both trial types are present for fitting."""
    return bool(beat_trials) and bool(interval_trials)


def _find_reference_frequency_grid(df, subjects, modalities, sessions):
    """Find the first valid standard/comparison grid in a dataframe."""
    for subject in subjects:
        for modality in modalities:
            for session in sessions:
                session_df = df[df['session'] == session]
                beat_trials = session_df[
                    (session_df['subject'] == subject) &
                    (session_df['modality'] == modality) &
                    (session_df['condition'] == 'beat')][[
                        'standard', 'comparison', 'answer']].values.tolist()
                interval_trials = session_df[
                    (session_df['subject'] == subject) &
                    (session_df['modality'] == modality) &
                    (session_df['condition'] == 'interval')][[
                        'standard', 'comparison', 'answer']].values.tolist()

                if not _has_trial_pair(beat_trials, interval_trials):
                    continue

                standards, comparisons, _, _, _, _ = perception_frequencies(
                    beat_trials, interval_trials)
                return standards, comparisons

    raise ValueError(
        'No subject/session/modality cell contains both beat and '
        'interval trials.')


def session_average_individual_estimates(
        subjects, output_dir, condition, sessions, sesstag,
        fit_max_abs_pse, fit_max_dl, fit_min_dl,
        estimator='mle_expit', modalities=None, raw_df_dir=None):
    """Average session-specific psychometric estimates.

    This avoids refitting a nonlinear psychometric function after pooling
    multiple sessions. Pooling can strongly change DL estimates when each
    single session has sparse, near-deterministic responses.
    """
    if modalities is None:
        modalities = ['auditory', 'visual']

    if raw_df_dir is None:
        raw_df_dir = os.path.join(
            os.path.abspath(output_dir), 'raw_dataframes')

    df_path = os.path.join(raw_df_dir, 'df_perception_' + sesstag + '.tsv')
    if not os.path.exists(df_path):
        raise FileNotFoundError(
            'Raw perception dataframe not found: ' + df_path)

    df = pd.read_csv(df_path, sep='\t')

    all_pse_audio = []
    all_dl_audio = []
    all_pse_visual = []
    all_dl_visual = []
    reference_standards, reference_comparisons = \
        _find_reference_frequency_grid(
            df, subjects, modalities, sessions)

    for subject in subjects:
        for modality in modalities:
            session_pse = []
            session_dl = []
            for session in sessions:
                session_df = df[df['session'] == session].copy()

                beat_trials = session_df[
                    (session_df['subject'] == subject) &
                    (session_df['modality'] == modality) &
                    (session_df['condition'] == 'beat')][[
                        'standard', 'comparison', 'answer']].values.tolist()

                interval_trials = session_df[
                    (session_df['subject'] == subject) &
                    (session_df['modality'] == modality) &
                    (session_df['condition'] == 'interval')][[
                        'standard', 'comparison', 'answer']].values.tolist()

                if not _has_trial_pair(beat_trials, interval_trials):
                    continue

                standards, comparisons, pse, dl = _fit_perception_estimates(
                    beat_trials, interval_trials, condition,
                    fit_max_abs_pse, fit_max_dl, fit_min_dl,
                    estimator=estimator)

                if not np.array_equal(reference_standards, standards):
                    raise ValueError('Session standards do not match.')
                if not np.array_equal(reference_comparisons, comparisons):
                    raise ValueError('Session comparisons do not match.')

                session_pse.append(pse)
                session_dl.append(dl)

            if session_pse:
                with np.errstate(invalid='ignore'):
                    mean_pse = np.nanmean(
                        np.asarray(session_pse, dtype=float), axis=0)
                    mean_dl = np.nanmean(
                        np.asarray(session_dl, dtype=float), axis=0)
                mean_pse = mean_pse.tolist()
                mean_dl = mean_dl.tolist()
            else:
                n_standards = len(reference_standards)
                mean_pse = [np.nan] * n_standards
                mean_dl = [np.nan] * n_standards

            if modality == 'auditory':
                all_pse_audio.append(mean_pse)
                all_dl_audio.append(mean_dl)
            else:
                assert modality == 'visual'
                all_pse_visual.append(mean_pse)
                all_dl_visual.append(mean_dl)

    return (reference_standards, reference_comparisons,
            all_pse_audio, all_dl_audio, all_pse_visual, all_dl_visual)


def session_average_group_frequencies(
        subjects, output_dir, condition, sessions, sesstag,
        modalities=None, raw_df_dir=None):
    """Average response frequencies across sessions within subject.

    This makes multi-session group psychometric plots consistent with
    the ANOVA input: sessions are summarized within subject first,
    rather than fitting or averaging after pooling raw trials.
    """
    if modalities is None:
        modalities = ['auditory', 'visual']

    if raw_df_dir is None:
        raw_df_dir = os.path.join(
            os.path.abspath(output_dir), 'raw_dataframes')

    df_path = os.path.join(raw_df_dir, 'df_perception_' + sesstag + '.tsv')
    if not os.path.exists(df_path):
        raise FileNotFoundError(
            'Raw perception dataframe not found: ' + df_path)

    df = pd.read_csv(df_path, sep='\t')

    all_rf1_audio = []
    all_rf2_audio = []
    all_rf1_visual = []
    all_rf2_visual = []
    reference_standards, reference_comparisons = \
        _find_reference_frequency_grid(
            df, subjects, modalities, sessions)

    for subject in subjects:
        for modality in modalities:
            session_rf1 = []
            session_rf2 = []

            for session in sessions:
                session_df = df[df['session'] == session].copy()

                beat_trials = session_df[
                    (session_df['subject'] == subject) &
                    (session_df['modality'] == modality) &
                    (session_df['condition'] == 'beat')][[
                        'standard', 'comparison', 'answer']].values.tolist()

                interval_trials = session_df[
                    (session_df['subject'] == subject) &
                    (session_df['modality'] == modality) &
                    (session_df['condition'] == 'interval')][[
                        'standard', 'comparison', 'answer']].values.tolist()

                if not _has_trial_pair(beat_trials, interval_trials):
                    continue

                if condition == 'beat':
                    standards, comparisons, n1, n2, _, _ = \
                        perception_frequencies(beat_trials, interval_trials)
                else:
                    assert condition == 'interval'
                    standards, comparisons, _, _, n1, n2 = \
                        perception_frequencies(beat_trials, interval_trials)

                if not np.array_equal(reference_standards, standards):
                    raise ValueError('Session standards do not match.')
                if not np.array_equal(reference_comparisons, comparisons):
                    raise ValueError('Session comparisons do not match.')

                rf1, rf2 = _relative_frequencies(n1, n2)
                session_rf1.append(rf1)
                session_rf2.append(rf2)

            if session_rf1:
                with np.errstate(invalid='ignore'):
                    mean_rf1 = np.nanmean(
                        np.asarray(session_rf1, dtype=float), axis=0)
                    mean_rf2 = np.nanmean(
                        np.asarray(session_rf2, dtype=float), axis=0)
                mean_rf1 = mean_rf1.tolist()
                mean_rf2 = mean_rf2.tolist()
            else:
                shape = (len(reference_standards), len(reference_comparisons))
                mean_rf1 = np.full(shape, np.nan).tolist()
                mean_rf2 = np.full(shape, np.nan).tolist()

            if modality == 'auditory':
                all_rf1_audio.append(mean_rf1)
                all_rf2_audio.append(mean_rf2)
            else:
                assert modality == 'visual'
                all_rf1_visual.append(mean_rf1)
                all_rf2_visual.append(mean_rf2)

    return (all_rf1_audio, all_rf2_audio, all_rf1_visual, all_rf2_visual,
            reference_standards, reference_comparisons)


def group_perception(all_rf1_audio, all_rf2_audio,
                     all_rf1_visual, all_rf2_visual,
                     standards, comparisons, condition, output_dir, sesstag,
                     session_label, fit_max_abs_pse, fit_max_dl, fit_min_dl,
                     estimator='mle_expit',
                     show_fit_estimates=True):

    # Use nanmean because missed/no-response trials can leave a
    # subject-specific standard/comparison cell undefined.  A plain
    # mean would propagate any one subject's NaN to the group curve and
    # can make sparse-session plots look empty.
    group_rf1_audio = np.nanmean(all_rf1_audio, axis=0)
    group_rf2_audio = np.nanmean(all_rf2_audio, axis=0)
    group_rf1_visual = np.nanmean(all_rf1_visual, axis=0)
    group_rf2_visual = np.nanmean(all_rf2_visual, axis=0)

    # ################## Plotting ###############################

    modalities = ['audio', 'visual']
    fig, ax = plt.subplots(1, len(modalities), figsize=(16, 8))

    # left   # the left side of the subplots of the figure
    # right  # the right side of the subplots of the figure
    # bottom # the bottom of the subplots of the figure
    # top    # the top of the subplots of the figure
    # wspace # the amount of width reserved for blank space between subplots
    # hspace # the amount of height reserved for white space between subplots
    plt.subplots_adjust(left=.085, bottom=.1, right=.975, wspace=.15)

    colors = ['cornflowerblue', 'blueviolet', 'magenta', 'orangered',
              'gold']

    group_pse = []
    group_dl = []
    for m, modality in enumerate(modalities):
        if modality == 'audio':
            rf1 = group_rf1_audio
            rf2 = group_rf2_audio
        else:
            assert modality == 'visual'
            rf1 = group_rf1_visual
            rf2 = group_rf2_visual

        modality_pse = []
        modality_dl = []
        for i, st in enumerate(standards):
            # Chose estimator
            if estimator == 'mle_cdf':
                func = loglik_cdf
                constant = stats.norm.ppf(0.75)
            else:
                assert estimator == 'mle_expit'
                func = loglik_expit
                constant = np.log(3)
            # Fit the model with a MLE estimator
            # fun: MLE estimator
            # x0: 1st arg of log_lik
            # args: 2nd and 3rd args of func
            opt_res = optimize.minimize(
                fun=func,
                x0=[np.mean(comparisons), 1.],
                args=(comparisons, rf2[i], rf1[i]),
                method='BFGS')

            # Estimates
            pse = opt_res.x[0]
            dl = opt_res.x[1] * constant
            is_valid = fit_is_valid(opt_res, pse, dl,
                                    fit_max_abs_pse,
                                    fit_max_dl, fit_min_dl)
            if is_valid:
                # Standard errors from Fisher information.
                se_pse = np.sqrt(np.diag(opt_res.hess_inv))[0]
                se_dl = np.sqrt(np.diag(opt_res.hess_inv))[1] * constant
                ci95_pse = se_pse * 1.96
                ci95_dl = se_dl * 1.96
                dFit = errFit(
                    opt_res.hess_inv,
                    opt_res.fun / (len(rf2[i]) - 2))
            else:
                pse = np.nan
                dl = np.nan
                ci95_pse = np.nan
                ci95_dl = np.nan
                dFit = np.array([np.nan, np.nan])

            print(modality, '-', condition)
            print('estimator:', estimator)
            print('standard:', st)
            print('valid fit:', is_valid)
            print('minimize:\n\tx: ', opt_res.x, '\n\tdx: ', dFit)

            # Plot each fit in one image
            # fig, ax = plt.subplots(1, 1)
            x = np.linspace(np.amin(comparisons), np.amax(comparisons),
                            100)
            # Plot data
            finite_pts = np.isfinite(rf2[i])
            ax[m].plot(comparisons[finite_pts], rf2[i][finite_pts], 'o',
                       color=colors[i], markersize=6, alpha=.5)
            # Plot fit
            if is_valid and estimator == 'mle_cdf':
                ax[m].plot(x, stats.norm(pse, opt_res.x[1]).cdf(x),
                           color=colors[i], linewidth=6, markersize=12,
                           alpha=.5, label='Standard = ' + str(st) + 'ms')
            elif is_valid:
                assert estimator == 'mle_expit'
                ax[m].plot(x, special.expit((x - pse) / opt_res.x[1]),
                           color=colors[i], linewidth=6, markersize=12,
                           alpha=.5, label='Standard = ' + str(st) + 'ms')
            # Add horizontal dashed line at y = 0.5
            ax[m].axhline(.5, linestyle='--', color='grey', linewidth=3)
            # Hide the right and top spines
            ax[m].spines['right'].set_visible(False)
            ax[m].spines['top'].set_visible(False)
            # Set x axis
            x_values = np.insert(comparisons, 3, 0)
            x_labels = [str(int(xl * 100)) for xl in x_values]
            ax[m].set_xticks(x_values, x_labels, fontsize=16)
            # Add estimates info.  These are the group-level fit
            # estimates for each standard, displayed as in the
            # individual psychometric plots.
            if show_fit_estimates:
                ax[m].text(-0.21, 1.04, 'For 95% CI,',
                           fontsize=10)
                if is_valid:
                    ax[m].text(
                        -0.21, 0.98 - i * 0.055,
                        'PSE=%.02f' % (pse * 100) +
                        '\u00B1%.02f' % (ci95_pse * 100) + '%; ' +
                        'DL=%.02f' % (dl * 100) +
                        '\u00B1%.02f' % (ci95_dl * 100) + '%',
                        fontsize=10, color=colors[i])

            modality_pse.append(pse)
            modality_dl.append(dl)

        group_pse.append(modality_pse)
        group_dl.append(modality_dl)

        # Add legend
        if m == 0:
            ax[m].legend(loc='lower right', frameon=False, prop={'size': 14})
            ax[m].set_title('Auditory Perception', weight='bold', pad=5,
                            fontsize=16)
        else:
            assert m == 1
            ax[m].set_title('Visual Perception', weight='bold', pad=5,
                            fontsize=16)

        # Name of x-axis
        fig.text(.485, .0275, 'Comparisons (%)', fontsize=16)
        # Name of y-axis
        fig.text(.0315, .125,
                 'Mean of Relative Frequency for "longer" responses (%)',
                 fontsize=16, rotation=90)
        # Set limits of y-axis
        y_values = np.linspace(0., 1., 6)
        y_labels = [str(int(yl * 100)) for yl in y_values]
        ax[m].set_yticks(y_values, y_labels, fontsize=16)
        ax[m].set_ylim([-.05, 1.05])

    # Title
    if estimator == 'mle_cdf':
        suffix = '(Estimator: MLE of Norm CDF)'
    else:
        assert estimator == 'mle_expit'
        suffix = '(Estimator: MLE of Logistic-Sigmoid Function)'
    plt.suptitle(
        'Group Mean of Relative Frequencies for the ' +
        condition.capitalize() + ' condition of the Perception Tasks: ' +
        session_label + ' ' + suffix, x=.5, y=.97, size=14,
        linespacing=.75)

    output_folder = os.path.join(output_dir, 'group_psychometric')
    # Create output_folder, if it does not exist
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Save figure
    plt.savefig(
        os.path.join(
            output_folder,
            'group_psychometric_' + condition + '_' + estimator + '_' +
            sesstag + '.png'),
        dpi=300,
        bbox_inches='tight')

    plt.close('all')

    return group_pse, group_dl


def plotfit_perception(x, y, estimator, output_dir, sesstag,
                       session_label):
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    plt.subplots_adjust(left=.085, bottom=.11, right=.975, wspace=.15, top=.8)

    all_vals = [
        v for modality in y for condition in modality for v in condition
        if np.isfinite(v)
    ]
    if all_vals:
        data_min, data_max = min(all_vals), max(all_vals)
        data_range = data_max - data_min if data_max != data_min else 1.0
        pad = data_range * 0.15
        y_min_auto = data_min - pad
        y_max_auto = data_max + pad
        raw_ticks = np.linspace(y_min_auto, y_max_auto, 5)
        tick_magnitude = 10 ** np.floor(np.log10(abs(data_range) + 1e-12))
        decimals = max(0, int(-np.floor(np.log10(tick_magnitude + 1e-12))))
        y_values = np.round(raw_ticks, max(decimals, 2))
        y_lim = (y_min_auto, y_max_auto)
    else:
        y_values = np.linspace(-.25, .25, 5)
        y_lim = (-.25, .25)

    colors = ['tab:blue', 'tab:orange']
    legend_labels = ['Beat', 'Interval']

    for m, modality_y in enumerate(y):
        for c, condition_y in enumerate(modality_y):
            condition_y = np.asarray(condition_y, dtype=float)
            valid = np.isfinite(condition_y)

            if np.sum(valid) >= 2:
                a, b = np.polyfit(np.asarray(x)[valid], condition_y[valid],
                                  deg=1)
                y_est = a * np.asarray(x) + b
                ax[m].plot(x, y_est, '-', color=colors[c], linewidth=12,
                           label=legend_labels[c], alpha=.5)

            ax[m].plot(np.asarray(x)[valid], condition_y[valid], 'bo',
                       color=colors[c], markersize=16, alpha=.5)
            ax[m].spines['right'].set_visible(False)
            ax[m].spines['top'].set_visible(False)
            ax[m].set_xticks(x, [str(xl) for xl in x], fontsize=24)
            ax[m].set_ylim(y_lim)
            y_labels = [str(int(yl * 100)) for yl in y_values]
            ax[m].set_yticks(y_values, y_labels, fontsize=24)
            ax[m].axhline(0., linestyle='--', color='grey', linewidth=12,
                          alpha=.5)

        if m == 0:
            ax[m].set_title('Auditory Perception', weight='bold', pad=-5,
                            fontsize=22)
        else:
            assert m == 1
            ax[m].legend(loc='upper right', frameon=False,
                         prop={'size': 16})
            ax[m].set_title('Visual Perception', weight='bold', pad=-5,
                            fontsize=22)

        fig.text(.47, .018, 'Standards (ms)', fontsize=24)
        fig.text(.0175, .35, 'Group PSE (%)', fontsize=24, rotation=90)
        fig.text(.42, .525, 'No Bias', fontsize=24, color='dimgrey')
        fig.text(.895, .525, 'No Bias', fontsize=24, color='dimgrey')

    if estimator == 'mle_cdf':
        suffix = '(Estimator: MLE of Norm CDF)'
    else:
        assert estimator == 'mle_expit'
        suffix = '(Estimator: MLE of Logistic-Sigmoid Function)'
    plt.suptitle(
        'Point of Subjective Equality (PSE) for the Perception Tasks: ' +
        session_label + '\n\n' + suffix,
        x=.5, y=.97, size=24, linespacing=.75)

    output_folder = os.path.join(output_dir, 'pse')
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    plt.savefig(
        os.path.join(
            output_folder,
            'pse-vs-standard_' + estimator + '_' + sesstag + '.png'),
        dpi=300,
        bbox_inches='tight')

    plt.close('all')


def dataframe(estim_pse, estim_dl, stand_numbers, output_dir, sesstag,
              subjects, estimator='mle_expit'):
    # Shape of pse and dl arrays:
    # (1, conditions, modality, subjects, standards)
    # estim_pse contains mle_expit results only (mle_cdf is used for the
    # PSE plot only and is never accumulated into this array).
    estim_pse = np.array(estim_pse, dtype=float)
    estim_dl = np.array(estim_dl, dtype=float)
    if estim_pse.shape != estim_dl.shape or estim_pse.ndim != 5:
        raise ValueError(
            'Expected PSE and DL arrays with shape '
            '(estimator, condition, modality, subject, standard).'
        )

    pse_flatten = np.ravel(estim_pse)
    dl_flatten = np.ravel(estim_dl)

    standards = np.tile(
        stand_numbers,
        estim_pse.shape[3] * estim_pse.shape[2] * estim_pse.shape[1] *
        estim_pse.shape[0])
    itag = ['sub-%02d' % s for s in subjects]
    stand_individuals = np.repeat(itag, len(stand_numbers))
    individuals = np.tile(
        stand_individuals,
        estim_pse.shape[2] * estim_pse.shape[1] * estim_pse.shape[0])
    stand_modalities = np.repeat(['audio', 'visual'], len(stand_individuals))
    modalities = np.tile(
        stand_modalities,
        estim_pse.shape[1] * estim_pse.shape[0])
    crossind_conditions = np.repeat(['beat', 'interval'],
                                    len(stand_modalities))
    conditions = np.tile(crossind_conditions, estim_pse.shape[0])
    sessions = np.repeat(sesstag, len(dl_flatten))

    table = np.vstack((
        dl_flatten, pse_flatten, standards, individuals, modalities,
        conditions, sessions)).T

    df = pd.DataFrame(table, columns=['DL_raw', 'PSE_raw', 'Standard',
                                      'Subject', 'Modality', 'Condition',
                                      'Session'])
    df['DL_raw'] = pd.to_numeric(df['DL_raw'], errors='coerce')
    df['PSE_raw'] = pd.to_numeric(df['PSE_raw'], errors='coerce')
    df['Standard'] = pd.to_numeric(df['Standard'], errors='coerce')

    df['DL'] = df['DL_raw']
    df['PSE'] = df['PSE_raw']
    df['FitValid'] = np.isfinite(df['DL']) & np.isfinite(df['PSE'])

    group_cols = ['Standard', 'Modality', 'Condition']
    for _, group_df in df.groupby(group_cols):
        idx = group_df.index

        ht_dl, lt_dl = outliers(group_df['DL'].values)
        df.loc[idx, 'DL'] = np.where(
            df.loc[idx, 'DL'] > ht_dl, np.nan, df.loc[idx, 'DL'])
        df.loc[idx, 'DL'] = np.where(
            df.loc[idx, 'DL'] < lt_dl, np.nan, df.loc[idx, 'DL'])

        ht_pse, lt_pse = outliers(group_df['PSE'].values)
        df.loc[idx, 'PSE'] = np.where(
            df.loc[idx, 'PSE'] > ht_pse, np.nan, df.loc[idx, 'PSE'])
        df.loc[idx, 'PSE'] = np.where(
            df.loc[idx, 'PSE'] < lt_pse, np.nan, df.loc[idx, 'PSE'])

    df['DLImputed'] = False
    df['PSEImputed'] = False

    output_folder = os.path.join(output_dir, 'anovas')
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    outpath = os.path.join(output_folder,
                           'df_perception_postfit_' + sesstag + '.tsv')
    df.to_csv(outpath, index=False, sep='\t')

    return df


def _format_pval(p, style='stars', show_ns=True):
    """Format a p-value for plot annotation.

    Parameters
    ----------
    p : float
        The p-value to format (use p-corr where available, else p-unc).
    style : {'stars', 'numeric'}
        'stars'   – significance symbols: *** / ** / * / n.s.
        'numeric' – numeric string: 'p<0.001' or 'p=0.032'.
    show_ns : bool
        Only relevant when style='stars'.  If True, non-significant
        comparisons are labelled 'n.s.'; if False they return '' so no
        annotation is drawn.  When style='numeric' every value is always
        shown, so this flag has no effect.

    Returns
    -------
    str – annotation string, or '' when the annotation should be skipped.
    """
    if style == 'numeric':
        if p < 0.001:
            return 'p<0.001'
        return f'p={p:.3f}'

    # style == 'stars'
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'n.s.' if show_ns else ''


def _annotate_bracket(ax, x1, x2, y_top, y_span, label,
                      fontsize=9, color='black', linewidth=0.8):
    """Draw a significance bracket between two box positions on *ax*.

    Parameters
    ----------
    ax : matplotlib Axes
    x1, x2 : float
        x-coordinates (data units) of the two boxes being compared.
    y_top : float
        y-coordinate (data units) at which the horizontal bar sits.
    y_span : float
        Total visible y-range of the axis (used to scale the tick height).
    label : str
        Text placed above the bracket midpoint.  If empty, nothing is drawn.
    """
    if not label:
        return

    tick_h = y_span * 0.025
    mid_x = (x1 + x2) / 2.0

    ax.plot([x1, x2], [y_top, y_top],
            color=color, linewidth=linewidth, clip_on=False)
    ax.plot([x1, x1], [y_top - tick_h, y_top],
            color=color, linewidth=linewidth, clip_on=False)
    ax.plot([x2, x2], [y_top - tick_h, y_top],
            color=color, linewidth=linewidth, clip_on=False)
    ax.text(mid_x, y_top + tick_h * 0.4, label,
            ha='center', va='bottom', fontsize=fontsize, color=color)


def twoway_repanova(df, output_dir, sesstag, min_valid_standards,
                    alternative='two-sided',
                    annot_style='stars',
                    annot_show_ns=True):
    """Run two-way repeated-measures ANOVA on DL and save an annotated boxplot.

    Parameters
    ----------
    annot_style : {'stars', 'numeric'}
        How p-values are rendered on the plot.
        'stars'   – *** / ** / * / n.s.
        'numeric' – p<0.001 / p=0.032 etc.
    annot_show_ns : bool
        When annot_style='stars', whether to label non-significant comparisons
        with 'n.s.'.  Has no effect when annot_style='numeric'.

    Annotation layout
    -----------------
    • A bracket drawn inside each subplot spans the two box positions and
      carries the within-modality beat-vs-interval p-value (interaction row).
    • Two italic lines of fig-level text below the subplots report the
      marginal Condition and Modality effects.
    """
    if isinstance(df, str):
        df = pd.read_csv(df, sep='\t')

    output_folder = os.path.join(output_dir, 'anovas/twoway')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cols = ['Condition', 'Modality', 'Subject', 'Standard', 'DL']
    df = df[cols].copy()

    cell_df = df.groupby(
        ['Condition', 'Modality', 'Subject'],
        as_index=False).agg(
            DL=('DL', 'mean'),
            ValidStandards=('DL', 'count'))

    excluded_cells = cell_df[
        cell_df['ValidStandards'] < min_valid_standards].copy()
    if not excluded_cells.empty:
        excluded_cells.to_csv(
            os.path.join(output_folder,
                         'twoway_excluded_cells_' + sesstag + '.tsv'),
            sep='\t', index=False)

    df = cell_df[cell_df['ValidStandards'] >= min_valid_standards].copy()

    grid_counts = df.groupby('Subject').size()
    complete_subjects = grid_counts[grid_counts == 4].index.tolist()
    incomplete_subjects = grid_counts[grid_counts < 4].index.tolist()

    if incomplete_subjects:
        excluded_df = pd.DataFrame({'Subject': incomplete_subjects})
        excluded_df.to_csv(
            os.path.join(output_folder,
                         'twoway_excluded_' + sesstag + '.tsv'),
            sep='\t', index=False)

    df = df[df['Subject'].isin(complete_subjects)].copy()
    if df['Subject'].nunique() < 2:
        msg = 'Skipped: repeated-measures ANOVA requires at least 2 subjects.'
        skip_df = pd.DataFrame({'Reason': [msg]})
        skip_df.to_csv(
            os.path.join(output_folder, 'twoway_anova_' + sesstag + '.tsv'),
            sep='\t', index=False)
        return None

    # ---- ANOVA -------------------------------------------------------
    model = AnovaRM(data=df, depvar='DL', subject='Subject',
                    within=['Modality', 'Condition'])
    results = model.fit()
    results.anova_table.to_csv(
        os.path.join(output_folder, 'twoway_anova_' + sesstag + '.tsv'),
        sep='\t')

    # ---- Post-hoc ----------------------------------------------------
    # 'cohen' is stable across pingouin versions ('eta-square' removed in
    # v0.6.x).  p-value column names changed from 'p-unc'/'p-corr' (hyphens,
    # v0.5.x) to 'p_unc'/'p_corr' (underscores, v0.6.x).
    # These names are detected at runtime.
    #
    # Two separate calls:
    #   posthoc     – within=['Condition','Modality'] – used for marginal
    #                 Condition and Modality effects.
    #   posthoc_rev – within=['Modality','Condition'] – interaction
    #                 rows where the 'Modality' column holds the fixed
    #                 level and A/B are
    #                 beat/interval. These rows are used for subplot brackets
    #                 and are the post-hoc rows saved to disk.
    posthoc = pg.pairwise_tests(
        data=df, dv='DL', within=['Condition', 'Modality'],
        subject='Subject', alternative=alternative, return_desc=True,
        padjust='holm', effsize='cohen')

    posthoc_rev = pg.pairwise_tests(
        data=df, dv='DL', within=['Modality', 'Condition'],
        subject='Subject', alternative=alternative, return_desc=True,
        padjust='holm', effsize='cohen')

    # Detect p-value column names (version-dependent).  Some pingouin
    # outputs omit corrected p-values when there is only one comparison.
    def _pvalue_columns(ph):
        p_corr = next((c for c in ['p-corr', 'p_corr']
                       if c in ph.columns), None)
        p_unc = next((c for c in ['p-unc', 'p_unc']
                      if c in ph.columns), None)
        if p_unc is None:
            raise KeyError('Could not find an uncorrected p-value column.')
        return p_corr, p_unc

    _pc, _pu = _pvalue_columns(posthoc)
    _pcr, _pur = _pvalue_columns(posthoc_rev)

    # ---- Annotation helpers ------------------------------------------
    def _ph_pval(mask, ph, pc, pu):
        """p-corr when available (non-NaN), otherwise p-unc."""
        row = ph[mask]
        if row.empty:
            return None
        p_unc = row[pu].values[0]
        if pc is None:
            return p_unc
        p_corr = row[pc].values[0]
        return (p_corr
                if (p_corr is not None and not np.isnan(p_corr))
                else p_unc)

    def _label(p):
        if p is None:
            return ''
        return _format_pval(p, annot_style, annot_show_ns)

    # Identify contrast strings in the forward table
    contrasts = posthoc['Contrast'].unique().tolist()
    cond_contrast = next(
        (c for c in contrasts if c.strip() == 'Condition'), None)
    mod_contrast = next(
        (c for c in contrasts if c.strip() == 'Modality'), None)

    # Marginal effects
    lbl_cond_marg = _label(
        _ph_pval(posthoc['Contrast'] == cond_contrast, posthoc, _pc, _pu)
        if cond_contrast else None)
    lbl_mod_marg = _label(
        _ph_pval(posthoc['Contrast'] == mod_contrast, posthoc, _pc, _pu)
        if mod_contrast else None)

    # Within-modality Condition brackets from the reversed call
    inter_contrast_rev = next(
        (c for c in posthoc_rev['Contrast'].unique()
         if 'Condition' in c and 'Modality' in c), None)

    if inter_contrast_rev is None:
        posthoc_within_modality = posthoc_rev.iloc[0:0].copy()
    else:
        posthoc_within_modality = posthoc_rev[
            posthoc_rev['Contrast'] == inter_contrast_rev
        ].copy()

    posthoc_within_modality.to_csv(
        os.path.join(output_folder, 'twoway_posthoc_' + sesstag + '.tsv'),
        sep='\t', index=False)

    modalities = [m for m in ['audio', 'visual']
                  if m in df.Modality.unique()]
    conditions = [c for c in ['beat', 'interval']
                  if c in df.Condition.unique()]

    annot_cond = {}
    for modality in modalities:
        if inter_contrast_rev is None:
            annot_cond[modality] = ''
            continue
        mask = ((posthoc_rev['Contrast'] == inter_contrast_rev) &
                (posthoc_rev['Modality'].astype(str) == modality))
        annot_cond[modality] = _label(
            _ph_pval(mask, posthoc_rev, _pcr, _pur))

    # ---- Layout arithmetic -------------------------------------------
    # Tighter padding than before: 5% margin around the data, plus only the
    # headroom that's actually needed for whichever brackets will be drawn.
    y_values_all = df['DL'].values
    y_min = np.nanmin(y_values_all)
    y_max = np.nanmax(y_values_all)
    y_data_range = y_max - y_min if y_max != y_min else 0.02

    y_pad_bot = y_data_range * 0.05    # small breathing room below the data
    y_pad_top = y_data_range * 0.05    # ... and above (always)

    has_cond_bracket = any(annot_cond.values())
    has_mod_bracket = bool(lbl_mod_marg)

    # Headroom for each bracket layer, only when that layer is drawn
    cond_pad = y_data_range * 0.18 if has_cond_bracket else 0.0
    mod_pad = y_data_range * 0.16 if has_mod_bracket else 0.0

    y_axis_bot = y_min - y_pad_bot
    y_axis_top = y_max + y_pad_top + cond_pad + mod_pad
    y_span = y_axis_top - y_axis_bot

    cond_bracket_y = y_max + y_pad_top + cond_pad * 0.45

    # ---- Figure geometry (figure-fraction coords) --------------------
    bottom = 0.15
    ax_h = 0.66
    ax_w = 0.37
    left0 = 0.16
    left1 = left0 + ax_w + 0.08

    # ---- Plot --------------------------------------------------------
    box_colors = [[.0, .66, .47, .5], [.89, .61, .06, .5]]

    fig = plt.figure(figsize=(4.5, 5.2))
    fig.subplots_adjust(top=0.85)

    for m, modality in enumerate(modalities):
        left = left0 if m == 0 else left1
        ax = plt.axes([left, bottom, ax_w, ax_h])
        width = .275
        x_labels = [str(cd).capitalize() for cd in conditions]

        dl_beat = df[(df.Modality == modality) &
                     (df.Condition == 'beat')].DL.values.tolist()
        dl_interval = df[(df.Modality == modality) &
                         (df.Condition == 'interval')].DL.values.tolist()

        beat = ax.boxplot(
            dl_beat, bootstrap=100, positions=[.2], widths=width,
            flierprops={'marker': '', 'markersize': 5}, patch_artist=True,
            medianprops=dict(color='black', linewidth=0.), notch=True,
            meanline=True, showmeans=True,
            meanprops=dict(color='tab:brown', linewidth=1.5))
        interval = ax.boxplot(
            dl_interval, bootstrap=100, positions=[.8], widths=width,
            flierprops={'marker': '', 'markersize': 5}, patch_artist=True,
            medianprops=dict(color='black', linewidth=0.), notch=True,
            meanline=True, showmeans=True,
            meanprops=dict(color='tab:brown', linewidth=1.5))

        for patch1, patch2 in zip(beat['boxes'], interval['boxes']):
            patch1.set_facecolor(box_colors[m])
            patch2.set_facecolor(box_colors[m])

        ax.set_xticks([.2, .8], x_labels, fontsize=11)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylim([y_axis_bot, y_axis_top])

        # Filled-rectangle legend above each subplot (figure-level so it
        # doesn't eat into the axis area).  We attach the legend artist
        # directly to the figure, anchored above the subplot in figure-
        # fraction coordinates.
        legend_label = 'Auditory' if m == 0 else 'Visual'
        legend_patch = Patch(facecolor=box_colors[m], edgecolor='black',
                             linewidth=0.8)
        leg = fig.legend([legend_patch], [legend_label],
                         loc='center',
                         bbox_to_anchor=(left + ax_w * 0.5,
                                         bottom + ax_h + 0.045),
                         bbox_transform=fig.transFigure,
                         frameon=False, prop={'size': 12},
                         handlelength=1.4, handleheight=1.1,
                         handletextpad=0.5)
        # Make sure the legend sits on top of everything else
        leg.set_zorder(5)

        if m == 0:
            ax.set_ylabel('Group DL', fontsize=14)
        else:
            ax.spines['left'].set_visible(False)
            ax.axes.get_yaxis().set_visible(False)

        # Within-modality Condition bracket (beat vs interval)
        _annotate_bracket(ax, .2, .8, cond_bracket_y, y_span,
                          annot_cond[modality], fontsize=9)

    # x-axis label centred between the two subplot centres.
    x_label_x = (left0 + left1 + ax_w) / 2.0
    fig.text(x_label_x, 0.065, 'Conditions', ha='center', size=14)

    # ---- Cross-axes Modality bracket (auditory vs visual) ------------
    if lbl_mod_marg:
        x_fig_left = left0 + ax_w * 0.5
        x_fig_right = left1 + ax_w * 0.5
        y_data_mod = y_axis_top - y_data_range * 0.03
        y_fig_mod = bottom + ax_h * (y_data_mod - y_axis_bot) / y_span
        tick_fig = ax_h * 0.025

        def _fig_line(x0, x1, y0, y1, **kw):
            fig.add_artist(Line2D([x0, x1], [y0, y1],
                                  transform=fig.transFigure,
                                  clip_on=False, **kw))

        lc = 'black'
        _fig_line(x_fig_left,  x_fig_right, y_fig_mod,            y_fig_mod,
                  linewidth=0.8, color=lc)
        _fig_line(x_fig_left,  x_fig_left,  y_fig_mod - tick_fig, y_fig_mod,
                  linewidth=0.8, color=lc)
        _fig_line(x_fig_right, x_fig_right, y_fig_mod - tick_fig, y_fig_mod,
                  linewidth=0.8, color=lc)
        fig.text((x_fig_left + x_fig_right) / 2,
                 y_fig_mod + tick_fig * 0.4,
                 lbl_mod_marg,
                 ha='center', va='bottom', fontsize=9, color=lc,
                 transform=fig.transFigure)

    # ---- Marginal Condition text (bottom-left, below the plot) -------
    if lbl_cond_marg:
        fig.text(0.02, 0.018,
                 f'Condition (marginal): {lbl_cond_marg}',
                 ha='left', va='bottom', fontsize=7.5,
                 color='dimgrey', style='italic')

    plt.savefig(
        os.path.join(output_folder,
                     'twoway_boxplot_' + sesstag + '.png'),
        dpi=300,
        bbox_inches='tight')
    plt.close('all')


# %%
# =========================== INPUTS ===================================

# ##################### Subjects' lists ################################
# All subjects
ALL_SUBJECTS = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36,
                37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

# All good subjects including img pilot (sub-04)
GOOD_SUBJECTS = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                 21, 22, 23, 24, 25, 26, 27, 28, 29, 32, 34, 35, 38, 39, 40,
                 41, 42, 43, 44, 45, 46, 47]

# Img subjects only (without pilot)
IMG_SUBJECTS = [3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26,
                28, 29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

# Subjects who did all behavioral sessions with the random condition...
# ... in the NTFD task and img sessions
BEHAVIMG_RAND_SUBJECTS = [16, 18, 20, 21, 22, 23, 26, 28, 29, 32, 34, 35, 38,
                          39, 40, 41, 42, 43, 44, 45, 46, 47]

# Second batch
ALL_SB_SUBJECTS = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]

GOOD_SB_SUBJECTS = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61, 62]

SB2_SUBJECTS = [50, 51, 52, 55, 57, 59]

SB3_SUBJECTS = []

# #######################################################################

# TASKS = ['Auditory Perception', 'Visual Perception']

N_TRIALS = 30

FIT_MAX_ABS_PSE = 0.50
FIT_MAX_DL = 0.50
FIT_MIN_DL = 0.0
MIN_VALID_STANDARDS = 1  # raised from 0 to exclude cells with no valid fits

# ---- Annotation options for the ANOVA boxplot -----------------------
# annot_style   : 'stars'   -> *** / ** / * / n.s.
#                 'numeric' -> p<0.001 / p=0.032 etc.
# annot_show_ns : only relevant when annot_style='stars'.
#                 True  -> non-significant comparisons are labelled 'n.s.'
#                 False -> non-significant comparisons are not annotated
ANNOT_STYLE = 'numeric'
ANNOT_SHOW_NS = False

# Show group-level PSE and DL estimates on the group psychometric
# plots, using the same display style as the individual plots.
SHOW_GROUP_FIT_ESTIMATES = True

# For any multi-session tag (e.g., behavses, behav12, behav13,
# behav23, imgses, allses), compute ANOVA input from the mean of
# session-specific fits instead of refitting pooled trials.
SESSION_AVERAGE_MULTISESSION_ANOVA = True


# %%
# ========================= PARAMETERS =================================

# #### First Batch ####

fb_sessions_dic = {
    'allses':   'All Sessions',
    'behavses': 'All Behavioral Sessions',
    'imgses':   'All Imaging Sessions',
    'ses-01':   'Session 1',
    'ses-02':   'Session 2',
    'ses-03':   'Session 3',
    'ses-04':   'Session 4',
    'ses-05':   'Session 5',
    'behav12':  'Sessions 1 and 2',
    'behav13':  'Sessions 1 and 3',
    'behav23':  'Sessions 2 and 3',
}

fb_subjects_dic = {
    'allses':   GOOD_SUBJECTS,
    'behavses': GOOD_SUBJECTS,
    'imgses':   IMG_SUBJECTS,
    'ses-01':   GOOD_SUBJECTS,
    'ses-02':   GOOD_SUBJECTS,
    'ses-03':   GOOD_SUBJECTS,
    'ses-04':   IMG_SUBJECTS,
    'ses-05':   IMG_SUBJECTS,
    'behav12':  GOOD_SUBJECTS,
    'behav13':  GOOD_SUBJECTS,
    'behav23':  GOOD_SUBJECTS,
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

# #### Map tag -> integer session list ####

sessions_list_dic = {
    'allses':   [1, 2, 3, 4, 5],
    'behavses': [1, 2, 3],
    'imgses':   [4, 5],
    'ses-01':   [1],
    'ses-02':   [2],
    'ses-03':   [3],
    'ses-04':   [4],
    'ses-05':   [5],
    'behav12':  [1, 2],
    'behav13':  [1, 3],
    'behav23':  [2, 3],
}

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory containing the raw perception dataframes
# (df_perception_<tag>.tsv files).  Set to None to use the default
# location: <RESULTS_FOLDER>/raw_dataframes/
RAW_DF_DIR = None


# %%
# ============================ RUN =====================================

if __name__ == "__main__":

    for batch_tag, sessions_dic, subjects_dic in [
        # ('first',  fb_sessions_dic,  fb_subjects_dic),
        ('second', sb_sessions_dic,  sb_subjects_dic),
    ]:

        results_subfolder = 'perception_results_' + batch_tag + '_batch'
        RESULTS_FOLDER = os.path.join(MAIN_DIR, results_subfolder)

        if not os.path.exists(RESULTS_FOLDER):
            os.mkdir(RESULTS_FOLDER)

        for tag, session_label in sessions_dic.items():

            SUBJECTS = subjects_dic[tag]
            SESSIONS = sessions_list_dic[tag]

            print('\n' + '=' * 60)
            print(f'Batch: {batch_tag}  |  Tag: {tag}  |  {session_label}')
            print('=' * 60)

            # ################# PERCEPTION ##############################

            estim_pse = []
            estim_dl = []
            for estimator in ['mle_cdf', 'mle_expit']:
                cond_pse = []
                cond_dl = []
                cond_ce = []
                cond_gpse = []
                for cond in ['beat', 'interval']:

                    # Compute individual psychometric functions
                    rfone_audio, rftwo_audio, rfone_visual, rftwo_visual, \
                        stand, comp, ipse_audio, idl_audio, ipse_visual, \
                        idl_visual = \
                        individual_perception(
                            SUBJECTS, MAIN_DIR, RESULTS_FOLDER,
                            cond, SESSIONS, tag, session_label,
                            FIT_MAX_ABS_PSE, FIT_MAX_DL,
                            FIT_MIN_DL, estimator=estimator,
                            raw_df_dir=RAW_DF_DIR)

                    # Compute group psychometric functions. For any
                    # multi-session tag, average response frequencies across
                    # sessions within subject before the group average.
                    # This mirrors the ANOVA logic, where session-specific
                    # fitted estimates are averaged within subject.
                    if (SESSION_AVERAGE_MULTISESSION_ANOVA and
                            len(SESSIONS) > 1):
                        print(
                            'Using session-averaged response frequencies for '
                            f'group psychometric plot: {tag} ({SESSIONS})')
                        group_rf = session_average_group_frequencies(
                            SUBJECTS, RESULTS_FOLDER, cond, SESSIONS, tag,
                            raw_df_dir=RAW_DF_DIR)
                        rfone_audio, rftwo_audio, rfone_visual, rftwo_visual, \
                            group_stand, group_comp = group_rf

                        if not np.array_equal(stand, group_stand):
                            raise ValueError(
                                'Pooled and session-average standards do '
                                'not match.')
                        if not np.array_equal(comp, group_comp):
                            raise ValueError(
                                'Pooled and session-average comparisons do '
                                'not match.')

                    gpse, _ = group_perception(
                        rfone_audio, rftwo_audio, rfone_visual, rftwo_visual,
                        stand, comp, cond,
                        RESULTS_FOLDER, tag, session_label,
                        FIT_MAX_ABS_PSE, FIT_MAX_DL,
                        FIT_MIN_DL, estimator=estimator,
                        show_fit_estimates=SHOW_GROUP_FIT_ESTIMATES)

                    # Start concatenating and appending.  For any
                    # multi-session ANOVA, use the average of
                    # session-specific fits.  This applies to
                    # behavioral multi-session tags and imaging tags.
                    if (SESSION_AVERAGE_MULTISESSION_ANOVA and
                            len(SESSIONS) > 1):
                        print(
                            'Using session-averaged fitted estimates for '
                            f'ANOVA input: {tag} ({SESSIONS})')
                        avg_stand, _, ipse_audio, idl_audio, \
                            ipse_visual, idl_visual = \
                            session_average_individual_estimates(
                                SUBJECTS, RESULTS_FOLDER, cond,
                                SESSIONS, tag, FIT_MAX_ABS_PSE,
                                FIT_MAX_DL, FIT_MIN_DL,
                                estimator=estimator,
                                raw_df_dir=RAW_DF_DIR)
                        if not np.array_equal(stand, avg_stand):
                            raise ValueError(
                                'Pooled and session-average standards do '
                                'not match.')

                    ipse = np.concatenate(([ipse_audio], [ipse_visual]),
                                          axis=0).tolist()
                    idl = np.concatenate(([idl_audio], [idl_visual]),
                                         axis=0).tolist()

                    cond_pse.append(ipse)
                    cond_dl.append(idl)
                    cond_gpse.append(gpse)

                    if cond == 'interval' and estimator == 'mle_expit':
                        pass
                    else:
                        del rfone_audio
                        del rftwo_audio
                        del rfone_visual
                        del rftwo_visual
                        # del stand
                        del comp
                        del ipse
                        del idl

                # Only accumulate mle_expit results for the ANOVA;
                # mle_cdf is used solely for the PSE plot.
                if estimator == 'mle_expit':
                    estim_pse.append(cond_pse)
                    estim_dl.append(cond_dl)

                # Plot PSE
                mod_gpse = np.swapaxes(cond_gpse, 0, 1)
                plotfit_perception(stand, mod_gpse, estimator,
                                   RESULTS_FOLDER, tag, session_label)

                # Compute ANOVAs and plot DL
                if estimator == 'mle_cdf':
                    continue
                else:
                    db = dataframe(estim_pse, estim_dl, stand,
                                   RESULTS_FOLDER, tag, SUBJECTS)
                    twoway_repanova(db, RESULTS_FOLDER, tag,
                                    MIN_VALID_STANDARDS,
                                    annot_style=ANNOT_STYLE,
                                    annot_show_ns=ANNOT_SHOW_NS)