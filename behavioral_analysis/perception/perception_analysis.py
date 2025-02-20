"""
Analysis of behavioral data for the Perception Tasks of the Music-SDTB project

author: Ana Luisa Pinho
e-mail: agrilopi@uwo.ca

Created: February, 2023
Last update: February, 2025

Compatibility: Python 3.10.14
"""

import sys
import os

import warnings

import numpy as np
import pandas as pd

import pingouin as pg
from scipy import stats, optimize, special
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# setting path
sys.path.append('../../')
# importing
from utils import parse_logfile

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
            theoretical_isi5 = data[dt+8][8]
            if data[dt+10][5] == 'feedback' and \
               data[dt+10][11] in ['o', 'p', 'b', 'y']:
                # rt = int(data[dt+9][7]) + int(data[dt+10][10])
                answer = data[dt+10][11]
            elif data[dt+10][5] == 'feedback' and \
                 data[dt+10][11] == 'None':
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

    standards = np.sort(np.unique(np.array(beat_trials)[:, 0]))
    standards = np.array(standards, dtype='int')
    comparisons = np.sort(np.unique([round((bt[1] - bt[0]) / bt[0], 2)
                                     for bt in beat_trials]))
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
    """
    This function minimizes the negative log-likelihood function,
    which is equivalent to maximizing the log-likelihood function.
    """
    # If the standard deviation parameter is negative, return a large value:
    if par_vec[1] < 0:
        return(1e8)
    # The likelihood function values:
    lik = stats.norm.cdf(y, 
                         loc = par_vec[0],
                         scale = par_vec[1])
    # This is similar to calculating the likelihood for Y - XB
    # res = y - par_vec[0] - par_vec[1] * x
    # lik = norm.cdf(res, loc = 0, sd = par_vec[2])

    # If all logarithms are zero, return a large value
    if any(v == 0 for v in lik):
        lik = np.where(lik==0., -1e-8, lik)

    return(-sum(np.multiply(n2, np.log(lik[np.nonzero(lik)])))
           -sum(np.multiply(n1, np.log(1-lik[np.nonzero(lik)]))))


def loglik_expit(par_vec, y, n2, n1):
    """
    This function minimizes the negative log-likelihood function,
    which is equivalent to maximizing the log-likelihood function.
    """
    # If the standard deviation parameter is negative, return a large value:
    if par_vec[1] < 0:
        return(1e8)
    # The likelihood function values:
    lik = special.expit((y - par_vec[0]) / par_vec[1])
    minus_lik = special.expit(-(y - par_vec[0]) / par_vec[1])

    # If all logarithms are zero, return a large value
    if any(v == 0 for v in lik):
        lik = np.where(lik==0., -1e-8, lik)
    if any(w == 0 for w in minus_lik):
        minus_lik = np.where(minus_lik==0., -1e-8, minus_lik)

    return(-sum(np.multiply(n2, np.log(lik[np.nonzero(lik)])))
           -sum(np.multiply(n1, np.log(minus_lik[np.nonzero(minus_lik)]))))


def errFit(hess_inv, resVariance):
    """Error of the fit parameters"""
    return np.sqrt(np.diag(hess_inv * resVariance))


def outliers(arr):
    q75, q25 = np.percentile(arr, [75, 25])
    iqr = q75 - q25
    high_thresh = q75 + 1.7*iqr
    low_thresh = q25 - 1.7*iqr

    return high_thresh, low_thresh


def individual_perception(
        subjects, this_dir, output_dir, condition, n_trials,
        sessions, sesstag, estimator='mle_expit',
        modalities=['auditory', 'visual']):

    df_path = os.path.join(
        os.path.abspath(
            os.path.join(this_dir, 'perception_results',
                         'raw_dataframes')),
        'df_perception.tsv')
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

            # Count the number of trials
            filtered_df = df[
                (df['subject'] == subject) &
                (df['modality'] == modality) &
                (df['standard'] == 459)][
                round(((df['comparison'] - df['standard']) / df['standard']),
                      2) == .2]
            total_beat_trials = filtered_df[
                (filtered_df['condition'] == 'beat')].shape[0]
            total_interval_trials = filtered_df[
                (filtered_df['condition'] == 'interval')].shape[0]
            
            if condition == 'beat':
                # Calculate frequencies of comparisons per standard
                standards, comparisons, n1_beat, n2_beat, _, _ = \
                    perception_frequencies(beat_trials, interval_trials)

                rf1 = [[n1/total_beat_trials for n1 in n1b] for n1b in n1_beat]
                rf2 = [[n2/total_beat_trials for n2 in n2b] for n2b in n2_beat]
            else:
                assert condition == 'interval'
                standards, comparisons, _, _, n1_interval, n2_interval = \
                    perception_frequencies(beat_trials, interval_trials)

                rf1 = [[n1/total_interval_trials for n1 in n1i]
                       for n1i in n1_interval]
                rf2 = [[n2/total_interval_trials for n2 in n2i]
                       for n2i in n2_interval]

            # Aggregate data
            if modality == 'auditory':
                all_rf1_audio.append(rf1)
                all_rf2_audio.append(rf2)
            else:
                assert modality == 'visual'
                all_rf1_visual.append(rf1)
                all_rf2_visual.append(rf2)

            # ################## Plotting ###############################
            colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
                      'tab:purple']
            if s == 0 and m == 0:
                fig = plt.figure(figsize=(16, 100))

            # Define subplot of bar charts and its position in the fig
            # plt.axes([left, bottom, width, height])
            ax = plt.axes([.1 + m*.46, .9685 - s*.023, .428, .0125])

            std_pse_audio = []
            std_dl_audio = []
            std_pse_visual = []
            std_dl_visual = []
            for i, st in enumerate(standards):
                # Choose estimator
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
                    fun = func,
                    x0 = [np.mean(comparisons), .1],
                    args = (comparisons, rf2[i], rf1[i]))

                # Estimates
                pse = opt_res.x[0]
                dl = opt_res.x[1] * constant
                # Standard Errors estimated from Fisher information
                se_pse = np.sqrt(np.diag(opt_res.hess_inv))[0]
                se_dl = np.sqrt(
                    np.diag(opt_res.hess_inv))[1] * constant
                # Correlation between mu (pse) and sigma (opt_res.x[1])
                # r = opt_res.hess_inv[0][1] / \
                #     np.sqrt(np.prod(np.diag(opt_res.hess_inv)))
                # 95%CIs
                ci95_pse = se_pse * 1.96
                ci95_dl = se_dl * 1.96

                # Append
                if modality == 'auditory':
                    std_pse_audio.append(pse)
                    std_dl_audio.append(dl)
                else:
                    assert modality == 'visual'
                    std_pse_visual.append(pse)
                    std_dl_visual.append(dl)

                # Plot
                # Plot each fit in one image
                # fig, ax = plt.subplots(1, 1)
                x = np.linspace(np.amin(comparisons), np.amax(comparisons),
                                100)
                # Plot data
                # ax.plot(comparisons, rf2[i], 'bo', color=colors[i],
                #         markersize=3)
                # Plot fit
                ax.plot(x, stats.norm(pse, opt_res.x[1]).cdf(x),
                        color=colors[i], label='Standard = ' + str(st) + 'ms')
                # Add horizontal dashed line at y = 0.5
                ax.axhline(.5, linestyle='--', color='silver', linewidth=1)
                # Hide the right and top spines
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                # Set x axis
                x_values = np.insert(comparisons, 3, 0)
                x_labels = [str(int(xl*100)) + '%' for xl in x_values]
                ax.set_xticks(x_values, x_labels)
                # Add estimates info
                ax.text(-.21, 1.53, 'For 95% CI,', fontsize=7.5)
                ax.text(-.21, 1.41 - i*.098,
                        'PSE=%.02f' % (pse*100) +
                        '\u00B1%.02f' % (ci95_pse*100) + '%; ' +
                        'DL=%.02f' % (dl*100) +
                        '\u00B1%.02f' % (ci95_dl*100) + '%', fontsize=7.5,
                        color=colors[i])
                # Set limits of y-axis
                ax.set_ylim([-.1, 1.1])

            # Add legend
            if s == 0:
                if m == 0:
                    ax.legend(loc='lower right', frameon=False,
                              prop={'size': 6})
                    ax.set_title('Auditory Perception', weight='bold', pad=60,
                                 fontsize=16)
                else:
                    assert m == 1
                    ax.set_title('Visual Perception', weight='bold', pad=60,
                                 fontsize=16)

            # Name of x-axis
            fig.text(.495, .0025, 'Comparisons (%)', fontsize=14)
            # Name of y-axis
            fig.text(.062, .46, 'Relative Frequency of "longer" responses',
                     fontsize=14, rotation=90)

            # Append
            if modality == 'auditory':
                all_pse_audio.append(std_pse_audio)
                all_dl_audio.append(std_dl_audio)
            else:
                assert modality == 'visual'
                all_pse_visual.append(std_pse_visual)
                all_dl_visual.append(std_dl_visual)

        fig.text(.03, .9765 - s*.023, 'Subject %d' % subject, ha='center',
                 fontsize=10, weight='bold')

    # Title
    if estimator == 'mle_cdf':
        suffix = '(Estimator: MLE of Norm CDF)'
    else:
        assert estimator == 'mle_expit'
        suffix = '(Estimator: MLE of Logistic-Sigmoid Function)'
    plt.suptitle(
        'Individual Relative Frequencies for the ' + condition.capitalize() +
        ' condition of the Perception Tasks: ' + sessions_dic[sesstag] + ' ' +
        suffix, x=.5, y=.9975, size=16, linespacing=.75)

    output_folder = os.path.join(output_dir, 'individual_psychometric')
    # Create output_folder, if it does not exist
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    # Save figure
    plt.savefig(os.path.join(
        this_dir, output_folder,
        'individual_psychometric_' + condition + '_' + estimator + '_' +
        sesstag + '.pdf'))

    plt.close('all')

    return (all_rf1_audio, all_rf2_audio, all_rf1_visual, all_rf2_visual,
            standards, comparisons, all_pse_audio, all_dl_audio,
            all_pse_visual, all_dl_visual)


def group_perception(all_rf1_audio, all_rf2_audio,
                     all_rf1_visual, all_rf2_visual,
                     standards, comparisons, condition, output_dir, sesstag,
                     estimator = 'mle_expit'):

    group_rf1_audio = np.mean(all_rf1_audio, axis=0)
    group_rf2_audio = np.mean(all_rf2_audio, axis=0)
    group_rf1_visual = np.mean(all_rf1_visual, axis=0)
    group_rf2_visual = np.mean(all_rf2_visual, axis=0)

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
            # Standard Errors estimated from Fisher information
            se_pse = np.sqrt(np.diag(opt_res.hess_inv))[0]
            se_dl = np.sqrt(
                np.diag(opt_res.hess_inv))[1] * constant
            # Correlation between mu (pse) and sigma (opt_res.x[1])
            # r = opt_res.hess_inv[0][1] / \
            #     np.sqrt(np.prod(np.diag(opt_res.hess_inv)))
            # 95%CIs
            ci95_pse = se_pse * 1.96
            ci95_dl = se_dl * 1.96

            # Estimate the goodness of the fit
            dFit = errFit(opt_res.hess_inv,
                          opt_res.fun/(len(rf2[i]) - \
                                       len([np.mean(comparisons), 1.])))

            print(modality, '-', condition)
            print('estimator:', estimator)
            print('standard:', st)
            print('minimize:\n\tx: ', opt_res.x, '\n\tdx: ', dFit)

            # Plot each fit in one image
            # fig, ax = plt.subplots(1, 1)
            x = np.linspace(np.amin(comparisons), np.amax(comparisons),
                            100)
            # Plot data
            ax[m].plot(comparisons, rf2[i], 'bo', color=colors[i],
                       markersize=6, alpha=.5)
            # Plot fit
            if estimator == 'mle_cdf':
                ax[m].plot(x, stats.norm(pse, opt_res.x[1]).cdf(x),
                           color=colors[i], linewidth=6, markersize=12,
                           alpha=.5, label='Standard = ' + str(st) + 'ms')
            else:
                assert estimator == 'mle_expit'
                ax[m].plot(x, special.expit((x - opt_res.x[0]) / opt_res.x[1]),
                           color=colors[i], linewidth=6, markersize=12,
                           alpha=.5, label='Standard = ' + str(st) + 'ms')
            # Add horizontal dashed line at y = 0.5
            ax[m].axhline(.5, linestyle='--', color='grey', linewidth=3)
            # Hide the right and top spines
            ax[m].spines['right'].set_visible(False)
            ax[m].spines['top'].set_visible(False)
            # Set x axis
            x_values = np.insert(comparisons, 3, 0)
            x_labels = [str(int(xl*100)) for xl in x_values]
            ax[m].set_xticks(x_values, x_labels, fontsize=16)
            # Add estimates info
            # ax[m].text(-.21, 1.03, 'For 95% CI,', fontsize=10)
            # ax[m].text(-.21, .98 - i*.05,
            #            'PSE=%.02f' % (pse*100) +
            #            '\u00B1%.02f' % (ci95_pse*100) + '%; ' +
            #            'DL=%.02f' % (dl*100) +
            #            '\u00B1%.02f' % (ci95_dl*100) + '%', fontsize=10,
            #            color=colors[i])

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
        y_labels = [str(int(yl*100)) for yl in y_values]
        ax[m].set_yticks(y_values, y_labels, fontsize=16)

    # Title
    if estimator == 'mle_cdf':
        suffix = '(Estimator: MLE of Norm CDF)'
    else:
        assert estimator == 'mle_expit'
        suffix = '(Estimator: MLE of Logistic-Sigmoid Function)'
    plt.suptitle(
        'Group Mean of Relative Frequencies for the ' +
        condition.capitalize() + ' condition of the Perception Tasks: ' +
        sessions_dic[sesstag] + ' ' + suffix, x=.5, y=.97, size=14,
        linespacing=.75)
    plt.title(sessions_dic[sesstag])

    output_folder = os.path.join(output_dir, 'group_psychometric')
    # Create output_folder, if it does not exist
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Save figure
    plt.savefig(os.path.join(
        output_folder,
        'group_psychometric_' + condition + '_' + estimator + '_' + sesstag + \
        '.pdf'))

    plt.close('all')

    return group_pse, group_dl


def plotfit_perception(x, y, estimator, output_dir, sesstag):
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    # left   # the left side of the subplots of the figure
    # right  # the right side of the subplots of the figure
    # bottom # the bottom of the subplots of the figure
    # top    # the top of the subplots of the figure
    # wspace # the amount of width reserved for blank space between subplots
    # hspace # the amount of height reserved for white space between subplots
    plt.subplots_adjust(left=.085, bottom=.11, right=.975, wspace=.15, top=.8)

    colors = ['tab:blue', 'tab:orange']
    legend_labels = ['Beat', 'Interval']

    for m, modality_y in enumerate(y):
        for c, condition_y in enumerate(modality_y):
            # Linear fit
            a, b = np.polyfit(x, condition_y, deg=1)
            y_est = a * x + b
            # y_err = x.std() * \
            #     np.sqrt(1/len(x) + (x - x.mean())**2 / np.sum((x - x.mean())**2))

            # Plot the linear fit
            ax[m].plot(x, y_est, '-', color=colors[c], linewidth=12,
                       label=legend_labels[c], alpha=.5)
            # ax[0].fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2)
            ax[m].plot(x, condition_y, 'bo', color=colors[c],
                       markersize=16, alpha=.5)
            # Hide the right and top spines
            ax[m].spines['right'].set_visible(False)
            ax[m].spines['top'].set_visible(False)
            # Set x axis
            x_labels = [str(xl) for xl in x]
            ax[m].set_xticks(x, x_labels, fontsize=24)
            # Set limits of y-axis
            y_values = np.linspace(-0.16, 0.16, 9)
            y_labels = [str(int(yl*100)) for yl in y_values]
            ax[m].set_yticks(y_values, y_labels, fontsize=24)
            # Add horizontal dashed line at y = 0.5
            ax[m].axhline(0., linestyle='--', color='grey', linewidth=12,
                          alpha=.5)

        # Add legend
        if m == 0:
            ax[m].set_title('Auditory Perception', weight='bold', pad=-5,
                            fontsize=22)
        else:
            assert m == 1
            ax[m].legend(loc='upper right', frameon=False,
                         prop={'size': 16})
            ax[m].set_title('Visual Perception', weight='bold', pad=-5,
                            fontsize=22)

        # Name of x-axis
        fig.text(.47, .018, 'Standards (ms)', fontsize=24)
        # Name of y-axis
        fig.text(.0175, .35, 'Group PSE (%)', fontsize=24, rotation=90)
        # Legends for horizontal dashed lines
        fig.text(.42, .525, 'No Bias', fontsize=24, color='dimgrey')
        fig.text(.895, .525, 'No Bias', fontsize=24, color='dimgrey')

    # Title
    if estimator == 'mle_cdf':
        suffix = '(Estimator: MLE of Norm CDF)'
    else:
        assert estimator == 'mle_expit'
        suffix = '(Estimator: MLE of Logistic-Sigmoid Function)'
    plt.suptitle(
        'Point of Subjective Equality (PSE) for the Perception Tasks: ' + \
        sessions_dic[sesstag] + '\n\n' + suffix,
        x=.5, y=.97, size=24, linespacing=.75)

    # plt.suptitle(
    #     'Point of Subjective Equality (PSE) for the Perception Tasks',
    #     x=.5, y=.97, size=26, linespacing=.75)

    output_folder = os.path.join(output_dir, 'pse')
    # Create output_folder, if it does not exist
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    # Save figure
    plt.savefig(os.path.join(
        output_folder,
        'pse-vs-standard_' + estimator + '_' + sesstag + '.pdf'))

    plt.close('all')


def dataframe(estim_pse, estim_dl, stand_numbers, output_dir, sesstag,
              estimator='mle_expit'):
    # Shape of pse and dl arrays:
    # (estimators, conditions, modality, subjects, standards)

    # Stack multidimensional numpy array to produce a dataframe
    estim_pse = np.array(estim_pse)
    estim_dl = np.array(estim_dl)
    pse_flatten = np.ravel(estim_pse)
    dl_flatten = np.ravel(estim_dl)

    # ## Standards column
    standards = np.tile(
        stand_numbers,
        estim_pse.shape[3] * estim_pse.shape[2] * estim_pse.shape[1] * \
        estim_pse.shape[0])
    # ## Individual column
    itag = ['sub-%02d' % s for s in SUBJECTS]
    stand_individuals = np.repeat(itag, len(stand_numbers))
    individuals = np.tile(
        stand_individuals,
        estim_pse.shape[2] * estim_pse.shape[1] * estim_pse.shape[0])
    # ## Modality column
    stand_modalities = np.repeat(['audio', 'visual'], len(stand_individuals))
    modalities = np.tile(
        stand_modalities,
        estim_pse.shape[1] * estim_pse.shape[0])
    # ## Conditions column
    crossind_conditions = np.repeat(['beat', 'interval'],
                                    len(stand_modalities))
    conditions = np.tile(crossind_conditions, estim_pse.shape[0])
    # ## Estimator column
    estimators = np.repeat(['mle_cdf', 'mle_expit'],
                           len(crossind_conditions))
    # ## Session
    sessions = np.repeat(sesstag, len(dl_flatten))

    # ## Build tables and dataframes
    table = np.vstack((dl_flatten, pse_flatten, standards, individuals, 
                       modalities, conditions, estimators, sessions)).T

    df = pd.DataFrame(table, columns=['DL', 'PSE', 'Standard', 'Subject',
                                      'Modality', 'Condition', 'Estimator',
                                      'Session'])
    df['DL'] = df['DL'].apply(pd.to_numeric)
    df['PSE'] = df['PSE'].apply(pd.to_numeric)
    df = df[df.Estimator == estimator]

    # Replace outliers by median across subjects
    ht_dl, lt_dl = outliers(dl_flatten)
    df['DL'] = np.where(df['DL'] > ht_dl, np.nan, df['DL'])
    df['DL'] = np.where(df['DL'] < lt_dl, np.nan, df['DL'])

    ht_pse, lt_pse = outliers(pse_flatten)
    df['PSE'] = np.where(df['PSE'] > ht_pse, np.nan, df['PSE'])
    df['PSE'] = np.where(df['PSE'] < lt_pse, np.nan, df['PSE'])

    for index, row in df.iterrows():
        if np.isnan(df.loc[index, 'DL']):
            std_val = df.loc[index, 'Standard']
            mod_val = df.loc[index, 'Modality']
            cond_val = df.loc[index, 'Condition']
            dl = np.nanmedian(df[df.Standard == std_val][
                df.Modality == mod_val][df.Condition == cond_val].DL.values)
            df.loc[index, 'DL'] = dl
        if np.isnan(df.loc[index, 'PSE']):
            std_val = df.loc[index, 'Standard']
            mod_val = df.loc[index, 'Modality']
            cond_val = df.loc[index, 'Condition']
            pse = np.nanmedian(df[df.Standard == std_val][
                df.Modality == mod_val][df.Condition == cond_val].PSE.values)
            df.loc[index, 'PSE'] = pse

    output_folder = os.path.join(output_dir, 'anovas')
    # Create output_folder, if it does not exist
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    # Save dataframe
    outpath = os.path.join(output_folder, 'df_perception_' + sesstag + '.tsv')
    df.to_csv(outpath, index=False, sep='\t')

    return df


def twoway_repanova(df, output_dir, sesstag, alternative='two-sided'):
    # Open dataframe
    if isinstance(df, str):
        df = pd.read_csv(df, sep='\t')

    # Averaged DL across Standards, i.e. grouped by ...
    # ... Condition, Modality and Subject and averaged afterwards
    df = df.drop(['Standard'], axis=1)
    df = df.drop(['Estimator'], axis=1)
    df = df.drop(['Session'], axis=1)
    df = df.groupby(['Condition', 'Modality', 'Subject']).mean().reset_index()

    # Create AnovaRM object
    model = AnovaRM(data=df, depvar='DL', subject='Subject',
                    within=['Modality', 'Condition'])

    # Run the 2-way repeated measures ANOVA
    results = model.fit()

    output_folder = os.path.join(output_dir, 'anovas/twoway')
    # Create output_folder, if it does not exist
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    # Save ANOVA results in a TSV file
    results.anova_table.to_csv(
        os.path.join(output_folder, 'twoway_anova_' + sesstag + '.tsv'),
        sep='\t')

    # Posthoc pairwise tests
    posthoc_results = pg.pairwise_tests(
        data=df, dv='DL', within=['Condition', 'Modality'],
        subject='Subject', alternative=alternative, return_desc=True,
        padjust='holm', effsize='eta-square')

    # Save Posthoc results
    posthoc_results.to_csv(
        os.path.join(output_folder, 'twoway_posthoc_' + sesstag + '.tsv'),
        sep='\t', index=False)

    # Plot
    modalities = np.unique(df.Modality).tolist()
    conditions = np.unique(df.Condition).tolist()

    for m, modality in enumerate(modalities):
        if modality == 'audio':
            fig = plt.figure(figsize=(3.75, 4))

        # Define subplot of bar charts and its position in the fig
        # plt.axes([left, bottom, width, height])
        ax = plt.axes([.175 + m*.425, .15, .39, .775])

        x_labels = [str(cd).capitalize() for cd in conditions]
        x = np.arange(len(x_labels))  # the label locations
        width = .275  # the width of the bars

        dl_beat = df[df.Modality==modality][
            df.Condition=='beat'].DL.values.tolist()

        dl_interval = df[df.Modality==modality][
            df.Condition=='interval'].DL.values.tolist()

        beat = ax.boxplot(dl_beat,
                          bootstrap=100,
                          positions=[.2],
                          widths=width,
                          flierprops={'marker': '', 'markersize': 5},
                          patch_artist=True,
                          medianprops = dict(color="black",linewidth=0.),
                          notch=True,
                          meanline=True,
                          showmeans=True,
                          meanprops = dict(color="tab:brown",linewidth=1.5))
        interval = ax.boxplot(dl_interval,
                              bootstrap=100,
                              positions=[.8],
                              widths=width,
                              flierprops={'marker': '', 'markersize': 5},
                              patch_artist=True,
                              medianprops = dict(color="black",linewidth=0.),
                              notch=True,
                              meanline=True,
                              showmeans=True,
                              meanprops = dict(color="tab:brown",linewidth=1.5))

        colors = [[.0, .66, .47, .5], [.89, .61, .06, .5]]
        for patch1, patch2 in zip(beat['boxes'], interval['boxes']):
            patch1.set_facecolor(colors[m])
            patch2.set_facecolor(colors[m])

        # Set ticks labels in x-axis
        ax.set_xticks([.2, .8], x_labels, fontsize=11)

        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # For the first plot,
        if m == 0:
            # Place legend
            ax.legend([beat["boxes"][0]], ['Auditory'],
                       loc=(.075, .92), frameon=False,
                       prop={'size': 12})
            # # Title of each plot
            # ax.set_title('Auditory Perception', fontweight='semibold',
            #              size=9, y=.95)
            # Set name for y-axis
            ax.set_ylabel('Group DL', fontsize=14)
            # Copy axis object to draw connection patch
            ax0 = ax
            # Draw small vertical line for annotation
            if sesstag in ['allses', 'behavses', 'imgses',
                           'ses-01', 'ses-02', 'ses-03', 'ses-04']:
                plt.vlines(.5, .19, .195, colors='k', linewidths=1.)
            else:
                assert sesstag == 'ses-05'
                plt.vlines(.77, .16, .165, colors='k', linewidths=1.)

        # For the second plot
        else:
            # Place legend
            ax.legend([beat["boxes"][0]], ['Visual'],
                       loc=(.075, .92), frameon=False,
                       prop={'size': 12})
            # ... remove y frame on the left
            ax.spines['left'].set_visible(False)
            # ... remove labels and ticks
            ax.axes.get_yaxis().set_visible(False)
            # # Title of each plot
            # ax.set_title('Visual Perception', fontweight='semibold', size=9,
            #              y=.95)
            # Draw small vertical line for annotation
            if sesstag in ['allses', 'behavses', 'imgses',
                           'ses-01', 'ses-02', 'ses-03', 'ses-04']:
                plt.vlines(.5, .19, .195, colors='k', linewidths=1.)
            else:
                assert sesstag == 'ses-05'
                plt.vlines(.77, .16, .165, colors='k', linewidths=1.)

        # Set limits of ticks in y axis
        plt.ylim([.01, .25])

        # Set name for x-axis
        fig.text(.435, .025, 'Conditions', size=14)

    # Annotation
    if sesstag in ['allses', 'behavses', 'imgses',
                   'ses-01', 'ses-02', 'ses-03']:
        xa = (.5, .195)
        xb = (.5, .195)
        fig.text(.53, .75, '****', size=12)
    elif sesstag == 'ses-04':
        xa = (.5, .195)
        xb = (.5, .195)
        fig.text(.57, .71, '*', size=12)
    else:
        assert sesstag == 'ses-05'
        xa = (.77, .165)
        xb = (.77, .165)
        fig.text(.585, .65, '****', size=12)

    con1 = ConnectionPatch(xyA=xa, xyB=xb,
                           coordsA="data", coordsB="data", axesA=ax0, axesB=ax,
                           color='black', linewidth=1.)
    ax.add_artist(con1)

    # Title
    # plt.suptitle(
    #     'Descriptive Stats of Group DL for 2-way RM-ANOVA',
    #     x=.5, y=.98, size=10, linespacing=.75)

    # Save figure
    plt.savefig(os.path.join(output_folder,
                             'twoway_boxplot_' + sesstag + '.pdf'))

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

# #######################################################################

# TASKS = ['Auditory Perception', 'Visual Perception']

N_TRIALS = 30

# ### For 'All Sessions' ###
# SUBJECTS = BEHAVIMG_RAND_SUBJECTS
# SESSIONS = [1, 2, 3, 4, 5]
# tag = 'allses'

# # ### For 'All Behavioral Sessionss' ###
# SUBJECTS = BEHAVIMG_RAND_SUBJECTS
# SESSIONS = [1, 2, 3]
# tag = 'behavses'

# # ### For 'All Imaging Sessionss' ###
SUBJECTS = BEHAVIMG_RAND_SUBJECTS
SESSIONS = [4, 5]
tag = 'imgses'

# ### For first behav session: 'ses-01' ###
# SUBJECTS = GOOD_SUBJECTS
# SESSIONS = [1]
# tag = 'ses-01'

# ### For second behav session: 'ses-02' ###
# SUBJECTS = GOOD_SUBJECTS
# SESSIONS = [2]
# tag = 'ses-02'

# ### For third behav session: 'ses-03' ###
# SUBJECTS = GOOD_SUBJECTS
# SESSIONS = [3]
# tag = 'ses-03'

# ### For first img session: 'ses-04' ###
# SUBJECTS = IMG_SUBJECTS
# SESSIONS = [4]
# tag = 'ses-04'

# ### For second img session: 'ses-05' ###
# SUBJECTS = IMG_SUBJECTS
# SESSIONS = [5]
# tag = 'ses-05'

sessions_dic = {'allses': 'All Sessions',
                'behavses': 'All Behavioral Sessions',
                'imgses': 'All Imaging Sessions',
                'ses-01': 'Session 1',
                'ses-02': 'Session 2',
                'ses-03': 'Session 3',
                'ses-04': 'Session 4',
                'ses-05': 'Session 5'}

# %%
# ========================= PARAMETERS =================================

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FOLDER = os.path.join(MAIN_DIR, 'perception_results')

# %%
# ============================ RUN =====================================

if __name__ == "__main__":

    if not os.path.exists(RESULTS_FOLDER):
        os.mkdir(RESULTS_FOLDER)

    # # ################### PERCEPTION ###################################

    estim_pse = []
    estim_dl = []
    for estimator in ['mle_cdf', 'mle_expit']:
        cond_pse = []
        cond_dl = []
        cond_ce = []
        cond_gpse = []
        for cond in ['beat', 'interval']:

            # Compute individual psychometric functions
            rfone_audio, rftwo_audio, rfone_visual, rftwo_visual, stand, \
                comp, ipse_audio, idl_audio, ipse_visual, idl_visual = \
                individual_perception(SUBJECTS, MAIN_DIR, RESULTS_FOLDER,
                                      cond, N_TRIALS, SESSIONS, tag,
                                      estimator=estimator)

            # Compute group psychometric functions
            gpse, _ = group_perception(rfone_audio, rftwo_audio, rfone_visual,
                                       rftwo_visual, stand, comp, cond,
                                       RESULTS_FOLDER, tag, estimator=estimator)

            # Start concatenating and appending
            ipse = np.concatenate(([ipse_audio], [ipse_visual]),
                                  axis = 0).tolist()
            idl = np.concatenate(([idl_audio], [idl_visual]), axis = 0).tolist()

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

        estim_pse.append(cond_pse)
        estim_dl.append(cond_dl)

        # Plot PSE
        mod_gpse = np.swapaxes(cond_gpse, 0, 1)
        plotfit_perception(stand, mod_gpse, estimator, RESULTS_FOLDER, tag)

        # Compute ANOVAS and plot DL
        if estimator == 'mle_cdf':
            continue
        else:
            db = dataframe(estim_pse, estim_dl, stand, RESULTS_FOLDER, tag)
            twoway_repanova(db, RESULTS_FOLDER, tag)
