"""
Analysis of behavioral data for the Perception Tasks of the Music-SDTB project

author: Ana Luisa Pinho
e-mail: agrilopi@uwo.ca

Created: February 2023
Last update: June 2023

Compatibility: Python 3.10.4
"""

import sys
import os

import warnings

import numpy as np
import pandas as pd

import pingouin as pg
import seaborn as sns

from scipy import stats, optimize, special
from matplotlib import pyplot as plt
from statannotations.Annotator import Annotator
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# setting path
sys.path.append('../')
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
        subjects, this_dir, output_dir, sesstype, condition, n_trials,
        estimator='mle_expit',
        tasks = ['Auditory Perception', 'Visual Perception']):

    all_rf1_audio = []
    all_rf2_audio = []
    all_rf1_visual = []
    all_rf2_visual = []
    all_pse_audio = []
    all_dl_audio = []
    all_pse_visual = []
    all_dl_visual = []
    for s, subject in enumerate(subjects):
        for t, task in enumerate(tasks):
            if task not in ['Auditory Perception', 'Visual Perception']:
                raise NameError('Task not valid!')

            data = parse_logfile(this_dir, subject, sesstype, task, n_trials)
            trials = perception_data(data)
            beat_trials, interval_trials, _ = filter_trialtype(trials,
                                                               'perception')

            if condition == 'beat':
            # Calculate frequencies of comparisons per standard
                standards, comparisons, n1_beat, n2_beat, _, _ = \
                    perception_frequencies(beat_trials, interval_trials)

                rf1 = [[n1/8 for n1 in n1b] for n1b in n1_beat]
                rf2 = [[n2/8 for n2 in n2b] for n2b in n2_beat]
            else:
                assert condition == 'interval'
                standards, comparisons, _, _, n1_interval, n2_interval = \
                    perception_frequencies(beat_trials, interval_trials)

                rf1 = [[n1/8 for n1 in n1i] for n1i in n1_interval]
                rf2 = [[n2/8 for n2 in n2i] for n2i in n2_interval]

            # Aggregate data
            if task == 'Auditory Perception':
                all_rf1_audio.append(rf1)
                all_rf2_audio.append(rf2)
            else:
                assert task == 'Visual Perception'
                all_rf1_visual.append(rf1)
                all_rf2_visual.append(rf2)

            # ################## Plotting ###############################
            colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
                      'tab:purple']
            if s == 0 and t == 0:
                fig = plt.figure(figsize=(16, 200))

            # Define subplot of bar charts and its position in the fig
            # plt.axes([left, bottom, width, height])
            ax = plt.axes([.1 + t*.46, .9685 - s*.02, .428, .0125])

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
                if task == 'Auditory Perception':
                    std_pse_audio.append(pse)
                    std_dl_audio.append(dl)
                else:
                    assert task == 'Visual Perception'
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
                if t == 0:
                    ax.legend(loc='lower right', frameon=False,
                              prop={'size': 6})
                    ax.set_title('Auditory Perception', weight='bold', pad=60,
                                 fontsize=16)
                else:
                    assert t == 1
                    ax.set_title('Visual Perception', weight='bold', pad=60,
                                 fontsize=16)

            # Name of x-axis
            fig.text(.495, .0025, 'Comparisons (%)', fontsize=14)
            # Name of y-axis
            fig.text(.062, .46, 'Relative Frequency of "longer" responses',
                     fontsize=14, rotation=90)

            # Append
            if task == 'Auditory Perception':
                all_pse_audio.append(std_pse_audio)
                all_dl_audio.append(std_dl_audio)
            else:
                assert task == 'Visual Perception'
                all_pse_visual.append(std_pse_visual)
                all_dl_visual.append(std_dl_visual)

        fig.text(.03, .9765 - s*.02, 'Subject %d' % subject, ha='center',
                 fontsize=10, weight='bold')

    # Title
    if estimator == 'mle_cdf':
        suffix = '(Estimator: MLE of Norm CDF)'
    else:
        assert estimator == 'mle_expit'
        suffix = '(Estimator: MLE of Logistic-Sigmoid Function)'
    plt.suptitle('Individual Relative Frequencies for the ' +
                 condition.capitalize() +
                 ' condition of the Perception Tasks ' + suffix, x=.5, y=.9975,
                 size=18, linespacing=.75)
    # plt.show()

    # Save figure
    plt.savefig(os.path.join(
        this_dir, output_dir,
        'individual_perception_' + condition + '_' + estimator + '.pdf'))

    plt.close('all')

    return (all_rf1_audio, all_rf2_audio, all_rf1_visual, all_rf2_visual,
            standards, comparisons, all_pse_audio, all_dl_audio,
            all_pse_visual, all_dl_visual)


def group_perception(all_rf1_audio, all_rf2_audio,
                     all_rf1_visual, all_rf2_visual,
                     standards, comparisons, condition, this_dir, output_dir,
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
        condition.capitalize() + ' condition of the Perception Tasks ' +
        suffix, x=.5, y=.97, size=16, linespacing=.75)

    # Save figure
    plt.savefig(os.path.join(this_dir, output_dir,
        'group_perception_' + condition + '_' + estimator + '.pdf'))

    plt.close('all')

    return group_pse, group_dl


def plotfit_perception(x, y, estimator, this_dir, output_dir):
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    # left   # the left side of the subplots of the figure
    # right  # the right side of the subplots of the figure
    # bottom # the bottom of the subplots of the figure
    # top    # the top of the subplots of the figure
    # wspace # the amount of width reserved for blank space between subplots
    # hspace # the amount of height reserved for white space between subplots
    plt.subplots_adjust(left=.085, bottom=.11, right=.975, wspace=.15)

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
                            fontsize=24)
        else:
            assert m == 1
            ax[m].legend(loc='upper right', frameon=False,
                         prop={'size': 16})
            ax[m].set_title('Visual Perception', weight='bold', pad=-5,
                            fontsize=24)

        # Name of x-axis
        fig.text(.47, .018, 'Standards (ms)', fontsize=24)
        # Name of y-axis
        fig.text(.0175, .35, 'Group PSE (%)', fontsize=24, rotation=90)
        # Legends for horizontal dashed lines
        fig.text(.42, .525, 'No Bias', fontsize=24, color='dimgrey')
        fig.text(.895, .525, 'No Bias', fontsize=24, color='dimgrey')

    # Title
    # if estimator == 'mle_cdf':
    #     suffix = '(Estimator: MLE of Norm CDF)'
    # else:
    #     assert estimator == 'mle_expit'
    #     suffix = '(Estimator: MLE of Logistic-Sigmoid Function)'
    # plt.suptitle(
    #     'Point of Subjective Equality (PSE) for the Perception Tasks' +
    #     '\n\n' + suffix, x=.5, y=.97, size=26, linespacing=.75)

    plt.suptitle(
        'Point of Subjective Equality (PSE) for the Perception Tasks',
        x=.5, y=.97, size=26, linespacing=.75)

    # Save figure
    plt.savefig(os.path.join(this_dir, output_dir,
                             'pse-vs-standard_' + estimator + '.pdf'))

    plt.close('all')


def dataframe(estim_pse, estim_dl, stand_numbers, this_dir, output_dir,
              estimator='mle_expit'):
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

    # ## Build tables and dataframes
    table = np.vstack((dl_flatten, standards, modalities, individuals,
                       conditions, estimators)).T

    df = pd.DataFrame(table, columns=['DL', 'Standard', 'Modality',
                                      'Subject', 'Condition', 'Estimator'])
    df['DL'] = df['DL'].apply(pd.to_numeric)
    df = df[df.Estimator == estimator]

    # Replace outliers by median
    ht, lt = outliers(dl_flatten)
    df['DL'] = np.where(df['DL'] > ht, np.nan, df['DL'])
    df['DL'] = np.where(df['DL'] < lt, np.nan, df['DL'])
    for index, row in df.iterrows():
        if df.loc[index, 'DL'] == np.nan:
            std_val = df.loc[index, 'Standard']
            mod_val = df.loc[index, 'Modality']
            cond_val = df.loc[index, 'Condition']
            dl = np.median(df[df.Standard == std_val][
                df.Modality == mod_val][df.Condition == cond_val].DL.values)
            df.loc[index, 'DL'] = dl

    return df


def twoway_repanova(df, stand_numbers, this_dir, output_dir):
    for s, st in enumerate(stand_numbers):
        # Compute a two-way repeated-measures ANOVA
        aov = pg.rm_anova(
            data=df[df.Standard == str(st)],
            dv='DL',
            within=['Modality', 'Condition'],
            subject='Subject',
            detailed=True)
        aov.to_csv(
            os.path.join(MAIN_DIR, PLOTS_FOLDER, 'anovas/twoway',
                         'anova_DL_' + str(st) + '.csv'), sep='\t')

        print('Normality for Modality' + str(st) + ': ',
              pg.normality(df[df.Standard == str(st)], group='Modality',
                           dv='DL', method='normaltest'))

        print('Normality for Condition' + str(st) + ': ',
              pg.normality(df[df.Standard == str(st)], group='Condition',
                           dv='DL', method='normaltest'))

        spher, W, chisq, dof, pval = pg.sphericity(
            df[df.Standard == str(st)], dv='DL', subject='Subject',
            within='Modality')
        print('Sphericity for Modality: ', spher, round(W, 3),
              round(chisq, 3), dof, round(pval, 3))

        spher, W, chisq, dof, pval = pg.sphericity(
            df[df.Standard == str(st)], dv='DL', subject='Subject',
            within='Condition')
        print('Sphericity for Condition: ', spher, round(W, 3),
              round(chisq, 3), dof, round(pval, 3))

        # Holm-corrected pairwise T-tests
        pairwise_tt = pg.pairwise_tests(
            data=df[df.Standard == str(st)],
            dv='DL',
            within=['Modality', 'Condition'],
            subject='Subject',
            alternative='two-sided',
            return_desc=True, padjust='holm', interaction=True)
        pairwise_tt.to_csv(os.path.join(
            MAIN_DIR, PLOTS_FOLDER, 'anovas/twoway',
            'pairwise_ttest_DL_' + str(st) + '.csv'), sep='\t')

        if s == 0:
            fig = plt.figure(figsize=(25, 5))

        # Define subplot of bar charts and its position in the fig
        # plt.axes([left, bottom, width, height])
        ax = plt.axes([.0345 + s*.198, .17, .165, .7])

        sns.set(style="ticks", rc={"lines.linewidth": 3.})

        if s == 3:
            ax = sns.pointplot(
                data=df[df.Standard == str(st)],
                x='Condition',
                y='DL',
                hue='Modality',
                dodge=True,
                capsize=.1,
                ci='sd',
                linestyles=["-", "--"],
                palette=['lime', 'indigo'])
        else:
            ax = sns.pointplot(
                data=df[df.Standard == str(st)],
                x='Condition',
                y='DL',
                hue='Modality',
                dodge=True,
                capsize=.1,
                ci='sd',
                palette=['lime', 'indigo'])

        # Set opacity
        plt.setp(ax.collections, alpha=.5) #for the markers
        plt.setp(ax.lines, alpha=.5)       #for the lines

        sns.despine(fig=fig, top=True, right=True, left=False, bottom=False)
        plt.ylim(-.3, .85)
        _ = plt.title('Standard = ' + str(st) + ' ms', x=.5, y=1.05,
                      fontsize=20)
        ax.legend(frameon=False, loc = 'lower right', fontsize=16)
        ax.set_xlabel("Condition", fontsize = 18)
        ax.set_ylabel("DL", fontsize = 18)
        ax.yaxis.labelpad = -4
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        # Annotate
        if s == 0:
            ymin = .075
            ymax = .2
            fig.text(.115, .48, '*', fontsize=18)
            fig.text(.0725, .75, 'Error bar: SD', fontsize=18)
        elif s == 1:
            ymin = .084
            ymax = .15
            fig.text(.31, .45, '**', fontsize=18)
        elif s == 2:
            ymin = .08
            ymax = .13
            fig.text(.508, .44, '**', fontsize=18)
        elif s == 3:
            ymin - .08
            ymax = .13
            fig.text(.704, .45, '***', fontsize=18)
            pair = [('audio', 'visual')]
            annotator = Annotator(
                ax, pair, data=df[df.Standard == str(st)],
                x='Modality', y='DL')
            annotator.configure(
                test=None, text_format="star",
                test_short_name="pttest", fontsize=16.)
            annotator.set_pvalues([pairwise_tt['p-unc'][1]])
            annotator.annotate()
            # Add second legend
            fig.text(.64, .81, 'Dashed: significant interaction',
                     fontsize=14)
            fig.text(.64, .76, 'Continuous: non-significant interaction',
                     fontsize=14)
        else:
            assert s == 4
            ymin = .093
            fig.text(.903, .433, '**', fontsize=18)
            ymax = .13

        plt.vlines(.5, ymin, ymax, color='black')
        # Save figure
        if s == len(stand_numbers) - 1:
            plt.savefig(os.path.join(this_dir, output_dir, MAIN_DIR,
                                     PLOTS_FOLDER, 'anovas/twoway',
                                     'anovaplot_DL.png'))
    plt.close('all')


def threeway_repanova(df, this_dir, output_dir):
    # Create AnovaRM object
    model = AnovaRM(data=df, depvar='DL', subject='Subject',
                    within=['Modality', 'Condition', 'Standard'])

    # Run the 3-way repeated measures ANOVA
    results = model.fit()

    # Save ANOVA results in a TSV file
    results.anova_table.to_csv(
        os.path.join(this_dir, output_dir,
                     'anovas/threeway/threeway_anova_results.tsv'),
        sep='\t')

    # Perform pairwise Tukey HSD tests
    posthoc_modality = pairwise_tukeyhsd(df['DL'], df['Modality'],
                                         alpha=0.05)
    posthoc_condition = pairwise_tukeyhsd(df['DL'], df['Condition'],
                                          alpha=0.05)
    posthoc_standard = pairwise_tukeyhsd(df['DL'], df['Standard'],
                                         alpha=0.05)

    # Save posthoc results in a TSV file
    output_folder = os.path.join(this_dir, output_dir, 'anovas/threeway')

    with open(os.path.join(output_folder, 'posthoc_modality.tsv'), 'w') as fm:
        fm.write(posthoc_modality.summary().as_csv(sep='\t'))

    with open(os.path.join(output_folder, 'posthoc_condition.tsv'), 'w') as fc:
        fc.write(posthoc_condition.summary().as_csv(sep='\t'))

    with open(os.path.join(output_folder, 'posthoc_standard.tsv'), 'w') as fs:
        fs.write(posthoc_standard.summary().as_csv(sep='\t'))

    # Plot
    modalities = np.unique(df.Modality).tolist()
    conditions = np.unique(df.Condition).tolist()
    standards = np.unique(df.Standard).tolist()

    for m, modality in enumerate(modalities):
        if modality == 'audio':
            fig = plt.figure(figsize=(8, 4))

        # Define subplot of bar charts and its position in the fig
        # plt.axes([left, bottom, width, height])
        ax = plt.axes([.075 + m*.45, .15, .43, .75])

        x_labels = [str(st) for st in standards]
        x = np.arange(len(x_labels))  # the label locations
        width = 0.35  # the width of the bars

        dl_beat = [df[df.Modality==modality][df.Condition=='beat'][
            df.Standard==st].DL.values.tolist() for st in standards]

        dl_interval = [df[df.Modality==modality][df.Condition=='interval'][
            df.Standard==st].DL.values.tolist() for st in standards]

        beat = ax.boxplot(dl_beat,
                          bootstrap=100,
                          positions=np.arange(len(x))*2. - width,
                          widths=0.6,
                          flierprops={'marker': '', 'markersize': 5},
                          patch_artist=True)
        interval = ax.boxplot(dl_interval,
                              bootstrap=100,
                              positions=np.arange(len(x))*2. + width,
                              widths=0.6,
                              flierprops={'marker': '', 'markersize': 5},
                              patch_artist=True)

        # Fill boxes with colors
        colors = ['b', 'y']
        for patch1, patch2 in zip(beat['boxes'], interval['boxes']):
            patch1.set_facecolor(colors[0])
            patch2.set_facecolor(colors[1])

        # Set ticks labels in x-axis
        ax.set_xticks(x*2., x_labels)

        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # For the first plot,
        if m == 0:
            # Place legend
            ax.legend([beat["boxes"][0], interval["boxes"][0]],
                      ['Beat', 'Interval'],
                      loc='upper right', frameon=False,
                      prop={'size': 12})
            # Title of each plot
            ax.set_title('Auditory Conditions', fontweight='semibold',
                         size=10, y=.95)
            # Set name for y-axis
            ax.set_ylabel('Group DL')
        # For the second plot
        else:
            # ... remove y frame on the left
            ax.spines['left'].set_visible(False)
            # ... remove labels and ticks
            ax.axes.get_yaxis().set_visible(False)
            # Title of each plot
            ax.set_title('Visual Conditions', fontweight='semibold', size=10,
                         y=.95)

        # Set limits of ticks in y axis
        plt.ylim([-.05, .5])

        # Set name for x-axis
        fig.text(.45, .025, 'Standards (ms)', size=12)

    # Title
    plt.suptitle(
        'Descriptive Stats of Group DL for 3-way Repeated Measures ANOVA',
        x=.5, y=.98, size=12, linespacing=.75)

    # Save figure
    plt.savefig(os.path.join(output_folder, 'threeway_boxplot.pdf'))


# %%
# =========================== INPUTS ===================================

SUBJECTS = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
            22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39,
            40, 41, 42, 43, 44, 45, 46]
# SUBJECTS = [3, 7, 8, 10]

# TASKS = ['Auditory Perception', 'Visual Perception']

SESSTYPES = ['behavioral_session', 'imaging_session']
# SESSTYPES = ['imaging_session']

PLOTS_FOLDER = 'perception_results'

N_TRIALS = 30

# %%
# ========================= PARAMETERS =================================

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))

# %%
# ============================ RUN =====================================

if __name__ == "__main__":

    if not os.path.exists(os.path.join(MAIN_DIR, PLOTS_FOLDER)):
        os.makedirs(os.path.join(MAIN_DIR, PLOTS_FOLDER, 'anovas/twoway'))
        os.makedirs(os.path.join(MAIN_DIR, PLOTS_FOLDER, 'anovas/threeway'))

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
                    individual_perception(SUBJECTS, MAIN_DIR, PLOTS_FOLDER,
                                          SESSTYPES, cond, N_TRIALS,
                                          estimator=estimator)

            # Compute group psychometric functions
            gpse, _ = group_perception(rfone_audio, rftwo_audio, rfone_visual,
                                       rftwo_visual, stand, comp, cond,
                                       MAIN_DIR, PLOTS_FOLDER,
                                       estimator=estimator)

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

        # Plot PSE as a function of the standard
        mod_gpse = np.swapaxes(cond_gpse, 0, 1)
        plotfit_perception(stand, mod_gpse, estimator, MAIN_DIR, PLOTS_FOLDER)

        # Compute ANOVAS
        if estimator == 'mle_cdf':
            continue
        else:
            db = dataframe(estim_pse, estim_dl, stand, MAIN_DIR, PLOTS_FOLDER)
            # twoway_repanova(db, stand, MAIN_DIR, PLOTS_FOLDER)
            threeway_repanova(db, MAIN_DIR, PLOTS_FOLDER)
