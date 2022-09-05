"""
Script to extract behavioral data from logfiles for the Music-SDTB project

author: Ana Luisa Pinho
e-mail: agrilopi@uwo.ca

Created: August 2022
Last update: September 2022

Compatibility: Python 3.7.11

"""
import os
import glob
import csv

import numpy as np
import pandas as pd
from scipy import stats, optimize

from matplotlib import pyplot as plt
from matplotlib import patches as mpatches

import seaborn as sns
from statannotations.Annotator import Annotator


# %%
# =========================== UTILS ====================================


def adjacent_values(vals, q1, q3):
    vals.sort()
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)

    return lower_adjacent_value, upper_adjacent_value


def customize_vplot(datum, ax, pos):
    q1, median, q3 = np.percentile(datum, [25, 50, 75])
    whiskers = np.array([adjacent_values(datum, q1, q3)])
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]
    ax.scatter(pos, median, marker='o', color='white', s=6, zorder=3)
    ax.vlines(pos, q1, q3, color='k', linestyle='-', lw=5)
    ax.vlines(pos, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)


def set_axis_style(ax, labels):
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')


def change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)


# %%
# ======================== MAIN FUNCTIONS ==============================


def parse_logfile(parent_dir, subject_no, sesstype, n_sess, task_name,
                  ttl=False, concatenate=True):

    sessions = ['sess-%02d' %s for s in np.arange(1, n_sess + 1)]
    allsessions = []
    for session in sessions:
        logpath = os.path.join(parent_dir, 'sub-%02d' % subject_no, session)
        logfiles = glob.glob(os.path.join(logpath, '*.xpd'))
        logfiles.sort()
        inputs_lists = [[line for line in csv.reader(open(logfile),
                                                     delimiter=',')]
                        for logfile in logfiles]
        # Pick log files of selected task
        allruns = []
        for i, inputs_list in enumerate(inputs_lists, 1):
            ttag = task_name + ' - ' + sesstype
            if ttag in inputs_list[8][0][9:]:
                liste = inputs_list
                # Extract trial information from log file
                for r, row in enumerate(liste):
                    if row[0] == str(subject_no):
                        break
                    else:
                        continue
                if not ttl:
                    trials_info = liste[r+1:]
                    trials_info = [line
                                   for line in trials_info if line[2] != 2]
                else:
                    trials_info = liste[r:]
                if concatenate:
                    allruns.extend(trials_info)
                else:
                    allruns.append(trials_info)
                if i == len(inputs_lists):
                    break

            if i == len(inputs_lists) and not allruns:
                raise NameError('Log file for selected task does not exist!')

        if concatenate:
            allsessions.extend(allruns)
        else:
            allsessions.append(allruns)

    return allsessions


def production_data(data):
    trials = []
    for dt, datum in enumerate(data):
        if datum[5] == 'interval_1':
            condition = datum[4]
            theoretical_isi1 = int(datum[8])
            real_isi1 = int(datum[9])
            if data[dt+8][5] == 'feedback' and data[dt+8][11] == 'o':
                rt = int(data[dt+7][7]) + int(data[dt+8][10])
            elif data[dt+8][5] == 'feedback' and data[dt+8][10] == 'None':
                rt = np.nan
            else:
                raise ValueError('No feedback entry!')
            trials.append([condition, theoretical_isi1, real_isi1, rt])

    return trials


def perception_data(data):
    trials = []
    for dt, datum in enumerate(data):
        if datum[5] == 'interval_1':
            condition = datum[4]
            theoretical_isi1 = datum[8]
            theoretical_isi5 = data[dt+8][8]
            if data[dt+10][5] == 'feedback' and \
               data[dt+10][11] in ['o', 'p']:
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


def ntfd_data(data):
    trials = []
    for dt, datum in enumerate(data):
        if datum[5] == 'feedback':
            condition = datum[4]
            theoretical_isi1 = int(data[dt-2][8])
            if datum[11] in ['o', 'p']:
                rt = int(data[dt-1][7]) + int(datum[10])
            elif datum[10] == 'None':
                rt = np.nan
            else:
                raise ValueError('No feedback entry!')
            trials.append([condition, theoretical_isi1, rt])

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
             if bt[0] == standard and bt[2] == 'p']
        diff_standard_beat_longer = \
            [round((bt[1] - bt[0]) / bt[0], 2)
             for bt in beat_trials
             if bt[0] == standard and bt[2] == 'o']
        diff_standard_interval_shorter = \
            [round((it[1] - it[0]) / it[0], 2)
             for it in interval_trials
             if it[0] == standard and it[2] == 'p']
        diff_standard_interval_longer = \
            [round((it[1] - it[0]) / it[0], 2)
             for it in interval_trials
             if it[0] == standard and it[2] == 'o']
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

    return standards, comparisons, n1_beat, n2_beat, n1_interval, n2_interval


def log_lik(par_vec, y, n2, n1):
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
    # Logarithm of zero = -Inf
    return(-sum(np.multiply(n2, np.log(lik[np.nonzero(lik)])))
           -sum(np.multiply(n1, np.log(1-lik[np.nonzero(lik)]))))


def individual_production_isi_sync(
        subjects, this_dir, sesstype, n_sess, sync_type, flatten=True,
        tasks = ['Auditory Production', 'Visual Production']):

    allsub_beat_audio = []
    allsub_interval_audio = []
    allsub_beat_visual = []
    allsub_interval_visual = []
    for s, subject in enumerate(subjects):
        for t, task in enumerate(tasks):
            if task not in ['Auditory Production', 'Visual Production']:
                raise NameError('Task not valid!')

            data = parse_logfile(this_dir, subject, sesstype, n_sess, task)
            trials = production_data(data)
            beat_trials, interval_trials, _ = filter_trialtype(trials,
                                                               'production')

            # ############# Asynchronies per ISI #######################
            isi1s = np.unique(np.array(beat_trials)[:, 0]).astype('int')

            ss_isi_beat = []
            as_isi_beat = []
            for i in isi1s:
                ss_beat = []
                as_beat = []
                for beat_trial in beat_trials:
                    if beat_trial[0] == i:
                        if ~np.any(np.isnan(beat_trial)):
                            ssb = round((beat_trial[2] - beat_trial[1]) / \
                                        beat_trial[1], 2)
                            asb = abs(ssb)
                        else:
                            ssb = np.nan
                            asb = np.nan
                        ss_beat.append(ssb)
                        as_beat.append(asb)
                # Replace missing values (nan's) by median of the sample
                if np.any(np.isnan(ss_beat)):
                    miss_sbval = np.nanmedian(ss_beat)
                    ss_beat = np.where(np.isnan(ss_beat), miss_sbval, ss_beat)
                if np.any(np.isnan(as_beat)):
                    miss_abval = np.nanmedian(as_beat)
                    as_beat = np.where(np.isnan(as_beat), miss_abval, as_beat)
                # Append isi array
                ss_isi_beat.append(ss_beat)
                as_isi_beat.append(as_beat)

            ss_isi_interval = []
            as_isi_interval = []
            for i in isi1s:
                ss_interval = []
                as_interval = []
                for interval_trial in interval_trials:
                    if interval_trial[0] == i:
                        if ~np.any(np.isnan(interval_trial)):
                            ssi = round((interval_trial[2] - \
                                         interval_trial[1]) / \
                                        interval_trial[1], 2)
                            asi = abs(ssi)
                        else:
                            ssi = np.nan
                            asi = np.nan
                        ss_interval.append(ssi)
                        as_interval.append(asi)
                # Replace missing values (nan's) by median of the isi sample
                if np.any(np.isnan(ss_interval)):
                    miss_sival = np.nanmedian(ss_interval)
                    ss_interval = np.where(np.isnan(ss_interval), miss_sival,
                                           ss_interval)
                if np.any(np.isnan(as_interval)):
                    miss_aival = np.nanmedian(as_interval)
                    as_interval = np.where(np.isnan(as_interval), miss_aival,
                                           as_interval)
                # Append isi array
                ss_isi_interval.append(ss_interval)
                as_isi_interval.append(as_interval)

            # ################## Plotting ###############################
            if s == 0 and t == 0:
                fig = plt.figure(figsize=(8, 40))

            # Define subplot of bar charts and its position in the fig
            # plt.axes([left, bottom, width, height])
            ax = plt.axes([.235 + t*.42, .9175 - s*.0525, .3, .0425])

            x_labels = [str(k) for k in isi1s]
            x = np.arange(len(x_labels))  # the label locations
            width = 0.35  # the width of the bars

            # Transform in Symlog
            logbeat = []
            loginterval = []
            shift = 2
            if sync_type == 'signed':
                for lsbeat in ss_isi_beat:
                    logv = np.abs(lsbeat)*(10.**shift)
                    logv[np.where(logv<1.)] = 1.
                    logv = np.sign(lsbeat)*np.log10(logv)
                    logbeat.append(logv)
                for lsint in ss_isi_interval:
                    logv = np.abs(lsint)*(10.**shift)
                    logv[np.where(logv<1.)] = 1.
                    logv = np.sign(lsint)*np.log10(logv)
                    loginterval.append(logv)
            else:
                assert sync_type == 'absolute'
                for lsbeat in as_isi_beat:
                    logv = np.abs(lsbeat)*(10.**shift)
                    logv[np.where(logv<1.)] = 1.
                    logv = np.sign(lsbeat)*np.log10(logv)
                    logbeat.append(logv)
                for lsint in as_isi_interval:
                    logv = np.abs(lsint)*(10.**shift)
                    logv[np.where(logv<1.)] = 1.
                    logv = np.sign(lsint)*np.log10(logv)
                    loginterval.append(logv)


            beat = ax.boxplot(logbeat,
                              bootstrap=100,
                              positions=np.arange(len(x))*2. - width,
                              widths=0.6,
                              flierprops={'marker': '+', 'markersize': 5},
                              patch_artist=True)
            interval = ax.boxplot(loginterval,
                                  bootstrap=100,
                                  positions=np.arange(len(x))*2. + width,
                                  widths=0.6,
                                  flierprops={'marker': '+', 'markersize': 5},
                                  patch_artist=True)
            # Overplot the mean, with horizontal alignment
            # in the center of each box
            for j in np.arange(len(x)):
                medbeat = beat['medians'][j]
                medinterval = interval['medians'][j]
                ax.plot(np.average(medbeat.get_xdata()),
                        np.average(ss_isi_beat[j]),
                        color='w', marker='*', markeredgecolor='k')
                ax.plot(np.average(medinterval.get_xdata()),
                        np.average(ss_isi_interval[j]),
                        color='w', marker='*', markeredgecolor='k')

            # Fill boxes with colors
            colors1 = ['tab:blue', 'lightblue']
            colors2 = ['purple', 'thistle']
            for patch1, patch2 in zip(beat['boxes'], interval['boxes']):
                if sync_type == 'signed':
                    patch1.set_facecolor(colors1[0])
                    patch2.set_facecolor(colors1[1])
                else:
                    assert sync_type == 'absolute'
                    patch1.set_facecolor(colors2[0])
                    patch2.set_facecolor(colors2[1])

            # x-label at the bottom
            if s == len(subjects) - 1:
                fig.text(.5, .01, ' ISIs (ms)', size=18)

            # x-tick labels with the standards
            ax.set_xticks(x*2., x_labels)

            if sync_type == 'signed':
                plt.ylim([-3., 3.])
                if (t % 2) == 0:
                    ax.set_ylabel('SymLog10(Asynchrony)')
            else:
                assert sync_type == 'absolute'
                plt.ylim([-.3, 3.])
                if (t % 2) == 0:
                    ax.set_ylabel('Log10(Asynchrony)')

            if s == 0:
                ax.set_title(task, pad=30, weight='bold')
                if t == 0:
                    ax.legend(frameon=False, loc = 'best',
                              prop={'size': 8})
                    ax.legend([beat["boxes"][0], interval["boxes"][0]],
                              ['Beat', 'Interval'],
                              loc='upper right', prop={'size': 8})
                    fig.text(.26, 0.956, '*', color='white',
                             backgroundcolor='silver', weight='roman',
                             size='medium')
                    fig.text(.275, 0.956, ' Mean', color='black',
                             weight='roman', size='x-small')

            # Hide the right and top spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            # Aggregate all data
            if task == 'Auditory Production' and sync_type == 'signed':
                allsub_beat_audio.append(ss_isi_beat)
                allsub_interval_audio.append(ss_isi_interval)
            elif task == 'Visual Production' and sync_type == 'signed':
                allsub_beat_visual.append(ss_isi_beat)
                allsub_interval_visual.append(ss_isi_interval)
            elif task == 'Auditory Production' and sync_type == 'absolute':
                allsub_beat_audio.append(as_isi_beat)
                allsub_interval_audio.append(as_isi_interval)
            else:
                assert task == 'Visual Production' and sync_type == 'absolute'
                allsub_beat_visual.append(as_isi_beat)
                allsub_interval_visual.append(as_isi_interval)

        fig.text(.07, .94 - s * .0525, 'Subject %d' % subject, ha='center',
                 fontsize=12, weight='bold')

    # Title
    if sync_type == 'signed':
        plt.suptitle(
            'Individual Signed Asynchrony for the Production tasks',
            x=.5, y=.99, size=14, linespacing=.75)
    else:
        assert sync_type == 'absolute'
        plt.suptitle(
            'Individual Absolute Asynchrony for the Production tasks',
            x=.5, y=.99, size=14, linespacing=.75)

    # plt.show()
    # Save figure
    plt.savefig(os.path.join(
        this_dir, 'production_individual_isi_' + sync_type + '_asynch.pdf'))

    # Flatten the data arrays
    if flatten:
        allsub_beat_audio = np.ravel(allsub_beat_audio)
        allsub_interval_audio = np.ravel(allsub_interval_audio)
        allsub_beat_visual = np.ravel(allsub_beat_visual)
        allsub_interval_visual = np.ravel(allsub_interval_visual)

    return (allsub_beat_audio, allsub_interval_audio, allsub_beat_visual,
            allsub_interval_visual, isi1s)


def individual_production_isi_rts(
        subjects, this_dir, sesstype, n_sess, flatten=True,
        tasks = ['Auditory Production', 'Visual Production']):

    allsub_beat_audio = []
    allsub_interval_audio = []
    allsub_beat_visual = []
    allsub_interval_visual = []
    for s, subject in enumerate(subjects):
        for t, task in enumerate(tasks):
            if task not in ['Auditory Production', 'Visual Production']:
                raise NameError('Task not valid!')

            data = parse_logfile(this_dir, subject, sesstype, n_sess, task)
            trials = production_data(data)
            beat_trials, interval_trials, _ = filter_trialtype(trials,
                                                               'production')

            # Filter necessary data
            beat_trials = [np.delete(trial, 1).tolist()
                           for trial in beat_trials]
            interval_trials = [np.delete(trial, 1).tolist()
                               for trial in interval_trials]

            # ############## Extract RT's per ISI ###################### 
            isi1s = np.unique(np.array(beat_trials)[:, 0]).astype('int')

            rt_isi1_grouped_beat = []
            for i in isi1s:
                rts_beat = []
                for beat_trial in beat_trials:
                    if beat_trial[0] == i:
                        if ~np.any(np.isnan(beat_trial)):
                            rts_beat.append(beat_trial[1])
                        else:
                            rts_beat.append(np.nan)
                # Replace missing values (nan's) by median of the sample
                if np.any(np.isnan(rts_beat)):
                    miss_bval = np.nanmedian(rts_beat)
                    rts_beat = np.where(np.isnan(rts_beat), miss_bval,
                                        rts_beat)
                # Append isi array
                rt_isi1_grouped_beat.append(rts_beat)

            rt_isi1_grouped_interval = []
            for j in isi1s:
                rts_interval = []
                for interval_trial in interval_trials:
                    if interval_trial[0] == j:
                        if ~np.any(np.isnan(interval_trial)):
                            rts_interval.append(interval_trial[1])
                        else:
                            rts_interval.append(np.nan)
                # Replace missing values (nan's) by median of the isi sample
                if np.any(np.isnan(rts_interval)):
                    miss_ival = np.nanmedian(rts_interval)
                    rts_interval = np.where(np.isnan(rts_interval), miss_ival,
                                            rts_interval)
                # Append isi array
                rt_isi1_grouped_interval.append(rts_interval)

            # ################## Plotting ###############################
            if s == 0 and t == 0:
                fig = plt.figure(figsize=(8, 40))

            # Define subplot of bar charts and its position in the fig
            # plt.axes([left, bottom, width, height])
            ax = plt.axes([.235 + t*.42, .9175 - s*.0525, .3, .0425])

            x_labels = [str(k) for k in isi1s]
            x = np.arange(len(x_labels))  # the label locations
            width = 0.35  # the width of the bars

            # Transform in the LogSpace
            logbeat = [np.log10(i) for i in rt_isi1_grouped_beat]
            loginterval = [np.log10(j) for j in rt_isi1_grouped_interval]

            beat = ax.boxplot(logbeat,
                              bootstrap=100,
                              positions=np.arange(len(x))*2. - width,
                              widths=0.6,
                              flierprops={'marker': '+', 'markersize': 5},
                              patch_artist=True)
            interval = ax.boxplot(loginterval,
                                  bootstrap=100,
                                  positions=np.arange(len(x))*2. + width,
                                  widths=0.6,
                                  flierprops={'marker': '+', 'markersize': 5},
                                  patch_artist=True)

            # Overplot the mean, with horizontal alignment
            # in the center of each box
            for j in np.arange(len(x)):
                medbeat = beat['medians'][j]
                medinterval = interval['medians'][j]
                ax.plot(np.average(medbeat.get_xdata()),
                        np.average(rt_isi1_grouped_beat[j]),
                        color='w', marker='*', markeredgecolor='k')
                ax.plot(np.average(medinterval.get_xdata()),
                        np.average(rt_isi1_grouped_interval[j]),
                        color='w', marker='*', markeredgecolor='k')

            # Fill boxes with colors
            colors = ['b', 'y']
            for patch1, patch2 in zip(beat['boxes'], interval['boxes']):
                patch1.set_facecolor(colors[0])
                patch2.set_facecolor(colors[1])

            if s == len(subjects) - 1:
                fig.text(.5, .01, ' ISIs (ms)', size=18)

            ax.set_xticks(x*2., x_labels)
            plt.ylim([2., 3.35])

            if (t % 2) == 0:
                ax.set_ylabel('Log10(Response Time)')

            if s == 0:
                ax.set_title(task, pad=30, weight='bold')
                if t == 0:
                    ax.legend(frameon=False, loc = 'best',
                              prop={'size': 12})
                    ax.legend([beat["boxes"][0], interval["boxes"][0]],
                              ['Beat', 'Interval'],
                              loc='upper right')
                    fig.text(.26, 0.956, '*', color='white',
                             backgroundcolor='silver', weight='roman',
                             size='medium')
                    fig.text(.275, 0.956, ' Mean', color='black',
                             weight='roman', size='x-small')

            # Hide the right and top spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            # Aggregate data
            if task == 'Auditory Production':
                allsub_beat_audio.append(rt_isi1_grouped_beat)
                allsub_interval_audio.append(rt_isi1_grouped_interval)
            else:
                assert task == 'Visual Production'
                allsub_beat_visual.append(rt_isi1_grouped_beat)
                allsub_interval_visual.append(rt_isi1_grouped_interval)

        fig.text(.07, .94 - s * .0525, 'Subject %d' % subject, ha='center',
                 fontsize=12, weight='bold')

    # Title
    plt.suptitle(
        'Individual Response Time for the Production tasks', x=.5, y=.99,
        size=14, linespacing=.75)

    # plt.show()
    # Save figure
    plt.savefig(os.path.join(this_dir,
                             'production_individual_isi_responsetime.pdf'))

    # Flatten the data arrays
    if flatten:
        allsub_beat_audio = np.ravel(allsub_beat_audio).tolist()
        allsub_interval_audio = np.ravel(allsub_interval_audio).tolist()
        allsub_beat_visual = np.ravel(allsub_beat_visual).tolist()
        allsub_interval_visual = np.ravel(allsub_interval_visual).tolist()

    return (allsub_beat_audio, allsub_interval_audio, allsub_beat_visual,
            allsub_interval_visual, isi1s)


def individual_perception(
        subjects, this_dir, sesstype, n_sess,
        tasks = ['Auditory Perception', 'Visual Perception']):

    all_rf1_audio = []
    all_rf2_audio = []
    all_rf1_visual = []
    all_rf2_visual = []
    for s, subject in enumerate(subjects):
        for t, task in enumerate(tasks):
            if task not in ['Auditory Perception', 'Visual Perception']:
                raise NameError('Task not valid!')

            data = parse_logfile(this_dir, subject, sesstype, n_sess, task)
            trials = perception_data(data)
            beat_trials, interval_trials, _ = filter_trialtype(trials,
                                                               'perception')

            # Calculate frequencies of comparisons per standard
            standards, comparisons, n1_beat, n2_beat, n1_interval, \
                n2_interval = perception_frequencies(beat_trials,
                                                     interval_trials)

            rf1 = [[n1/8 for n1 in n1b] for n1b in n1_beat]
            rf2 = [[n2/8 for n2 in n2b] for n2b in n2_beat]

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
                fig = plt.figure(figsize=(16, 40))

            # Define subplot of bar charts and its position in the fig
            # plt.axes([left, bottom, width, height])
            ax = plt.axes([.1 + t*.46, .92 - s*.0525, .425, .0325])

            for i, st in enumerate(standards):
                # Fit the model with a MLE estimator
                # fun: MLE estimator
                # x0: 1st arg of log_lik
                # args: 2nd and 3rd args of log_lik
                opt_res = optimize.minimize(
                    fun = log_lik,
                    x0 = [np.mean(comparisons), 1.],
                    args = (comparisons, rf2[i], rf1[i]))

                # Estimates
                pse = opt_res.x[0]
                dl = opt_res.x[1] * stats.norm.ppf(0.75)
                # Standard Errors estimated from Fisher information
                se_pse = np.sqrt(np.diag(opt_res.hess_inv))[0]
                se_dl = np.sqrt(
                    np.diag(opt_res.hess_inv))[1] * stats.norm.ppf(0.75)
                # Correlation between mu (pse) and sigma (opt_res.x[1])
                # r = opt_res.hess_inv[0][1] / \
                #     np.sqrt(np.prod(np.diag(opt_res.hess_inv)))
                # 95%CIs
                ci95_pse = se_pse * 1.96
                ci95_dl = se_dl * 1.96

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
                # Hide the right and top spines
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                # Set x axis
                x_values = np.insert(comparisons, 3, 0)
                x_labels = [str(int(xl*100)) + '%' for xl in x_values]
                ax.set_xticks(x_values, x_labels)
                # Add estimates info
                ax.text(-.21, 1.35, 'For 95% CI,', fontsize=7.5)
                ax.text(-.21, 1.23 - i*.105,
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
                    ax.set_title('Auditory Perception', weight='bold', pad=50,
                                 fontsize=16)
                else:
                    assert t ==1
                    ax.set_title('Visual Perception', weight='bold', pad=50,
                                 fontsize=16)

            # Name of x-axis
            fig.text(.495, .009, 'Comparisons (%)', fontsize=14)
            # Name of y-axis
            fig.text(.062, .44, 'Relative Frequency of "longer" responses',
                     fontsize=14, rotation=90)

        fig.text(.03, .935 - s*.0525, 'Subject %d' % subject, ha='center',
                 fontsize=10, weight='bold')

    # Title
    plt.suptitle('Individual Relative Frequencies for the Perception Tasks',
                 x=.5, y=.991, size=18, linespacing=.75)
    # plt.show()

    # Save figure
    plt.savefig(os.path.join(this_dir, 'individual_perception_responses.pdf'))

    return (all_rf1_audio, all_rf2_audio, all_rf1_visual, all_rf2_visual,
            standards, comparisons)


def individual_ntfd_rts(subjects, this_dir, sesstype, n_sess, flatten=True,
                        tasks = ['Auditory No-Temporal Feature Discrimination',
                                 'Visual No-Temporal Feature Discrimination']):

    allsub_beat_audio = []
    allsub_interval_audio = []
    allsub_random_audio = []
    allsub_beat_visual = []
    allsub_interval_visual = []
    allsub_random_visual =[]
    for s, subject in enumerate(subjects):
        for t, task in enumerate(tasks):
            if task not in ['Auditory No-Temporal Feature Discrimination',
                            'Visual No-Temporal Feature Discrimination']:
                raise NameError('Task not valid!')

            data = parse_logfile(this_dir, subject, sesstype, n_sess, task)
            if subject == 2 and \
               task == 'Visual No-Temporal Feature Discrimination':
                data = data[:476]
            trials = ntfd_data(data)
            beat_trials, interval_trials, random_trials = \
                filter_trialtype(trials, 'ntfd')

            # ############## Extract RT's ######################
            beat_trials = np.array([bt[1] for bt in beat_trials])
            interval_trials = np.array([it[1] for it in interval_trials])
            random_trials = np.array([rt[1] for rt in random_trials])

            # Replace missing values (nan's) by median of the all sample
            if np.any(np.isnan(beat_trials)):
                miss_bval = np.nanmedian(beat_trials)
                beat_trials = np.where(np.isnan(beat_trials),
                                       miss_bval, beat_trials)

            if np.any(np.isnan(interval_trials)):
                miss_ival = np.nanmedian(interval_trials)
                interval_trials = np.where(np.isnan(interval_trials),
                                           miss_ival, interval_trials)

            if np.any(np.isnan(random_trials)):
                miss_rval = np.nanmedian(random_trials)
                random_trials = np.where(np.isnan(random_trials),
                                         miss_rval, random_trials)

            # ################## Plotting ###############################
            if s == 0 and t == 0:
                fig = plt.figure(figsize=(8, 15))

            # Define subplot of bar charts and its position in the fig
            # plt.axes([left, bottom, width, height])
            ax = plt.axes([.235 + t*.42, .78 - s*.145, .3, .11])

            labels = ['beat', 'interval', 'random']
            x = [.2, .4, .6]  # the label locations
            width = .175  # the width of the bars
            ntfd_plt = ax.bar(x,
                              [round(beat_trials.mean(0), 2),
                               round(interval_trials.mean(0), 2),
                               round(random_trials.mean(0), 2)],
                              width=width, alpha=.5,
                              color=['tab:blue', 'tab:orange', 'green'],
                              yerr=[round(beat_trials.std(0), 2),
                                    round(interval_trials.std(0), 2),
                                    round(random_trials.std(0), 2)],
                              error_kw=dict(capsize=2), label=labels)
            # Add means values on the top of the bar
            ax.bar_label(ntfd_plt, label_type='center')
            ax.set_xticks(x, labels)
            plt.xlim([0., .8])
            plt.ylim([0., 850.])

            if s == 0:
                if t == 0:
                    ax.set_title('Auditory NTFD', weight='bold', pad=30)
                    fig.text(.25, .88, 'Error bars: SD', fontsize=12)
                else:
                    assert t ==1
                    ax.set_title('Visual NTFD', weight='bold', pad=30)

            # Hide the right and top spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            # Aggregate data to compute the paired sample t-test
            if task == 'Auditory No-Temporal Feature Discrimination':
                allsub_beat_audio.append(beat_trials.tolist())
                allsub_interval_audio.append(interval_trials.tolist())
                allsub_random_audio.append(random_trials.tolist())
            else:
                assert task == 'Visual No-Temporal Feature Discrimination'
                allsub_beat_visual.append(beat_trials.tolist())
                allsub_interval_visual.append(interval_trials.tolist())
                allsub_random_visual.append(random_trials.tolist())

        fig.text(.07, .84 - s*.145, 'Subject %d' % subject, ha='center',
                 fontsize=12, weight='bold')
    fig.text(.1675, .42, 'Reaction Time (ms)', ha='center', fontsize=12,
             rotation = 90)

    # Title
    plt.suptitle('Individual Mean and Standard Deviation of Reaction Time ' + \
                 'for the NTFD tasks', x=.5, y=.965, size=14, linespacing=.75)
    # plt.show()

    # Save figure
    plt.savefig(os.path.join(this_dir, 'ntfd_individual_rt.pdf'))

    # Flatten the data arrays
    if flatten:
        allsub_beat_audio = np.ravel(allsub_beat_audio).tolist()
        allsub_interval_audio = np.ravel(allsub_interval_audio).tolist()
        allsub_random_audio = np.ravel(allsub_random_audio).tolist()
        allsub_beat_visual = np.ravel(allsub_beat_visual).tolist()
        allsub_interval_visual = np.ravel(allsub_interval_visual).tolist()
        allsub_random_visual = np.ravel(allsub_random_visual).tolist()

    return (allsub_beat_audio, allsub_interval_audio, allsub_random_audio,
            allsub_beat_visual, allsub_interval_visual, allsub_random_visual)


def individual_ntfd_isi_rts(
        subjects, this_dir, sesstype, n_sess, flatten=True,
        tasks = ['Auditory No-Temporal Feature Discrimination',
                 'Visual No-Temporal Feature Discrimination']):

    allsub_beat_audio = []
    allsub_interval_audio = []
    allsub_beat_visual = []
    allsub_interval_visual = []
    for s, subject in enumerate(subjects):
        for t, task in enumerate(tasks):
            if task not in ['Auditory No-Temporal Feature Discrimination',
                            'Visual No-Temporal Feature Discrimination']:
                raise NameError('Task not valid!')

            data = parse_logfile(this_dir, subject, sesstype, n_sess, task)
            if subject == 2 and \
               task == 'Visual No-Temporal Feature Discrimination':
                data = data[:476]
            trials = ntfd_data(data)
            beat_trials, interval_trials, _ = filter_trialtype(trials, 'ntfd')

            # ############## Extract RT's per ISI ######################
            isi1s = np.unique(np.array(beat_trials)[:, 0]).astype('int')

            rt_isi1_grouped_beat = []
            for i in isi1s:
                rts_beat = []
                for beat_trial in beat_trials:
                    if beat_trial[0] == i:
                        if ~np.any(np.isnan(beat_trial)):
                            rts_beat.append(beat_trial[1])
                        else:
                            rts_beat.append(np.nan)
                # Replace missing values (nan's) by median of the isi sample
                if np.any(np.isnan(rts_beat)):
                    miss_bval = np.nanmedian(rts_beat)
                    rts_beat = np.where(np.isnan(rts_beat), miss_bval,
                                        rts_beat).tolist()
                rt_isi1_grouped_beat.append(rts_beat)

            rt_isi1_grouped_interval = []
            for j in isi1s:
                rts_interval = []
                for interval_trial in interval_trials:
                    if interval_trial[0] == j:
                        if ~np.any(np.isnan(interval_trial)):
                            rts_interval.append(interval_trial[1])
                        else:
                            rts_interval.append(np.nan)
                # Replace missing values (nan's) by median of the sample
                if np.any(np.isnan(rts_interval)):
                    miss_ival = np.nanmedian(rts_interval)
                    rts_interval = np.where(np.isnan(rts_interval), miss_ival,
                                            rts_interval).tolist()
                rt_isi1_grouped_interval.append(rts_interval)

            # ################## Plotting set 1 ########################
            if s == 0 and t == 0:
                fig = plt.figure(figsize=(8, 40))

            # Define subplot of bar charts and its position in the fig
            # plt.axes([left, bottom, width, height])
            ax = plt.axes([.235 + t*.42, .9175 - s*.0525, .3, .0425])

            x_labels = [str(k) for k in isi1s]
            x = np.arange(len(x_labels))  # the label locations
            width = 0.35  # the width of the bars

            # Transform in the LogSpace
            logbeat = [np.log10(i) for i in rt_isi1_grouped_beat]
            loginterval = [np.log10(j) for j in rt_isi1_grouped_interval]

            beat = ax.boxplot(logbeat,
                              bootstrap=100,
                              positions=np.arange(len(x))*2. - width,
                              widths=0.6,
                              flierprops={'marker': '+', 'markersize': 5},
                              patch_artist=True)
            interval = ax.boxplot(loginterval,
                                  bootstrap=100,
                                  positions=np.arange(len(x))*2. + width,
                                  widths=0.6,
                                  flierprops={'marker': '+', 'markersize': 5},
                                  patch_artist=True)

            # Overplot the mean, with horizontal alignment
            # in the center of each box
            for j in np.arange(len(x)):
                medbeat = beat['medians'][j]
                medinterval = interval['medians'][j]
                ax.plot(np.average(medbeat.get_xdata()),
                        np.average(rt_isi1_grouped_beat[j]),
                        color='w', marker='*', markeredgecolor='k')
                ax.plot(np.average(medinterval.get_xdata()),
                        np.average(rt_isi1_grouped_interval[j]),
                        color='w', marker='*', markeredgecolor='k')

            # Fill boxes with colors
            colors = ['b', 'y']
            for patch1, patch2 in zip(beat['boxes'], interval['boxes']):
                patch1.set_facecolor(colors[0])
                patch2.set_facecolor(colors[1])

            if s == len(subjects) - 1:
                fig.text(.5, .01, ' ISIs (ms)', size=18)

            ax.set_xticks(x*2., x_labels)
            plt.ylim([2., 3.35])

            if (t % 2) == 0:
                ax.set_ylabel('Log10(Reaction Time)')

            if s == 0:
                if t == 0:
                    ax.set_title('Auditory NTFD', pad=30, weight='bold')
                    ax.legend(frameon=False, loc = 'upper left',
                              prop={'size': 12})
                    ax.legend([beat["boxes"][0], interval["boxes"][0]],
                              ['Beat', 'Interval'],
                              loc='upper right')
                    fig.text(.26, 0.956, '*', color='white',
                             backgroundcolor='silver', weight='roman',
                             size='medium')
                    fig.text(.275, 0.956, ' Mean', color='black',
                             weight='roman', size='x-small')
                else:
                    assert t == 1
                    ax.set_title('Visual NTFD', pad=30, weight='bold')

            # Hide the right and top spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            # Aggregate data
            diff = 36 - np.array(rt_isi1_grouped_interval).shape[1]
            if diff != 0:
                # Add missing data for subjects who have less data because of
                # the introduction of the random condition
                rt_isi1_grouped_beat = [
                    np.append(rb, np.repeat(np.median(rb), diff)).tolist()
                    for rb in rt_isi1_grouped_beat]
                rt_isi1_grouped_interval = [
                    np.append(ri, np.repeat(np.median(ri), diff)).tolist()
                    for ri in rt_isi1_grouped_interval]
            if task == 'Auditory No-Temporal Feature Discrimination':
                allsub_beat_audio.append(rt_isi1_grouped_beat)
                allsub_interval_audio.append(rt_isi1_grouped_interval)
            else:
                assert task == 'Visual No-Temporal Feature Discrimination'
                allsub_beat_visual.append(rt_isi1_grouped_beat)
                allsub_interval_visual.append(rt_isi1_grouped_interval)

        fig.text(.07, .94 - s * .0525, 'Subject %d' % subject, ha='center',
                 fontsize=12, weight='bold')

    # Title
    plt.suptitle(
        'Individual Reaction Time for the NTFD tasks',
        x=.5, y=.99, size=14, linespacing=.75)

    # plt.show()
    # Save figure
    plt.savefig(os.path.join(this_dir, 'ntfd_individual_isi_rt.pdf'))

    # Flatten the data arrays
    if flatten:
        allsub_beat_audio = np.ravel(allsub_beat_audio).tolist()
        allsub_interval_audio = np.ravel(allsub_interval_audio).tolist()
        allsub_beat_visual = np.ravel(allsub_beat_visual).tolist()
        allsub_interval_visual = np.ravel(allsub_interval_visual).tolist()

    return (allsub_beat_audio, allsub_interval_audio, allsub_beat_visual,
            allsub_interval_visual, isi1s)


def ginput_reshape(audio_beat, audio_interval, visual_beat, visual_interval):
    # Reshape (n_subjects, n_isi, n_trials) --> (n_isi, n_subjects*n_trials)

    s_audio_beat = np.swapaxes(audio_beat, 0, 1)
    s_audio_interval = np.swapaxes(audio_interval, 0, 1)
    s_visual_beat = np.swapaxes(visual_beat, 0, 1)
    s_visual_interval = np.swapaxes(visual_interval, 0, 1)

    rs_audio_beat = np.reshape(
        s_audio_beat,
        (s_audio_beat.shape[0],
         s_audio_beat.shape[1]*s_audio_beat.shape[2]))

    rs_audio_interval = np.reshape(
        s_audio_interval,
        (s_audio_interval.shape[0],
         s_audio_interval.shape[1]*s_audio_interval.shape[2]))

    rs_visual_beat = np.reshape(
        s_visual_beat,
        (s_visual_beat.shape[0],
         s_visual_beat.shape[1]*s_visual_beat.shape[2]))

    rs_visual_interval = np.reshape(
        s_visual_interval,
        (s_visual_interval.shape[0],
         s_visual_interval.shape[1]*s_visual_interval.shape[2]))

    return rs_audio_beat, rs_audio_interval, rs_visual_beat, rs_visual_interval


def plot_violin(audio_beat, audio_interval,
                visual_beat, visual_interval,
                isi1s, ylim_b, ylim_t, y_label,
                title, this_dir, fname):

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    # left   # the left side of the subplots of the figure
    # right  # the right side of the subplots of the figure
    # bottom # the bottom of the subplots of the figure
    # top    # the top of the subplots of the figure
    # wspace # the amount of width reserved for blank space between subplots
    # hspace # the amount of height reserved for white space between subplots
    plt.subplots_adjust(left=.1, right=.98, bottom=.15, wspace=.075)

    for i, (isi_audio_beat, isi_audio_interval) in enumerate(
            zip(audio_beat, audio_interval)):
        pos_ab = [i*2 - .4]
        pos_ai = [i*2 + .4]
        v1_ab = ax1.violinplot(isi_audio_beat, pos_ab, showmeans=True,
                               showmedians=False, showextrema=True, widths=.75)
        v1_ai = ax1.violinplot(isi_audio_interval, pos_ai, showmeans=True,
                               showmedians=False, showextrema=True, widths=.75)
        customize_vplot(isi_audio_beat, ax1, pos_ab)
        customize_vplot(isi_audio_interval, ax1, pos_ai)

        for vab in v1_ab['bodies']:
            vab.set_facecolor('tab:blue')
            vab.set_edgecolor('black')
            vab.set_alpha(1)

        for vai in v1_ai['bodies']:
            vai.set_facecolor('tab:orange')
            vai.set_edgecolor('black')
            vai.set_alpha(1)

        labels = []
        cb = vab.get_facecolor()
        ci = vai.get_facecolor()
        labels.append((mpatches.Patch(color=cb), 'Beat'))
        labels.append((mpatches.Patch(color=ci), 'Interval'))

        v1_ab['cmaxes'].set_color('black')
        v1_ab['cmins'].set_color('black')
        v1_ab['cbars'].set_color('black')
        v1_ab['cmeans'].set_color('black')

        v1_ai['cmaxes'].set_color('black')
        v1_ai['cmins'].set_color('black')
        v1_ai['cbars'].set_color('black')
        v1_ai['cmeans'].set_color('black')

    for j, (isi_visual_beat, isi_visual_interval) in enumerate(
            zip(visual_beat, visual_interval)):
        pos_vb = [j*2 - .4]
        pos_vi = [j*2 + .4]
        v2_ab = ax2.violinplot(isi_visual_beat, pos_vb, showmeans=True,
                               showmedians=False, showextrema=True, widths=.75)
        v2_ai = ax2.violinplot(isi_visual_interval, pos_vi, showmeans=True,
                               showmedians=False, showextrema=True, widths=.75)
        customize_vplot(isi_visual_beat, ax2, pos_vb)
        customize_vplot(isi_visual_interval, ax2, pos_vi)

        for vab in v2_ab['bodies']:
            vab.set_facecolor('tab:blue')
            vab.set_edgecolor('black')
            vab.set_alpha(1)

        for vai in v2_ai['bodies']:
            vai.set_facecolor('tab:orange')
            vai.set_edgecolor('black')
            vai.set_alpha(1)

        v2_ab['cmaxes'].set_color('black')
        v2_ab['cmins'].set_color('black')
        v2_ab['cbars'].set_color('black')
        v2_ab['cmeans'].set_color('black')

        v2_ai['cmaxes'].set_color('black')
        v2_ai['cmins'].set_color('black')
        v2_ai['cbars'].set_color('black')
        v2_ai['cmeans'].set_color('black')

    # Hide the right and top spines
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    # Label x-axis
    x_labels = [str(standard) for standard in standards]
    pos = np.arange(len(standards))*2
    ax1.set_xticks(pos, x_labels)
    ax2.set_xticks(pos, x_labels)

    # Set limits of y-axis
    ax1.set_ylim(bottom=ylim_b, top=ylim_t)
    ax2.set_ylim(bottom=ylim_b, top=ylim_t)
    # Set y label
    ax1.set_ylabel(y_label, labelpad=.5)
    # Remove y frame, labels and spines of second plot
    ax2.spines['left'].set_visible(False)
    ax2.axes.get_yaxis().set_visible(False)

    # Title of each plot
    ax1.set_title('Auditory Conditions', fontweight='semibold', size=10,
                  y=-.175)
    ax2.set_title('Visual Conditions', fontweight='semibold', size=10,
                  y=-.175)

    # Add legend
    ax1.legend(*zip(*labels), loc='best', frameon=False)
    fig.text(.75, 0.84, 'white circle: median', size=8)
    fig.text(.75, 0.8, 'hline: mean', size=8)

    # Title
    plt.suptitle(title, size=10, linespacing=.75)

    # Save figure
    plt.savefig(os.path.join(this_dir, fname + '.pdf'))


def plot_pttest_isi(audio_beat, audio_interval, visual_beat, visual_interval,
                    pval_audio, pval_visual,
                    isi1s, y, ylim_b, ylim_t, yshift,
                    title, this_dir, fname):

    # Concatenate data
    data_audio = [np.append(audio_beat[j], audio_interval[j]).tolist()
                  for j in np.arange(len(audio_beat))]
    data_visual = [np.append(visual_beat[j], visual_interval[j]).tolist()
                   for j in np.arange(len(visual_beat))]

    modalities = ['audio', 'visual']
    fig, ax = plt.subplots(1, len(modalities))

    # left   # the left side of the subplots of the figure
    # right  # the right side of the subplots of the figure
    # bottom # the bottom of the subplots of the figure
    # top    # the top of the subplots of the figure
    # wspace # the amount of width reserved for blank space between subplots
    # hspace # the amount of height reserved for white space between subplots
    plt.subplots_adjust(left=.12, right=.99, bottom=.15, wspace=.075)

    # Prepare the data
    x = 'Standard'
    z = 'Conditions'
    for m, modality in enumerate(modalities):
        if modality == 'audio':
            n_isi = np.array(data_audio).shape[0]
            n_repeat = np.array(data_audio).shape[1]
            standard = [np.repeat(str(isi1), n_repeat) for isi1 in isi1s]
            conditions = [
                np.append(np.repeat('Beat', n_repeat / 2),
                np.repeat('Interval', n_repeat / 2)).tolist()
                for j in np.arange(n_isi)]
            data_list = data_audio
            pvalue = pval_audio
            x_label = 'Auditory Conditions'
        else:
            assert modality == 'visual'
            n_isi = np.array(data_visual).shape[0]
            n_repeat = np.array(data_visual).shape[1]
            standard = [np.repeat(str(isi1), n_repeat) for isi1 in isi1s]
            conditions = [
                np.append(np.repeat('Beat', n_repeat / 2),
                np.repeat('Interval', n_repeat / 2)).tolist()
                for j in np.arange(n_isi)]
            data_list = data_visual
            pvalue = pval_visual
            x_label = 'Visual Conditions'
        d = {x: np.ravel(standard),
             y: np.ravel(data_list),
             z: np.ravel(conditions)}
        df = pd.DataFrame(data=d)

        # Create bar plot
        sns.barplot(ax=ax[m],
            x=x,
            y=y,
            hue=z,
            data=df,
            estimator=np.mean,
            ci=95, # 1.96 * standard error (95% confidence interval)
            errcolor="black", errwidth=1.5, capsize = 0.2, alpha=0.5)

        # Annotate
        pairs = tuple([[(str(isi1), 'Beat'), (str(isi1), 'Interval')]
                       for isi1 in isi1s])
        annotator = Annotator(ax[m], pairs, data=df, x=x, y=y, hue=z)
        annotator.configure(test=None, text_format="simple",
                            test_short_name="pttest", fontsize=4.5)
        annotator.set_pvalues(pvalue)
        annotator.annotate()

        # Set limits of y-axis
        ax[m].set_ylim(bottom=ylim_b, top=ylim_t)

        # Remove frame of legend
        ax[m].legend(frameon=False)

        # For the second (right) plot, ...
        if m ==1:
            # ... remove labels and ticks
            ax[m].axes.get_yaxis().set_visible(False)
            # ... remove y frame
            ax[m].spines['left'].set_visible(False)
            # ... remove legend
            ax[m].legend([],[], frameon=False)

        # Change x label
        ax[m].set_xlabel(x_label, fontweight='semibold', labelpad=15)

        # Display means rounded to two decimals on the top
        for p in ax[m].patches:
            ax[m].text(p.get_x() + p.get_width()/2.,
                       p.get_height() + np.sign(p.get_height()) * yshift,
                       '{:.2e}'.format(p.get_height()), fontsize=2.5,
                       fontweight='bold', color='black', ha='center',
                       va='bottom')

        # Change width of seaborn barplots
        change_width(ax[m], .4)

        # Hide the right and top spines
        ax[m].spines['right'].set_visible(False)
        ax[m].spines['top'].set_visible(False)

    # Title
    plt.suptitle(title, size=10, linespacing=.75)
    plt.title('95% CI for the Mean', size=8, x=-.15)

    # plt.show()
    # Save figure
    plt.savefig(os.path.join(this_dir, fname + '.pdf'))


def plot_pttest(data_audio, data_visual,
                pval_audio_bi, pval_audio_br, pval_audio_ir,
                pval_visual_bi, pval_visual_br, pval_visual_ir,
                y, ylim_b, ylim_t, yshift, title, this_dir, fname):

    modalities = ['audio', 'visual']
    fig, ax = plt.subplots(1, len(modalities))

    # left   # the left side of the subplots of the figure
    # right  # the right side of the subplots of the figure
    # bottom # the bottom of the subplots of the figure
    # top    # the top of the subplots of the figure
    # wspace # the amount of width reserved for blank space between subplots
    # hspace # the amount of height reserved for white space between subplots
    plt.subplots_adjust(left=.15, bottom=.15, wspace=.25)

    # Define subplot of bar charts and its position in the fig
    # plt.axes([left, bottom, width, height])
    # ax = plt.axes([.225, .145, .65, .65])

    # Prepare the data
    x = 'Conditions'
    pval_audio = [pval_audio_bi, pval_audio_br, pval_audio_ir]
    pval_visual = [pval_visual_bi, pval_visual_br, pval_visual_ir]
    for m, modality in enumerate(modalities):
        if modality == 'audio':
            data_list = data_audio
            pvalue = pval_audio
            x_label = 'Auditory Conditions'
        else:
            assert modality == 'visual'
            data_list = data_visual
            pvalue = pval_visual
            x_label = 'Visual Conditions'
        conditions = np.repeat('Beat', len(data_list) / 3).tolist() + \
            np.repeat('Interval', len(data_list) / 3).tolist() + \
            np.repeat('Random', len(data_list) / 3).tolist()
        d = {x: conditions, y: data_list}
        df = pd.DataFrame(data=d)

        # Create bar plot
        sns.barplot(ax=ax[m],
            x=x,
            y=y,
            data=df,
            estimator=np.mean,
            ci=95, # 1.96 * standard error (95% confidence interval)
            errcolor="black", errwidth=1.5, capsize = 0.2, alpha=0.5)

        # Annotate
        pairs = [('Beat', 'Interval'), ('Beat', 'Random'), ('Interval', 'Random')]
        annotator = Annotator(ax[m], pairs, data=df, x=x, y=y)
        annotator.configure(test=None, text_format="simple",
                            test_short_name="pttest", fontsize=7.)
        annotator.set_pvalues(pvalue)
        annotator.annotate()

        # Set limits of y-axis
        ax[m].set_ylim(bottom=ylim_b, top=ylim_t)

        if m ==1:
            # Remove labels and ticks
            ax[m].axes.get_yaxis().set_visible(False)
            # Remove y frame
            ax[m].spines['left'].set_visible(False)
            # Change x label
            ax[m].set_xlabel('Visual Conditions', fontweight='semibold',
                             labelpad=15)
        else:
            assert m == 0
            ax[m].set_xlabel('Auditory Conditions', fontweight='semibold',
                             labelpad=15)

        # Display means rounded to two decimals on the top
        # ax.bar_label(ax.containers[0], padding=-50)
        for p in ax[m].patches:
            ax[m].text(p.get_x() + p.get_width()/2., p.get_height() + yshift,
                       '{:.2e}'.format(p.get_height()), fontsize=7.,
                       color='black', ha='center', va='bottom')

        # Change width of seaborn barplots
        change_width(ax[m], .7)

        # Hide the right and top spines
        ax[m].spines['right'].set_visible(False)
        ax[m].spines['top'].set_visible(False)

    # Title
    plt.suptitle(title, size=10, linespacing=.75)
    plt.title('95% CI for the Mean', size=8, x=-.15)

    # plt.show()
    # Save figure
    plt.savefig(os.path.join(this_dir, fname + '.pdf'))


def group_perception(all_rf1_audio, all_rf2_audio,
                     all_rf1_visual, all_rf2_visual,
                     standards, comparisons):

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

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
              'tab:purple']

    for m, modality in enumerate(modalities):
        if modality == 'audio':
            rf1 = group_rf1_audio
            rf2 = group_rf2_audio
        else:
            assert modality == 'visual'
            rf1 = group_rf1_visual
            rf2 = group_rf2_visual
        for i, st in enumerate(standards):
            # Fit the model with a MLE estimator
            # fun: MLE estimator
            # x0: 1st arg of log_lik
            # args: 2nd and 3rd args of log_lik
            opt_res = optimize.minimize(
                fun = log_lik,
                x0 = [np.mean(comparisons), 1.],
                args = (comparisons, rf2[i], rf1[i]))

            # Estimates
            pse = opt_res.x[0]
            dl = opt_res.x[1] * stats.norm.ppf(0.75)
            # Standard Errors estimated from Fisher information
            se_pse = np.sqrt(np.diag(opt_res.hess_inv))[0]
            se_dl = np.sqrt(
                np.diag(opt_res.hess_inv))[1] * stats.norm.ppf(0.75)
            # Correlation between mu (pse) and sigma (opt_res.x[1])
            # r = opt_res.hess_inv[0][1] / \
            #     np.sqrt(np.prod(np.diag(opt_res.hess_inv)))
            # 95%CIs
            ci95_pse = se_pse * 1.96
            ci95_dl = se_dl * 1.96

            # Plot each fit in one image
            # fig, ax = plt.subplots(1, 1)
            x = np.linspace(np.amin(comparisons), np.amax(comparisons),
                            100)
            # Plot data
            # ax.plot(comparisons, rf2[i], 'bo', color=colors[i],
            #         markersize=3)
            # Plot fit
            ax[m].plot(x, stats.norm(pse, opt_res.x[1]).cdf(x),
                       color=colors[i], label='Standard = ' + str(st) + 'ms')
            # Hide the right and top spines
            ax[m].spines['right'].set_visible(False)
            ax[m].spines['top'].set_visible(False)
            # Set x axis
            x_values = np.insert(comparisons, 3, 0)
            x_labels = [str(int(xl*100)) + '%' for xl in x_values]
            ax[m].set_xticks(x_values, x_labels)
            # Add estimates info
            ax[m].text(-.21, 1.03, 'For 95% CI,', fontsize=10)
            ax[m].text(-.21, .98 - i*.05,
                       'PSE=%.02f' % (pse*100) +
                       '\u00B1%.02f' % (ci95_pse*100) + '%; ' +
                       'DL=%.02f' % (dl*100) +
                       '\u00B1%.02f' % (ci95_dl*100) + '%', fontsize=10,
                       color=colors[i])

        # Add legend
        if m == 0:
                ax[m].legend(loc='lower right', frameon=False,
                             prop={'size': 10})
                ax[m].set_title('Auditory Perception', weight='bold', pad=5,
                                fontsize=16)
        else:
            assert m == 1
            ax[m].set_title('Visual Perception', weight='bold', pad=5,
                            fontsize=16)

        # Name of x-axis
        fig.text(.485, .0275, 'Comparisons (%)', fontsize=14)
        # Name of y-axis
        fig.text(.035, .19, 'Mean of Relative Frequency for "longer" responses',
                 fontsize=14, rotation=90)
        # Set limits of y-axis
        ax[m].set_ylim([-.1, 1.1])

    # Title
    plt.suptitle('Group Mean of Relative Frequencies for the Perception Tasks',
                 x=.5, y=.97, size=18, linespacing=.75)
    # plt.show()

    # Save figure
    plt.savefig(os.path.join('group_perception_responses.pdf'))

    return (all_rf1_audio, all_rf2_audio, all_rf1_visual, all_rf2_visual,
            standards, comparisons)


# %%
# =========================== INPUTS ===================================

SUBJECTS = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
# RAND_SUBJECTS = [16, 17, 18, 19, 20, 21]
# SUBJECTS = [4]

# TASKS = ['Auditory Production',
#          'Auditory Perception',
#          'Auditory No-Temporal Feature Discrimination',
#          'Visual Production',
#          'Visual Perception',
#          'Visual No-Temporal Feature Discrimination']

SESSTYPE = 'behavioral session'
N_SESSIONS = 3

# %%
# ========================= PARAMETERS =================================

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))

# %%^
# ============================ RUN =====================================

if __name__ == "__main__":

    # # ############### PRODUCTION SYNCHRONIES ###########################

    # # ### Individual analysis per standard --- box plots
    # ssync_audio_beat, ssync_audio_interval, ssync_visual_beat, \
    #     ssync_visual_interval, standards = individual_production_isi_sync(
    #         SUBJECTS, MAIN_DIR, SESSTYPE, N_SESSIONS, 'signed', flatten=False)

    # async_audio_beat, async_audio_interval, async_visual_beat, \
    #     async_visual_interval, standards = individual_production_isi_sync(
    #         SUBJECTS, MAIN_DIR, SESSTYPE, N_SESSIONS, 'absolute',
    #         flatten=False)

    # # ## Reshape

    # # Signed asynchronies
    # rs_ssync_audio_beat, rs_ssync_audio_interval, \
    #     rs_ssync_visual_beat, rs_ssync_visual_interval = ginput_reshape(
    #         ssync_audio_beat, ssync_audio_interval,
    #         ssync_visual_beat, ssync_visual_interval)

    # # Absolute asynchronies
    # rs_async_audio_beat, rs_async_audio_interval, \
    #     rs_async_visual_beat, rs_async_visual_interval = ginput_reshape(
    #         async_audio_beat, async_audio_interval,
    #         async_visual_beat, async_visual_interval)

    # # ### Group Analyses per standard --- violin plots

    # # Signed asynchronies
    # plot_violin(
    #     rs_ssync_audio_beat, rs_ssync_audio_interval,
    #     rs_ssync_visual_beat, rs_ssync_visual_interval,
    #     standards, -1., 4., 'Asynchrony',
    #     'Group Distribution of Signed-Asynchrony for the Production Tasks',
    #     MAIN_DIR,
    #     'production_groupviolin_signed_asynch')

    # # Absolute asynchronies
    # plot_violin(
    #     rs_async_audio_beat, rs_async_audio_interval,
    #     rs_async_visual_beat, rs_async_visual_interval,
    #     standards, -.05, 4., 'Asynchrony',
    #     'Group Distribution of Absolute-Asynchrony for the Production Tasks',
    #     MAIN_DIR,
    #     'production_groupviolin_absolute_asynch')

    # # ### Group Analyses per standard --- bar plots + paired t-test

    # # Signed asynchronies
    # _, pssync_audio = stats.ttest_rel(
    #     rs_ssync_audio_beat, rs_ssync_audio_interval,
    #     axis=1, alternative='two-sided')

    # _, pssync_visual = stats.ttest_rel(
    #     rs_ssync_visual_beat, rs_ssync_visual_interval,
    #     axis=1, alternative='two-sided')

    # ssync_title = 'Group Mean of Signed Asynchrony for the Production tasks'
    # ssync_f = 'paired-ttest_signed_asynch'
    # plot_pttest_isi(rs_ssync_audio_beat, rs_ssync_audio_interval,
    #                 rs_ssync_visual_beat, rs_ssync_visual_interval,
    #                 pssync_audio, pssync_visual,
    #                 standards, 'Signed Asynchrony', -.125, .275, -.039,
    #                 ssync_title, MAIN_DIR, ssync_f)

    # # Absolute asynchronies
    # _, pasync_audio = stats.ttest_rel(
    #     rs_async_audio_beat, rs_async_audio_interval,
    #     axis=1, alternative='two-sided')

    # _, pasync_visual = stats.ttest_rel(
    #     rs_async_visual_beat, rs_async_visual_interval,
    #     axis=1, alternative='two-sided')

    # async_title = 'Group Mean of Absolute Asynchrony for the Production tasks'
    # async_f = 'paired-ttest_absolute_asynch'
    # plot_pttest_isi(rs_async_audio_beat, rs_async_audio_interval,
    #                 rs_async_visual_beat, rs_async_visual_interval,
    #                 pasync_audio, pasync_visual,
    #                 standards, 'Absolute Asynchrony', -0., .275, -.04,
    #                 async_title, MAIN_DIR, async_f)

    # # # ############## PRODUCTION RESPONSE TIME ########################

    # # ### Individual analysis per standard --- box plots
    # rtsprod_audio_beat, rtsprod_audio_interval, rtsprod_visual_beat, \
    #     rtsprod_visual_interval, standards = individual_production_isi_rts(
    #         SUBJECTS, MAIN_DIR, SESSTYPE, N_SESSIONS, flatten=False)

    # # ## Reshape
    # rs_rtsprod_audio_beat, rs_rtsprod_audio_interval, \
    #     rs_rtsprod_visual_beat, rs_rtsprod_visual_interval = ginput_reshape(
    #         rtsprod_audio_beat, rtsprod_audio_interval,
    #         rtsprod_visual_beat, rtsprod_visual_interval)

    # # ### Group Analyses per standard --- violin plots
    # plot_violin(
    #     rs_rtsprod_audio_beat, rs_rtsprod_audio_interval,
    #     rs_rtsprod_visual_beat, rs_rtsprod_visual_interval,
    #     standards, 0., 2250., 'Response Time (ms)',
    #     'Group Distribution of Response Time for the Production Tasks',
    #     MAIN_DIR,
    #     'production_groupviolin_responsetime')

    # # ### Group Analyses per standard --- bar plots + paired t-test
    # _, prtprod_audio = stats.ttest_rel(
    #     rs_rtsprod_audio_beat, rs_rtsprod_audio_interval,
    #     axis=1, alternative='two-sided')

    # _, prtprod_visual = stats.ttest_rel(
    #     rs_rtsprod_visual_beat, rs_rtsprod_visual_interval,
    #     axis=1, alternative='two-sided')

    # rtprod_title = 'Group Mean of Response Time for the Production tasks'
    # rtprod_f = 'paired-ttest_responsetime_production'
    # plot_pttest_isi(rs_rtsprod_audio_beat, rs_rtsprod_audio_interval,
    #                 rs_rtsprod_visual_beat, rs_rtsprod_visual_interval,
    #                 prtprod_audio, prtprod_visual,
    #                 standards, 'Response Time (ms)', 0., 900., -100.,
    #                 rtprod_title, MAIN_DIR, rtprod_f)

    # # ################### PERCEPTION ###################################

    rfone_audio, rftwo_audio, rfone_visual, rftwo_visual, stand, comp = \
        individual_perception(SUBJECTS, MAIN_DIR, SESSTYPE, N_SESSIONS)

    group_perception(rfone_audio, rftwo_audio, rfone_visual, rftwo_visual,
                     stand, comp)

    # # # ################### NTFD RT'S ####################################

    # # ### Individual analysis merging all standards --- bar plots
    # m_rtsntfd_audio_beat, m_rtsntfd_audio_interval, m_rtsntfd_audio_random, \
    #     m_rtsntfd_visual_beat, m_rtsntfd_visual_interval, \
    #     m_rtsntfd_visual_random = individual_ntfd_rts(
    #         RAND_SUBJECTS, MAIN_DIR, SESSTYPE, N_SESSIONS)

    # m_rtsntfd_audio = m_rtsntfd_audio_beat + m_rtsntfd_audio_interval + \
    #     m_rtsntfd_audio_random
    # m_rtsntfd_visual = m_rtsntfd_visual_beat + m_rtsntfd_visual_interval + \
    #     m_rtsntfd_visual_random

    # _, pntfd_audio_bi = stats.ttest_rel(
    #     m_rtsntfd_audio_beat, m_rtsntfd_audio_interval,
    #     axis=0, alternative='two-sided')

    # _, pntfd_audio_br = stats.ttest_rel(
    #     m_rtsntfd_audio_beat, m_rtsntfd_audio_random,
    #     axis=0, alternative='two-sided')

    # _, pntfd_audio_ir = stats.ttest_rel(
    #     m_rtsntfd_audio_interval, m_rtsntfd_audio_random,
    #     axis=0, alternative='two-sided')

    # _, pntfd_visual_bi = stats.ttest_rel(
    #     m_rtsntfd_visual_beat, m_rtsntfd_visual_interval,
    #     axis=0, alternative='two-sided')

    # _, pntfd_visual_br = stats.ttest_rel(
    #     m_rtsntfd_visual_beat, m_rtsntfd_visual_random,
    #     axis=0, alternative='two-sided')

    # _, pntfd_visual_ir = stats.ttest_rel(
    #     m_rtsntfd_visual_interval, m_rtsntfd_visual_random,
    #     axis=0, alternative='two-sided')

    # ntfd_title = 'Group Mean of Reaction Time for the NTFD tasks'
    # ntfd_f = 'paired-ttest_merged-rt_ntfd'
    # plot_pttest(m_rtsntfd_audio, m_rtsntfd_visual,
    #             pntfd_audio_bi, pntfd_audio_br, pntfd_audio_ir,
    #             pntfd_visual_bi, pntfd_visual_br, pntfd_visual_ir,
    #             'Reaction Time (ms)', 0., 750., -100.,
    #             ntfd_title, MAIN_DIR, ntfd_f)

    # # ### Individual analysis per standards --- box plots
    # rtsntfd_audio_beat, rtsntfd_audio_interval, rtsntfd_visual_beat, \
    #     rtsntfd_visual_interval, standards = individual_ntfd_isi_rts(
    #         SUBJECTS, MAIN_DIR, SESSTYPE, N_SESSIONS, flatten=False)

    # # ## Reshape
    # rs_rtsntfd_audio_beat, rs_rtsntfd_audio_interval, \
    #     rs_rtsntfd_visual_beat, rs_rtsntfd_visual_interval = ginput_reshape(
    #         rtsntfd_audio_beat, rtsntfd_audio_interval,
    #         rtsntfd_visual_beat, rtsntfd_visual_interval)

    # # ### Group Analyses per standard --- violin plots
    # plot_violin(
    #     rs_rtsntfd_audio_beat, rs_rtsntfd_audio_interval,
    #     rs_rtsntfd_visual_beat, rs_rtsntfd_visual_interval,
    #     standards, 0., 2250., 'Reaction Time (ms)',
    #     'Group Distribution of Reaction Time for the NTFD Tasks',
    #     MAIN_DIR,
    #     'ntfd_groupviolin_rt')

    # # ### Group Analyses per standard --- bar plots + paired t-test
    # _, prtntfd_audio = stats.ttest_rel(
    #     rs_rtsntfd_audio_beat, rs_rtsntfd_audio_interval,
    #     axis=1, alternative='two-sided')

    # _, prtntfd_visual = stats.ttest_rel(
    #     rs_rtsntfd_visual_beat, rs_rtsntfd_visual_interval,
    #     axis=1, alternative='two-sided')

    # rtntfd_title = 'Group Mean of Reaction Time for the NTFD tasks'
    # rtntfd_f = 'paired-ttest_rt_ntfd'
    # plot_pttest_isi(rs_rtsntfd_audio_beat, rs_rtsntfd_audio_interval,
    #                 rs_rtsntfd_visual_beat, rs_rtsntfd_visual_interval,
    #                 prtntfd_audio, prtntfd_visual,
    #                 standards, 'Reaction Time (ms)', 0., 650., -100.,
    #                 rtntfd_title, MAIN_DIR, rtntfd_f)
