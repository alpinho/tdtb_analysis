"""
Script to extract behavioral data from logfiles for the Music-SDTB project

author: Ana Luisa Pinho
e-mail: agrilopi@uwo.ca

Created: August 2022
Last update: August 2022

Compatibility: Python 3.7.11

"""
import os
import glob
import csv

import numpy as np
import pandas as pd
from scipy import stats

from matplotlib import pyplot as plt
from matplotlib import patches as mpatches

import seaborn as sns
from statannotations.Annotator import Annotator


# %%
# =========================== UTILS ====================================


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


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
            real_isi1 = datum[9]
            real_isi5 = data[dt+8][9]
            if data[dt+10][5] == 'feedback' and \
               data[dt+10][11] in ['o', 'p']:
                # rt = int(data[dt+9][7]) + int(data[dt+10][10])
                answer = data[dt+10][11]
            elif data[dt+10][5] == 'feedback' and \
                 data[dt+10][11] == 'None':
                continue
            else:
                raise ValueError('No feedback entry!')
            trials.append([condition, real_isi1, real_isi5, answer])

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


def perception_frequencies(isi_diff_condition, condition_trials):

    idiffs = np.unique(isi_diff_condition)

    frequencies = []
    for idiff in idiffs:
        count = 0
        responses = []
        for i, idiff_cond in enumerate(isi_diff_condition):
            if idiff_cond == idiff:
                count += 1
                responses.append(condition_trials[i][2])
        frequency = responses.count('o') / count
        frequencies.append(frequency)

    idiffs = idiffs * 100
    idiffs = ['%d' % i + '%' for i in idiffs]
    frequencies = [round(f * 100, 0) for f in frequencies]

    return idiffs, frequencies


def individual_production_isi_sync(
        subjects, this_dir, sesstype, n_sess, sync_type,
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
                # Replace missing values (nan's) by median of the sample
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
                fig = plt.figure(figsize=(8, 36))

            # Define subplot of bar charts and its position in the fig
            # plt.axes([left, bottom, width, height])
            ax = plt.axes([.235 + t*.42, .88 - s*.07, .3, .05])

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

            # if s == len(subjects) - 1:
            #     ax.set_xticks(x, x_labels)
            # else:
            #     # ax.set_xticks([x[0] + .4, x[1] - .4], '')
            #     ax.tick_params(bottom=False)
            #     ax.spines['bottom'].set_visible(False)
            if s == len(subjects) - 1:
                fig.text(.5, .02, ' ISIs (ms)', size=18)

            ax.set_xticks(x*2., x_labels)

            # if sync_type == 'signed':
            #     y_ticks = np.arange(-1., 4., .5)
            # else:
            #     assert sync_type == 'absolute'
            #     y_ticks = np.arange(0., 4., .5)
            # y_labels = np.array([str(y_tick) if (y % 2) != 0 else ''
            #                      for y, y_tick in enumerate(y_ticks)])
            # ax.set_yticks(y_ticks, y_labels)
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
                ax.set_title(task, pad=60, weight='bold')
                if t == 0:
                    ax.legend(frameon=False, loc = 'upper right',
                              prop={'size': 8})
                    ax.legend([beat["boxes"][0], interval["boxes"][0]],
                              ['Beat', 'Interval'],
                              loc='upper right', prop={'size': 8})
                    fig.text(.27, 0.925, '*', color='white',
                             backgroundcolor='silver', weight='roman',
                             size='medium')
                    fig.text(.285, 0.925, ' Mean', color='black',
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

        fig.text(.07, .905 - s * .07, 'Subject %d' % subject, ha='center',
                 fontsize=12, weight='bold')

    # plt.show()
    # Save figure
    plt.savefig(os.path.join(
        this_dir, 'production_individual_isi_' + sync_type + '_asynch.pdf'))

    # Flatten the data arrays
    allsub_beat_audio = np.ravel(allsub_beat_audio)
    allsub_interval_audio = np.ravel(allsub_interval_audio)
    allsub_beat_visual = np.ravel(allsub_beat_visual)
    allsub_interval_visual = np.ravel(allsub_interval_visual)

    return (allsub_beat_audio, allsub_interval_audio, allsub_beat_visual,
            allsub_interval_visual)


def individual_production_isi_rts(
        subjects, this_dir, sesstype, n_sess,
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
                # Replace missing values (nan's) by median of the sample
                if np.any(np.isnan(rts_interval)):
                    miss_ival = np.nanmedian(rts_interval)
                    rts_interval = np.where(np.isnan(rts_interval), miss_ival,
                                            rts_interval)
                # Append isi array
                rt_isi1_grouped_interval.append(rts_interval)

            # ################## Plotting ###############################
            if s == 0 and t == 0:
                fig = plt.figure(figsize=(8, 36))

            # Define subplot of bar charts and its position in the fig
            # plt.axes([left, bottom, width, height])
            ax = plt.axes([.235 + t*.42, .88 - s*.07, .3, .05])

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

            # if s == len(subjects) - 1:
            #     ax.set_xticks(x, x_labels)
            # else:
            #     # ax.set_xticks([x[0] + .4, x[1] - .4], '')
            #     ax.tick_params(bottom=False)
            #     ax.spines['bottom'].set_visible(False)
            if s == len(subjects) - 1:
                fig.text(.5, .02, ' ISIs (ms)', size=18)

            ax.set_xticks(x*2., x_labels)
            plt.ylim([2., 3.35])

            if (t % 2) == 0:
                ax.set_ylabel('Log10(RT)')

            if s == 0:
                ax.set_title(task, pad=60, weight='bold')
                if t == 0:
                    ax.legend(frameon=False, loc = 'upper left',
                              prop={'size': 12})
                    ax.legend([beat["boxes"][0], interval["boxes"][0]],
                              ['Beat', 'Interval'],
                              loc='upper right')
                    fig.text(.27, 0.923, '*', color='white',
                             backgroundcolor='silver', weight='roman',
                             size='medium')
                    fig.text(.285, 0.9225, ' Mean', color='black',
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

        fig.text(.07, .905 - s * .07, 'Subject %d' % subject, ha='center',
                 fontsize=12, weight='bold')

    # plt.show()
    # Save figure
    plt.savefig(os.path.join(this_dir, 'production_individual_isi_rts.pdf'))

    # Flatten the data arrays
    allsub_beat_audio = np.ravel(allsub_beat_audio)
    allsub_interval_audio = np.ravel(allsub_interval_audio)
    allsub_beat_visual = np.ravel(allsub_beat_visual)
    allsub_interval_visual = np.ravel(allsub_interval_visual)

    return (allsub_beat_audio, allsub_interval_audio, allsub_beat_visual,
            allsub_interval_visual)


def individual_perception(
        subjects, this_dir, sesstype, n_sess,
        tasks = ['Auditory Perception', 'Visual Perception']):
    for s, subject in enumerate(subjects):
        for t, task in enumerate(tasks):
            if task not in ['Auditory Perception', 'Visual Perception']:
                raise NameError('Task not valid!')

            data = parse_logfile(this_dir, subject, sesstype, n_sess, task)
            trials = perception_data(data)
            beat_trials, interval_trials, _ = filter_trialtype(trials,
                                                               'perception')

            # ######### Compute perception frequencies #################
            isi_diff_beat = [round((bt[1] - bt[0]) / bt[0], 2)
                             for bt in beat_trials]
            isi_diff_interval = [round((it[1] - it[0]) / it[0], 2)
                                 for it in interval_trials]

            x_vals, y_beat_vals = perception_frequencies(isi_diff_beat,
                                                         beat_trials)
            _, y_interval_val = perception_frequencies(isi_diff_interval,
                                                       interval_trials)

            # ################## Plotting ###############################
            if s == 0 and t == 0:
                fig = plt.figure(figsize=(8, 36))

            # Define subplot of bar charts and its position in the fig
            # plt.axes([left, bottom, width, height])
            ax = plt.axes([.235 + t*.42, .9 - s*.095, .3, .05])

            ax.plot(x_vals, y_beat_vals, marker='o', markersize=5,
                    color='b', label='Beat', linewidth=3)
            ax.plot(x_vals, y_interval_val, marker='o', markersize=5,
                    color='y', label='Interval', linewidth=3)

            # X axis
            x_labels = np.arange(len(x_vals))
            plt.xticks(x_labels)
            # if s != len(subjects) - 1:
            #     plt.xticks([])
            #     ax.tick_params(bottom=False)
            #     ax.spines['bottom'].set_visible(False)

            # Y axis
            y_vals = np.arange(0, 1.5, .5) * 100
            y_vals = y_vals.astype('int')
            y_labels = ['%d' % yv + '%' for yv in y_vals]
            # y_labels = ['shorter', 'equal', 'longer']
            if t == 0:
                ax.set_yticks(y_vals, y_labels)
            else:
                ax.set_yticks(y_vals, '')

            # Title per column with Task name and legend
            if s == 0:
                ax.set_title(task, pad=50, weight='bold')
                if t == 0:
                    ax.legend(frameon=False, loc = 'lower right',
                              prop={'size': 10})

            # Hide the right and top spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        fig.text(.07, .9275 - s * .095, 'Subject %d' % subject, ha='center',
                 fontsize=10, weight='bold')

    fig.text(.3, .02, 'Proportion of time difference between RTs and ISI1 (%)',
             fontsize=12)
    fig.text(.14, .4, '% of "longer" responses', fontsize=12, rotation=90)

    # plt.show()

    # Save figure
    plt.savefig(os.path.join(this_dir, 'perception_responses.pdf'))


def individual_ntfd_rts(subjects, this_dir, sesstype, n_sess,
                        tasks = ['Auditory No-Temporal Feature Discrimination',
                                 'Visual No-Temporal Feature Discrimination']):

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

            beat_trials = np.array(beat_trials).ravel()
            interval_trials = np.array(interval_trials).ravel()
            random_trials = np.array(random_trials).ravel()

            # ################## Plotting ###############################
            if s == 0 and t == 0:
                fig = plt.figure(figsize=(8, 4))

            # Define subplot of bar charts and its position in the fig
            # plt.axes([left, bottom, width, height])
            ax = plt.axes([.235 + t*.42, .15, .3, .5])

            labels = ['beat', 'interval', 'random']
            x = [.2, .4, .6]  # the label locations
            width = .175  # the width of the bars
            ntfd_plt = ax.bar(x,
                              [round(beat_trials.mean(0), 2),
                               round(interval_trials.mean(0), 2),
                               round(random_trials.mean(0), 2)],
                              width=width,
                              color=['b', 'y', 'm'],
                              yerr=[round(beat_trials.std(0), 2),
                                    round(interval_trials.std(0), 2),
                                    round(random_trials.std(0), 2)],
                              error_kw=dict(capsize=2), label=labels)
            ax.bar_label(ntfd_plt, padding=3)
            ax.set_xticks(x, labels)
            plt.xlim([0., .8])
            plt.ylim([0., 1500.])

            if s == 0:
                if t == 0:
                    ax.set_title('Auditory NTFD', weight='bold', pad=40)
                    fig.text(.25, .625, 'Error bars: SD', fontsize=12)
                else:
                    assert t ==1
                    ax.set_title('Visual NTFD', weight='bold', pad=40)

            # Hide the right and top spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        fig.text(.07, .4, 'Subject %d' % subject, ha='center',
                 fontsize=12, weight='bold')
    fig.text(.155, .25, 'Mean of RT (ms)', ha='center',
             fontsize=12, rotation = 90)

    # plt.show()

    # Save figure
    plt.savefig(os.path.join(this_dir, 'ntfd_individual_rts_sub-16.pdf'))


def individual_ntfd_isi_rts(
        subjects, this_dir, sesstype, n_sess,
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
                # Replace missing values (nan's) by median of the sample
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
                fig = plt.figure(figsize=(8, 36))

            # Define subplot of bar charts and its position in the fig
            # plt.axes([left, bottom, width, height])
            ax = plt.axes([.235 + t*.42, .88 - s*.07, .3, .05])

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

            # if s == len(subjects) - 1:
            #     ax.set_xticks(x, x_labels)
            # else:
            #     # ax.set_xticks([x[0] + .4, x[1] - .4], '')
            #     ax.tick_params(bottom=False)
            #     ax.spines['bottom'].set_visible(False)
            if s == len(subjects) - 1:
                fig.text(.5, .02, ' ISIs (ms)', size=18)

            ax.set_xticks(x*2., x_labels)
            plt.ylim([2., 3.35])

            if (t % 2) == 0:
                ax.set_ylabel('Log10(RT)')

            if s == 0:
                if t == 0:
                    ax.set_title('Auditory NTFD', pad=60, weight='bold')
                    ax.legend(frameon=False, loc = 'upper left',
                              prop={'size': 12})
                    ax.legend([beat["boxes"][0], interval["boxes"][0]],
                              ['Beat', 'Interval'],
                              loc='upper right')
                    fig.text(.27, 0.923, '*', color='white',
                             backgroundcolor='silver', weight='roman',
                             size='medium')
                    fig.text(.285, 0.9225, ' Mean', color='black',
                             weight='roman', size='x-small')
                else:
                    assert t == 1
                    ax.set_title('Visual NTFD', pad=60, weight='bold')

            # Hide the right and top spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            # Aggregate data to compute the paired sample t-test
            rt_isi1_grouped_beat = np.ravel(rt_isi1_grouped_beat).tolist()
            rt_isi1_grouped_interval = np.ravel(
                rt_isi1_grouped_interval).tolist()
            if task == 'Auditory No-Temporal Feature Discrimination':
                allsub_beat_audio.extend(rt_isi1_grouped_beat)
                allsub_interval_audio.extend(rt_isi1_grouped_interval)
            else:
                assert task == 'Visual No-Temporal Feature Discrimination'
                allsub_beat_visual.extend(rt_isi1_grouped_beat)
                allsub_interval_visual.extend(rt_isi1_grouped_interval)

        fig.text(.07, .905 - s * .07, 'Subject %d' % subject, ha='center',
                 fontsize=12, weight='bold')

    # plt.show()
    # Save figure
    plt.savefig(os.path.join(this_dir, 'ntfd_individual_isi_rts.pdf'))

    return (allsub_beat_audio, allsub_interval_audio, allsub_beat_visual,
            allsub_interval_visual)


def plot_violin(allaudio_beat, allaudio_interval,
                allvisual_beat, allvisual_interval,
                title, y_label, this_dir, fname, loc):

    data = [allaudio_beat, allaudio_interval,
            allvisual_beat, allvisual_interval]
    pos = np.array([1.15, 1.85, 3.15, 3.85])

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4),
                           sharey=True)
    ax.set_title(title, size=10)
    parts = ax.violinplot(data, pos, showmeans=True, showmedians=False,
                          showextrema=True)

    labels = []
    for pc in parts['bodies'][:2]:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(1)
        color = parts["bodies"][0].get_facecolor().flatten()
        labels.append((mpatches.Patch(color=color), 'Auditory'))

    for pc in parts['bodies'][2:]:
        pc.set_facecolor('#dede00')
        pc.set_edgecolor('black')
        pc.set_alpha(1)
        color = parts["bodies"][2].get_facecolor().flatten()
        labels.append((mpatches.Patch(color=color), 'Visual'))

    quartile1 = []
    medians = []
    quartile3 = []
    for datum in data:
        q1, median, q3 = np.percentile(datum, [25, 50, 75])
        quartile1.append(q1)
        medians.append(median)
        quartile3.append(q3)
    quartile1 = np.array(quartile1)
    medians = np.array(medians)
    quartile3 = np.array(quartile3)

    whiskers = np.array([
        adjacent_values(sorted_array, q1, q3)
        for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

    # inds = np.arange(1, len(medians) + 1)
    ax.scatter(pos, medians, marker='o', color='white', s=6, zorder=3)
    ax.vlines(pos, quartile1, quartile3, color='k', linestyle='-', lw=5)
    ax.vlines(pos, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

    # set style for the axes
    x_labels = ['Beat', 'Interval', 'Beat', 'Interval']
    ax.set_xticks(pos, x_labels)
    plt.ylabel(y_label)

    plt.subplots_adjust(left=.2, bottom=0.1, top=.85, wspace=0.05)

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Add legend
    labels.remove(labels[1])
    labels.remove(labels[-1])
    plt.legend(*zip(*labels), loc=loc)
    fig.text(.01, 0.96, 'white circle: median', size=8)
    fig.text(.01, 0.92, 'hline: mean', size=8)

    # plt.show()

    # Save figure
    plt.savefig(os.path.join(this_dir, fname + '.pdf'))


def plot_pairedttest(data_audio, data_visual, pval_audio, pval_visual,
                     ylim_b, ylim_t, title, this_dir, fname):

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
    y = 'Signed Asynchrony'
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
        conditions = np.repeat('Beat', len(data_list) / 2).tolist() + \
            np.repeat('Interval', len(data_list) / 2).tolist()
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
        pairs = [(('Beat', 'Interval'))]
        annotator = Annotator(ax[m], pairs, data=df, x=x, y=y)
        annotator.configure(test=None, text_format="simple",
                            test_short_name="Paired t-test")
        annotator.set_pvalues([pvalue])
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
        yshift = -.02
        for p in ax[m].patches:
            ax[m].text(p.get_x() + p.get_width()/2., p.get_height() + yshift,
                       '{:.2e}'.format(p.get_height()), fontsize=10,
                       color='black', ha='center', va='bottom')

        # Change width of seaborn barplots
        change_width(ax[m], .6)

        # Hide the right and top spines
        ax[m].spines['right'].set_visible(False)
        ax[m].spines['top'].set_visible(False)

    # Title
    plt.suptitle(title, size=10, linespacing=.75)
    plt.title('95% CI for the Mean', size=8, x=-.15)

    # plt.show()
    # Save figure
    plt.savefig(os.path.join(this_dir, fname + '.pdf'))


# %%
# =========================== INPUTS ===================================

SUBJECTS = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
# SUBJECTS = [16]

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

    # ############### PRODUCTION SYNCHRONIES ###########################

    ssync_audio_beat, ssync_audio_interval, ssync_visual_beat, \
        ssync_visual_interval = individual_production_isi_sync(
            SUBJECTS, MAIN_DIR, SESSTYPE, N_SESSIONS, 'signed')

    async_audio_beat, async_audio_interval, async_visual_beat, \
        async_visual_interval = individual_production_isi_sync(
            SUBJECTS, MAIN_DIR, SESSTYPE, N_SESSIONS, 'absolute')

    # ###############

    # plot_violin(
    #     ssync_audio_beat, ssync_audio_interval,
    #     ssync_visual_beat, ssync_visual_interval,
    #     'Group Signed-Asynchrony for Production Tasks',
    #     'Asynchrony',
    #     MAIN_DIR,
    #     'production_groupviolin_signed_asynch',
    #     'upper left')

    # plot_violin(
    #     async_audio_beat, async_audio_interval,
    #     async_visual_beat, async_visual_interval,
    #     'Group Absolute-Asynchrony for Production Tasks',
    #     'Asynchrony',
    #     MAIN_DIR,
    #     'production_groupviolin_absolute_asynch',
    #     'upper left')

    # ###############

    # Compute and plot paired-sample t-test for production asynchronies

    tssync_audio, pssync_audio = stats.ttest_rel(
        ssync_audio_beat, ssync_audio_interval, alternative='two-sided')
    ssync_audio = ssync_audio_beat.tolist() + ssync_audio_interval.tolist()

    tssync_visual, pssync_visual = stats.ttest_rel(
        ssync_visual_beat, ssync_visual_interval, alternative='two-sided')
    ssync_visual = ssync_visual_beat.tolist() + ssync_visual_interval.tolist()

    ssync_title = 'Group Mean of Signed Asynchrony for the Production tasks'
    ssync_f = 'paired-ttest_signed_asynch'

    plot_pairedttest(ssync_audio, ssync_visual, pssync_audio, pssync_visual,
                     -.03, .14, ssync_title, MAIN_DIR, ssync_f)

    # tasync_audio, pasync_audio = stats.ttest_rel(
    #     async_audio_beat, async_audio_interval, alternative='two-sided')
    # tasync_visual, pasync_visual = stats.ttest_rel(
    #     async_visual_beat, async_visual_interval, alternative='two-sided')

    # # ################# PRODUCTION RT'S ################################

    # rts_audio_beat, rts_audio_interval, \
    #     rts_visual_beat, rts_visual_interval = individual_production_isi_rts(
    #         SUBJECTS, MAIN_DIR, SESSTYPE, N_SESSIONS)

    # plot_violin(
    #     rts_audio_beat, rts_audio_interval,
    #     rts_visual_beat, rts_visual_interval,
    #     'Group RTs for Production Tasks',
    #     'RTs (ms)',
    #     MAIN_DIR,
    #     'production_groupviolin_rts',
    #     'upper center')

    # # ###############

    # # # Compute paired-sample t-test for production RT's
    # trt_audio, prt_audio = stats.ttest_rel(
    #     rts_audio_beat, rts_audio_interval, alternative='two-sided')
    # trt_visual, prt_visual = stats.ttest_rel(
    #     rts_visual_beat, rts_visual_interval, alternative='two-sided')

    # # ################### PERCEPTION ###################################

    # # individual_perception(SUBJECTS, MAIN_DIR, SESSTYPE, N_SESSIONS)

    # # ################### NTFD RT'S ####################################

    # individual_ntfd_rts([16], MAIN_DIR, SESSTYPE, N_SESSIONS)

    # ntfd_audio_beat, ntfd_audio_interval, \
    #     ntfd_visual_beat, ntfd_visual_interval = individual_ntfd_isi_rts(
    #         SUBJECTS, MAIN_DIR, SESSTYPE, N_SESSIONS)

    # plot_violin(
    #     ntfd_audio_beat, ntfd_audio_interval,
    #     ntfd_visual_beat, ntfd_visual_interval,
    #     'Group RTs for NTFD Tasks',
    #     'RTs (ms)',
    #     MAIN_DIR,
    #     'ntfd_groupviolin_rts',
    #     'upper center')

    # # ###############

    # # Compute paired-sample t-test for NTFD tasks
    # tntfd_audio, pntfd_audio = stats.ttest_rel(
    #     ntfd_audio_beat, ntfd_audio_interval, alternative='two-sided')
    # tntfd_visual, pntfd_visual = stats.ttest_rel(
    #     ntfd_visual_beat, ntfd_visual_interval, alternative='two-sided')
