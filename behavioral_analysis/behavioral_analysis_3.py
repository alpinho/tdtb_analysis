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

from scipy import stats

from matplotlib import pyplot as plt


# %%
# =========================== FUNCTIONS ================================


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
            if datum[11] in ['o', 'p']:
                rt = int(data[dt-1][7]) + int(datum[10])
            elif datum[10] == 'None':
                continue
            else:
                raise ValueError('No feedback entry!')
            trials.append([condition, rt])

    return trials


def filter_trialtype(trs, category):
    beat = [tr[1:] for tr in trs if tr[0][:4] == 'beat']
    interval = [tr[1:] for tr in trs if tr[0][:8] == 'interval']

    if category in ['production', 'ntfd']:
        beat = [list(map(int, b)) if ~np.any(np.isnan(b)) else b
                for b in beat]
        interval = [list(map(int, i)) if ~np.any(np.isnan(i)) else i
                    for i in interval]
    else:
        assert category == 'perception'
        beat = [[int(b[0]), int(b[1]), b[2]] for b in beat]
        interval = [[int(i[0]), int(i[1]), i[2]] for i in interval]

    return beat, interval


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


def individual_production_sync(
        subjects, this_dir, sesstype, n_sess, sync_type, mode='mean',
        tasks = ['Auditory Production', 'Visual Production']):

    # allsub_beat_audio = []
    # allsub_intv_audio = []
    # allsub_beat_visual = []
    # allsub_intv_visual = []
    for s, subject in enumerate(subjects):
        for t, task in enumerate(tasks):
            if task not in ['Auditory Production', 'Visual Production']:
                raise NameError('Task not valid!')

            data = parse_logfile(this_dir, subject, sesstype, n_sess, task)
            trials = production_data(data)
            beat_trials, interval_trials = filter_trialtype(trials,
                                                            'production')

            # ############# Assynchronies per ISI ######################
            isi1s = np.unique(np.array(beat_trials)[:, 0]).astype('int')

            ss_isi_beat = []
            as_isi_beat = []
            for i in isi1s:
                ss_beat = []
                as_beat = []
                for b, beat_trial in enumerate(beat_trials):
                    if beat_trial[0] == i:
                        if ~np.any(np.isnan(beat_trial)):
                            ssb = round((beat_trial[2] - beat_trial[1]) / \
                                        beat_trial[1], 2)
                            asb = abs(ssb)
                        # else:
                        #     ssb = np.nan
                        #     asb = np.nan
                        ss_beat.append(ssb)
                        as_beat.append(asb)
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
                        # else:
                        #     ssi = np.nan
                        #     asi = np.nan
                        ss_interval.append(ssi)
                        as_interval.append(asi)
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

            if sync_type == 'signed':
                beat = ax.boxplot(ss_isi_beat,
                                  bootstrap=100,
                                  positions=np.arange(len(x))*2. - width,
                                  widths=0.6,
                                  flierprops={'marker': '+', 'markersize': 5},
                                  patch_artist=True)
                interval = ax.boxplot(ss_isi_interval,
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
            else:
                assert sync_type == 'absolute'
                beat = ax.boxplot(as_isi_beat,
                                  bootstrap=100,
                                  positions=np.arange(len(x))*2. - width,
                                  widths=0.6,
                                  flierprops={'marker': '+', 'markersize': 5},
                                  patch_artist=True)
                interval = ax.boxplot(as_isi_interval,
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
                            np.average(as_isi_beat[j]),
                            color='w', marker='*', markeredgecolor='k')
                    ax.plot(np.average(medinterval.get_xdata()),
                            np.average(as_isi_interval[j]),
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

            if sync_type == 'signed':
                y_ticks = np.arange(-1., 4., .5)
            else:
                assert sync_type == 'absolute'
                y_ticks = np.arange(0., 4., .5)
            y_labels = np.array([str(y_tick) if (y % 2) != 0 else ''
                                 for y, y_tick in enumerate(y_ticks)])
            ax.set_yticks(y_ticks, y_labels)
            # plt.ylim([-1., 3.7])

            if (t % 2) == 0:
                ax.set_ylabel('Asynchrony')

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

            # # Aggregate data to compute the paired sample t-test
            # if task == 'Auditory Production' and sync_type == 'signed':
            #     allsub_beat_audio.append(round(ssb.mean(0), 2))
            #     allsub_intv_audio.append(round(ssi.mean(0), 2))
            # elif task == 'Visual Production' and sync_type == 'signed':
            #     allsub_beat_visual.append(round(ssb.mean(0), 2))
            #     allsub_intv_visual.append(round(ssi.mean(0), 2))
            # elif task == 'Auditory Production' and sync_type == 'absolute':
            #     allsub_beat_audio.append(round(asb.mean(0), 2))
            #     allsub_intv_audio.append(round(asi.mean(0), 2))
            # else:
            #     assert task == 'Visual Production' and sync_type == 'absolute'
            #     allsub_beat_visual.append(round(asb.mean(0), 2))
            #     allsub_intv_visual.append(round(asi.mean(0), 2))

        fig.text(.07, .905 - s * .07, 'Subject %d' % subject, ha='center',
                 fontsize=12, weight='bold')

    # plt.show()

    # Save figure
    plt.savefig(os.path.join(
        this_dir, 'production_individual_' + sync_type + '_assynch.pdf'))

    # return (allsub_beat_audio, allsub_intv_audio, allsub_beat_visual,
    #         allsub_intv_visual)


def individual_production_rts(
        subjects, this_dir, sesstype, n_sess,
        tasks = ['Auditory Production', 'Visual Production']):

    for s, subject in enumerate(subjects):
        for t, task in enumerate(tasks):
            if task not in ['Auditory Production', 'Visual Production']:
                raise NameError('Task not valid!')

            data = parse_logfile(this_dir, subject, sesstype, n_sess, task)
            trials = production_data(data)
            beat_trials, interval_trials = filter_trialtype(trials,
                                                            'production')
            # Filter necessary data
            beat_trials = [np.delete(trial, 1).tolist()
                           for trial in beat_trials]
            interval_trials = [np.delete(trial, 1).tolist()
                               for trial in interval_trials]

            # ############## Compute RT's per ISI ###################### 
            isi1s = np.unique(np.array(beat_trials)[:, 0]).astype('int')

            rt_isi1_grouped_beat = []
            for i in isi1s:
                rts_beat = []
                for beat_trial in beat_trials:
                    if beat_trial[0] == i:
                        if ~np.any(np.isnan(beat_trial)):
                            rts_beat.append(beat_trial[1])
                rt_isi1_grouped_beat.append(rts_beat)

            rt_isi1_grouped_interval = []
            for j in isi1s:
                rts_interval = []
                for interval_trial in interval_trials:
                    if interval_trial[0] == j:
                        if ~np.any(np.isnan(interval_trial)):
                            rts_interval.append(interval_trial[1])
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

            beat = ax.boxplot(rt_isi1_grouped_beat,
                              bootstrap=100,
                              positions=np.arange(len(x))*2. - width,
                              widths=0.6,
                              flierprops={'marker': '+', 'markersize': 5},
                              patch_artist=True)
            interval = ax.boxplot(rt_isi1_grouped_interval,
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
                fig.text(.5, .02, ' ISIs', size=18)

            ax.set_xticks(x*2., x_labels)

            y_ticks = np.linspace(0., 2200, 6, dtype='int')
            y_labels = np.array([str(y_tick) if (y % 2) != 0 else ''
                                 for y, y_tick in enumerate(y_ticks)])
            ax.set_yticks(y_ticks, y_labels)
            # plt.ylim([0., 2100.])

            if (t % 2) == 0:
                ax.set_ylabel('RTs (ms)')

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

        fig.text(.07, .905 - s * .07, 'Subject %d' % subject, ha='center',
                 fontsize=12, weight='bold')

    # plt.show()

    # Save figure
    plt.savefig(os.path.join(this_dir, 'production_individual_rts.pdf'))


def individual_perception(
        subjects, this_dir, sesstype, n_sess,
        tasks = ['Auditory Perception', 'Visual Perception']):
    for s, subject in enumerate(subjects):
        for t, task in enumerate(tasks):
            if task not in ['Auditory Perception', 'Visual Perception']:
                raise NameError('Task not valid!')

            data = parse_logfile(this_dir, subject, sesstype, n_sess, task)
            trials = perception_data(data)
            beat_trials, interval_trials = filter_trialtype(trials,
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


def individual_ntfd(subjects, this_dir, sesstype, n_sess,
                    tasks = ['Auditory No-Temporal Feature Discrimination',
                             'Visual No-Temporal Feature Discrimination']):

    allsub_beat_audio = []
    allsub_intv_audio = []
    allsub_beat_visual = []
    allsub_intv_visual = []
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
            beat_trials, interval_trials = filter_trialtype(trials, 'ntfd')

            beat_trials = np.array(beat_trials).ravel()
            interval_trials = np.array(interval_trials).ravel()

            # ################## Plotting ###############################
            if s == 0 and t == 0:
                fig = plt.figure(figsize=(8, 36))

            # Define subplot of bar charts and its position in the fig
            # plt.axes([left, bottom, width, height])
            ax = plt.axes([.235 + t*.42, .9 - s*.095, .3, .05])

            labels = ['beat', 'interval']
            x = [.2, .6]  # the label locations
            width = .2  # the width of the bars
            ntfd_plt = ax.bar(x,
                              [round(beat_trials.mean(0), 2),
                               round(interval_trials.mean(0), 2)],
                              width=width,
                              color=['b', 'y'],
                              yerr=[round(beat_trials.std(0), 2),
                                    round(interval_trials.std(0), 2)],
                              error_kw=dict(capsize=2), label=labels)
            ax.bar_label(ntfd_plt, padding=3)
            ax.set_xticks(x, labels)
            plt.xlim([0., .8])
            plt.ylim([0., 1000.])

            if s == 0:
                if task == 'Auditory No-Temporal Feature Discrimination':
                    ax.set_title('Auditory NTFD', weight='bold', pad=20)
                else:
                    assert task == 'Visual No-Temporal Feature Discrimination'
                    ax.set_title('Visual NTFD', weight='bold', pad=20)
                if t == 0:
                    fig.text(.25, .945, 'Error bars: SD', fontsize=12)

            # Hide the right and top spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            # Aggregate data to compute the paired sample t-test
            if task == 'Auditory No-Temporal Feature Discrimination':
                allsub_beat_audio.append(round(beat_trials.mean(0), 2))
                allsub_intv_audio.append(round(interval_trials.mean(0), 2))
            else:
                assert task == 'Visual No-Temporal Feature Discrimination'
                allsub_beat_visual.append(round(beat_trials.mean(0), 2))
                allsub_intv_visual.append(round(interval_trials.mean(0), 2))

        fig.text(.07, .9275 - s * .095, 'Subject %d' % subject, ha='center',
                 fontsize=12, weight='bold')
    fig.text(.155, .45, 'Mean of RT (ms)', ha='center',
             fontsize=12, rotation = 90)

    # plt.show()

    # Save figure
    plt.savefig(os.path.join(this_dir, 'ntfd_rt.pdf'))

    return (allsub_beat_audio, allsub_intv_audio, allsub_beat_visual,
            allsub_intv_visual)


# %%
# =========================== INPUTS ===================================

SUBJECTS = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
# SUBJECTS = [8]

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

    # ssync_audio_beat, ssync_audio_intv, \
    #     ssync_visual_beat, ssync_visual_intv = \
    # individual_production_sync(SUBJECTS, MAIN_DIR, SESSTYPE,
    #                            N_SESSIONS, 'signed', mode='mean')

    # async_audio_beat, async_audio_intv, \
    #     async_visual_beat, async_visual_intv = \
    # individual_production_sync(SUBJECTS, MAIN_DIR, SESSTYPE,
    #                            N_SESSIONS, 'absolute', mode='mean')

    # ssync_audio_beat, ssync_audio_intv, \
    #     ssync_visual_beat, ssync_visual_intv = \
    #         individual_production_sync(SUBJECTS, MAIN_DIR, SESSTYPE,
    #                                    N_SESSIONS, 'signed', mode='std')

    # async_audio_beat, async_audio_intv, \
    #     async_visual_beat, async_visual_intv = \
    #         individual_production_sync(SUBJECTS, MAIN_DIR, SESSTYPE,
    #                                    N_SESSIONS, 'absolute', mode='std')

    # Compute paired-sample t-test for production synchronies
    # tssync_audio, pssync_audio = stats.ttest_rel(
    #     ssync_audio_beat, ssync_audio_intv, alternative='less')
    # tssync_visual, pssync_visual = stats.ttest_rel(
    #     ssync_visual_beat, ssync_visual_intv, alternative='less')

    # tasync_audio, pasync_audio = stats.ttest_rel(
    #     async_audio_beat, async_audio_intv, alternative='less')
    # tasync_visual, pasync_visual = stats.ttest_rel(
    #     async_visual_beat, async_visual_intv, alternative='less')

    # ################# PRODUCTION RT'S ##############################

    individual_production_rts(SUBJECTS, MAIN_DIR, SESSTYPE, N_SESSIONS)
    # individual_production_rts(SUBJECTS, MAIN_DIR, SESSTYPE, N_SESSIONS,
    #                           mode='std')
    # individual_perception(SUBJECTS, MAIN_DIR, SESSTYPE, N_SESSIONS)
    # ntdf_audio_beat, ntfd_audio_intv, ntfd_visual_beat, ntfd_visual_intv = \
    #     individual_ntfd(SUBJECTS, MAIN_DIR, SESSTYPE, N_SESSIONS)

    # Compute paired-sample t-test for NTFD tasks
    # tntfd_audio, pntfd_audio = stats.ttest_rel(
    #     ntdf_audio_beat, ntfd_audio_intv, alternative='less')
    # tntfd_visual, pntfd_visual = stats.ttest_rel(
    #     ntfd_visual_beat, ntfd_visual_intv, alternative='less')

