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


def production_synchronies(subjects, this_dir, sesstype, n_sess, sync_type,
                           tasks = ['Auditory Production', 'Visual Production']):
    for s, subject in enumerate(subjects):
        for t, task in enumerate(tasks):
            if task not in ['Auditory Production', 'Visual Production']:
                raise NameError('Task not valid!')
            data = parse_logfile(this_dir, subject, sesstype, n_sess, task)

            trials = []
            for dt, datum in enumerate(data):
                if datum[5] == 'interval_1':
                    condition = datum[4]
                    real_isi1 = int(datum[9])
                    if data[dt+8][5] == 'feedback' and data[dt+8][11] == 'o':
                        rt = int(data[dt+7][7]) + int(data[dt+8][10])
                    elif data[dt+8][5] == 'feedback' and \
                         data[dt+8][10] == 'None':
                        continue
                    else:
                        raise ValueError('No feedback entry!')
                    trials.append([condition, real_isi1, rt])

            beat_trials, interval_trials = filter_trialtype(trials,
                                                            'production')
            # ################# Synchronies ############################
            ssb = np.array([round((bt[1]-bt[0])/bt[0], 2)
                            for bt in beat_trials])
            ssi = np.array([round((it[1]-it[0])/it[0], 2)
                            for it in interval_trials])

            # #### Plotting #####
            if s == 0 and t == 0:
                fig = plt.figure(figsize=(8, 36))

            # Define subplot of bar charts and its position in the fig
            # plt.axes([left, bottom, width, height])
            ax = plt.axes([.235 + t*.42, .9 - s*.095, .3, .05])

            labels = ['beat', 'interval']
            x = np.arange(len(labels))  # the label locations

            if sync_type == 'signed':
                signed = ax.bar([x[0] + .4, x[1] - .4],
                       [round(ssb.mean(0), 2), round(ssi.mean(0), 2)],
                       width=.15, yerr=[ssb.std(0), ssi.std(0)],
                       error_kw=dict(capsize=2), facecolor='tab:blue',
                       label='Signed Asynchrony')
                ax.bar_label(signed, padding=3)
                plt.ylim([-.8, .8])
            else:
                assert sync_type == 'absolute'

                asb = np.array([abs(ssb) for ssb in ssb])
                asi = np.array([abs(ssi) for ssi in ssi])

                absolute = ax.bar([x[0] + .4, x[1] - .4],
                       [round(asb.mean(0), 2), round(asi.mean(0), 2)],
                       width=.15, yerr=[asb.std(0), asi.std(0)],
                       error_kw=dict(capsize=2), facecolor='tab:orange',
                       label='Absolute Asynchrony')
                ax.bar_label(absolute, padding=3)
                plt.ylim([-.8, .8])

            # ax.set_ylabel('Mean of assynchrony (ms)')
            if s == 0:
                ax.set_title(task, pad=30, weight='bold')
                if t == 0:
                    if sync_type == 'signed':
                        ax.legend(frameon=False, loc = 'upper left',
                                  prop={'size': 12})
                        fig.text(.25, .91, 'Error bars: SD', fontsize=12)
                    else:
                        assert sync_type == 'absolute'
                        ax.legend(frameon=False, loc = 'upper left',
                                  prop={'size': 12})
                        fig.text(.25, .91, 'Error bars: SD', fontsize=12)

            ax.set_xticks([x[0] + .4, x[1] - .4], labels)
            # if s == len(subjects) - 1:
            #     ax.set_xticks([x[0] + .4, x[1] - .4], labels)
            # else:
            #     ax.set_xticks([x[0] + .4, x[1] - .4], '')
            #     ax.tick_params(bottom=False)
            #     ax.spines['bottom'].set_visible(False)

            # Hide the right and top spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        fig.text(.07, .9275 - s * .095, 'Subject %d' % subject, ha='center',
                 fontsize=12, weight='bold')
    fig.text(.15, .41, 'Mean of Asynchrony', ha='center',
             fontsize=14, rotation = 90)

    # plt.show()

    # Save figure
    plt.savefig(os.path.join(
        this_dir, 'production_' + sync_type + '_assynchronies.pdf'))


def production_isi_rts(subjects, this_dir, sesstype, n_sess,
                       tasks = ['Auditory Production', 'Visual Production'],
                       mode = 'mean'):
    for s, subject in enumerate(subjects):
        for t, task in enumerate(tasks):
            if task not in ['Auditory Production', 'Visual Production']:
                raise NameError('Task not valid!')
            data = parse_logfile(this_dir, subject, sesstype, n_sess, task)

            trials = []
            for dt, datum in enumerate(data):
                if datum[5] == 'interval_1':
                    condition = datum[4]
                    theoretical_isi1 = int(datum[8])
                    if data[dt+8][5] == 'feedback' and data[dt+8][11] == 'o':
                        rt = int(data[dt+7][7]) + int(data[dt+8][10])
                    elif data[dt+8][5] == 'feedback' and \
                         data[dt+8][10] == 'None':
                        rt = np.nan
                    else:
                        raise ValueError('No feedback entry!')
                    trials.append([condition, theoretical_isi1, rt])

            beat_trials, interval_trials = filter_trialtype(trials,
                                                            'production')
            isi1s = np.unique(np.array(beat_trials)[:, 0]).astype('int')

            rt_isi1_grouped_beat = []
            for i in isi1s:
                rts_beat = []
                for beat_trial in beat_trials:
                    if beat_trial[0] == i:
                        rts_beat.append(beat_trial[1])
                rt_isi1_grouped_beat.append(rts_beat)

            rt_isi1_grouped_interval = []
            for j in isi1s:
                rts_interval = []
                for interval_trial in interval_trials:
                    if interval_trial[0] == j:
                        rts_interval.append(interval_trial[1])
                rt_isi1_grouped_interval.append(rts_interval)

            if mode == 'mean':
                rt_isi1_beat = np.around(
                    np.nanmean(rt_isi1_grouped_beat, axis=1), decimals=0)
                rt_isi1_interval = np.around(
                    np.nanmean(rt_isi1_grouped_interval, axis=1), decimals=0)
            else:
                assert mode == 'std'
                rt_isi1_beat = np.around(
                    np.nanstd(rt_isi1_grouped_beat, axis=1), decimals=0)
                rt_isi1_interval = np.around(
                    np.nanstd(rt_isi1_grouped_interval, axis=1), decimals=0)

            # #### Plotting #####
            if s == 0 and t == 0:
                fig = plt.figure(figsize=(8, 36))

            # Define subplot of bar charts and its position in the fig
            # plt.axes([left, bottom, width, height])
            ax = plt.axes([.235 + t*.42, .9 - s*.095, .3, .05])

            x_labels = [str(k) for k in isi1s]
            x = np.arange(len(x_labels))  # the label locations
            width = 0.35  # the width of the bars

            beat_plot = ax.bar(x - width/2, rt_isi1_beat, width=width,
                               label='Beat')

            interval_plot = ax.bar(x + width/2, rt_isi1_interval, width=width,
                                   label='Interval')

            ax.bar_label(beat_plot, padding=3, fontsize=4)
            ax.bar_label(interval_plot, padding=3, fontsize=4)
            if mode == 'mean':
                plt.ylim([0., 850])
            else:
                assert mode == 'std'
                plt.ylim([0., 400])

            if s == 0:
                ax.set_title(task, pad=20, weight='bold')
                if t == 0:
                    ax.legend(frameon=False, loc='upper left',
                              prop={'size': 10})

            ax.set_xticks(x, x_labels)
            # if s == len(subjects) - 1:
            #     ax.set_xticks(x, x_labels)
            # else:
            #     ax.set_xticks(x, '')
            #     ax.tick_params(bottom=False)
            #     ax.spines['bottom'].set_visible(False)

            # Hide the right and top spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        fig.text(.07, .9275 - s * .095, 'Subject %d' % subject, ha='center',
                 fontsize=12, weight='bold')

    fig.text(.5, .02, 'ISI1=ISI (ms)', fontsize=12)
    if mode == 'mean':
        fig.text(.16, .45, 'Mean of RTs (ms)',
                 ha='center', fontsize=14, rotation = 90)
        # Save figure
        plt.savefig(os.path.join(this_dir, 'production_rts_isis_mean.pdf'))
    else:
        assert mode == 'std'
        fig.text(.16, .4, 'Standard Deviation of RTs (ms)',
                 ha='center', fontsize=14, rotation = 90)
        # Save figure
        plt.savefig(os.path.join(this_dir, 'production_rts_isis_std.pdf'))


def perception_results(subjects, this_dir, sesstype, n_sess,
                       tasks = ['Auditory Perception', 'Visual Perception']):
    for s, subject in enumerate(subjects):
        for t, task in enumerate(tasks):
            if task not in ['Auditory Perception', 'Visual Perception']:
                raise NameError('Task not valid!')
            data = parse_logfile(this_dir, subject, sesstype, n_sess, task)
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

            beat_trials, interval_trials = filter_trialtype(trials,
                                                            'perception')

            isi_diff_beat = [round((bt[1] - bt[0]) / bt[0], 2)
                             for bt in beat_trials]
            isi_diff_interval = [round((it[1] - it[0]) / it[0], 2)
                                 for it in interval_trials]

            x_vals, y_beat_vals = perception_frequencies(isi_diff_beat,
                                                         beat_trials)
            _, y_interval_val = perception_frequencies(isi_diff_interval,
                                                       interval_trials)

            # #### Plotting ####
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


def ntfd_results(subjects, this_dir, sesstype, n_sess,
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

            beat_trials, interval_trials = filter_trialtype(trials, 'ntfd')
            beat_trials = np.array(beat_trials).ravel()
            interval_trials = np.array(interval_trials).ravel()

            # #### Plotting ####
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

        fig.text(.07, .9275 - s * .095, 'Subject %d' % subject, ha='center',
                 fontsize=12, weight='bold')
    fig.text(.155, .45, 'Mean of RT (ms)', ha='center',
             fontsize=12, rotation = 90)

    # plt.show()

    # Save figure
    plt.savefig(os.path.join(this_dir, 'ntfd_rt.pdf'))


# %%
# =========================== INPUTS ===================================

SUBJECTS = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13]
# SUBJECTS = [3]

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
    production_synchronies(SUBJECTS, MAIN_DIR, SESSTYPE, N_SESSIONS, 'signed')
    production_synchronies(SUBJECTS, MAIN_DIR, SESSTYPE, N_SESSIONS,
                           'absolute')
    production_isi_rts(SUBJECTS, MAIN_DIR, SESSTYPE, N_SESSIONS, mode='mean')
    production_isi_rts(SUBJECTS, MAIN_DIR, SESSTYPE, N_SESSIONS, mode='std')
    perception_results(SUBJECTS, MAIN_DIR, SESSTYPE, N_SESSIONS)
    ntfd_results(SUBJECTS, MAIN_DIR, SESSTYPE, N_SESSIONS)




