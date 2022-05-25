"""
Script to extract behavioral data from logfiles for the Music-SDTB project

author: Ana Luisa Pinho
e-mail: agrilopi@uwo.ca

Created: May 2022
Last update: May 2022

Compatibility: Python 3.7.11

"""
import os
import glob
import csv
import numpy as np

from matplotlib import pyplot as plt


# %%
# =========================== FUNCTIONS ================================


def parse_logfile(parent_dir, subject_no, task_name, ttl=False):
    logpath = os.path.join(parent_dir, 'sub-%02d' % subject_no, 'logfiles')
    logfiles = glob.glob(os.path.join(logpath, '*.xpd'))
    logfiles.sort()
    inputs_lists = [[line for line in csv.reader(open(logfile), delimiter=',')]
                    for logfile in logfiles]
    # Pick log file of selected task
    for i, inputs_list in enumerate(inputs_lists, 1):
        if inputs_list[6][0][9:] == task_name:
            liste = inputs_list
            break
        if i == len(inputs_lists):
            raise NameError('Log file for selected task does not exist!')
    # Extract trial information from log file
    for r, row in enumerate(liste):
        if row[0] == str(subject_no):
            break
        else:
            continue
    if not ttl:
        trials_info = liste[r+1:]
    else:
        trials_info = liste[r:]

    return trials_info


def filter_trialtype(trs, category):
    beat = [tr[1:] for tr in trs if tr[0][:4] == 'beat']
    interval = [tr[1:] for tr in trs if tr[0][:8] == 'interval']

    if category in ['production', 'ntfd']:
        beat = [list(map(int, b)) for b in beat]
        interval = [list(map(int, i)) for i in interval]
    else:
        assert category == 'perception'
        beat = [[int(b[0]), int(b[1]), int(b[2]), b[3]] for b in beat]
        interval = [[int(i[0]), int(i[1]), int(i[2]), i[3]] for i in interval]

    return beat, interval


def perception_frequencies(isi_diff_condition, condition_trials, min_val,
                           max_val, n_steps):
    time_diff, step = np.linspace(min_val, max_val, num=n_steps, dtype=int,
                                  retstep=True)
    print('Step: ', step)
    frequencies = []
    for s in np.arange(len(time_diff) - 1):
        responses = []
        for idb, btr in zip(isi_diff_condition, condition_trials):
            if time_diff[s] <= idb < time_diff[s+1]:
                responses.append(btr[3])
        occurr = responses.count('o')
        if occurr == 0:
            percent_occurr = 0
        else:
            percent_occurr = (occurr/len(responses))*100
        frequencies.append(percent_occurr)

    time_val = time_diff + step/2
    time_val = time_val[:-1]

    return time_diff, time_val, frequencies


def production_results(subjects, this_dir,
                       tasks = ['Auditory Production', 'Visual Production']):
    for s, subject in enumerate(subjects):
        for t, task in enumerate(tasks):
            if task not in ['Auditory Production', 'Visual Production']:
                raise NameError('Task not valid!')
            data = parse_logfile(this_dir, subject, task)
            trials = []
            # good_trials = []
            for dt, datum in enumerate(data):
                if datum[4] == 'isi_1':
                    condition = datum[3]
                    isi1 = datum[6]
                    # isi2 = data[d+2][6]
                    # isi3 = data[d+4][6]
                    # isi4 = data[d+6][6]
                    if data[dt+8][4] == 'feedback' and data[dt+8][8] == 'o':
                        rt = data[dt+8][7]
                    elif data[dt+8][4] == 'feedback' and \
                         data[dt+8][8] == 'None':
                        continue
                    else:
                        raise ValueError('No feedback entry!')
                    trials.append([condition, isi1, rt])
                    # if isi1 != isi3:
                    #     continue
                    # if condition[:4] == 'beat' and isi2 != isi4:
                    #     continue
                    # good_trials.append([condition, isi1])

            beat_trials, interval_trials = filter_trialtype(trials,
                                                            'production')

            # #### Synchronies ####
            ssb = np.array([round((bt[0]-bt[1])/bt[0], 2)
                            for bt in beat_trials])
            ssi = np.array([round((it[0]-it[1])/it[0], 2)
                            for it in interval_trials])

            asb = np.array([abs(ssb) for ssb in ssb])
            asi = np.array([abs(ssi) for ssi in ssi])

            # #### Plotting ####
            if s == 0 and t == 0:
                fig = plt.figure(figsize=(8, 12))

            # Define subplot of bar charts and its position in the fig
            # plt.axes([left, bottom, width, height])
            ax = plt.axes([.2 + t*.425, .75 - s*.23, .36, .2])

            labels = ['beat', 'interval']
            x = np.arange(len(labels))  # the label locations
            width = 0.35  # the width of the bars
            signed = ax.bar(x - width/2,
                            [round(ssb.mean(0), 2), round(ssi.mean(0), 2)],
                            width=width, yerr=[ssb.std(0), ssi.std(0)],
                            error_kw=dict(capsize=2), label='Signed assynchrony')
            absolute = ax.bar(x + width/2,
                              [round(asb.mean(0), 2), round(asi.mean(0), 2)],
                              width=width, yerr=[asb.std(0), asi.std(0)],
                              error_kw=dict(capsize=2),
                              label='Absolute assynchrony')
            plt.ylim([-.6, .7])
            # ax.bar_label(signed, padding=3)
            # ax.bar_label(absolute, padding=3)

            # ax.set_ylabel('Mean of assynchrony (ms)')
            if s == 0:
                ax.set_title(task)
                if t == 0:
                    ax.legend(frameon=False, loc = 'lower left',
                              prop={'size': 8})
                    fig.text(.211, .79, 'Error bars: SD', fontsize=9)
            if s == len(subjects) - 1:
                ax.set_xticks(x, labels)
            else:
                ax.set_xticks(x, '')
                ax.tick_params(bottom=False)
                ax.spines['bottom'].set_visible(False)

            # Hide the right and top spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        fig.text(.06, .862 - s * .233, 'Subject %d' % subject, ha='center',
                 fontsize=12, weight='bold')
    fig.text(.13, .435, 'Mean of assynchrony', ha='center',
             fontsize=14, rotation = 90)

    # plt.show()

    # Save figure
    plt.savefig(os.path.join(this_dir, 'production_assynchronies.pdf'))


def perception_results(subjects, this_dir,
                       tasks = ['Auditory Perception', 'Visual Perception'],
                       step_size=5):
    for s, subject in enumerate(subjects):
        for t, task in enumerate(tasks):
            if task not in ['Auditory Perception', 'Visual Perception']:
                raise NameError('Task not valid!')
            data = parse_logfile(this_dir, subject, task)
            trials = []
            for dt, datum in enumerate(data):
                if datum[4] == 'isi_1':
                    condition = datum[3]
                    isi1 = datum[6]
                    isi5 = data[dt+8][6]
                    if data[dt+10][4] == 'feedback' and \
                       data[dt+10][8] in ['o', 'p']:
                        rt = data[dt+10][7]
                        answer = data[dt+10][8]
                    elif data[dt+10][4] == 'feedback' and \
                         data[dt+10][8] == 'None':
                        continue
                    else:
                        raise ValueError('No feedback entry!')
                    trials.append([condition, isi1, isi5, rt, answer])
            beat_trials, interval_trials = filter_trialtype(trials,
                                                            'perception')
            isi_diff_beat = [bt[1] - bt[0] for bt in beat_trials]
            isi_diff_interval = [it[1] - it[0] for it in interval_trials]

            x_labels, x_vals, y_beat_vals = perception_frequencies(
                isi_diff_beat, beat_trials, -400, 400, step_size)
            _, _, y_interval_val = perception_frequencies(
                isi_diff_interval, interval_trials, -400, 400, step_size)

            # #### Plotting ####
            if s == 0 and t == 0:
                fig = plt.figure(figsize=(8, 12))

            # Define subplot of bar charts and its position in the fig
            # plt.axes([left, bottom, width, height])
            ax = plt.axes([.2 + t*.41, .75 - s*.22, .36, .2])

            ax.plot(x_vals, y_beat_vals, marker='o', markersize=3,
                    color='b', label='Beat')
            ax.plot(x_vals, y_interval_val, marker='o', markersize=3,
                    color='y', label='Interval')

            # X axis
            plt.xticks(x_labels)
            if s != len(subjects) - 1:
                plt.xticks([])
                ax.tick_params(bottom=False)
                ax.spines['bottom'].set_visible(False)

            # Y axis
            y_vals = np.arange(0, 1.5, .5) * 100
            y_labels = ['shorter', 'equal', 'longer']
            if t == 0:
                ax.set_yticks(y_vals, y_labels)
            else:
                ax.set_yticks(y_vals, '')

            # Title per column with Task name and legend
            if s == 0:
                ax.set_title(task)
                if t == 0:
                    ax.legend(frameon=False, loc = 'upper left',
                              prop={'size': 10})

            # Hide the right and top spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        fig.text(.055, .875 - s * .25, 'Subject %d' % subject, ha='center',
                 fontsize=10, weight='bold')

    fig.text(.5, .02, 'Time difference (ms)', fontsize=12)
    fig.text(.1, .415, '% of responses', fontsize=12, rotation=90)

    # plt.show()

    # Save figure
    plt.savefig(os.path.join(this_dir, 'perception_responses.pdf'))


def ntfd_results(subjects, this_dir,
                 tasks = ['Auditory No-Temporal Feature Discrimination',
                          'Visual No-Temporal Feature Discrimination']):
    for s, subject in enumerate(subjects):
        for t, task in enumerate(tasks):
            if task not in ['Auditory No-Temporal Feature Discrimination',
                            'Visual No-Temporal Feature Discrimination']:
                raise NameError('Task not valid!')
            data = parse_logfile(this_dir, subject, task)
            if subject == 2 and \
               task == 'Visual No-Temporal Feature Discrimination':
                data = data[:476]
            trials = []
            for dt, datum in enumerate(data):
                if datum[4] == 'feedback':
                    condition = datum[3]
                    if datum[8] in ['o', 'p']:
                        rt = datum[7]
                    elif datum[8] == 'None':
                        continue
                    else:
                        raise ValueError('No feedback entry!')
                    trials.append([condition, rt])

            beat_trials, interval_trials = filter_trialtype(trials, 'ntfd')
            beat_trials = np.array(beat_trials).ravel()
            interval_trials = np.array(interval_trials).ravel()

            # #### Plotting ####
            if s == 0 and t == 0:
                fig = plt.figure(figsize=(8, 6))

            # Define subplot of bar charts and its position in the fig
            # plt.axes([left, bottom, width, height])
            ax = plt.axes([.2 + t*.425, .55 - s*.4, .36, .3])

            labels = ['beat', 'interval']
            x = [.2, .6]  # the label locations
            width = 0.2  # the width of the bars
            ax.bar(x,
                   [round(beat_trials.mean(0), 2),
                    round(interval_trials.mean(0), 2)],
                   width=width,
                   color=['b', 'y'],
                   yerr=[round(beat_trials.std(0), 2),
                         round(interval_trials.std(0), 2)],
                   error_kw=dict(capsize=2), label=labels)
            ax.set_xticks(x, labels)
            plt.xlim([0., .8])
            plt.ylim([0., 800.])

            if s == 0:
                ax.set_title(task, fontsize=10, pad=20)
                if t == 0:
                    fig.text(.211, .835, 'Error bars: SD', fontsize=9)

            # Hide the right and top spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        fig.text(.065, .7 - s * .4, 'Subject %d' % subject, ha='center',
                 fontsize=12, weight='bold')
    fig.text(.132, .42, 'Mean of RT (ms)', ha='center',
             fontsize=12, rotation = 90)

    # plt.show()

    # Save figure
    plt.savefig(os.path.join(this_dir, 'ntfd_rt.pdf'))


# %%
# =========================== INPUTS ===================================

SUBJECTS = [1, 2, 3, 4]
# SUBJECTS = [2, 4]
# SUBJECTS = [4]

# TASKS = ['Auditory Production',
#          'Auditory Perception',
#          'Auditory No-Temporal Feature Discrimination',
#          'Visual Production',
#          'Visual Perception',
#          'Visual No-Temporal Feature Discrimination']

# TASKS = ['Auditory Perception']


# %%
# ========================= PARAMETERS =================================

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))

# %%
# ============================ RUN =====================================

if __name__ == "__main__":
    # production_results(SUBJECTS, MAIN_DIR)
    perception_results(SUBJECTS, MAIN_DIR, step_size=5)
    # ntfd_results(SUBJECTS, MAIN_DIR)




