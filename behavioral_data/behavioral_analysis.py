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


# %%
# =========================== INPUTS ===================================

# subjects = [1, 2, 3, 4]
subjects = [1]

# tasks = ['Auditory Production',
#          'Auditory Perception',
#          'Auditory No-Temporal Feature Discrimination',
#          'Visual Production',
#          'Visual Perception',
#          'Visual No-Temporal Feature Discrimination']

tasks = ['Auditory Production']

# %%
# ========================= PARAMETERS =================================

this_dir = os.path.dirname(os.path.abspath(__file__))

# Subject, task, condition, reaction time, response

# %%
# ============================ RUN =====================================

if __name__ == "__main__":
    for subject in subjects:
        for task in tasks:
            data = parse_logfile(this_dir, subject, task)
            # ****************** Production Tasks **********************
            # trials = np.empty((0, 3))
            trials = []
            # good_trials = []
            for d, datum in enumerate(data):
                if datum[4] == 'isi_1':
                    condition = datum[3]
                    isi1 = datum[6]
                    # isi2 = data[d+2][6]
                    # isi3 = data[d+4][6]
                    # isi4 = data[d+6][6]
                    if data[d+8][4] == 'feedback' and data[d+8][8] == 'o':
                        rt = data[d+8][7]
                    else:
                        raise ValueError('No feedback entry!')
                    # trials = np.append(trials, [[condition, isi1, rt]], axis=0)
                    trials.append([condition, isi1, rt])
                    # if isi1 != isi3:
                    #     continue
                    # if condition[:4] == 'beat' and isi2 != isi4:
                    #     continue
                    # good_trials.append([condition, isi1])
            beat_trials = [trial[1:] for trial in trials
                           if trial[0][:4] == 'beat']
            beat_trials = [list(map(int, beat_trial))
                           for beat_trial in beat_trials]
            interval_trials = [trial[1:] for trial in trials
                               if trial[0][:8] == 'interval']
            interval_trials = [list(map(int, interval_trial))
                               for interval_trial in interval_trials]
            # #### Synchronies ####
            signed_sync_beat = [round((bt[0]-bt[1])/bt[0], 2)
                                for bt in beat_trials]
            signed_sync_interval = [round((it[0]-it[1])/it[0], 2)
                                    for it in interval_trials]

            abs_sync_beat = [abs(ssb) for ssb in signed_sync_beat]
            abs_sync_interval = [abs(ssi) for ssi in signed_sync_interval]

            ssb_mean = round(np.mean(signed_sync_beat), 2)
            asb_mean = round(np.mean(abs_sync_beat), 2)

            ssi_mean = round(np.mean(signed_sync_interval), 2)
            asi_mean = round(np.mean(abs_sync_interval), 2)

            ssb_std = round(np.std(signed_sync_beat), 2)
            asb_std = round(np.std(abs_sync_beat), 2)

            ssi_std = round(np.std(signed_sync_interval), 2)
            asi_std = round(np.std(abs_sync_interval), 2)

            labels = ['beat', 'interval']
            x = np.arange(len(labels))  # the label locations
            width = 0.35  # the width of the bars

            fig, ax = plt.subplots()
            ax.legend(frameon=False)
            signed = ax.bar(x - width/2, [ssb_mean, ssi_mean], width,
                            yerr=[ssb_std, ssi_std],
                            error_kw=dict(capsize=2),
                            label='Signed synchrony')
            absolute = ax.bar(x + width/2, [asb_mean, asi_mean], width,
                              yerr=[asb_std, asi_std],
                              error_kw=dict(capsize=2),
                              label='Absolute synchrony')

            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_ylabel('Mean of assynchrony (ms)')
            ax.set_title(tasks[0])
            ax.set_xticks(x, labels)
            ax.legend(frameon=False)

            # ax.bar_label(signed, padding=3)
            # ax.bar_label(absolute, padding=3)

            # Hide the right and top spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            fig.tight_layout()

            plt.show()




