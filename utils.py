import os
import glob
import csv

import numpy as np


def parse_logfile(parent_dir, subject_no, sesstype, n_sess, tasks,
                  ttl=False, concatenate=True):

    sessions = ['ses-%02d' %s for s in np.arange(1, n_sess + 1)]
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
            for task_name in tasks:
                ttag = task_name + ' - ' + sesstype
                # print('ttag: ', ttag)
                if ttag in inputs_list[8][0][9:]:
                    # print('inputs: ', inputs_list[8][0][9:])
                    liste = inputs_list
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
                    if concatenate:
                        allruns.extend(trials_info)
                    else:
                        allruns.append(trials_info)
                    if i == len(inputs_lists):
                        break

                if i == len(inputs_lists) and not allruns:
                    raise NameError(
                        'Log file for selected task does not exist!')

        if concatenate:
            allsessions.extend(allruns)
        else:
            allsessions.append(allruns)

    return allsessions


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


def ffx(audio_beat, audio_interval, visual_beat, visual_interval):
    # Inputs shape (n_subjects, n_isi, n_trials)
    # Computes mean of elements in the third dimension
    # Swaps dimensions and returns array w/ shape (n_isi, n_subjects)

    mean_audio_beat = np.array(audio_beat).mean(2)
    mean_audio_interval = np.array(audio_interval).mean(2)
    mean_visual_beat = np.array(visual_beat).mean(2)
    mean_visual_interval = np.array(visual_interval).mean(2)

    ffx_audio_beat = np.swapaxes(mean_audio_beat, 0, 1)
    ffx_audio_interval = np.swapaxes(mean_audio_interval, 0, 1)
    ffx_visual_beat = np.swapaxes(mean_visual_beat, 0, 1)
    ffx_visual_interval = np.swapaxes(mean_visual_interval, 0, 1)

    return (ffx_audio_beat, ffx_audio_interval, ffx_visual_beat,
            ffx_visual_interval)
