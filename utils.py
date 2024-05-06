import os
import glob
import csv

import numpy as np


def parse_logfile(parent_dir, subject_no, sesstypes, task, n_trials,
                  ttl=False, concatenate=True, sessions=None):

    allsessions = []
    for sesstype in sesstypes:
        sesstype_path = os.path.join(parent_dir, 'sub-%02d' % subject_no,
                                     sesstype + 's')
        # If the participant only did behaviour
        if sesstype == 'imaging_session' and not os.path.exists(sesstype_path):
            break

        # Do not consider behavioural data from imaging session of sub-04
        # Comment to extract paradigm descriptors
        elif sesstype == 'imaging_session' and subject_no == 4:
            break
        elif sessions is None:
            sessions = []
            sessions = os.listdir(sesstype_path)
            sessions.sort()
        else:
            assert sessions is not None

        for session in sessions:
            logpath = os.path.join(sesstype_path, session)
            logfiles = glob.glob(os.path.join(logpath, '*.xpd'))
            logfiles.sort()
            inputs_lists = [[line for line in csv.reader(open(logfile),
                                                         delimiter=',')]
                            for logfile in logfiles]

            # Pick log files of selected task
            allruns = []
            for i, inputs_list in enumerate(inputs_lists, 1):
                sesstype_tag = sesstype.replace('_', ' ')
                ttag = task + ' - ' + sesstype_tag
                # print('ttag: ', ttag)
                if ttag in inputs_list[8][0][9:] or \
                   ttag in inputs_list[9][0][9:]:
                    liste = inputs_list
                    # Extract trial information from log file
                    for r, row in enumerate(liste):
                        if row[0] == str(subject_no):
                            break
                        else:
                            continue
                    # Remove TTL line or not?
                    if not ttl:
                        trials_info = liste[r+1:]
                    else:
                        trials_info = liste[r:]

                    # For behavioral sessions
                    if sesstype == 'behavioral_session':
                        trials_info = trials_info[:-1]
                    # For imaging sessions
                    else:
                        assert sesstype == 'imaging_session'
                        if ttag in inputs_list[8][0][9:]:
                            for li, line in enumerate(trials_info):
                                if li and \
                                   trials_info[li-1][3] == str(n_trials) and \
                                   line[4] == 'fixcross':
                                    break
                                else:
                                    continue
                            trials_info = trials_info[:li+2]
                        elif ttag in inputs_list[9][0][9:]:
                            for li, line in enumerate(trials_info):
                                if line[4] == 'baseline' and \
                                   trials_info[li-2][3] == str(n_trials):
                                    break
                                else:
                                    continue
                            trials_info = trials_info[li+1:]
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


def ffx(audio_beat, audio_interval, visual_beat, visual_interval,
        metric='mean'):
    # Inputs shape (n_subjects, n_isi, n_trials)
    # Computes mean of elements in the third dimension
    # Swaps dimensions and returns array w/ shape (n_isi, n_subjects)

    if metric == 'mean':
        ffx_audio_beat = [[np.mean(ab2)
                           for ab2 in ab1] for ab1 in audio_beat]
        ffx_audio_interval = [[np.mean(ai2)
                               for ai2 in ai1] for ai1 in audio_interval]
        ffx_visual_beat = [[np.mean(vb2)
                             for vb2 in vb1] for vb1 in visual_beat]
        ffx_visual_interval = [[np.mean(vi2)
                                for vi2 in vi1] for vi1 in visual_interval]
    else:
        assert metric == 'std'
        ffx_audio_beat = [[np.std(ab2)
                           for ab2 in ab1] for ab1 in audio_beat]
        ffx_audio_interval = [[np.std(ai2)
                               for ai2 in ai1] for ai1 in audio_interval]
        ffx_visual_beat = [[np.std(vb2)
                             for vb2 in vb1] for vb1 in visual_beat]
        ffx_visual_interval = [[np.std(vi2)
                                for vi2 in vi1] for vi1 in visual_interval]

    ffx_audio_beat = np.swapaxes(ffx_audio_beat, 0, 1)
    ffx_audio_interval = np.swapaxes(ffx_audio_interval, 0, 1)
    ffx_visual_beat = np.swapaxes(ffx_visual_beat, 0, 1)
    ffx_visual_interval = np.swapaxes(ffx_visual_interval, 0, 1)

    return (ffx_audio_beat, ffx_audio_interval, ffx_visual_beat,
            ffx_visual_interval)


def resize_arrays(arr):
    """
    Resize numpy arrays when there is less trials per isi because
    the participant only did the behavioral sessions
    """
    maxlength = np.amax([np.array(arr0).shape[0] for arr0 in arr])
    new_arr = [
        np.append(arr0, np.repeat('n/a', maxlength - len(arr0))).tolist()
        if len(arr0) < maxlength else arr0 for arr0 in arr]

    return new_arr
