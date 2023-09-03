"""
Extraction of paradigm descriptors for the Music-SDTB Tasks

Author: Ana Luisa Pinho
Email: agrilopi@uwo.ca

Creation: January 2023
Last Update: September 2023

Compatibility: Python 3.10.4

"""

import sys

import os
import glob
import csv

import numpy as np

import itertools, collections

# setting path
sys.path.append('../')
# importing
from utils import parse_logfile


# %%
# ========================== FUNCTIONS =================================


def convert(strnum):
    converted_num = round(float(strnum)/1e3, 3)

    return converted_num


def consume(iterator, n):
    '''
    Advance the iterator n-steps ahead.
    If n is none, consume entirely.
    '''
    collections.deque(itertools.islice(iterator, n), maxlen=0)


def merge_rest_conditions(onsets, durations, names):
    idx = 0
    res = collections.defaultdict(list)

    # Grouping Consecutives
    for key, sub in itertools.groupby(names):
        ele = len(list(sub))

        # Append strt index, and till index
        res[key].append((idx, idx + ele - 1))
        idx += ele

    valid_pairs = [pair for pair in dict(res)['rest']
                   if pair[0] != pair[1]]

    di = {}
    di = dict(valid_pairs)
    first_idxs = list(di.keys())

    new_onsets = []
    new_durations = []
    new_names = []
    sequence = np.arange(len(names)).__iter__()
    for nm in sequence:
        new_onsets.append(onsets[nm])
        new_names.append(names[nm])
        if nm in first_idxs:
            durations_subset = [float(d) for d in durations[nm:di[nm]+1]]
            new_duration = round(sum(durations_subset), 3)
            new_durations.append(str(new_duration))
            # Update list under iteration, i.e. 'sequence' list
            consume(sequence, di[nm]-nm)
        else:
            new_durations.append(durations[nm])

    return new_onsets, new_durations, new_names


def extraction_dbb(data, cat, header, events_dir, ttl = True, flag=0,
                   merge_decision=False, merge_rest=False):

    for ses_datum in data:
        for run_datum in ses_datum:
            onset = []
            duration = []
            trial_type = []
            trial_type_md = []
            if ttl:
                assert run_datum[0][4] == 'ttl'
                offset = convert(run_datum[0][6])
                onset.append(str(float(0)))
                initial_rest = convert(run_datum[0][7])
                duration.append(str(initial_rest))
                trial_type.append('rest')
                trial_type_md.append('rest')
                run_datum = run_datum[1:]
            subject_number = int(run_datum[0][0])
            session_number = int(run_datum[0][1])
            run_number = int(run_datum[0][2])

            for rw, row in enumerate(run_datum):
                if rw == 0 or \
                   (run_datum[rw][3] == '1' and run_datum[rw-1][3] == '30') \
                   or ((run_datum[rw-1][4] in ['fixcross', 'baseline']) and
                    row[4] not in 'final_baseline'):

                    if cat == 'Production':
                        feedback_onset = convert(run_datum[rw+9][6])
                        feedback_duration = convert(run_datum[rw+9][7])
                        if run_datum[rw+9][10] == 'None':
                            rt = None
                        else:
                            rt = convert(run_datum[rw+9][10])
                    else:
                        assert cat in ['Perception',
                                       'No-Temporal Feature Discrimination']
                        feedback_onset = convert(run_datum[rw+11][6])
                        feedback_duration = convert(run_datum[rw+11][7])
                        if run_datum[rw+11][10] == 'None':
                            rt = None
                        else:
                            rt = convert(run_datum[rw+11][10])

                    # Onset and duration for the encoding
                    onset_encod = round(convert(row[6]) - offset, 3)
                    duration_encod = round(feedback_onset - convert(row[6]), 3)
                    onset.append(str(onset_encod))
                    duration.append(str(duration_encod))

                    # Onset and duration for the decision, response and rest
                    onset_decis = round(feedback_onset - offset, 3)
                    onset.append(str(onset_decis))

                    if rt is None:
                        # Duration for the decision
                        duration_decis = feedback_duration
                        duration.append(str(duration_decis))
                    else:
                        # Duration for the decision
                        duration_decis = round(rt + EVENT_DURATION, 3)
                        duration.append(str(duration_decis))

                        # Onset and duration for the rest
                        onset_rest = round(feedback_onset + rt +
                                           EVENT_DURATION - offset, 3)
                        onset.append(str(onset_rest))
                        duration_rest = round(feedback_duration - rt -
                                              EVENT_DURATION, 3)
                        duration.append(str(duration_rest))

                    # Trial types for all conditions
                    if row[4][:4] == 'beat' and row[5][:4] == 'beep':
                        trial_type.append('auditory_beat_encoding')
                        trial_type_md.append('auditory_beat_encoding')
                        trial_type.append('auditory_beat_decision')
                    elif row[4][:4] == 'beat' and row[5][:4] == 'rect':
                        trial_type.append('visual_beat_encoding')
                        trial_type_md.append('visual_beat_encoding')
                        trial_type.append('visual_beat_decision')
                    elif row[4][:4] == 'inte' and row[5][:4] == 'beep':
                        trial_type.append('auditory_interval_encoding')
                        trial_type_md.append('auditory_interval_encoding')
                        trial_type.append('auditory_interval_decision')
                    elif row[4][:4] == 'inte' and row[5][:4] == 'rect':
                        trial_type.append('visual_interval_encoding')
                        trial_type_md.append('visual_interval_encoding')
                        trial_type.append('visual_interval_decision')
                    elif row[4][:4] == 'rand' and row[5][:4] == 'beep':
                        trial_type.append('auditory_random_encoding')
                        trial_type_md.append('auditory_random_encoding')
                        trial_type.append('auditory_random_decision')
                    elif row[4][:4] == 'rand' and row[5][:4] == 'rect':
                        trial_type.append('visual_random_encoding')
                        trial_type_md.append('visual_random_encoding')
                        trial_type.append('visual_random_decision')
                    else:
                        raise NameError(
                            'Trial type does not exist for this trial!')

                    trial_type_md.append('decision')
                    if rt is not None:
                        trial_type.append('rest')
                        trial_type_md.append('rest')

                elif row[4] in ['fixcross', 'baseline', 'final_baseline']:
                    onset_rest = round(convert(row[6]) - offset, 3)
                    onset.append(str(onset_rest))
                    duration_rest = convert(row[7])
                    duration.append(str(duration_rest))
                    trial_type.append('rest')
                    trial_type_md.append('rest')
                else:
                    pass

            if merge_rest:
                if merge_decision:
                    onset_mr, duration_mr, trial_type_md_mr = \
                        merge_rest_conditions(onset, duration, trial_type_md)
                    del onset
                    del duration
                    del trial_type_md
                    onset = onset_mr
                    duration = duration_mr
                    trial_type_md = trial_type_md_mr
                else:
                    onset_mr, duration_mr, trial_type_mr = \
                        merge_rest_conditions(onset, duration, trial_type)
                    del onset
                    del duration
                    del trial_type_md
                    onset = onset_mr
                    duration = duration_mr
                    trial_type = trial_type_mr

            liste = np.empty((0, len(header)))
            if merge_decision:
                liste = np.vstack((
                    header, np.vstack((onset, duration, trial_type_md)).T))
            else:
                liste = np.vstack((
                    header, np.vstack((onset, duration, trial_type)).T))

            subjsess_dir = os.path.join(events_dir,
                                        'sub-%02d' % subject_number,
                                        'ses-%02d' % session_number)

            if not os.path.exists(subjsess_dir):
                os.makedirs(subjsess_dir)
            else:
                if flag == 0 and run_number == 1:
                    for f in glob.glob(subjsess_dir + '/*_events.tsv'):
                        os.remove(f)

            if cat == 'Production':
                cattag = 'prod'
            elif cat == 'Perception':
                cattag = 'percep'
            else:
                assert cat == 'No-Temporal Feature Discrimination'
                cattag = 'ntfd'

            fname = 'sub-%02d' % subject_number + \
                '_ses-%02d' % session_number + '_task-' + cattag + \
                '_run-%02d' % run_number + '_events.tsv'

            output_path = os.path.join(subjsess_dir, fname)

            # Save liste in the output file
            with open(output_path, 'w') as fp:
                a = csv.writer(fp, delimiter='\t')
                a.writerows(liste)


def extraction_drbb(data, cat, header, events_dir, ttl = True, flag=0,
                    merge_decision=False, merge_rest=False):

    for ses_datum in data:
        for run_datum in ses_datum:
            onset = []
            duration = []
            trial_type = []
            trial_type_md = []
            if ttl:
                assert run_datum[0][4] == 'ttl'
                offset = convert(run_datum[0][6])
                onset.append(str(float(0)))
                initial_rest = convert(run_datum[0][7])
                duration.append(str(initial_rest))
                trial_type.append('rest')
                trial_type_md.append('rest')
                run_datum = run_datum[1:]
            subject_number = int(run_datum[0][0])
            session_number = int(run_datum[0][1])
            run_number = int(run_datum[0][2])

            for rw, row in enumerate(run_datum):
                if rw == 0 or \
                   (run_datum[rw][3] == '1' and run_datum[rw-1][3] == '30') \
                   or ((run_datum[rw-1][4] in ['fixcross', 'baseline']) and
                    row[4] not in 'final_baseline'):

                    if cat == 'Production':
                        feedback_onset = convert(run_datum[rw+9][6])
                        feedback_duration = convert(run_datum[rw+9][7])
                        if run_datum[rw+9][10] == 'None':
                            rt = None
                        else:
                            rt = convert(run_datum[rw+9][10])
                    else:
                        assert cat in ['Perception',
                                       'No-Temporal Feature Discrimination']
                        feedback_onset = convert(run_datum[rw+11][6])
                        feedback_duration = convert(run_datum[rw+11][7])
                        if run_datum[rw+11][10] == 'None':
                            rt = None
                        else:
                            rt = convert(run_datum[rw+11][10])

                    # Onset and duration for the encoding
                    onset_encod = round(convert(row[6]) - offset, 3)
                    duration_encod = round(feedback_onset - convert(row[6]), 3)
                    onset.append(str(onset_encod))
                    duration.append(str(duration_encod))

                    # Onset and duration for the decision, response and rest
                    onset_decis = round(feedback_onset - offset, 3)
                    onset.append(str(onset_decis))

                    if rt is None:
                        # Duration for the decision
                        duration_decis = feedback_duration
                        duration.append(str(duration_decis))
                    else:
                        # Duration for the decision
                        duration_decis = round(rt, 3)
                        duration.append(str(duration_decis))

                        # Onset and duration for the response
                        onset_resp = round(feedback_onset + rt - offset, 3)
                        onset.append(str(onset_resp))
                        duration_resp = round(EVENT_DURATION, 3)
                        duration.append(str(duration_resp))

                        # Onset and duration for the rest
                        onset_rest = round(feedback_onset + rt +
                                           EVENT_DURATION - offset, 3)
                        onset.append(str(onset_rest))
                        duration_rest = round(feedback_duration - rt -
                                              EVENT_DURATION, 3)
                        duration.append(str(duration_rest))

                    # Trial types for all conditions
                    if row[4][:4] == 'beat' and row[5][:4] == 'beep':
                        trial_type.append('auditory_beat_encoding')
                        trial_type_md.append('auditory_beat_encoding')
                        trial_type.append('auditory_beat_decision')
                    elif row[4][:4] == 'beat' and row[5][:4] == 'rect':
                        trial_type.append('visual_beat_encoding')
                        trial_type_md.append('visual_beat_encoding')
                        trial_type.append('visual_beat_decision')
                    elif row[4][:4] == 'inte' and row[5][:4] == 'beep':
                        trial_type.append('auditory_interval_encoding')
                        trial_type_md.append('auditory_interval_encoding')
                        trial_type.append('auditory_interval_decision')
                    elif row[4][:4] == 'inte' and row[5][:4] == 'rect':
                        trial_type.append('visual_interval_encoding')
                        trial_type_md.append('visual_interval_encoding')
                        trial_type.append('visual_interval_decision')
                    elif row[4][:4] == 'rand' and row[5][:4] == 'beep':
                        trial_type.append('auditory_random_encoding')
                        trial_type_md.append('auditory_random_encoding')
                        trial_type.append('auditory_random_decision')
                    elif row[4][:4] == 'rand' and row[5][:4] == 'rect':
                        trial_type.append('visual_random_encoding')
                        trial_type_md.append('visual_random_encoding')
                        trial_type.append('visual_random_decision')
                    else:
                        raise NameError(
                            'Trial type does not exist for this trial!')

                    trial_type_md.append('decision')
                    if rt is not None:
                        trial_type.append('response')
                        trial_type_md.append('response')
                        trial_type.append('rest')
                        trial_type_md.append('rest')

                elif row[4] in ['fixcross', 'baseline', 'final_baseline']:
                    onset_rest = round(convert(row[6]) - offset, 3)
                    onset.append(str(onset_rest))
                    duration_rest = convert(row[7])
                    duration.append(str(duration_rest))
                    trial_type.append('rest')
                    trial_type_md.append('rest')
                else:
                    pass

            if merge_rest:
                if merge_decision:
                    onset_mr, duration_mr, trial_type_md_mr = \
                        merge_rest_conditions(onset, duration, trial_type_md)
                    del onset
                    del duration
                    del trial_type_md
                    onset = onset_mr
                    duration = duration_mr
                    trial_type_md = trial_type_md_mr
                else:
                    onset_mr, duration_mr, trial_type_mr = \
                        merge_rest_conditions(onset, duration, trial_type)
                    del onset
                    del duration
                    del trial_type_md
                    onset = onset_mr
                    duration = duration_mr
                    trial_type = trial_type_mr

            liste = np.empty((0, len(header)))
            if merge_decision:
                liste = np.vstack((
                    header, np.vstack((onset, duration, trial_type_md)).T))
            else:
                liste = np.vstack((
                    header, np.vstack((onset, duration, trial_type)).T))

            subjsess_dir = os.path.join(events_dir,
                                        'sub-%02d' % subject_number,
                                        'ses-%02d' % session_number)

            if not os.path.exists(subjsess_dir):
                os.makedirs(subjsess_dir)
            else:
                if flag == 0 and run_number == 1:
                    for f in glob.glob(subjsess_dir + '/*_events.tsv'):
                        os.remove(f)

            if cat == 'Production':
                cattag = 'prod'
            elif cat == 'Perception':
                cattag = 'percep'
            else:
                assert cat == 'No-Temporal Feature Discrimination'
                cattag = 'ntfd'

            fname = 'sub-%02d' % subject_number + \
                '_ses-%02d' % session_number + '_task-' + cattag + \
                '_run-%02d' % run_number + '_drbb_events.tsv'

            output_path = os.path.join(subjsess_dir, fname)

            # Save liste in the output file
            with open(output_path, 'w') as fp:
                a = csv.writer(fp, delimiter='\t')
                a.writerows(liste)



# %%
# =========================== INPUTS ===================================

# All subjects
SUBJECTS = [3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 23, 28, 29,
            32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

# All Subjects without pilot (sub-04)
# SUBJECTS = [3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 23, 28, 29,
#             32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

# Working list
# SUBJECTS = [4]

CATEGORIES = ['Production', 'Perception', 'No-Temporal Feature Discrimination']
MODALITIES = ['Auditory', 'Visual']
SESSTYPE = ['imaging_session']
N_TRIALS = 30
LOGFOLDER = 'logfiles'
EVENTSFOLDER = 'events'
HEADER = ['onset', 'duration', 'trial_type']
EVENT_DURATION = .08

# %%
# ========================= PARAMETERS =================================

all_tasks = np.ravel([[m + ' ' + c for c in CATEGORIES]
                      for m in MODALITIES]).tolist()
main_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(main_dir, os.pardir))
logpath = os.path.join(parent_dir, LOGFOLDER)
eventspath = os.path.join(main_dir, EVENTSFOLDER)

# %%
# ============================ RUN =====================================

if __name__ == "__main__":
    for subject in SUBJECTS:
        for c, category in enumerate(CATEGORIES):
            tasks = [task_name for task_name in all_tasks
                     if category in task_name]
            behavioral_data = []
            for task in tasks:
                task_data = []
                task_data = parse_logfile(logpath, subject, SESSTYPE, task,
                                          N_TRIALS, ttl=True,
                                          concatenate=False)
                if category == 'No-Temporal Feature Discrimination':
                    task_data[0].extend([[], []])
                behavioral_data.append(task_data)

            # Re-order behavioral data in a chronological way per run
            ## (tasks, sessions, runs) --> (sessions, runs, tasks)
            sessions = np.moveaxis(behavioral_data, 0, -1)
            ## (sessions, runs, tasks) --> (sessions, runs)
            chrono_data = []
            for session in sessions:
                session = [ir for ir in session if np.any(ir) != []]
                chrono_session = []
                for img_run in session:
                    chrono_run = []
                    for i in np.arange(0, len(img_run), 2):
                        if img_run[i][0][4] == 'ttl':
                            chrono_run = np.vstack(
                                [img_run[i], img_run[i+1]]).tolist()
                        else:
                            chrono_run = np.vstack(
                                [img_run[i+1], img_run[i]]).tolist()
                    chrono_session.append(chrono_run)
                chrono_data.append(chrono_session)

            # For Production tasks,
            if c == 0:
                # With flag = 0, all previous event files will be deleted
                # in every dir subj, sess and run dependent (FOR ALL TASKS)
                # Do flag = 0 only in the first function call of this
                # if statement
                extraction_drbb(chrono_data, category, HEADER, eventspath,
                                flag=0, merge_decision=True, merge_rest=True)
            # For Remaining tasks
            else:
                extraction_drbb(chrono_data, category, HEADER, eventspath,
                                flag=1, merge_decision=True, merge_rest=True)
