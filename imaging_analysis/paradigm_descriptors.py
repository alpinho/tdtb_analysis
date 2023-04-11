"""
Extraction of paradigm descriptors for the Music-SDTB Tasks

Author: Ana Luisa Pinho
Email: agrilopi@uwo.ca

Creation: January 2023
Last Update: April 2023

Compatibility: Python 3.10.4

"""

import sys

import os
import glob
import csv

import numpy as np

# setting path
sys.path.append('../')
# importing
from utils import parse_logfile


# %%
# ========================== FUNCTIONS =================================


def convert(strnum):
    converted_num = round(float(strnum)/1e3, 3)

    return converted_num


def extraction(data, cat, header, events_dir, ttl = True, flag=0,
               merge_decision=0):
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
                if rw == 0 or ((run_datum[rw-1][4] == 'fixcross' or \
                                run_datum[rw-1][4] == 'baseline') and \
                               row[4] not in 'final_baseline'):

                    # Onset and duration for the encoding
                    onset_encod = round(convert(row[6]) - offset, 3)

                    if cat == 'Production':
                        if run_datum[rw+9][10] == 'None':
                            duration_encod = round(
                                convert(run_datum[rw+9][6]) - \
                                convert(row[6]), 3)
                            onset_decis = round(
                                convert(run_datum[rw+9][6]) - offset, 3)
                            duration_decis = round(
                                convert(run_datum[rw+9][7]), 3)
                        else:
                            duration_encod = round(
                                convert(run_datum[rw+9][6]) + \
                                convert(run_datum[rw+9][10]) - \
                                convert(row[6]), 3)
                            onset_decis = round(
                                convert(run_datum[rw+9][6]) + \
                                convert(run_datum[rw+9][10]) - offset, 3)
                            duration_decis = round(
                                convert(run_datum[rw+10][6]) - \
                                convert(run_datum[rw+9][6]) - \
                                convert(run_datum[rw+9][10]), 3)
                    else:
                        assert cat in ['Perception',
                                       'No-Temporal Feature Discrimination']
                        duration_encod = round(
                            convert(run_datum[rw+11][6]) - convert(row[6]), 3)
                        onset_decis = round(
                            convert(run_datum[rw+11][6]) - offset, 3)
                        duration_decis = round(convert(run_datum[rw+11][7]), 3)

                    onset.append(str(onset_encod))
                    duration.append(str(duration_encod))
                    onset.append(str(onset_decis))
                    duration.append(str(duration_decis))

                    # Trial types for all conditions
                    if row[4][:4] == 'beat' and row[5][:4] == 'beep':
                        trial_type.append('auditory_beat_encoding')
                        trial_type_md.append('auditory_beat_encoding')
                        trial_type.append('auditory_beat_decision')
                        trial_type_md.append('decision')
                    elif row[4][:4] == 'beat' and row[5][:4] == 'rect':
                        trial_type.append('visual_beat_encoding')
                        trial_type_md.append('visual_beat_encoding')
                        trial_type.append('visual_beat_decision')
                        trial_type_md.append('decision')
                    elif row[4][:4] == 'inte' and row[5][:4] == 'beep':
                        trial_type.append('auditory_interval_encoding')
                        trial_type_md.append('auditory_interval_encoding')
                        trial_type.append('auditory_interval_decision')
                        trial_type_md.append('decision')
                    elif row[4][:4] == 'inte' and row[5][:4] == 'rect':
                        trial_type.append('visual_interval_encoding')
                        trial_type_md.append('visual_interval_encoding')
                        trial_type.append('visual_interval_decision')
                        trial_type_md.append('decision')
                    elif row[4][:4] == 'rand' and row[5][:4] == 'beep':
                        trial_type.append('auditory_random_encoding')
                        trial_type_md.append('auditory_random_encoding')
                        trial_type.append('auditory_random_decision')
                        trial_type_md.append('decision')
                    elif row[4][:4] == 'rand' and row[5][:4] == 'rect':
                        trial_type.append('visual_random_encoding')
                        trial_type_md.append('visual_random_encoding')
                        trial_type.append('visual_random_decision')
                        trial_type_md.append('decision')
                    else:
                        raise NameError(
                            'Trial type does not exist for this trial!')
                elif row[4] in ['fixcross', 'baseline', 'final_baseline']:
                    onset_rest = round(convert(row[6]) - offset, 3)
                    onset.append(str(onset_rest))
                    duration_rest = convert(row[7])
                    duration.append(str(duration_rest))
                    trial_type.append('rest')
                    trial_type_md.append('rest')
                else:
                    pass

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


def extraction_dr(data, cat, header, events_dir, ttl = True, flag=0,
                  merge_decision=0):
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
                if rw == 0 or ((run_datum[rw-1][4] == 'fixcross' or \
                                run_datum[rw-1][4] == 'baseline') and \
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
                '_run-%02d' % run_number + '_dr_events.tsv'

            output_path = os.path.join(subjsess_dir, fname)

            # Save liste in the output file
            with open(output_path, 'w') as fp:
                a = csv.writer(fp, delimiter='\t')
                a.writerows(liste)


def extraction_split(data, cat, header, events_dir, ttl = True,
                     merge_decision = 0):
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
                if rw == 0 or ((run_datum[rw-1][4] == 'fixcross' or \
                                run_datum[rw-1][4] == 'baseline') and \
                               row[4] not in 'final_baseline'):

                    # Onset and duration for the encoding
                    onset_encod = round(convert(row[6]) - offset, 3)

                    if cat == 'Production':
                        if run_datum[rw+9][10] == 'None':
                            duration_encod = round(
                                convert(run_datum[rw+9][6]) - \
                                convert(row[6]), 3)
                            onset_decis = round(
                                convert(run_datum[rw+9][6]) - offset, 3)
                            duration_decis = round(
                                convert(run_datum[rw+9][7]), 3)
                        else:
                            duration_encod = round(
                                convert(run_datum[rw+9][6]) + \
                                convert(run_datum[rw+9][10]) - \
                                convert(row[6]), 3)
                            onset_decis = round(
                                convert(run_datum[rw+9][6]) + \
                                convert(run_datum[rw+9][10]) - offset, 3)
                            duration_decis = round(
                                convert(run_datum[rw+10][6]) - \
                                convert(run_datum[rw+9][6]) - \
                                convert(run_datum[rw+9][10]), 3)
                    else:
                        assert cat in ['Perception',
                                       'No-Temporal Feature Discrimination']
                        duration_encod = round(
                            convert(run_datum[rw+11][6]) - convert(row[6]), 3)
                        onset_decis = round(
                            convert(run_datum[rw+11][6]) - offset, 3)
                        duration_decis = round(convert(run_datum[rw+11][7]), 3)

                    onset.append(str(onset_encod))
                    duration.append(str(duration_encod))
                    onset.append(str(onset_decis))
                    duration.append(str(duration_decis))

                    # Trial types for all conditions
                    if row[4] in ['beat01', 'beat02', 'beat03'] \
                       and row[5][:4] == 'beep':
                        trial_type.append('auditory_beat_encoding_low')
                        trial_type_md.append('auditory_beat_encoding_low')
                        trial_type.append('auditory_beat_decision')
                    elif row[4] in ['beat04', 'beat05'] \
                         and row[5][:4] == 'beep':
                        trial_type.append('auditory_beat_encoding_high')
                        trial_type_md.append('auditory_beat_encoding_high')
                        trial_type.append('auditory_beat_decision')
                    elif row[4] in ['beat01', 'beat02', 'beat03'] \
                         and row[5][:4] == 'rect':
                        trial_type.append('visual_beat_encoding_low')
                        trial_type_md.append('visual_beat_encoding_low')
                        trial_type.append('visual_beat_decision')
                    elif row[4] in ['beat04', 'beat05'] \
                         and row[5][:4] == 'rect':
                        trial_type.append('visual_beat_encoding_high')
                        trial_type_md.append('visual_beat_encoding_high')
                        trial_type.append('visual_beat_decision')
                    elif row[4] in ['interval01', 'interval02', 'interval03'] \
                         and row[5][:4] == 'beep':
                        trial_type.append('auditory_interval_encoding_low')
                        trial_type_md.append('auditory_interval_encoding_low')
                        trial_type.append('auditory_interval_decision')
                    elif row[4] in ['interval04', 'interval05'] \
                         and row[5][:4] == 'beep':
                        trial_type.append('auditory_interval_encoding_high')
                        trial_type_md.append('auditory_interval_encoding_high')
                        trial_type.append('auditory_interval_decision')
                    elif row[4] in ['interval01', 'interval02', 'interval03'] \
                         and row[5][:4] == 'rect':
                        trial_type.append('visual_interval_encoding_low')
                        trial_type_md.append('visual_interval_encoding_low')
                        trial_type.append('visual_interval_decision')
                    elif row[4] in ['interval04', 'interval05'] \
                         and row[5][:4] == 'rect':
                        trial_type.append('visual_interval_encoding_high')
                        trial_type_md.append('visual_interval_encoding_high')
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
                elif row[4] in ['fixcross', 'baseline', 'final_baseline']:
                    onset_rest = round(convert(row[6]) - offset, 3)
                    onset.append(str(onset_rest))
                    duration_rest = convert(row[7])
                    duration.append(str(duration_rest))
                    trial_type.append('rest')
                    trial_type_md.append('rest')
                else:
                    pass

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

            if cat == 'Production':
                cattag = 'prod'
            elif cat == 'Perception':
                cattag = 'percep'
            else:
                assert cat == 'No-Temporal Feature Discrimination'
                cattag = 'ntfd'

            fname = 'sub-%02d' % subject_number + \
                '_ses-%02d' % session_number + '_task-' + cattag + \
                '_run-%02d' % run_number + '_splitdesign_events.tsv'

            output_path = os.path.join(subjsess_dir, fname)

            # Save liste in the output file
            with open(output_path, 'w') as fp:
                a = csv.writer(fp, delimiter='\t')
                a.writerows(liste)


# %%
# =========================== INPUTS ===================================

SUBJECTS = [3, 4, 7, 8, 10]
# SUBJECTS = [3]

CATEGORIES = ['Production', 'Perception', 'No-Temporal Feature Discrimination']
MODALITIES = ['Auditory', 'Visual']
SESSTYPE = 'imaging session'
N_SESSIONS = 2
LOGFOLDER = 'logfiles'
EVENTSFOLDER = 'events'
HEADER = ['onset', 'duration', 'trial_type']
EVENT_DURATION = 0.08

# %%
# ========================= PARAMETERS =================================

all_tasks = np.ravel([[m + ' ' + c for c in CATEGORIES]
                      for m in MODALITIES]).tolist()
main_dir = os.path.dirname(os.path.abspath(__file__))
logpath = os.path.join(main_dir, LOGFOLDER)
eventspath = os.path.join(main_dir, EVENTSFOLDER)

# %%
# ============================ RUN =====================================

if __name__ == "__main__":
    for subject in SUBJECTS:
        for c, category in enumerate(CATEGORIES):
            tasks = [task for task in all_tasks if category in task]
            behavioral_data = parse_logfile(logpath, subject, SESSTYPE,
                                            N_SESSIONS, tasks, ttl=True,
                                            concatenate=False)
            if c == 0:
                extraction(behavioral_data, category, HEADER, eventspath,
                           merge_decision=1)
                extraction_dr(behavioral_data, category, HEADER, eventspath,
                           merge_decision=1)
                extraction_split(behavioral_data, category, HEADER, eventspath,
                                 merge_decision=1)
            else:
                extraction(behavioral_data, category, HEADER, eventspath,
                           flag=1, merge_decision=1)
                extraction_dr(behavioral_data, category, HEADER, eventspath,
                           flag=1, merge_decision=1)
                extraction_split(behavioral_data, category, HEADER, eventspath,
                                 merge_decision=1)
