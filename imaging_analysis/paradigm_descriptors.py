"""
Extraction of paradigm descriptors for the Music-SDTB Tasks

Author: Ana Luisa Pinho
Email: agrilopi@uwo.ca

Creation: January 2023
Last Update: January 2023

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


def extraction(data, cat, header, events_dir, ttl = True, flag=0):
    for ses_datum in data:
        for run_datum in ses_datum:
            onset = []
            duration = []
            trial_type = []
            if ttl:
                assert run_datum[0][4] == 'ttl'
                onset.append(run_datum[0][6])
                duration.append(run_datum[0][7])
                trial_type.append('rest')
                run_datum = run_datum[1:]
            subject_number = int(run_datum[0][0])
            session_number = int(run_datum[0][1])
            run_number = int(run_datum[0][2])
            for rw, row in enumerate(run_datum):
                if rw == 0 or ((run_datum[rw-1][4] == 'fixcross' or \
                                run_datum[rw-1][4] == 'baseline') and \
                               row[4] not in 'final_baseline'):
                    # Onset and duration for evaluation
                    onset.append(row[6])
                    duration_eval = int(run_datum[rw+8][6]) - int(row[6])
                    duration.append(str(duration_eval))
                    # Onset and duration for judgment
                    onset.append(run_datum[rw+8][6])
                    if cat == 'Production':
                        duration.append(run_datum[rw+8][7])
                    else:
                        assert cat in ['Perception',
                                       'No-Temporal Feature Discrimination']
                        duration_judg = int(run_datum[rw+11][6]) - \
                            int(run_datum[rw+8][6])
                        duration.append(str(duration_judg)) 
                    # Onset and duration for response
                    onset.append(run_datum[rw+9][6])
                    duration.append(run_datum[rw+9][7])
                    # Trial types for all conditions
                    if row[4][:4] == 'beat' and row[5][:4] == 'beep':
                        trial_type.append('auditory_beat_evaluation')
                        trial_type.append('auditory_beat_judgment')
                        trial_type.append('auditory_beat_response')
                    elif row[4][:4] == 'beat' and row[5][:4] == 'rect':
                        trial_type.append('visual_beat_evaluation')
                        trial_type.append('visual_beat_judgment')
                        trial_type.append('visual_beat_response')
                    elif row[4][:4] == 'inte' and row[5][:4] == 'beep':
                        trial_type.append('auditory_interval_evaluation')
                        trial_type.append('auditory_interval_judgment')
                        trial_type.append('auditory_interval_response')
                    elif row[4][:4] == 'inte' and row[5][:4] == 'rect':
                        trial_type.append('visual_interval_evaluation')
                        trial_type.append('visual_interval_judgment')
                        trial_type.append('visual_interval_response')
                    elif row[4][:4] == 'rand' and row[5][:4] == 'beep':
                        trial_type.append('visual_random_evaluation')
                        trial_type.append('visual_random_judgment')
                        trial_type.append('visual_random_response')
                    elif row[4][:4] == 'rand' and row[5][:4] == 'rect':
                        trial_type.append('visual_random_evaluation')
                        trial_type.append('visual_random_judgment')
                        trial_type.append('visual_random_response')
                    else:
                        raise NameError(
                            'Condition does not exist for this trial!')
                elif row[4] in ['fixcross', 'baseline', 'final_baseline']:
                    onset.append(row[6])
                    duration.append(row[7])
                    trial_type.append('rest')
                else:
                    pass

            liste = np.empty((0, len(header)))
            liste = np.vstack((header,
                               np.vstack((onset, duration, trial_type)).T))

            subject_dir = os.path.join(events_dir, 'sub-%02d' % subject_number)

            if not flag:
                if not os.path.exists(subject_dir):
                    os.makedirs(subject_dir)
                else:
                    for f in glob.glob(subject_dir + '/*_events.tsv'):
                        os.remove(f)
                flag = 1

            if cat == 'No-Temporal Feature Discrimination':
                fname = 'sub-%02d' % subject_number + \
                    '_ses-%02d' % session_number + \
                    '_task-ntfd_run-%02d' % run_number + \
                    '_events.tsv'
            else:
                fname = 'sub-%02d' % subject_number + \
                    '_ses-%02d' % session_number + \
                    '_task-' + cat.lower() + '_run-%02d' % run_number + \
                    '_events.tsv'
            output_path = os.path.join(subject_dir, fname)

            # Save liste in the output file
            with open(output_path, 'w') as fp:
                a = csv.writer(fp, delimiter='\t')
                a.writerows(liste)

# %%
# =========================== INPUTS ===================================

SUBJECTS = [3, 4, 7, 8]
# SUBJECTS = [3]

CATEGORIES = ['Production', 'Perception', 'No-Temporal Feature Discrimination']
MODALITIES = ['Auditory', 'Visual']
SESSTYPE = 'imaging session'
N_SESSIONS = 2
LOGFOLDER = 'logfiles'
EVENTSFOLDER = 'events'
HEADER = ['onset', 'duration', 'trial_type']

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
                extraction(behavioral_data, category, HEADER, eventspath)
            else:
                extraction(behavioral_data, category, HEADER, eventspath,
                           flag=1)
