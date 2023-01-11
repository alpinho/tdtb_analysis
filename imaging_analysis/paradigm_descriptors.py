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


def production(data, header, events_dir, ttl = True):
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
                    duration.append(run_datum[rw+8][7])
                    # Onset and duration for response
                    onset.append(run_datum[rw+9][6])
                    duration.append(run_datum[rw+9][7])
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
                    else:
                        pass
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
            if not os.path.exists(subject_dir):
                os.makedirs(subject_dir)

            fname = 'sub-%02d' % subject_number + \
                '_ses-%02d' % session_number + \
                '_task-production_run-%02d' % run_number + '_events.tsv'
            output_path = os.path.join(subject_dir, fname)

            # Save liste in the output file
            with open(output_path, 'w') as fp:
                a = csv.writer(fp, delimiter='\t')
                a.writerows(liste)

# %%
# =========================== INPUTS ===================================

# SUBJECTS = [3, 8]
SUBJECTS = [3]

TASKS = ['Auditory Production',
         'Auditory Perception',
         'Auditory No-Temporal Feature Discrimination',
         'Visual Production',
         'Visual Perception',
         'Visual No-Temporal Feature Discrimination']

SESSTYPE = 'imaging session'
N_SESSIONS = 2
LOGFOLDER = 'logfiles'
EVENTSFOLDER = 'events'
HEADER = ['onset', 'duration', 'trial_type']

# %%
# ========================= PARAMETERS =================================

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
logpath = os.path.join(MAIN_DIR, LOGFOLDER)
eventspath = os.path.join(MAIN_DIR, EVENTSFOLDER)

# %%
# ============================ RUN =====================================

if __name__ == "__main__":
    behavioral_data = parse_logfile(logpath, SUBJECTS[0], SESSTYPE, N_SESSIONS,
                                    [TASKS[0], TASKS[3]], ttl = True,
                                    concatenate=False)

    production(behavioral_data, HEADER, eventspath)

# logpath = os.path.join(MAIN_DIR, 'logfiles', 'sub-03', 'sess-01')
# logfiles = glob.glob(os.path.join(logpath, '*.xpd'))
# logfiles.sort()
# inputs_lists = [list(csv.reader(open(logfile), delimiter=','))
#                 for logfile in logfiles]

# for inputs_list in inputs_lists:
#     for r, row in enumerate(inputs_list):
#         if row[0][9:][:-18] in TASKS:
#             task_name = row[0][9:][:-18]
#             if task_name[:4] == 'Audi':
#                 modality = 'audio'
#             else:
#                 modality = 'visual'
#         if row[0] == 'subject_id':
#             liste = inputs_list[r+1:]
#             0/0
