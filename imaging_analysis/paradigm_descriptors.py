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


def production(data):
    trials = []
    for dt, datum in enumerate(data):
    #     if datum[5] == 'interval_1':
    #         condition = datum[4]
    #         theoretical_isi1 = int(datum[8])
    #         real_isi1 = int(datum[9])
    #         if data[dt+8][5] == 'feedback' and data[dt+8][11] == 'o':
    #             rt = int(data[dt+7][7]) + int(data[dt+8][10])
    #         elif data[dt+8][5] == 'feedback' and data[dt+8][10] == 'None':
    #             rt = np.nan
    #         else:
    #             raise ValueError('No feedback entry!')
    #         trials.append([condition, theoretical_isi1, real_isi1, rt])

    # return trials

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

# %%
# ========================= PARAMETERS =================================

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
logpath = os.path.join(MAIN_DIR, LOGFOLDER)

# %%
# ============================ RUN =====================================

if __name__ == "__main__":
    data = parse_logfile(logpath, SUBJECTS[0], SESSTYPE, N_SESSIONS, TASKS[2],
                         concatenate=False)
    
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
