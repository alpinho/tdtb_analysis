"""
Analysis of behavioral data for the Perception Tasks of the Music-SDTB project

author: Ana Luisa Pinho
e-mail: agrilopi@uwo.ca

Created: 28th of January 2025
Last update: January 2025

Compatibility: Python 3.10.14
"""

import sys
import os

import warnings

import numpy as np
import pandas as pd

# setting path
sys.path.append('../../')
# importing
from utils import parse_logfile

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# %%
# ======================== MAIN FUNCTIONS ==============================


def perception_data(data):
    trials = []
    for dt, datum in enumerate(data):
        if datum[5] == 'interval_1':
            condition = datum[4]
            theoretical_isi1 = datum[8]
            theoretical_isi5 = data[dt+8][8]
            if data[dt+10][5] == 'feedback' and \
               data[dt+10][11] in ['o', 'p', 'b', 'y']:
                rt = int(data[dt+9][7]) + int(data[dt+10][10])
                answer = data[dt+10][11]
            elif data[dt+10][5] == 'feedback' and \
                 data[dt+10][11] == 'None':
                rt = np.nan
                answer = np.nan
            else:
                raise ValueError('No feedback entry!')
            trials.append([condition[:-2], theoretical_isi1, theoretical_isi5,
                           answer, rt])

    return trials


def create_dataframe(
        subjects, this_dir, output_dir, sesstype, n_trials, sesstag,
        n_columns, estimator='mle_expit', sessions=None,
        tasks = ['Auditory Perception', 'Visual Perception']):

    logfiles_dir = os.path.join(
        os.path.abspath(os.path.join(this_dir, os.pardir, os.pardir)),
        'logfiles')

    trials_arr = np.empty((0, n_columns))
    for s, subject in enumerate(subjects):
        for t, task in enumerate(tasks):
            if task not in ['Auditory Perception', 'Visual Perception']:
                raise NameError('Task not valid!')

            data = parse_logfile(logfiles_dir, subject, sesstype, task,
                                 n_trials, sessions=sessions)
            trials = perception_data(data)

            submod = np.array([subject, task.partition(' ')[0].lower()])
            submod_col = np.tile(submod, (np.array(trials).shape[0], 1))
            table = np.hstack((submod_col, np.array(trials))).tolist()
            
            trials_arr = np.vstack((trials_arr, table))

    df = pd.DataFrame(trials_arr, columns=[
        'Subject', 'Modality', 'Condition', 'Standard', 'Comparison', 'Answer',
        'Response Time'])

    # Save dataframe
    outpath = os.path.join(output_dir, 'df_perception_' + sesstag + '.tsv')
    df.to_csv(outpath, index=False, sep='\t')


# %%
# =========================== INPUTS ===================================

# ################## Note about subjects ###############################
# All subjects
# SUBJECTS = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
#             22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39,
#             40, 41, 42, 43, 44, 45, 46, 47]

# All good subjects including img pilot (sub-04)
# SUBJECTS = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
#             22, 23, 24, 25, 26, 27, 28, 29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 
#             44, 45, 46, 47]

# Img subjects only
# SUBJECTS = [3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26, 28,
#             29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

# #######################################################################

# TASKS = ['Auditory Perception', 'Visual Perception']

N_TRIALS = 30

# ### For 'All Sessions' ###
# SUBJECTS = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
#             22, 23, 24, 25, 26, 27, 28, 29, 32, 34, 35, 38, 39, 40, 41, 42, 43,
#             44, 45, 46, 47]
# SESSTYPES = ['behavioral_session', 'imaging_session']
# SESSIONS = None
# tag = 'allses'

# ### For first behav session: 'ses-01' ###
# SUBJECTS = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
#             22, 23, 24, 25, 26, 27, 28, 29, 32, 34, 35, 38, 39, 40, 41, 42, 43,
#             44, 45, 46, 47]
# SESSTYPES = ['behavioral_session']
# SESSIONS = ['ses-01']
# tag = SESSIONS[0]

# ### For second behav session: 'ses-02' ###
# SUBJECTS = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
#             22, 23, 24, 25, 26, 27, 28, 29, 32, 34, 35, 38, 39, 40, 41, 42, 43,
#             44, 45, 46, 47]
# SESSTYPES = ['behavioral_session']
# SESSIONS = ['ses-02']
# tag = SESSIONS[0]

# ### For third behav session: 'ses-03' ###
# SUBJECTS = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
#             22, 23, 24, 25, 26, 27, 28, 29, 32, 34, 35, 38, 39, 40, 41, 42, 43,
#             44, 45, 46, 47]
# SESSTYPES = ['behavioral_session']
# SESSIONS = ['ses-03']
# tag = SESSIONS[0]

# ### For first img session: 'ses-04' ###
# SUBJECTS = [3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26, 28,
#             29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
# SESSTYPES = ['imaging_session']
# SESSIONS = ['ses-01']
# tag = 'ses-04'

# ### For second img session: 'ses-05' ###
SUBJECTS = [3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26, 28,
            29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
SESSTYPES = ['imaging_session']
SESSIONS = ['ses-02']
tag = 'ses-05'

sessions_dic = {'allses': 'All Sessions',
                'ses-01': 'Session 1',
                'ses-02': 'Session 2',
                'ses-03': 'Session 3',
                'ses-04': 'Session 4',
                'ses-05': 'Session 5'}

# %%
# ========================= PARAMETERS =================================

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FOLDER = os.path.join(MAIN_DIR, 'perception_results/raw_dataframes')

# %%
# ============================ RUN =====================================

if __name__ == "__main__":

    if not os.path.exists(RESULTS_FOLDER):
        os.mkdir(RESULTS_FOLDER)

    create_dataframe(
        SUBJECTS, MAIN_DIR, RESULTS_FOLDER, SESSTYPES, N_TRIALS, tag, 7,
        sessions=SESSIONS, tasks=['Auditory Perception', 'Visual Perception'])
