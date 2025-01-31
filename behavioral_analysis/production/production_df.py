"""
Create dataframe of data from Production Tasks of the
Music-SDTB project

author: Ana Luisa Pinho
e-mail: agrilopi@uwo.ca

Created: May 4, 2024
Last update: January 2025

Compatibility: Python 3.10.14
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# setting path
sys.path.append('../../')
# importing
from utils import parse_logfile

import numpy as np
import pandas as pd


# %%
# ======================== MAIN FUNCTIONS ==============================


def production_data(data):
    trials = []
    for dt, datum in enumerate(data):
        if datum[5] == 'interval_1':
            subject = datum[0]
            session = datum[1]
            run_id = datum[2]
            condition = datum[4]
            theoretical_isi1 = int(datum[8])
            if data[dt+8][5] == 'feedback' and data[dt+8][11] in ['o', 'b']:
                rt = int(data[dt+7][7]) + int(data[dt+8][10])
            elif data[dt+8][5] == 'feedback' and data[dt+8][10] == 'None':
                rt = np.nan
            else:
                raise ValueError('No feedback entry!')
            trials.append([subject, session, run_id, condition[:-2],
                           theoretical_isi1, rt])

    return trials


def symlog_transform(arr, shift):
    """About this function, consult:
    https://pythonmatplotlibtips.blogspot.com/2018/11/x-symlog-with-shift.html
    """
    logv = np.abs(arr)*(10.**shift)
    logv[np.where(logv < 1.)] = 1.
    logv = np.sign(arr)*np.log10(logv)

    return logv


def production_dataframe(subjects, this_dir, output_dir, sesstype, n_trials,
                         sesstag, n_columns, sessions=None,
                         tasks=['Auditory Production', 'Visual Production']):

    logfiles_dir = os.path.join(
        os.path.abspath(os.path.join(this_dir, os.pardir, os.pardir)),
        'logfiles')

    trials_arr = np.empty((0, n_columns))
    for s, subject in enumerate(subjects):
        for t, task in enumerate(tasks):
            if task not in ['Auditory Production', 'Visual Production']:
                raise NameError('Task not valid!')
            
            data = parse_logfile(logfiles_dir, subject, sesstype, task,
                                 n_trials, sessions=sessions)
            trials = production_data(data)

            # Get beat and interval trials to stack them later in groups
            # of beat and interval trials
            beat_trials = np.array([
                tr for tr in trials if tr[3][:4] == 'beat'], dtype=object)
            interval_trials = np.array([
                tr for tr in trials if tr[3][:8] == 'interval'], dtype=object)

            # Compute asynchronies, but return NaN if any value in...
            # ... the row is NaN
            beat_numeric = beat_trials[:, 4:].astype(float)
            interval_numeric = interval_trials[:, 4:].astype(float)
            with np.errstate(invalid='ignore'):  # Avoid warnings for NaN operations
                ss_beat = np.where(
                    np.isnan(beat_numeric).any(axis=1), np.nan,
                    np.round((beat_numeric[:, 1] - beat_numeric[:, 0]) /
                             beat_numeric[:, 0], 2))
                ss_interval = np.where(
                    np.isnan(interval_numeric).any(axis=1), np.nan,
                    np.round((interval_numeric[:, 1] - interval_numeric[:, 0]) /
                             interval_numeric[:, 0], 2))

            # Append asynchronies as the last elements of the row
            beat_trials = np.hstack((beat_trials,
                                     ss_beat.reshape(-1, 1)
                                     ))
            interval_trials = np.hstack((interval_trials,
                                         ss_interval.reshape(-1, 1)
                                         ))

            # Append modality info in the third position of the row
            modality = np.array([task.partition(' ')[0].lower()])
            
            mbeat = np.repeat(modality, beat_trials.shape[0])
            table_beat = np.insert(beat_trials, 3, mbeat, axis=1)

            minterval = np.repeat(modality, interval_trials.shape[0])
            table_interval = np.insert(interval_trials, 3, minterval, axis=1)

            # Stack
            trials_arr = np.vstack((trials_arr, table_beat))
            trials_arr = np.vstack((trials_arr, table_interval))

    df = pd.DataFrame(trials_arr, columns=[
        'Subject', 'Session', 'Run', 'Modality', 'Condition', 'Standard',
        'Response Time', 'Signed Asynchrony'])

    # Save dataframe
    outpath = os.path.join(output_dir, 'df_production_' + sesstag + '.tsv')
    df.to_csv(outpath, index=False, sep='\t', na_rep='NaN')


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

# TASKS = ['Visual Production']

N_TRIALS = 30

# ### For 'All Sessions' ###
SUBJECTS = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
            22, 23, 24, 25, 26, 27, 28, 29, 32, 34, 35, 38, 39, 40, 41, 42, 43,
            44, 45, 46, 47]
SESSTYPES = ['behavioral_session', 'imaging_session']
SESSIONS = None
tag = 'allses'

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
# SUBJECTS = [3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26, 28,
#             29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
# SESSTYPES = ['imaging_session']
# SESSIONS = ['ses-02']
# tag = 'ses-05'

# %%
# ========================= PARAMETERS =================================

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FOLDER = os.path.join(MAIN_DIR, 'production_results', 'dataframes')

# %%
# ============================ RUN =====================================

if __name__ == "__main__":

    if not os.path.exists(RESULTS_FOLDER):
        os.mkdir(RESULTS_FOLDER)

    # Create the dataframe
    production_dataframe(SUBJECTS, MAIN_DIR, RESULTS_FOLDER, SESSTYPES,
                         N_TRIALS, tag, 8, sessions=SESSIONS)
