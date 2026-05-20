"""
Create dataframe of data from Production Tasks of the
Music-SDTB project

author: Ana Luisa Pinho
e-mail: agrilopi@uwo.ca

Created: May 4, 2024
Last update: May 2026

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
                         subjects_batch,
                         sesstag=None,
                         sessions=None,
                         audio_latency=0,
                         visual_latency=0,
                         button_press=0,
                         tasks=['Auditory Production', 'Visual Production']):

    # Define columns of dataframe
    df = pd.DataFrame(columns=[
        'subject', 'session', 'run', 'modality', 'condition', 'standard',
        'response_time', 'response_time_corrected', 'signed_asynchrony'])

    logfiles_dir = os.path.join(
        os.path.abspath(os.path.join(this_dir, os.pardir, os.pardir)),
        'logfiles')

    trials_arr = np.empty((0, df.columns.size))
    for s, subject in enumerate(subjects):
        for t, task in enumerate(tasks):
            if task not in ['Auditory Production', 'Visual Production']:
                raise NameError('Task not valid!')

            data = parse_logfile(logfiles_dir, subject, sesstype, task,
                                 n_trials, sessions=sessions,
                                 renumber_sessions=True)
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

            modality = np.array([task.partition(' ')[0].lower()])

            if modality[0] == 'auditory':
                latency = audio_latency
            elif modality[0] == 'visual':
                latency = visual_latency
            else:
                raise NameError('Modality not valid!')

            beat_corr = beat_numeric[:, 1] - latency - button_press
            interval_corr = interval_numeric[:, 1] - latency - button_press

            with np.errstate(invalid='ignore'):  # Avoid warnings for NaN operations
                ss_beat = np.where(
                    np.isnan(beat_numeric).any(axis=1), np.nan,
                    np.round((beat_corr - beat_numeric[:, 0]) /
                             beat_numeric[:, 0], 2))
                ss_interval = np.where(
                    np.isnan(interval_numeric).any(axis=1), np.nan,
                    np.round((interval_corr - interval_numeric[:, 0]) /
                             interval_numeric[:, 0], 2))

            # Append corrected response times and asynchronies as the last
            # elements of the row
            beat_trials = np.hstack((beat_trials,
                                     beat_corr.reshape(-1, 1),
                                     ss_beat.reshape(-1, 1)
                                     ))
            interval_trials = np.hstack((interval_trials,
                                         interval_corr.reshape(-1, 1),
                                         ss_interval.reshape(-1, 1)
                                         ))

            # Append modality info in the third position of the row
            mbeat = np.repeat(modality, beat_trials.shape[0])
            table_beat = np.insert(beat_trials, 3, mbeat, axis=1)

            minterval = np.repeat(modality, interval_trials.shape[0])
            table_interval = np.insert(interval_trials, 3, minterval, axis=1)

            # Stack
            trials_arr = np.vstack((trials_arr, table_beat))
            trials_arr = np.vstack((trials_arr, table_interval))

    # Add data to dataframe
    df = pd.DataFrame(trials_arr, columns=df.columns)

    # Save dataframe
    if sesstag:
        outpath = os.path.join(
            output_dir, 
            subjects_batch + '_' + sesstag + '.tsv')
    else:
        outpath = os.path.join(output_dir, subjects_batch + '.tsv')
    df.to_csv(outpath, index=False, sep='\t', na_rep='NaN')


# %%
# =========================== INPUTS ===================================

# ##################### Subjects' lists ################################
# All subjects
ALL_SUBJECTS = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36,
                37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

# All good subjects including img pilot (sub-04)
GOOD_SUBJECTS = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                 21, 22, 23, 24, 25, 26, 27, 28, 29, 32, 34, 35, 38, 39, 40,
                 41, 42, 43, 44, 45, 46, 47]

# Img subjects only (without pilot)
IMG_SUBJECTS = [3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26,
                28, 29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

# Second batch
SB_SUBJECTS = [48]

# #######################################################################

# TASKS = ['Visual Production']

N_TRIALS = 30

AUDIO_LATENCY = 63 # Expy: 133 / Psychopy: 63
VISUAL_LATENCY = 35 # Expy: 35
BUTTON_PRESS = 20 # 20

# ### For 'All Sessions' ###
# SUBJECTS = GOOD_SUBJECTS
# SESSTYPES = ['behavioral_session', 'imaging_session']
# SESSIONS = None
# tag = 'allses'
# batch = 'df_production' # 'df_production' / 'df_production_sb'

# ### For first behav session: 'ses-01' ###
SUBJECTS = SB_SUBJECTS # GOOD_SUBJECTS / SB_SUBJECTS
SESSTYPES = ['behavioral_session']
SESSIONS = ['ses-01']
tag = SESSIONS[0]
batch = 'df_production_sb' # 'df_production' / 'df_production_sb'

# ### For second behav session: 'ses-02' ###
# SUBJECTS = GOOD_SUBJECTS
# SESSTYPES = ['behavioral_session']
# SESSIONS = ['ses-02']
# tag = SESSIONS[0]
# batch = 'df_production' # 'df_production' / 'df_production_sb'

# ### For third behav session: 'ses-03' ###
# SUBJECTS = GOOD_SUBJECTS
# SESSTYPES = ['behavioral_session']
# SESSIONS = ['ses-03']
# tag = SESSIONS[0]
# batch = 'df_production' # 'df_production' / 'df_production_sb'

# ### For first img session: 'ses-04' ###
# SUBJECTS = IMG_SUBJECTS
# SESSTYPES = ['imaging_session']
# SESSIONS = ['ses-01']
# tag = 'ses-04'
# batch = 'df_production' # 'df_production' / 'df_production_sb'

# ### For second img session: 'ses-05' ###
# SUBJECTS = IMG_SUBJECTS
# SESSTYPES = ['imaging_session']
# SESSIONS = ['ses-02']
# tag = 'ses-05'
# batch = 'df_production' # 'df_production' / 'df_production_sb'

# %%
# ========================= PARAMETERS =================================

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FOLDER = os.path.join(MAIN_DIR, 'production_results', 'dataframes')

# %%
# ============================ RUN =====================================

if __name__ == "__main__":

    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)

    # Create the dataframe
    production_dataframe(SUBJECTS, MAIN_DIR, RESULTS_FOLDER, SESSTYPES,
                         N_TRIALS, batch,
                         sesstag=None, 
                         sessions=SESSIONS,
                         audio_latency=AUDIO_LATENCY,
                         visual_latency=VISUAL_LATENCY,
                         button_press=BUTTON_PRESS)