"""
Create dataframe of raw data from Perception Tasks of the
Music-SDTB project

author: Ana Luisa Pinho
e-mail: agrilopi@uwo.ca

Created: 28th of January 2025
Last update: May 2026

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
            subject = datum[0]
            session = datum[1]
            run_id = datum[2]
            condition = datum[4]
            theoretical_isi1 = int(datum[8])
            theoretical_isi5 = int(data[dt+8][8])
            if (
                    data[dt + 10][5] == 'feedback' and
                    data[dt + 10][11] in ['o', 'p', 'b', 'y']):
                rt = int(data[dt+9][7]) + int(data[dt+10][10])
                answer = data[dt+10][11]
            elif (
                    data[dt+10][5] == 'feedback' and 
                    data[dt+10][11] in ['None', '-']):
                rt = np.nan
                answer = 'None'
            else:
                raise ValueError('No feedback entry!')
            trials.append([subject, session, run_id, condition[:-2],
                           theoretical_isi1, theoretical_isi5, rt, answer])

    return trials


def perception_dataframe(subjects, this_dir, output_dir, sesstype, n_trials,
                         sesstag=None, sessions=None,
                         tasks=['Auditory Perception', 'Visual Perception']):

    # Define columns of dataframe
    df = pd.DataFrame(columns=[
        'subject', 'session', 'run', 'modality', 'condition', 'standard',
        'comparison', 'response_time', 'answer'])

    logfiles_dir = os.path.join(
        os.path.abspath(os.path.join(this_dir, os.pardir, os.pardir)),
        'logfiles')

    trials_arr = np.empty((0, df.columns.size))
    for s, subject in enumerate(subjects):
        for t, task in enumerate(tasks):
            if task not in ['Auditory Perception', 'Visual Perception']:
                raise NameError('Task not valid!')

            data = parse_logfile(logfiles_dir, subject, sesstype, task,
                                 n_trials, sessions=sessions,
                                 renumber_sessions=True)
            trials = perception_data(data)

            # Get beat and interval trials to stack them later in groups
            # of beat and interval trials
            beat_trials = np.array([
                tr for tr in trials if tr[3][:4] == 'beat'], dtype=object)
            interval_trials = np.array([
                tr for tr in trials if tr[3][:8] == 'interval'], dtype=object)

            # Append modality info in the third position of the row
            modality = np.array([task.partition(' ')[0].lower()])

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
        outpath = os.path.join(output_dir, 'df_perception_' + sesstag + '.tsv')
    else:
        outpath = os.path.join(output_dir, 'df_perception.tsv')
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
SB_SUBJECTS = [48, 49, 50, 51, 52, 53]

# #######################################################################

sessions_dic = {
    'allses': 'All Sessions',
    'ses-01': 'Session 1',
    'ses-02': 'Session 2',
    'ses-03': 'Session 3',
    'ses-04': 'Session 4',
    'ses-05': 'Session 5',
}

# TASKS = ['Auditory Perception', 'Visual Perception']

N_TRIALS = 30

# ### For 'All Sessions' ###
# SUBJECTS = GOOD_SUBJECTS
# SESSTYPES = ['behavioral_session', 'imaging_session']
# SESSIONS = None
# tag = 'allses'

# ### For 'All Behavioral Sessions' ###
# SUBJECTS = GOOD_SUBJECTS
# SESSTYPES = ['behavioral_session']
# SESSIONS = None
# tag = 'behavses'

# ### For 'All Imaging Sessions' ###
# SUBJECTS = IMG_SUBJECTS
# SESSTYPES = ['imaging_session']
# SESSIONS = None
# tag = 'imgses'

# ### For first behav session: 'ses-01' ###
SUBJECTS = SB_SUBJECTS  # SB_SUBJECTS / GOOD_SUBJECTS
SESSTYPES = ['behavioral_session']
SESSIONS = ['ses-01']
tag = SESSIONS[0]

# ### For second behav session: 'ses-02' ###
# SUBJECTS = GOOD_SUBJECTS
# SESSTYPES = ['behavioral_session']
# SESSIONS = ['ses-02']
# tag = SESSIONS[0]

# ### For third behav session: 'ses-03' ###
# SUBJECTS = GOOD_SUBJECTS
# SESSTYPES = ['behavioral_session']
# SESSIONS = ['ses-03']
# tag = SESSIONS[0]

# ### For first img session: 'ses-04' ###
# SUBJECTS = IMG_SUBJECTS
# SESSTYPES = ['imaging_session']
# SESSIONS = ['ses-01']
# tag = 'ses-04'

# ### For second img session: 'ses-05' ###
# SUBJECTS = IMG_SUBJECTS
# SESSTYPES = ['imaging_session']
# SESSIONS = ['ses-02']
# tag = 'ses-05'

# ### For first and second behav sessions: ###
# ### 'ses-01' and 'ses-02' ###
# SUBJECTS = GOOD_SUBJECTS
# SESSTYPES = ['behavioral_session']
# SESSIONS = ['ses-01', 'ses-02']
# tag = 'behav12'

# ### For first and third behav sessions: ###
# ### 'ses-01' and 'ses-03' ###
# SUBJECTS = GOOD_SUBJECTS
# SESSTYPES = ['behavioral_session']
# SESSIONS = ['ses-01', 'ses-03']
# tag = 'behav13'

# ### For second and third behav sessions: ###
# ### 'ses-02' and 'ses-03' ###
# SUBJECTS = GOOD_SUBJECTS
# SESSTYPES = ['behavioral_session']
# SESSIONS = ['ses-02', 'ses-03']
# tag = 'behav23'

# %%
# ========================= PARAMETERS =================================

if SUBJECTS == SB_SUBJECTS:
    batch_tag = 'second'
else:
    batch_tag = 'first'
results_subfolder = 'perception_results_' + batch_tag + '_batch'

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FOLDER = os.path.join(MAIN_DIR, 
                              results_subfolder,
                              'raw_dataframes')
# %%
# ============================ RUN =====================================

if __name__ == "__main__":

    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)

    # Create the dataframe
    perception_dataframe(SUBJECTS, MAIN_DIR, RESULTS_FOLDER, SESSTYPES,
                         N_TRIALS, sesstag=tag, sessions=SESSIONS)
