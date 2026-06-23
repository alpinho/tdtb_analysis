"""
Parse logfiles and create dataframes for NTFD Tasks of the
Music-SDTB project

author: Ana Luisa Pinho
e-mail: agrilopi@uwo.ca

Created: May 4, 2024
Last update: June 2026

Compatibility: Python 3.10.14
"""

import os
import sys
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


def ntfd_data(data):
    trials = []
    for dt, datum in enumerate(data):
        if datum[5] == 'feedback':
            subject = datum[0]
            session = datum[1]
            run_id = datum[2]
            condition = datum[4]
            stim = data[dt - 1][5]
            theoretical_isi1 = int(data[dt - 2][8])
            if datum[11] in ['o', 'p', 'b', 'y']:
                rt = int(data[dt - 1][7]) + int(datum[10])
                answer = datum[11]
            elif datum[10] in ['None', '-']:
                rt = np.nan
                answer = 'None'
            else:
                raise ValueError('No feedback entry!')
            trials.append([subject, session, run_id, condition[:-2], stim,
                           theoretical_isi1, rt, answer])

    return trials


def success(trials, subject):
    if subject == 4:
        high_cir = ['o', 'y']
        low_tri = ['p', 'b']
    else:
        high_cir = ['o', 'b']
        low_tri = ['p', 'y']
    scores = []
    for trial in trials:
        if trial[7] in high_cir and trial[4] in ['beep_880hz', 'circle']:
            scores.append(1)
        elif trial[7] in low_tri and trial[4] in [
                'beep_220hz', 'triangle']:
            scores.append(1)
        elif trial[7] == 'None':
            scores.append(np.nan)
        else:
            scores.append(0)

    return scores


def ntfd_dataframe(subjects, this_dir, output_dir, sesstype, n_trials,
                   sesstag=None, sessions=None,
                   tasks=['Auditory No-Temporal Feature Discrimination',
                          'Visual No-Temporal Feature Discrimination']):

    # Define columns of dataframe
    df = pd.DataFrame(columns=[
        'subject', 'session', 'run', 'task', 'modality', 'condition',
        'standard', 'reaction_time', 'answer', 'score'])

    logfiles_dir = os.path.join(
        os.path.abspath(os.path.join(this_dir, os.pardir, os.pardir)),
        'logfiles')

    trials_arr = np.empty((0, df.columns.size))
    for subject in subjects:
        for task in tasks:
            if task not in ['Auditory No-Temporal Feature Discrimination',
                            'Visual No-Temporal Feature Discrimination']:
                raise NameError('Task not valid!')

            data = parse_logfile(logfiles_dir, subject, sesstype, task,
                                 n_trials, sessions=sessions,
                                 renumber_sessions=True)
            if subject == 2 and \
               task == 'Visual No-Temporal Feature Discrimination':
                data = data[:476]
            trials = ntfd_data(data)

            success_scores = success(trials, subject)

            # Convert into array and preserve data types
            trials = np.array(trials, dtype=object)
            success_scores = np.array(success_scores, dtype=object)

            # Remove stim element from trials
            trials = np.delete(trials, 4, axis=1)

            # Stack success_scores as the last column
            trials_extended = np.column_stack((trials, success_scores))

            # Get beat, interval and random trials to stack them later in
            # groups of beat, interval and random trials
            beat_trials = np.array([
                tr for tr in trials_extended if tr[3][:4] == 'beat'])
            interval_trials = np.array([
                tr for tr in trials_extended if tr[3][:8] == 'interval'])
            random_trials = np.array([
                tr for tr in trials_extended if tr[3][:6] == 'random'])

            # Append task info in the third position of the row
            ntfd_tag = np.array(['NTFD'])
            ntfd_rand_tag = np.array(['NTFD Rand'])

            tbeat = np.repeat(ntfd_tag, beat_trials.shape[0])
            beat_trials = np.insert(beat_trials, 3, tbeat, axis=1)

            tinterval = np.repeat(ntfd_tag, interval_trials.shape[0])
            interval_trials = np.insert(interval_trials, 3, tinterval, axis=1)

            # Append modality info after task
            modality = np.array([task.partition(' ')[0].lower()])

            mbeat = np.repeat(modality, beat_trials.shape[0])
            table_beat = np.insert(beat_trials, 4, mbeat, axis=1)

            minterval = np.repeat(modality, interval_trials.shape[0])
            table_interval = np.insert(interval_trials, 4, minterval, axis=1)

            # Stack
            trials_arr = np.vstack((trials_arr, table_beat))
            trials_arr = np.vstack((trials_arr, table_interval))

            # Do the same for random trials if they exist
            if random_trials.size != 0:
                trandom = np.repeat(ntfd_rand_tag, random_trials.shape[0])
                random_trials = np.insert(random_trials, 3, trandom, axis=1)

                mrandom = np.repeat(modality, random_trials.shape[0])
                table_random = np.insert(random_trials, 4, mrandom, axis=1)
                trials_arr = np.vstack((trials_arr, table_random))

    # Add data to dataframe
    df = pd.DataFrame(trials_arr, columns=df.columns)

    # Save dataframe
    if sesstag:
        outpath = os.path.join(output_dir, 'df_ntfd_' + sesstag + '.tsv')
    else:
        outpath = os.path.join(output_dir, 'df_ntfd.tsv')
    df.to_csv(outpath, index=False, sep='\t', na_rep='NaN')


# %%
# =========================== INPUTS ===================================

# ##################### Subjects' lists ################################
# All subjects
ALL_SUBJECTS = [
    3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36,
    37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

# All good subjects including img pilot (sub-04)
GOOD_SUBJECTS = [
    3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 32, 34, 35, 38, 39, 40,
    41, 42, 43, 44, 45, 46, 47]

# Img subjects only (without pilot)
IMG_SUBJECTS = [
    3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26,
    28, 29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

# Second batch
ALL_SB_SUBJECTS = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

GOOD_SB_SUBJECTS = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59]

SB2_SUBJECTS = [51, 52, 57]

SB3_SUBJECTS = []

# #######################################################################

# TASKS = ['Auditory No-Temporal Feature Discrimination',
#          'Visual No-Temporal Feature Discrimination']

# Total number of trials per run
N_TRIALS = 30
# Total number of trials per isi per condition (without random condition)...
# ... across all runs of every behavioral sessions
N_ISI_TRIALS_BEHAV = 36  # (3*4*3) --> (n_trials * n_ntfd_runs * n_sessions)
# Total number of trials per isi per condition (without random condition)...
# ... across all runs of every imaging sessions
N_ISI_TRIALS_IMG = 16  # (3*2*2 + 2*2*1) -->
                       # --> (n_trials * n_ntfd_runs * n_sessions)

# ### For 'All Sessions' ###
# SUBJECTS = GOOD_SUBJECTS
# SESSTYPES = ['behavioral_session', 'imaging_session']
# SESSIONS = None
# tag = 'allses'

# ### For 'All Behavioral Sessions' ###
# SUBJECTS = GOOD_SB_SUBJECTS  # GOOD_SUBJECTS / GOOD_SB_SUBJECTS
# SESSTYPES = ['behavioral_session']
# SESSIONS = None
# tag = 'behavses'

# ### For 'All Imaging Sessions' ###
# SUBJECTS = IMG_SUBJECTS
# SESSTYPES = ['imaging_session']
# SESSIONS = None
# tag = 'imgses'

# ### For first behav session: 'ses-01' ###
# SUBJECTS = GOOD_SB_SUBJECTS  # GOOD_SUBJECTS / GOOD_SB_SUBJECTS
# SESSTYPES = ['behavioral_session']
# SESSIONS = ['ses-01']
# tag = SESSIONS[0]

# ### For second behav session: 'ses-02' ###
SUBJECTS = SB2_SUBJECTS
SESSTYPES = ['behavioral_session']
SESSIONS = ['ses-02']
tag = SESSIONS[0]

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

if SUBJECTS in [GOOD_SB_SUBJECTS, ALL_SB_SUBJECTS, SB2_SUBJECTS, SB3_SUBJECTS]:
    batch_tag = 'second'
else:
    batch_tag = 'first'
results_subfolder = 'ntfd_results_' + batch_tag + '_batch'

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FOLDER = os.path.join(MAIN_DIR, results_subfolder, 'dataframes')

if SESSTYPES == ['behavioral_session', 'imaging_session']:
    N_ISI_TRIALS = N_ISI_TRIALS_BEHAV + N_ISI_TRIALS_IMG
elif SESSTYPES == ['behavioral_session']:
    N_ISI_TRIALS = N_ISI_TRIALS_BEHAV
else:
    assert SESSTYPES == ['imaging_session']
    N_ISI_TRIALS = N_ISI_TRIALS_IMG

# %%
# ============================ RUN =====================================

if __name__ == "__main__":

    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)

    # Create dataframes
    ntfd_dataframe(SUBJECTS, MAIN_DIR, RESULTS_FOLDER, SESSTYPES, N_TRIALS,
                   sesstag=tag, sessions=SESSIONS)