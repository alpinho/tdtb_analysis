"""
Create dataframe of raw data from Perception Tasks of the
TDTB project

author: Ana Luisa Pinho
e-mail: agrilopi@uwo.ca

Created: 28th of January 2025
Last update: July 2026

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
ALL_SB_SUBJECTS = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]

GOOD_SB_SUBJECTS = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61, 62]

SB2_SUBJECTS = [50, 51, 52, 55, 57, 59]

SB3_SUBJECTS = []

# ##################### Trial counts ###################################
N_TRIALS = 30

# ################# Session-aggregation definitions ####################
# Maps each session tag to (SESSTYPES, SESSIONS), the two arguments that
# select which logfiles perception_dataframe reads. The mapping is
# batch-independent: it defines what each aggregation means, while the
# per-batch dictionaries below decide which subjects each tag uses.
#
#   key       -- the session tag; also the output filename suffix, i.e.
#                df_perception_<tag>.tsv.
#   SESSTYPES -- list of session types to read: 'behavioral_session',
#                'imaging_session', or both.
#   SESSIONS  -- list of logfile session ids within those types (e.g.
#                'ses-01'), or None to take every session of the type(s).
#                Imaging ids restart at 'ses-01', so imaging 'ses-01' and
#                'ses-02' are renumbered to output sessions 4 and 5.
#
# Entries:
#   'allses'   -- all behavioural and imaging sessions together (1-5).
#   'behavses' -- all behavioural sessions (1-3).
#   'imgses'   -- all imaging sessions (4-5).
#   'ses-01'   -- behavioural session 1.
#   'ses-02'   -- behavioural session 2.
#   'ses-03'   -- behavioural session 3.
#   'ses-04'   -- imaging session 1 (logfile 'ses-01'), output session 4.
#   'ses-05'   -- imaging session 2 (logfile 'ses-02'), output session 5.
#   'behav12'  -- behavioural sessions 1 and 2.
#   'behav13'  -- behavioural sessions 1 and 3.
#   'behav23'  -- behavioural sessions 2 and 3.
SESSION_CONFIG = {
    'allses':   (['behavioral_session', 'imaging_session'], None),
    'behavses': (['behavioral_session'], None),
    'imgses':   (['imaging_session'], None),
    'ses-01':   (['behavioral_session'], ['ses-01']),
    'ses-02':   (['behavioral_session'], ['ses-02']),
    'ses-03':   (['behavioral_session'], ['ses-03']),
    'ses-04':   (['imaging_session'], ['ses-01']),
    'ses-05':   (['imaging_session'], ['ses-02']),
    'behav12':  (['behavioral_session'], ['ses-01', 'ses-02']),
    'behav13':  (['behavioral_session'], ['ses-01', 'ses-03']),
    'behav23':  (['behavioral_session'], ['ses-02', 'ses-03']),
}

# ##################### Per-batch subjects #############################
# For each batch, map every session tag to its subject list. The keys also
# decide which aggregations are generated for that batch (e.g. the second
# batch has no imaging sessions).
fb_subjects_dic = {
    'allses':   GOOD_SUBJECTS,
    'behavses': GOOD_SUBJECTS,
    'imgses':   IMG_SUBJECTS,
    'ses-01':   GOOD_SUBJECTS,
    'ses-02':   GOOD_SUBJECTS,
    'ses-03':   GOOD_SUBJECTS,
    'ses-04':   IMG_SUBJECTS,
    'ses-05':   IMG_SUBJECTS,
    'behav12':  GOOD_SUBJECTS,
    'behav13':  GOOD_SUBJECTS,
    'behav23':  GOOD_SUBJECTS,
}

sb_subjects_dic = {
    'behavses': GOOD_SB_SUBJECTS,
    'ses-01':   GOOD_SB_SUBJECTS,
    'ses-02':   SB2_SUBJECTS,
}

batch_dic = {
    'first':  {'subjects': fb_subjects_dic,
               'results_subfolder': 'perception_results_first_batch'},
    'second': {'subjects': sb_subjects_dic,
               'results_subfolder': 'perception_results_second_batch'},
}

# ##################### Run selection ##################################
# Batches to generate: ['first'], ['second'], or ['first', 'second'].
# BATCHES_TO_RUN = ['first', 'second']
BATCHES_TO_RUN = ['second']

# Session aggregations to generate. Set to None to run every tag available
# for each batch, or list a subset, e.g. ['imgses'] or ['allses', 'imgses'].
# Tags not defined for a given batch are skipped.
SESSION_TAGS_TO_RUN = None


# %%
# ========================= PARAMETERS =================================

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))


# %%
# ============================ RUN =====================================

if __name__ == "__main__":

    # Every per-batch subject tag must be defined in SESSION_CONFIG.
    for _dic in (fb_subjects_dic, sb_subjects_dic):
        _unknown = set(_dic) - set(SESSION_CONFIG)
        if _unknown:
            raise KeyError(
                'Session tags missing from SESSION_CONFIG: ' +
                str(sorted(_unknown)))

    for batch_tag in BATCHES_TO_RUN:
        batch_info = batch_dic[batch_tag]
        subjects_dic = batch_info['subjects']
        results_folder = os.path.join(
            MAIN_DIR, batch_info['results_subfolder'], 'raw_dataframes')

        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        if SESSION_TAGS_TO_RUN is None:
            tags = list(subjects_dic)
        else:
            tags = SESSION_TAGS_TO_RUN

        print('\n' + '=' * 64)
        print(f'Batch: {batch_tag}  |  output: {results_folder}')
        print('=' * 64)

        for tag in tags:
            if tag not in subjects_dic:
                print(f'  Skipping {tag!r}: not defined for the '
                      f'{batch_tag} batch.')
                continue

            subjects = subjects_dic[tag]
            sesstypes, sessions = SESSION_CONFIG[tag]

            print(f'  Generating df_perception_{tag}.tsv  '
                  f'(sesstypes={sesstypes}, sessions={sessions}, '
                  f'n_subjects={len(subjects)})')

            perception_dataframe(subjects, MAIN_DIR, results_folder, sesstypes,
                                 N_TRIALS, sesstag=tag, sessions=sessions)