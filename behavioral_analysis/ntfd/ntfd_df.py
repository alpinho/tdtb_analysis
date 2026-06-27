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

# The task variation is determined by the presence of random-condition trials
# in a run, not by the run number or the trial condition: if a run contains
# any random trial, every trial in that run (beat, interval and random) is
# labelled 'NTFD Rand'; otherwise the run is 'NTFD'. (session, run) keys are
# used because run numbers restart within each session.

def random_run_blocks(trials):
    # Set of (session, run) blocks that contain at least one random trial.
    # trials columns: 0 subject, 1 session, 2 run, 3 condition, ...
    blocks = set()
    for tr in trials:
        if tr[3][:6] == 'random':
            blocks.add((tr[1], tr[2]))
    return blocks


def task_label(session, run, rand_blocks):
    if (session, run) in rand_blocks:
        return 'NTFD Rand'
    return 'NTFD'


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

            # Determine which (session, run) blocks are NTFD Rand, i.e. the
            # runs that contain random trials. The whole run inherits that
            # label, so beat/interval trials in those runs become NTFD Rand.
            rand_blocks = random_run_blocks(trials_extended)

            # Append task info in the third position of the row, per trial.
            tbeat = np.array(
                [task_label(tr[1], tr[2], rand_blocks) for tr in beat_trials])
            beat_trials = np.insert(beat_trials, 3, tbeat, axis=1)

            tinterval = np.array(
                [task_label(tr[1], tr[2], rand_blocks)
                 for tr in interval_trials])
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
                trandom = np.array(
                    [task_label(tr[1], tr[2], rand_blocks)
                     for tr in random_trials])
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
ALL_SB_SUBJECTS = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]

GOOD_SB_SUBJECTS = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60]

SB2_SUBJECTS = [50, 51, 52, 57]

SB3_SUBJECTS = []

# ##################### Trial counts ###################################
# Total number of trials per run
N_TRIALS = 30
# Total number of trials per isi per condition (without random condition)...
# ... across all runs of every behavioral sessions
N_ISI_TRIALS_BEHAV = 36  # (3*4*3) --> (n_trials * n_ntfd_runs * n_sessions)
# Total number of trials per isi per condition (without random condition)...
# ... across all runs of every imaging sessions
N_ISI_TRIALS_IMG = 16  # (3*2*2 + 2*2*1) -->
                       # --> (n_trials * n_ntfd_runs * n_sessions)

# ################# Session-aggregation definitions ####################
# Maps each session tag to (SESSTYPES, SESSIONS), the two arguments that
# select which logfiles ntfd_dataframe reads. The mapping is
# batch-independent: it defines what each aggregation means, while the
# per-batch dictionaries below decide which subjects each tag uses.
#
#   key       -- the session tag; also the output filename suffix, i.e.
#                df_ntfd_<tag>.tsv.
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
               'results_subfolder': 'ntfd_results_first_batch'},
    'second': {'subjects': sb_subjects_dic,
               'results_subfolder': 'ntfd_results_second_batch'},
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
            MAIN_DIR, batch_info['results_subfolder'], 'dataframes')

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

            print(f'  Generating df_ntfd_{tag}.tsv  '
                  f'(sesstypes={sesstypes}, sessions={sessions}, '
                  f'n_subjects={len(subjects)})')

            ntfd_dataframe(subjects, MAIN_DIR, results_folder, sesstypes,
                           N_TRIALS, sesstag=tag, sessions=sessions)