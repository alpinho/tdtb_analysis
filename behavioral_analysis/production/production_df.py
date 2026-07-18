"""
Create dataframe of data from Production Tasks of the
TDTB project

author: Ana Luisa Pinho
e-mail: agrilopi@uwo.ca

Created: May 4, 2024
Last update: July 2026

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
            elif (
                  data[dt+8][5] == 'feedback' and 
                  data[dt+8][11] in ['None', '-']):
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
ALL_SB_SUBJECTS = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
                   62, 63]

GOOD_SB_SUBJECTS = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61,
                    62, 63]

SB2_SUBJECTS = [50, 51, 52, 55, 57, 59, 62]

SB3_SUBJECTS = []

# ##################### Trial counts ###################################
N_TRIALS = 30

# ##################### Latency correction #############################
# Two latency configurations ("input types") are generated per batch:
#   'latency_corrected' -- the batch's acquisition latencies (ms) are
#                          subtracted from the response times before the
#                          asynchronies are computed.
#   'uncorrected'       -- no correction (0/0/0).
# The latency values are defined per batch and input type in the *_inputs_dic
# dictionaries below, and are embedded in the output filename, e.g.
# df_production_fb_133_35_20_<tag>.tsv vs df_production_fb_0_0_0_<tag>.tsv.
# Only the audio latency differs between batches (first batch: Expyriment;
# second batch: PsychoPy); the visual and button-press latencies match.

# ################# Session-aggregation definitions ####################
# Maps each session tag to (SESSTYPES, SESSIONS), the two arguments that
# select which logfiles production_dataframe reads. The mapping is
# batch-independent: it defines what each aggregation means, while the
# per-batch dictionaries below decide which subjects and latencies each tag
# uses.
#
#   key       -- the session tag; also the trailing part of the output
#                filename:
#                df_production_<batch>_<audio>_<visual>_<button>_<tag>.tsv.
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

# ##################### Per-batch configuration ########################
# For each batch: the filename prefix code ('fb'/'sb'), the subject list per
# session tag, and the acquisition latencies (applied only when APPLY_LATENCY
# is True). The subject keys also decide which aggregations are generated for
# that batch (e.g. the second batch has no imaging sessions).
# >>> VERIFY THE SUBJECT ASSIGNMENTS <<< -- inferred from the previous manual
# blocks (matching ntfd_df.py / perception_df.py).
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

# Latency triples (audio, visual, button-press in ms) per input type.
# 'latency_corrected' differs by batch (audio 133 vs 63); 'uncorrected' is
# always 0/0/0.
fb_inputs_dic = {
    'latency_corrected': {'audio_latency': 133, 'visual_latency': 35,
                          'button_press': 20},   # Expyriment
    'uncorrected':       {'audio_latency': 0, 'visual_latency': 0,
                          'button_press': 0},
}

sb_inputs_dic = {
    'latency_corrected': {'audio_latency': 63, 'visual_latency': 35,
                          'button_press': 20},   # PsychoPy
    'uncorrected':       {'audio_latency': 0, 'visual_latency': 0,
                          'button_press': 0},
}

batch_dic = {
    'first':  {'prefix': 'fb', 'subjects': fb_subjects_dic,
               'inputs': fb_inputs_dic},
    'second': {'prefix': 'sb', 'subjects': sb_subjects_dic,
               'inputs': sb_inputs_dic},
}

# ##################### Run selection ##################################
# Batches to generate: ['first'], ['second'], or ['first', 'second'].
# BATCHES_TO_RUN = ['first', 'second']
BATCHES_TO_RUN = ['second']

# Latency input types to generate per batch: ['latency_corrected'],
# ['uncorrected'], or both.
# INPUT_TYPES_TO_RUN = ['latency_corrected', 'uncorrected']
INPUT_TYPES_TO_RUN = ['uncorrected']

# Session aggregations to generate. Set to None to run every tag available
# for each batch, or list a subset, e.g. ['imgses'] or ['allses', 'imgses'].
# Tags not defined for a given batch are skipped.
SESSION_TAGS_TO_RUN = None


# %%
# ========================= PARAMETERS =================================

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FOLDER = os.path.join(MAIN_DIR, 'production_results', 'dataframes')


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

    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)

    for batch_tag in BATCHES_TO_RUN:
        batch_info = batch_dic[batch_tag]
        subjects_dic = batch_info['subjects']
        inputs_dic = batch_info['inputs']

        if SESSION_TAGS_TO_RUN is None:
            tags = list(subjects_dic)
        else:
            tags = SESSION_TAGS_TO_RUN

        for input_type in INPUT_TYPES_TO_RUN:
            latencies = inputs_dic[input_type]
            audio_latency = latencies['audio_latency']
            visual_latency = latencies['visual_latency']
            button_press = latencies['button_press']

            subjects_batch = (
                f"df_production_{batch_info['prefix']}_"
                f"{audio_latency}_{visual_latency}_{button_press}")

            print('\n' + '=' * 64)
            print(f'Batch: {batch_tag}  |  input: {input_type}  |  '
                  f'prefix: {subjects_batch}')
            print('=' * 64)

            for tag in tags:
                if tag not in subjects_dic:
                    print(f'  Skipping {tag!r}: not defined for the '
                          f'{batch_tag} batch.')
                    continue

                subjects = subjects_dic[tag]
                sesstypes, sessions = SESSION_CONFIG[tag]

                print(f'  Generating {subjects_batch}_{tag}.tsv  '
                      f'(sesstypes={sesstypes}, sessions={sessions}, '
                      f'n_subjects={len(subjects)})')

                production_dataframe(
                    subjects, MAIN_DIR, RESULTS_FOLDER, sesstypes,
                    N_TRIALS, subjects_batch,
                    sesstag=tag,
                    sessions=sessions,
                    audio_latency=audio_latency,
                    visual_latency=visual_latency,
                    button_press=button_press)