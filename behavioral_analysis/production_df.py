"""
Parse logfiles and create dataframes for Production Tasks of the
 Music-SDTB project

author: Ana Luisa Pinho
e-mail: agrilopi@uwo.ca

Created: May 4, 2024
Last update: May 2024

Compatibility: Python 3.10.8
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# setting path
sys.path.append('../')
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
            condition = datum[4]
            theoretical_isi1 = int(datum[8])
            real_isi1 = int(datum[9])
            if data[dt+8][5] == 'feedback' and data[dt+8][11] in ['o', 'b']:
                rt = int(data[dt+7][7]) + int(data[dt+8][10])
            elif data[dt+8][5] == 'feedback' and data[dt+8][10] == 'None':
                rt = np.nan
            else:
                raise ValueError('No feedback entry!')
            trials.append([condition, theoretical_isi1, real_isi1, rt])

    return trials


def filter_trialtype(trs, category):
    beat = [tr[1:] for tr in trs if tr[0][:4] == 'beat']
    interval = [tr[1:] for tr in trs if tr[0][:8] == 'interval']
    random = [tr[1:] for tr in trs if tr[0][:6] == 'random']

    if category in ['production', 'ntfd']:
        beat = [list(map(int, b)) if ~np.any(np.isnan(b)) else b
                for b in beat]
        interval = [list(map(int, i)) if ~np.any(np.isnan(i)) else i
                    for i in interval]
        if random:
            random = [list(map(int, r)) if ~np.any(np.isnan(r)) else r
                      for r in random]
    else:
        assert category == 'perception'
        beat = [[int(b[0]), int(b[1]), b[2]] for b in beat]
        interval = [[int(i[0]), int(i[1]), i[2]] for i in interval]

    return beat, interval, random


def symlog_transform(arr, shift):
    """About this function, consult:
    https://pythonmatplotlibtips.blogspot.com/2018/11/x-symlog-with-shift.html
    """
    logv = np.abs(arr)*(10.**shift)
    logv[np.where(logv < 1.)] = 1.
    logv = np.sign(arr)*np.log10(logv)

    return logv


def resize_production_arr(arr):
    # Inputs shape (n_subjects, n_isi, n_trials)

    # Resize numpy arrays when there is less trials per isi because the participant
    # only did the behavioral sessions
    nt_max = np.ravel([[np.array(isi).shape[0] for isi in isis]
                       for isis in arr]).max()

    rlist = [[np.append(trials, np.repeat('n/a', nt_max - len(trials))).tolist()
              if len(trials) < nt_max else trials for trials in tisis]
             for tisis in arr]

    return rlist


def isi_async(subjects, sesstypes, this_dir, sync_type, n_trials,
              sessions=None,
              tasks=['Auditory Production', 'Visual Production']):

    logfiles_dir = os.path.join(
        os.path.abspath(os.path.join(this_dir, os.pardir)), 'logfiles')

    allsub_beat_audio = []
    allsub_interval_audio = []
    allsub_beat_visual = []
    allsub_interval_visual = []
    for s, subject in enumerate(subjects):
        for t, task in enumerate(tasks):
            if task not in ['Auditory Production', 'Visual Production']:
                raise NameError('Task not valid!')
            data = parse_logfile(logfiles_dir, subject, sesstypes, task,
                                 n_trials, sessions=sessions)
            trials = production_data(data)
            beat_trials, interval_trials, _ = filter_trialtype(trials,
                                                               'production')

            # ############# Asynchronies per ISI #######################
            isi1s = np.unique(np.array(beat_trials)[:, 0]).astype('int')

            ss_isi_beat = []
            as_isi_beat = []
            for i in isi1s:
                ss_beat = []
                as_beat = []
                for beat_trial in beat_trials:
                    if beat_trial[0] == i:
                        if ~np.any(np.isnan(beat_trial)):
                            ssb = round((beat_trial[2] - beat_trial[1]) / \
                                        beat_trial[1], 2)
                            asb = abs(ssb)
                        else:
                            ssb = np.nan
                            asb = np.nan
                        ss_beat.append(ssb)
                        as_beat.append(asb)
                # Replace missing values (nan's) by median of the sample
                # if np.any(np.isnan(ss_beat)):
                #     miss_sbval = np.nanmedian(ss_beat)
                #     ss_beat = np.where(np.isnan(ss_beat), miss_sbval,
                #                        ss_beat).tolist()
                # if np.any(np.isnan(as_beat)):
                #     miss_abval = np.nanmedian(as_beat)
                #     as_beat = np.where(np.isnan(as_beat), miss_abval,
                #                        as_beat).tolist()
                # Append isi array
                ss_isi_beat.append(ss_beat)
                as_isi_beat.append(as_beat)

            ss_isi_interval = []
            as_isi_interval = []
            for i in isi1s:
                ss_interval = []
                as_interval = []
                for interval_trial in interval_trials:
                    if interval_trial[0] == i:
                        if ~np.any(np.isnan(interval_trial)):
                            ssi = round((interval_trial[2] - \
                                         interval_trial[1]) / \
                                        interval_trial[1], 2)
                            asi = abs(ssi)
                        else:
                            ssi = np.nan
                            asi = np.nan
                        ss_interval.append(ssi)
                        as_interval.append(asi)
                # Replace missing values (nan's) by median of the isi sample
                # if np.any(np.isnan(ss_interval)):
                #     miss_sival = np.nanmedian(ss_interval)
                #     ss_interval = np.where(np.isnan(ss_interval), miss_sival,
                #                            ss_interval).tolist()
                # if np.any(np.isnan(as_interval)):
                #     miss_aival = np.nanmedian(as_interval)
                #     as_interval = np.where(np.isnan(as_interval), miss_aival,
                #                            as_interval).tolist()
                # Append isi array
                ss_isi_interval.append(ss_interval)
                as_isi_interval.append(as_interval)
            
            # Aggregate all data
            if task == 'Auditory Production' and sync_type == 'signed':
                allsub_beat_audio.append(ss_isi_beat)
                allsub_interval_audio.append(ss_isi_interval)
            elif task == 'Visual Production' and sync_type == 'signed':
                allsub_beat_visual.append(ss_isi_beat)
                allsub_interval_visual.append(ss_isi_interval)
            elif task == 'Auditory Production' and sync_type == 'absolute':
                allsub_beat_audio.append(as_isi_beat)
                allsub_interval_audio.append(as_isi_interval)
            else:
                assert task == 'Visual Production' and sync_type == 'absolute'
                allsub_beat_visual.append(as_isi_beat)
                allsub_interval_visual.append(as_isi_interval)


    return (allsub_beat_audio, allsub_interval_audio, allsub_beat_visual,
            allsub_interval_visual, isi1s)


def long_dataframe(sync_audio_beat, sync_audio_interval, sync_visual_beat,
                   sync_visual_interval, stand_numbers, subjects, output_dir,
                   sesstag):
    # Inputs shape (n_subjects, n_isi, n_trials)

    conditions_names = np.array(['beat', 'interval'])
    modalities_names = np.array(['audio', 'visual'])

    # Flatten the synchronies arrays
    sync_audio_beat_flatten = np.ravel(sync_audio_beat)
    sync_audio_interval_flatten = np.ravel(sync_audio_interval)
    sync_visual_beat_flatten = np.ravel(sync_visual_beat)
    sync_visual_interval_flatten = np.ravel(sync_visual_interval)

    # Stack synchronies in one single array
    asynchronies = np.hstack((sync_audio_beat_flatten,
                              sync_audio_interval_flatten,
                              sync_visual_beat_flatten,
                              sync_visual_interval_flatten))

    # ## Standards column
    trial_standards = np.repeat(
        stand_numbers, np.array(sync_audio_beat).shape[2])
    standards_col = np.tile(trial_standards, len(sync_audio_beat) * 4)

    # ## Subjects column
    standard_subjects = np.repeat(SUBJECTS, trial_standards.shape[0])
    subjects_col = np.tile(standard_subjects, 4)

    # ## Modality column
    modalities_col = np.repeat(modalities_names,
                               len(sync_audio_beat_flatten) * 2)

    # ## Conditions column
    conditions_stack = np.repeat(conditions_names,
                                 len(sync_audio_beat_flatten))
    conditions_col = np.tile(conditions_stack, 2)

    # Session column
    sessions_col = np.repeat(sesstag, len(asynchronies))

    # ## Build tables and dataframes
    table = np.vstack((asynchronies, standards_col, subjects_col,
                       modalities_col, conditions_col, sessions_col)).T

    df = pd.DataFrame(table, columns=['Asynchronies', 'Standard', 'Subject',
                                      'Modality', 'Condition', 'Session'])
    # df['Asynchronies'] = df['Asynchronies'].apply(pd.to_numeric)

    output_folder = os.path.join(output_dir, 'dataframes')
    # Create output_folder, if it does not exist
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    # Save dataframe
    outpath = os.path.join(output_folder, 'df_production_' + sesstag + '.tsv')
    df.to_csv(outpath, index=False, sep='\t')

    return df


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
RESULTS_FOLDER = os.path.join(MAIN_DIR, 'production_results')

# %%
# ============================ RUN =====================================

if __name__ == "__main__":

    if not os.path.exists(RESULTS_FOLDER):
        os.mkdir(RESULTS_FOLDER)

    # Compute asynchronies
    ssync_audio_beat, ssync_audio_interval, ssync_visual_beat, \
        ssync_visual_interval, standards = isi_async(
            SUBJECTS, SESSTYPES, MAIN_DIR, 'signed', N_TRIALS,
            sessions=SESSIONS)

    # Resize
    rsized_ssync_audio_beat = resize_production_arr(ssync_audio_beat)
    rsized_ssync_audio_interval = resize_production_arr(ssync_audio_interval)
    rsized_ssync_visual_beat = resize_production_arr(ssync_visual_beat)
    rsized_ssync_visual_interval = resize_production_arr(ssync_visual_interval)

    # Build and save dataframe in the long format
    long_dataframe(rsized_ssync_audio_beat, rsized_ssync_audio_interval,
                   rsized_ssync_visual_beat, rsized_ssync_visual_interval,
                   standards, SUBJECTS, RESULTS_FOLDER, tag)
