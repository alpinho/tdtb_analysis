"""
Parse logfiles and create dataframes for NTFD Tasks of the
 Music-SDTB project

author: Ana Luisa Pinho
e-mail: agrilopi@uwo.ca

Created: May 4, 2024
Last update: January, 2025

Compatibility: Python 3.10.14
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# setting path
sys.path.append('../')
# importing
from utils import parse_logfile, filter_trialtype

import numpy as np
import pandas as pd

# %%
# ======================== MAIN FUNCTIONS ==============================


def ntfd_data(data):
    trials = []
    for dt, datum in enumerate(data):
        if datum[5] == 'feedback':
            condition = datum[4]
            theoretical_isi1 = int(data[dt-2][8])
            if datum[11] in ['o', 'p', 'b', 'y']:
                rt = int(data[dt-1][7]) + int(datum[10])
            elif datum[10] == 'None':
                rt = np.nan
            else:
                raise ValueError('No feedback entry!')
            trials.append([condition, theoretical_isi1, rt])

    return trials


def success_trialtype_filter(data):
    trial_beat = [dt for dt in data if dt[4][:4] == 'beat']
    trial_interval = [dt for  dt in data if dt[4][:8] == 'interval']
    trial_random = [dt for dt in data if dt[4][:6] == 'random']

    return trial_beat, trial_interval, trial_random


def success(data, subject):
    if subject == 4:
        high_cir = ['o', 'y']
        low_tri = ['p', 'b']
    else:
        high_cir = ['o', 'b']
        low_tri = ['p', 'y']
    scores = []
    for dt, datum in enumerate(data):
        if datum[5] == 'feedback':
            answer = datum[11]
            dstimulus = data[dt-1][5]
            if answer in high_cir and dstimulus in ['beep_880hz', 'circle']:
                scores.append(1)
            elif answer in low_tri and dstimulus in ['beep_220hz', 'triangle']:
                scores.append(1)
            elif answer == 'None':
                scores.append(np.nan)
            else:
                scores.append(0)

    # Replace missing values (nan's) by median of the all sample
    # if np.any(np.isnan(scores)):
    #     missval = np.nanmedian(scores)
    #     scores = np.where(np.isnan(scores), missval, scores).tolist()

    return scores


def resize_ntfd_arr(arr):
    """
    Resize numpy arrays when there is less trials per isi because
    the participant only did the behavioral sessions
    """
    maxlength = np.amax([np.array(arr0).shape[0] for arr0 in arr])
    new_arr = [
        np.append(arr0, np.repeat('n/a', maxlength - len(arr0))).tolist()
        if len(arr0) < maxlength else arr0 for arr0 in arr]

    return new_arr


def ntfd_isi_dataframe(
        subjects, this_dir, output_dir, sesstype, n_trials, n_isi_trials,
        sesstag, sessions=None,
        tasks=['Auditory No-Temporal Feature Discrimination',
               'Visual No-Temporal Feature Discrimination']):

    logfiles_dir = os.path.join(
        os.path.abspath(os.path.join(this_dir, os.pardir)), 'logfiles')

    allsub_beat_audio = []
    allsub_interval_audio = []
    allsub_beat_visual = []
    allsub_interval_visual = []
    subjects_id = []
    for s, subject in enumerate(subjects):
        for t, task in enumerate(tasks):
            if task not in ['Auditory No-Temporal Feature Discrimination',
                            'Visual No-Temporal Feature Discrimination']:
                raise NameError('Task not valid!')

            data = parse_logfile(logfiles_dir, subject, sesstype, task,
                                 n_trials, sessions=sessions)
            if subject == 2 and \
               task == 'Visual No-Temporal Feature Discrimination':
                data = data[:476]
            trials = ntfd_data(data)
            beat_trials, interval_trials, _ = filter_trialtype(trials, 'ntfd')

            # ############## Extract RT's per ISI ######################
            isi1s = np.unique(np.array(beat_trials)[:, 0]).astype('int')

            rt_isi1_grouped_beat = []
            for i in isi1s:
                rts_beat = []
                for beat_trial in beat_trials:
                    if beat_trial[0] == i:
                        if ~np.any(np.isnan(beat_trial)):
                            rts_beat.append(beat_trial[1])
                        else:
                            rts_beat.append(np.nan)
                # Replace missing values (nan's) by median of the isi sample
                # if np.any(np.isnan(rts_beat)):
                #     miss_bval = np.nanmedian(rts_beat)
                #     rts_beat = np.where(np.isnan(rts_beat), miss_bval,
                #                         rts_beat).tolist()

                rt_isi1_grouped_beat.append(rts_beat)

            rt_isi1_grouped_interval = []
            for j in isi1s:
                rts_interval = []
                for interval_trial in interval_trials:
                    if interval_trial[0] == j:
                        if ~np.any(np.isnan(interval_trial)):
                            rts_interval.append(interval_trial[1])
                        else:
                            rts_interval.append(np.nan)
                # Replace missing values (nan's) by median of the sample
                # if np.any(np.isnan(rts_interval)):
                #     miss_ival = np.nanmedian(rts_interval)
                #     rts_interval = np.where(np.isnan(rts_interval), miss_ival,
                #                             rts_interval).tolist()

                rt_isi1_grouped_interval.append(rts_interval)

            # Aggregate data
            diff = n_isi_trials - np.array(rt_isi1_grouped_interval).shape[1]
            if diff != 0:
                # Add 'n/a' for subjects who have less data because of
                # the introduction of the random condition
                rt_isi1_grouped_beat = [
                    np.append(rb, np.repeat('n/a', diff)).tolist()
                    for rb in rt_isi1_grouped_beat]
                rt_isi1_grouped_interval = [
                    np.append(ri, np.repeat('n/a', diff)).tolist()
                    for ri in rt_isi1_grouped_interval]
            if task == 'Auditory No-Temporal Feature Discrimination':
                allsub_beat_audio.append(rt_isi1_grouped_beat)
                allsub_interval_audio.append(rt_isi1_grouped_interval)
            else:
                assert task == 'Visual No-Temporal Feature Discrimination'
                allsub_beat_visual.append(rt_isi1_grouped_beat)
                allsub_interval_visual.append(rt_isi1_grouped_interval)

    # Flatten RT's
    allsub_beat_audio_flatten = np.ravel(allsub_beat_audio)
    allsub_interval_audio_flatten = np.ravel(allsub_interval_audio)
    allsub_beat_visual_flatten = np.ravel(allsub_beat_visual)
    allsub_interval_visual_flatten = np.ravel(allsub_interval_visual)

    reaction_times = np.concatenate((allsub_beat_audio_flatten,
                                     allsub_interval_audio_flatten,
                                     allsub_beat_visual_flatten,
                                     allsub_interval_visual_flatten))

    # Standards column
    icm_standards = np.repeat(isi1s, n_isi_trials)
    cm_standards = np.tile(icm_standards, len(subjects))
    standards_col = np.tile(cm_standards, 4)

    # Subjects column
    cm_subjects = np.repeat(subjects, len(icm_standards))
    subjects_col = np.tile(cm_subjects, 4)

    # Conditions column
    beat_col = np.repeat('beat', len(allsub_beat_audio_flatten))
    interval_col = np.repeat('interval', len(allsub_interval_audio_flatten))
    cond_col = np.concatenate((beat_col, interval_col, beat_col, interval_col))

    # Modality column
    audio_col = np.repeat('audio', len(allsub_beat_audio_flatten) * 2)
    visual_col = np.repeat('visual', len(allsub_beat_visual_flatten) * 2)
    mod_col = np.concatenate((audio_col, visual_col))

    # Session column
    ses_col = np.repeat(sesstag, len(reaction_times))

    # Build dataframe
    table = np.vstack((reaction_times, standards_col, subjects_col, cond_col,
                       mod_col, ses_col)).T
    df = pd.DataFrame(table, columns=['RT', 'Standard', 'Subject', 'Condition',
                                      'Modality', 'Session'])
    # df['RT'] = df['RT'].apply(pd.to_numeric)

    output_folder = os.path.join(output_dir, 'dataframes')
    # Create output_folder, if it does not exist
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Save dataframe
    outpath = os.path.join(output_folder, 'df_ntfd_isi_' + sesstag + '.tsv')
    df.to_csv(outpath, index=False, sep='\t')

    return df


def ntfd_dataframe(subjects, this_dir, output_dir, sesstype, n_trials,
                   sesstag, sessions=None,
                   tasks=['Auditory No-Temporal Feature Discrimination',
                          'Visual No-Temporal Feature Discrimination']):

    logfiles_dir = os.path.join(
        os.path.abspath(os.path.join(this_dir, os.pardir)), 'logfiles')

    allsub_beat_audio = []
    allsub_interval_audio = []
    allsub_random_audio = []
    allsub_beat_visual = []
    allsub_interval_visual = []
    allsub_random_visual = []
    allsub_success_rate_audio_beat = []
    allsub_success_rate_audio_interval = []
    allsub_success_rate_audio_random  = []
    allsub_success_rate_visual_beat = []
    allsub_success_rate_visual_interval = []
    allsub_success_rate_visual_random  = []
    for s, subject in enumerate(subjects):
        for t, task in enumerate(tasks):
            if task not in ['Auditory No-Temporal Feature Discrimination',
                            'Visual No-Temporal Feature Discrimination']:
                raise NameError('Task not valid!')

            data = parse_logfile(logfiles_dir, subject, sesstype, task,
                                 n_trials, sessions=sessions)

            if subject == 2 and \
               task == 'Visual No-Temporal Feature Discrimination':
                data = data[:476]

            trials = ntfd_data(data)
            beat_trials, interval_trials, random_trials = \
                filter_trialtype(trials, 'ntfd')

            # ############## Extract RT's ######################
            beat_trials = np.array([bt[1] for bt in beat_trials])
            interval_trials = np.array([it[1] for it in interval_trials])
            random_trials = np.array([rt[1] for rt in random_trials])

            # # Replace missing values (nan's) by median of the all sample
            # if np.any(np.isnan(beat_trials)):
            #     miss_bval = np.nanmedian(beat_trials)
            #     beat_trials = np.where(np.isnan(beat_trials),
            #                            miss_bval, beat_trials)

            # if np.any(np.isnan(interval_trials)):
            #     miss_ival = np.nanmedian(interval_trials)
            #     interval_trials = np.where(np.isnan(interval_trials),
            #                                miss_ival, interval_trials)

            # if np.any(np.isnan(random_trials)):
            #     miss_rval = np.nanmedian(random_trials)
            #     random_trials = np.where(np.isnan(random_trials),
            #                              miss_rval, random_trials)

            # ######### Extract the Success Rate ###############
            ft_beat, ft_interval, ft_random = success_trialtype_filter(data)
            success_rate_beat = success(ft_beat, subject)
            success_rate_interval = success(ft_interval, subject)
            success_rate_random = success(ft_random, subject)

            # Aggregate data
            if task == 'Auditory No-Temporal Feature Discrimination':
                allsub_beat_audio.append(beat_trials.tolist())
                allsub_interval_audio.append(interval_trials.tolist())
                allsub_random_audio.append(random_trials.tolist())
                allsub_success_rate_audio_beat.append(success_rate_beat)
                allsub_success_rate_audio_interval.append(
                    success_rate_interval)
                allsub_success_rate_audio_random.append(success_rate_random)
            else:
                assert task == 'Visual No-Temporal Feature Discrimination'
                allsub_beat_visual.append(beat_trials.tolist())
                allsub_interval_visual.append(interval_trials.tolist())
                allsub_random_visual.append(random_trials.tolist())
                allsub_success_rate_visual_beat.append(success_rate_beat)
                allsub_success_rate_visual_interval.append(
                    success_rate_interval)
                allsub_success_rate_visual_random.append(success_rate_random)

    # Resize outputs with 'n/a' when there is less trials because...
    # ... the participant only did the behavioral sessions
    allsub_beat_audio = resize_ntfd_arr(allsub_beat_audio)
    allsub_interval_audio = resize_ntfd_arr(allsub_interval_audio)
    allsub_random_audio = resize_ntfd_arr(allsub_random_audio)
    allsub_beat_visual = resize_ntfd_arr(allsub_beat_visual)
    allsub_interval_visual = resize_ntfd_arr(allsub_interval_visual)
    allsub_random_visual = resize_ntfd_arr(allsub_random_visual)
    allsub_success_rate_audio_beat = resize_ntfd_arr(
        allsub_success_rate_audio_beat)
    allsub_success_rate_audio_interval = resize_ntfd_arr(
        allsub_success_rate_audio_interval)
    allsub_success_rate_audio_random = resize_ntfd_arr(
        allsub_success_rate_audio_random)
    allsub_success_rate_visual_beat = resize_ntfd_arr(
        allsub_success_rate_visual_beat)
    allsub_success_rate_visual_interval = resize_ntfd_arr(
        allsub_success_rate_visual_interval)
    allsub_success_rate_visual_random = resize_ntfd_arr(
        allsub_success_rate_visual_random)

    # Flatten RT's
    allsub_beat_audio_flatten = np.ravel(allsub_beat_audio).tolist()
    allsub_interval_audio_flatten = np.ravel(allsub_interval_audio).tolist()
    allsub_random_audio_flatten = np.ravel(allsub_random_audio).tolist()
    allsub_beat_visual_flatten = np.ravel(allsub_beat_visual).tolist()
    allsub_interval_visual_flatten = np.ravel(allsub_interval_visual).tolist()
    allsub_random_visual_flatten = np.ravel(allsub_random_visual).tolist()
    allsub_success_rate_audio_beat_flatten = np.ravel(
        allsub_success_rate_audio_beat).tolist()
    allsub_success_rate_audio_interval_flatten = np.ravel(
        allsub_success_rate_audio_interval).tolist()
    allsub_success_rate_audio_random_flatten = np.ravel(
        allsub_success_rate_audio_random).tolist()
    allsub_success_rate_visual_beat_flatten = np.ravel(
        allsub_success_rate_visual_beat).tolist()
    allsub_success_rate_visual_interval_flatten = np.ravel(
        allsub_success_rate_visual_interval).tolist()
    allsub_success_rate_visual_interval_flatten = np.ravel(
        allsub_success_rate_visual_interval).tolist()
    allsub_success_rate_visual_random_flatten = np.ravel(
        allsub_success_rate_visual_random).tolist()

    # Concatenate
    reaction_times = np.concatenate((allsub_beat_audio_flatten,
                                     allsub_interval_audio_flatten,
                                     allsub_random_audio_flatten,
                                     allsub_beat_visual_flatten,
                                     allsub_interval_visual_flatten,
                                     allsub_random_visual_flatten,))

    success_scores = np.concatenate((
        allsub_success_rate_audio_beat_flatten,
        allsub_success_rate_audio_interval_flatten,
        allsub_success_rate_audio_random_flatten,
        allsub_success_rate_visual_beat_flatten,
        allsub_success_rate_visual_interval_flatten,
        allsub_success_rate_visual_random_flatten))

    # Subjects column
    beat_subjects = np.repeat(subjects, np.array(allsub_beat_audio).shape[1])
    interval_subjects = np.repeat(
        subjects, np.array(allsub_interval_audio).shape[1])
    random_subjects = np.repeat(
        subjects, np.array(allsub_random_audio).shape[1])
    subjects_col = np.concatenate((
        beat_subjects, interval_subjects, random_subjects,
        beat_subjects, interval_subjects, random_subjects))

    # Conditions column
    beat_col = np.repeat('beat', len(allsub_beat_audio_flatten))
    interval_col = np.repeat('interval', len(allsub_interval_audio_flatten))
    random_col = np.repeat('random', len(allsub_random_audio_flatten))
    cond_col = np.concatenate((beat_col, interval_col, random_col,
                               beat_col, interval_col, random_col))

    # Modality column
    audio_col = np.repeat('audio', len(cond_col) / 2)
    visual_col = np.repeat('visual', len(cond_col) / 2)
    mod_col = np.concatenate((audio_col, visual_col))

    # Session column
    ses_col = np.repeat(sesstag, len(reaction_times))

    # Build dataframe
    table = np.vstack((reaction_times, success_scores, subjects_col, cond_col,
                       mod_col, ses_col)).T
    df = pd.DataFrame(table, columns=['RT', 'Score', 'Subject',
                                      'Condition', 'Modality', 'Session'])

    output_folder = os.path.join(output_dir, 'dataframes')
    # Create output_folder, if it does not exist
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Save dataframe
    outpath = os.path.join(output_folder, 'df_ntfd_' + sesstag + '.tsv')
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

# This set of subjects are those that for the behavioral experiments did
# the NTFD with the Random Condition
# RAND_SUBJECTS = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
#                  32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 
#                  47]
# RAND_SUBJECTS = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 32, 34, 
# 		 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

# #######################################################################

# TASKS = ['Auditory No-Temporal Feature Discrimination',
#          'Visual No-Temporal Feature Discrimination']

# Total number of trials per run
N_TRIALS = 30
# Total number of trials per isi per condition (without random condition)...
# ... across all runs of every behavioral sessions
N_ISI_TRIALS_BEHAV = 36 # (3*4*3) --> (n_trials * n_ntfd_runs * n_sessions)
# Total number of trials per isi per condition (without random condition)...
# ... across all runs of every imaging sessions
N_ISI_TRIALS_IMG = 16 # (3*2*2 + 2*2*1) --> (n_trials * n_ntfd_runs * n_sessions)

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
RESULTS_FOLDER = os.path.join(MAIN_DIR, 'ntfd_results')

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
        os.mkdir(RESULTS_FOLDER)

    # Dataframe with beat and interval (i.e. no random) only for every isi
    ntfd_isi_dataframe(SUBJECTS, MAIN_DIR, RESULTS_FOLDER, SESSTYPES,
                       N_TRIALS, N_ISI_TRIALS, tag, sessions=SESSIONS)

    # Dataframe with ffx of beat, interval and random
    ntfd_dataframe(SUBJECTS, MAIN_DIR, RESULTS_FOLDER, SESSTYPES, N_TRIALS,
                   tag, sessions=SESSIONS)
