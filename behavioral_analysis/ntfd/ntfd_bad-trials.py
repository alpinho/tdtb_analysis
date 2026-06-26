"""
Estimate bad trials (bad perfomances or missing trials) for
the NTFD Tasks of the Music-SDTB project

author: Ana Luisa Pinho
e-mail: agrilopi@uwo.ca

Created: January 21, 2025
Last update: June 2026

Compatibility: Python 3.10.14
"""

import os
import numpy as np
import pandas as pd


# %%
# ========================= FUNCTIONS ==================================

def filter_data(df, sessions, tasks):
    # Keep only the requested sessions and task variations. The NTFD / NTFD
    # Rand label is read from the 'task' column, which is set by the parser
    # (ntfd_df.py): a run that contains any random trial is NTFD Rand, every
    # other run is NTFD. Regenerate the dataframes with the current parser
    # before running this, or the split will reflect the old labelling.
    df_filtered = df[df['session'].isin(sessions) & df['task'].isin(tasks)]

    # Guard against an empty selection (e.g. sessions/tasks not in df)
    if df_filtered.empty:
        raise ValueError(
            f'No trials found for sessions {sessions} and tasks {tasks} '
            f'in this dataframe.')

    return df_filtered


def bad_trials(df, sessions, tasks):
    # Get arrays of reaction time and scores for the requested sessions/tasks
    df_filtered = filter_data(df, sessions, tasks)
    rt = df_filtered['reaction_time'].values
    scores = df_filtered['score'].values

    # Total number of trials (includes NaN values)
    total_trials = len(rt)

    # Number of missing trials (NaN)
    nan_trials = np.sum(np.isnan(rt))

    # Number of trials with response time lower than 100ms and higher
    # than 700ms
    bad_rt = ((rt < 100) | (rt > 700)).sum()

    # Number of wrong responses
    count_zeros = np.sum(scores == 0)

    # Number of trials where there was an automatic/extreme response
    # and a wrong response at the same time (element-wise on the masks)
    overlap_count = np.sum(((rt < 100) | (rt > 700)) & (scores == 0))

    # Calculate percentages
    percentage_nan = round((nan_trials / total_trials) * 100, 2)
    percentage_bad_rt = round((bad_rt / total_trials) * 100, 2)
    percentage_wrong = round((count_zeros / total_trials) * 100, 2)
    percentage_overlap = round((overlap_count / total_trials) * 100, 2)
    percentage_bad = round(((
        nan_trials + (bad_rt + count_zeros -
                      overlap_count)) / total_trials) * 100, 2)

    # Print results
    print(f'Percentage of NaN trials: {percentage_nan:.2f}%')
    print(f'Percentage of trials with too low and too long rts: '
          f'{percentage_bad_rt:.2f}%')
    print(f'Percentage of trials with wrong answers: {percentage_wrong:.2f}%')
    print(f'Percentage of automatic trials with wrong answers: '
          f'{percentage_overlap:.2f}%')
    print(f'Percentage of bad trials: {percentage_bad:.2f}%')


def low_accuracy(df, accuracy_sessions, threshold=0.9):
    # Behavioural accuracy QC, computed per participant over ALL trials in the
    # accuracy sessions. It is deliberately not split by NTFD / NTFD Rand:
    # in the behavioural sessions the run number does not cleanly separate the
    # two variations for the later cohort (random trials appear in all runs),
    # so a run-based split here would be unreliable.
    df_acc = df[df['session'].isin(accuracy_sessions)]

    # Accuracy per participant
    accuracy_per_subject = df_acc.groupby("subject")["score"].mean()

    # Participants below the accuracy threshold
    low_accuracy_subjects = accuracy_per_subject[
        accuracy_per_subject < threshold]

    # Display results
    if not low_accuracy_subjects.empty:
        print(f"Participants who scored less than {threshold:.0%} in "
              f"sessions {accuracy_sessions}:")
        print(low_accuracy_subjects)
    else:
        print(f"No participants scored below {threshold:.0%} in "
              f"sessions {accuracy_sessions}.")


def bad_trials_participant(df, sessions, tasks):
    # Keep only the requested sessions and task variations (NTFD / NTFD Rand)
    df_filtered = filter_data(df, sessions, tasks)

    # List to store individual participant metrics
    individual_metrics = []

    # Loop through each participant
    for subject, df_subject in df_filtered.groupby("subject"):
        # Get reaction times and scores for the participant
        rt = df_subject['reaction_time'].values
        scores = df_subject['score'].values

        # Total number of trials (includes NaN values)
        total_trials = len(rt)

        # Number of missing trials (NaN values)
        nan_trials = np.sum(np.isnan(rt))

        # Number of trials with response time lower than 100ms or higher
        # than 700ms
        bad_rt = ((rt < 100) | (rt > 700)).sum()

        # Number of wrong responses
        count_zeros = np.sum(scores == 0)

        # Number of trials where there was an automatic/extreme response
        # and a wrong response at the same time
        overlap_count = np.sum(((rt < 100) | (rt > 700)) & (scores == 0))

        # Calculate percentages
        percentage_nan = round((nan_trials / total_trials) * 100, 2)
        percentage_bad_rt = round((bad_rt / total_trials) * 100, 2)
        percentage_wrong = round((count_zeros / total_trials) * 100, 2)
        percentage_overlap = round((overlap_count / total_trials) * 100, 2)
        percentage_bad = round(((
            nan_trials +
            (bad_rt + count_zeros - overlap_count)) / total_trials) * 100, 2)

        # Append participant's data
        individual_metrics.append([subject, percentage_nan, percentage_bad_rt,
                                   percentage_wrong, percentage_overlap,
                                   percentage_bad])

    # Convert to DataFrame
    metrics_df = pd.DataFrame(individual_metrics,
                              columns=['subject', 'missing_trials_percentage',
                                       'bad_rt_percentage',
                                       'wrong_response_percentage',
                                       'overlap_percentage',
                                       'total_bad_trials_percentage'])

    # Compute mean and standard deviation for each category
    summary_stats = metrics_df.describe().loc[['mean', 'std']]

    # Display individual participant metrics
    print("Bad Trials Metrics per Participant:")
    print(metrics_df)

    # Display summary statistics (mean and SD)
    print("\nMean and Standard Deviation of Bad Trials Metrics:")
    print(summary_stats)


# %%
# =========================== INPUTS ===================================

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
DF_DIR = os.path.join(MAIN_DIR, 'ntfd_results_first_batch', 'dataframes')

# Sessions to analyse for the bad-trial metrics. Imaging sessions are 4 and 5;
# behavioural sessions are 1, 2 and 3. Explicit here instead of hardcoded
# inside the functions.
SESSIONS = [4, 5]

# Sessions used for the behavioural accuracy check (< 90%), via low_accuracy().
ACCURACY_SESSIONS = [4, 5]

# NTFD / NTFD Rand are read from the 'task' column, set by the parser
# (ntfd_df.py): a run containing any random trial is NTFD Rand. The script
# reports each group in turn, so the results are split between NTFD and NTFD
# Rand and also computed together.
TASK_GROUPS = {
    'NTFD': ['NTFD'],
    'NTFD Rand': ['NTFD Rand'],
    'Both': ['NTFD', 'NTFD Rand'],
}

# The 'allses' file holds every session, so SESSIONS above is the single
# source of truth for what is analysed.
DF_FNAME = 'df_ntfd_imgses.tsv'
DF_PATH = os.path.join(DF_DIR, DF_FNAME)

# %%
# ============================ RUN =====================================

if __name__ == '__main__':
    # Open dataframes
    db = pd.read_csv(DF_PATH, sep='\t')

    # ***************** Behavioural accuracy QC (once) ****************
    # low_accuracy(db, ACCURACY_SESSIONS)

    # Report each task group in turn (NTFD, NTFD Rand, both together)
    for label, tasks in TASK_GROUPS.items():
        print('\n' + '=' * 70)
        print(f'TASK: {label}  (tasks={tasks})')
        print('=' * 70)

        # ********************* Bad Trials *****************************
        # bad_trials(db, SESSIONS, tasks)

        # ************* Bad Trials per participant *********************
        bad_trials_participant(db, SESSIONS, tasks)