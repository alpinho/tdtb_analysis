"""
Estimate bad trials (bad perfomances or missing trials) for
the Perception Tasks of the Music-SDTB project

author: Ana Luisa Pinho
e-mail: agrilopi@uwo.ca

Created: January 31, 2025
Last update: June 2026

Compatibility: Python 3.10.14
"""

import os
import numpy as np
import pandas as pd


def missing_trials(df, sessions):
    # Get array of response times but only for the requested sessions
    rt = df[df['session'].isin(sessions)]['response_time'].values

    # Guard against an empty selection (e.g. requested sessions not in df)
    if rt.size == 0:
        raise ValueError(
            f'No trials found for sessions {sessions} in this dataframe.')

    # Total number of trials (includes NaN values)
    total_trials = len(rt)

    # Number of missing trials (NaN)
    nan_trials = np.sum(np.isnan(rt))

    # Number of trials with response time lower than 100ms
    automatic_responses = (rt < 100).sum()

    # Calculate percentages
    percentage_nan = round((nan_trials / total_trials) * 100, 2)
    percentage_automatic = round((
        automatic_responses / total_trials) * 100, 2)

    # Print results
    print(f'Percentage of missing trials: {percentage_nan:.2f}%')
    print(f'Percentage of automatic-response trials: '
      f'{percentage_automatic:.2f}%')


def missing_trials_participant(df, sessions):
    # Filter only the requested sessions
    df_filtered = df[df['session'].isin(sessions)]

    # Guard against an empty selection (e.g. requested sessions not in df)
    if df_filtered.empty:
        raise ValueError(
            f'No trials found for sessions {sessions} in this dataframe.')

    # Dictionary to store missing trial percentages per participant
    missing_trials_stats = []

    # Loop through each participant
    for subject, df_subject in df_filtered.groupby("subject"):
        # Get response times for the participant
        rt = df_subject['response_time'].values

        # Total number of trials (including NaN values)
        total_trials = len(rt)

        # Number of missing trials (NaN values)
        nan_trials = np.sum(np.isnan(rt))

        # Calculate percentage of missing trials for this participant
        percentage_nan = round((nan_trials / total_trials) * 100, 2)

        # Append data
        missing_trials_stats.append([subject, percentage_nan])

    # Convert to DataFrame
    missing_trials_df = pd.DataFrame(
        missing_trials_stats,
        columns=['subject', 'missing_trials_percentage'])

    # Compute mean and standard deviation of missing trial percentages
    mean_missing = missing_trials_df['missing_trials_percentage'].mean()
    std_missing = missing_trials_df['missing_trials_percentage'].std()

    # Print results
    print("Missing Trial Percentages per Participant:")
    print(missing_trials_df)

    print(f"\nMean percentage of missing trials: {mean_missing:.2f}%")
    print(f"Standard deviation of missing trials: {std_missing:.2f}%")


# %%
# =========================== INPUTS ===================================

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
DF_DIR = os.path.join(
    MAIN_DIR, 'perception_results_first_batch', 'raw_dataframes')

# Sessions to analyse. Imaging sessions are 4 and 5; behavioural sessions are
# 1, 2 and 3. Explicit here instead of hardcoded inside the functions.
SESSIONS = [4, 5]

# The 'allses' file holds every session, so the SESSIONS list above is the
# single source of truth for what is analysed.
DF_FNAME = 'df_perception_allses.tsv'
DF_PATH = os.path.join(DF_DIR, DF_FNAME)

# %%
# ============================ RUN =====================================

if __name__ == '__main__':
    # Open dataframes
    db = pd.read_csv(DF_PATH, sep='\t')

    # ************************* Bad Trials *****************************
    # missing_trials(db, SESSIONS)

    # ***************** Bad Trials per participant *********************
    missing_trials_participant(db, SESSIONS)