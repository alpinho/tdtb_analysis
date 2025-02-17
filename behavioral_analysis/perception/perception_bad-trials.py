"""
Estimate bad trials (bad perfomances or missing trials) for
the Perception Tasks of the Music-SDTB project

author: Ana Luisa Pinho
e-mail: agrilopi@uwo.ca

Created: January 31, 2025
Last update: February, 2025

Compatibility: Python 3.10.14
"""

import os
import numpy as np
import pandas as pd


def missing_trials(df):
    # Get arrays of response time and answers but only for sessions
    # 4 and 5 (imaging sessions)
    rt = df[df['session'].isin([4, 5])]['response_time'].values

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


def missing_trials_participant(df):
    # Filter only sessions 4 and 5 (imaging sessions)
    df_filtered = df[df['session'].isin([4, 5])]

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
DF_DIR = os.path.join(MAIN_DIR, 'perception_results', 'raw_dataframes')

DF_IMG_DIR = os.path.join(DF_DIR, 'df_perception_allses.tsv')

# %%
# ============================ RUN =====================================

if __name__ == '__main__':
    # Open dataframes
    db = pd.read_csv(DF_IMG_DIR, sep='\t')

    # ************************* Bad Trials *****************************
    missing_trials(db)

    # ***************** Bad Trials per participant *********************
    missing_trials_participant(db)
