"""
Estimate bad trials (bad perfomances or missing trials) for
the NTFD Tasks of the Music-SDTB project

author: Ana Luisa Pinho
e-mail: agrilopi@uwo.ca

Created: January 21, 2025
Last update: February, 2025

Compatibility: Python 3.10.14
"""

import os
import numpy as np
import pandas as pd


# %%
# ========================= FUNCTIONS ==================================

def bad_trials(df):
    # Get arrays of reaction time and scores  but only for sessions
    # 4 and 5 (imaging sessions)
    rt = df[df['session'].isin([4, 5])]['reaction_time'].values
    scores = df[df['session'].isin([4, 5])]['score'].values

    # Total number of trials (includes NaN values)
    total_trials = len(rt)

    # Number of missing trials (NaN)
    nan_trials = np.sum(np.isnan(rt))

    # Number of trials with response time lower than 100ms and higher
    # than 700ms
    bad_rt = ((rt < 100) | (rt > 700)).sum()

    # Number of wrong responses
    count_zeros = np.sum(scores == 0)

    # Number of trials where there was an automatic and wrong response
    # at the same time
    overlap_count = (bad_rt & count_zeros).sum()

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

    # ***********************************************************************
    # Assuming df is your DataFrame containing the data
    # Step 1: Filter only sessions 1 to 3
    df_filtered = df[df["session"].isin([1, 2, 3])]

    # Step 2: Compute accuracy for each participant
    accuracy_per_subject = df_filtered.groupby("subject")["score"].mean()

    # Step 3: Find participants with accuracy < 90%
    low_accuracy_subjects = accuracy_per_subject[accuracy_per_subject < 0.9]

    # Display results
    if not low_accuracy_subjects.empty:
        print("Participants who scored less than 90% in sessions 1 to 3:")
        print(low_accuracy_subjects)
    else:
        print("No participants scored below 90% in sessions 1 to 3.")


def bad_trials_participant(df):
    # Filter only sessions 4 and 5 (imaging sessions)
    df_filtered = df[df['session'].isin([4, 5])]

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
DF_DIR = os.path.join(MAIN_DIR, 'ntfd_results', 'dataframes')

DF_IMG_DIR = os.path.join(DF_DIR, 'df_ntfd_allses.tsv')

# %%
# ============================ RUN =====================================

if __name__ == '__main__':
    # Open dataframes
    db = pd.read_csv(DF_IMG_DIR, sep='\t')

    # ************************* Bad Trials *****************************
    # bad_trials(db)

    # ***************** Bad Trials per participant *********************
    bad_trials_participant(db)


