"""
Estimate bad trials (bad perfomances or missing trials) for
the Production Tasks of the Music-SDTB project

author: Ana Luisa Pinho
e-mail: agrilopi@uwo.ca

Created: January 21, 2025
Last update: June, 2026

Compatibility: Python 3.10.14
"""

import os
import numpy as np
import pandas as pd

pd.set_option('display.max_rows', None)      # all rows (your n=31 participants)
pd.set_option('display.max_columns', None)   # all columns
pd.set_option('display.width', None)         # don't wrap to terminal width
pd.set_option('display.max_colwidth', None)  # don't truncate long cell values


# %%
# ========================== FUNCTIONS =================================

def bad_trials(df, sessions):
    # Get array with asynchronies but only for the requested sessions
    asynchronies = df[df['session'].isin(sessions)][
        'signed_asynchrony'].values

    # Guard against an empty selection (e.g. requested sessions not in df)
    if asynchronies.size == 0:
        raise ValueError(
            f'No trials found for sessions {sessions} in this dataframe.')

    # Total number of trials (includes NaN values)
    total_trials = len(asynchronies)

    # Mean-center the asynchronies (ignoring NaN values)
    mean_asynchrony = np.nanmean(asynchronies)
    mean_centered_asynchronies = asynchronies - mean_asynchrony

    # Calculate the standard deviation for the mean-centered data
    # (ignoring NaN values)
    std_dev_centered = np.nanstd(mean_centered_asynchronies)

    # Define thresholds for mean-centered data
    lower_threshold = -3 * std_dev_centered
    upper_threshold = 3 * std_dev_centered

    # Count 'NaN' trials
    nan_trials = np.sum(np.isnan(mean_centered_asynchronies))

    # Count trials exceeding thresholds (ignoring NaN values)
    exceeding_trials = np.sum((mean_centered_asynchronies < lower_threshold) |
                              (mean_centered_asynchronies > upper_threshold))

    # Calculate percentages
    percentage_nan = round((nan_trials / total_trials) * 100, 2)
    percentage_exceeding = round((exceeding_trials / total_trials) * 100, 2)
    percentage_bad = round(((
        nan_trials + exceeding_trials) / total_trials) * 100, 2)

    # Print results
    print(f'Percentage of NaN trials: {percentage_nan:.2f}%')
    print(f'Percentage of trials exceeding 3SD: {percentage_exceeding:.2f}%')
    print(f'Percentage of bad trials: {percentage_bad:.2f}%')


def bad_trials_participant(df, sessions):
    # Filter only the requested sessions
    df_filtered = df[df['session'].isin(sessions)]

    # Guard against an empty selection (e.g. requested sessions not in df)
    if df_filtered.empty:
        raise ValueError(
            f'No trials found for sessions {sessions} in this dataframe.')

    # Dictionary to store bad trial percentages for each participant
    bad_trial_stats = []

    # Loop through each participant
    for subject, df_subject in df_filtered.groupby("subject"):
        # Get asynchronies for the participant
        asynchronies = df_subject['signed_asynchrony'].values

        # Total number of trials (includes NaN values)
        total_trials = len(asynchronies)

        # Mean-center the asynchronies (ignoring NaN values)
        mean_asynchrony = np.nanmean(asynchronies)
        mean_centered_asynchronies = asynchronies - mean_asynchrony

        # Calculate the standard deviation for the mean-centered data
        std_dev_centered = np.nanstd(mean_centered_asynchronies)

        # Define thresholds for mean-centered data
        lower_threshold = -3 * std_dev_centered
        upper_threshold = 3 * std_dev_centered

        # Count 'NaN' trials
        nan_trials = np.sum(np.isnan(mean_centered_asynchronies))
        percentage_nan = round((nan_trials / total_trials) * 100, 2)

        # Count trials exceeding thresholds (ignoring NaN values)
        exceeding_trials = np.sum((
            mean_centered_asynchronies < lower_threshold) |
            (mean_centered_asynchronies > upper_threshold))
        percentage_exceeding = round((
            exceeding_trials / total_trials) * 100, 2)

        # Total percentage of bad trials
        percentage_bad = round(((
            nan_trials + exceeding_trials) / total_trials) * 100, 2)

        # Append data
        bad_trial_stats.append([subject, percentage_nan, percentage_exceeding,
                                percentage_bad])

    # Convert to DataFrame
    bad_trials_df = pd.DataFrame(
        bad_trial_stats, columns=['subject', 'missing_trials_percentage',
                                  'exceeding_3SD_percentage',
                                  'total_bad_trials_percentage'])

    # Compute mean and standard deviation for each category
    mean_missing = bad_trials_df['missing_trials_percentage'].mean()
    std_missing = bad_trials_df['missing_trials_percentage'].std()

    mean_exceeding = bad_trials_df['exceeding_3SD_percentage'].mean()
    std_exceeding = bad_trials_df['exceeding_3SD_percentage'].std()

    mean_bad_trials = bad_trials_df['total_bad_trials_percentage'].mean()
    std_bad_trials = bad_trials_df['total_bad_trials_percentage'].std()

    # Print results
    print("Bad Trial Percentages per Participant:")
    print(bad_trials_df)

    print(f"\nMean and SD of Missing Trials:")
    print(f"Mean: {mean_missing:.2f}%, SD: {std_missing:.2f}%")

    print(f"\nMean and SD of Trials Exceeding ±3SD:")
    print(f"Mean: {mean_exceeding:.2f}%, SD: {std_exceeding:.2f}%")

    print(f"\nMean and SD of Total Bad Trials:")
    print(f"Mean: {mean_bad_trials:.2f}%, SD: {std_bad_trials:.2f}%")


def scores(df, sessions):
    # Define a good trial as having an absolute synchrony lower than 0.25
    df["good_trial"] = df["signed_asynchrony"].abs() < 0.25

    # Filter for the requested sessions
    df_filtered = df[df["session"].isin(sessions)]

    # Guard against an empty selection (e.g. requested sessions not in df)
    if df_filtered.empty:
        raise ValueError(
            f'No trials found for sessions {sessions} in this dataframe.')

    # Compute the percentage of good trials per participant and modality
    score_df = df_filtered.groupby([
        "subject", "modality"])["good_trial"].mean().reset_index()

    # Round the score to two decimal places
    score_df["score"] = score_df["good_trial"].round(2)

    # Drop the intermediate column for clarity
    score_df.drop(columns=["good_trial"], inplace=True)

    # Display or save the result
    print(score_df)
    score_df.to_csv("production_scores.csv", index=False, sep='\t')

    # *******************

    # Pivot the dataframe to have separate columns for auditory and...
    # ... visual scores
    score_pivot = score_df.pivot(
        index="subject", columns="modality", values="score").reset_index()

    # Rename columns for clarity
    score_pivot.columns.name = None  # Remove multi-index title
    score_pivot.rename(
        columns={"auditory": "auditory_score", "visual": "visual_score"},
        inplace=True)

    # Check if any participant scored below 0.70 in both tasks
    low_performers = score_pivot[(score_pivot["auditory_score"] < 0.70) &
                                 (score_pivot["visual_score"] < 0.70)]

    # Display the result
    if not low_performers.empty:
        print("Participants who scored below 0.70 in both tasks:")
        print(low_performers)
    else:
        print("No participant scored below 0.70 in both tasks.")
    

# %%
# =========================== INPUTS ===================================

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
DF_DIR = os.path.join(
    MAIN_DIR, 'production_results', 'dataframes')

# Feedback offset (auditory, visual, ntfd) in ms used to generate the
# dataframe. This selects the input file. Use (0, 0, 0) for no offset or
# (133, 35, 20) for the corrected feedback.
FEEDBACK_OFFSET = (0, 0, 0)

# Sessions to analyse for the bad-trial functions. Imaging sessions are 4 and
# 5; behavioural sessions are 1, 2 and 3. Explicit here instead of hardcoded
# inside the functions.
SESSIONS = [4, 5]

# Sessions used to compute the behavioural scores (scores() only).
SCORE_SESSIONS = [4, 5]

# Build the dataframe path from the feedback offset. The 'allses' file holds
# every session, so the SESSIONS list above is the single source of truth for
# what is analysed.
DF_FNAME = 'df_production_fb_{}_{}_{}_allses.tsv'.format(*FEEDBACK_OFFSET)
DF_PATH = os.path.join(DF_DIR, DF_FNAME)

# %%
# ============================ RUN =====================================

if __name__ == '__main__':
    # Open dataframes
    db = pd.read_csv(DF_PATH, sep='\t')

    # ************************* Bad Trials *****************************
    # bad_trials(db, SESSIONS)


    # ***************** BAD TRIALS PER PARTICIPANT *********************
    bad_trials_participant(db, SESSIONS)


    # ********************** Compute Scores ****************************
    # scores(db, SCORE_SESSIONS)