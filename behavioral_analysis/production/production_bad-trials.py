"""
Estimate bad trials (bad perfomances or missing trials) for
Production Tasks of the Music-SDTB project

author: Ana Luisa Pinho
e-mail: agrilopi@uwo.ca

Created: January 21, 2025
Last update: January 2025

Compatibility: Python 3.10.14
"""

import os
import numpy as np
import pandas as pd


# %%
# =========================== INPUTS ===================================

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
DF_DIR = os.path.join(MAIN_DIR, 'production_results', 'dataframes')

DF_IMG_DIR = os.path.join(DF_DIR, 'df_production_allses.tsv')

# %%
# ============================ RUN =====================================

# Open dataframes
df = pd.read_csv(DF_IMG_DIR, sep='\t')

# Get array with asynchronies but only for sessions 4 and 5 (imaging sessions)
asynchronies = df[df["session"].isin([4, 5])]["signed_asynchrony"].values

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
print(f"Percentage of NaN trials: {percentage_nan:.2f}%")
print(f"Percentage of trials exceeding 3SD: {percentage_exceeding:.2f}%")
print(f"Percentage of bad trials: {percentage_bad:.2f}%")
