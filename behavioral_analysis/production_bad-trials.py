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

DF_IMG1_DIR = os.path.join(DF_DIR, 'df_production_ses-04.tsv')
DF_IMG2_DIR = os.path.join(DF_DIR, 'df_production_ses-05.tsv')

# %%
# ============================ RUN =====================================

# Open dataframes
df1 = pd.read_csv(DF_IMG1_DIR, sep='\t')
df2 = pd.read_csv(DF_IMG2_DIR, sep='\t')

# Stacking DataFrames vertically
df = pd.concat([df1, df2], ignore_index=True)

# Get array with asynchronies
asynchronies = df['Asynchronies']

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
