"""
Estimate bad trials (bad perfomances or missing trials) for
the NTFD Tasks of the Music-SDTB project

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
DF_DIR = os.path.join(MAIN_DIR, 'ntfd_results', 'dataframes')

DF_IMG_DIR = os.path.join(DF_DIR, 'df_ntfd_allses.tsv')

# %%
# ============================ RUN =====================================

# Open dataframes
df = pd.read_csv(DF_IMG_DIR, sep='\t')

# Get arrays of reaction time and scores  but only for sessions 4 and 5
# (imaging sessions)
rt = df[df['session'].isin([4, 5])]['reaction_time'].values
scores = df[df['session'].isin([4, 5])]['score'].values

# Total number of trials (includes NaN values)
total_trials = len(rt)

# Number of missing trials (NaN)
nan_trials = np.sum(np.isnan(rt))

# Number of trials with response time lower than 100ms and higher than 700ms
bad_rt = ((rt < 100) | (rt > 700)).sum()

# Number of wrong responses
count_zeros = np.sum(scores == 0)

# Number of trials where there was an automatic and wrong response at the same
# time
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
