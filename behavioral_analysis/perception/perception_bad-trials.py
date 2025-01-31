"""
Estimate bad trials (bad perfomances or missing trials) for
the Perception Tasks of the Music-SDTB project

author: Ana Luisa Pinho
e-mail: agrilopi@uwo.ca

Created: January 31, 2025
Last update: January 2025

Compatibility: Python 3.10.14
"""

import os
import numpy as np
import pandas as pd


# %%
# =========================== INPUTS ===================================

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
DF_DIR = os.path.join(MAIN_DIR, 'perception_results', 'raw_dataframes')

DF_IMG_DIR = os.path.join(DF_DIR, 'df_perception_allses.tsv')

# %%
# ============================ RUN =====================================

# Open dataframes
df = pd.read_csv(DF_IMG_DIR, sep='\t')

# Get arrays of response time and answers but only for sessions 4 and 5
# (imaging sessions)
rt = df[df['session'].isin([4, 5])]['response_time'].values

# Total number of trials (includes NaN values)
total_trials = len(rt)

# Number of missing trials (NaN)
nan_trials = np.sum(np.isnan(rt))

# Number of trials with response time lower than 100ms
automatic_responses = (rt < 100).sum()

# Calculate percentages
percentage_nan = round((nan_trials / total_trials) * 100, 2)
percentage_automatic = round((automatic_responses / total_trials) * 100, 2)

# Print results
print(f'Percentage of missing trials: {percentage_nan:.2f}%')
print(f'Percentage of automatic-response trials: {percentage_automatic:.2f}%')
