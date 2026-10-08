"""
Lists of subjects per batch, shared by the scripts that process the
questionnaires (demographic_data, goldmsi_scores and postses_results).

The lists mirror the ones in the other behavioral_analysis scripts (e.g.
production/production_df.py, where BEHAV_SUBJECTS is called
GOOD_SUBJECTS); update both places when a list changes.

Author: Ana Luisa Pinho
e-mail: alpinho@uwo.ca

Created: October 2026

Compatibility: Python 3.10.14
"""

# *********************** First Batch *********************************
# Expyriment / Implicit

# All subjects
ALL_SUBJECTS = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36,
                37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

# All good subjects including img pilot (sub-04)
BEHAV_SUBJECTS = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                  20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 32, 34, 35, 38, 39,
                  40, 41, 42, 43, 44, 45, 46, 47]

# Img subjects only (this is a subset of BEHAV_SUBJECTS)
IMG_SUBJECTS = [3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26,
                28, 29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

# *********************** Second Batch ********************************
# Psychopy / Implicit

# First session completed
GOOD_SB_SUBJECTS = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61,
                    62, 63, 76, 77]

# *********************** Third Batch *********************************
# Psychopy / Explicit

# First session completed
GOOD_TB_SUBJECTS = [64, 65, 66, 67, 68, 70, 71, 72, 73, 74, 75]
