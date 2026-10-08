"""
Extract demographic data (age and gender) from the demographic forms

author: Ana Luisa Pinho
e-mail: agrilopi@uwo.ca

Created: November 19, 2024
Last update: October 2026

Compatibility: Python 3.10.14
"""

import os
import sys

import pandas as pd

# The subject lists are in the same folder as this script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from subject_lists import (ALL_SUBJECTS, BEHAV_SUBJECTS,  # noqa: E402
                           IMG_SUBJECTS, GOOD_SB_SUBJECTS,
                           GOOD_TB_SUBJECTS)

# %%
# =========================== INPUTS ===================================
# Subject lists per batch are shared with the other questionnaire
# scripts, in subject_lists.py

# Outputs are written to the demographic_data folder, next to this
# script (created if missing). The curated sheets (sub-XX folders)
# contain participant data and are kept outside the repository, in
# the private OneDrive folder of the project; change INPUT_DIR if
# your copy of that folder is elsewhere.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'demographic_data')
INPUT_DIR = os.path.expanduser('~/OneDrive/tdtb_admin/tdtb_private/forms')

# One output table per group of subjects (written to OUTPUT_DIR)
GROUPS = {'all_subjects_fb': ALL_SUBJECTS,                # first batch
          'behavioral_subjects_fb': BEHAV_SUBJECTS,       # first batch
          'imaging_subjects_fb': IMG_SUBJECTS,            # first batch
          'behavioral_subjects_sb': GOOD_SB_SUBJECTS,     # second batch
          'behavioral_subjects_tb': GOOD_TB_SUBJECTS}     # third batch

# Table with the participants of all batches, with a column for the batch
ALL_BATCHES = {'fb': BEHAV_SUBJECTS, 'sb': GOOD_SB_SUBJECTS,
               'tb': GOOD_TB_SUBJECTS}
ALL_BATCHES_FNAME = 'behavioral_subjects_all'

AGE_QUESTION = 'Age:'
GENDER_QUESTION = 'Are you (circle one):'

# Missing values follow the BIDS convention
MISSING = 'n/a'

# %%
# ========================== FUNCTIONS =================================


def read_demographics(subject):
    """Return the age and gender answers of one subject."""
    subject_demographic_path = os.path.join(
        INPUT_DIR, 'sub-%02d' % subject,
        'demographic_info_sub-%02d.tsv' % subject)

    # Keep every answer as a string; "n/a" is read as missing
    data = pd.read_csv(subject_demographic_path, delimiter='\t',
                       names=['Questions', 'Answers'], dtype=str,
                       keep_default_na=False, na_values=[MISSING])

    for question in (AGE_QUESTION, GENDER_QUESTION):
        if question not in data['Questions'].values:
            raise ValueError(
                'The question "%s" does not exist in the DataFrame for '
                'subject %02d.' % (question, subject))

    age = data.loc[data['Questions'] == AGE_QUESTION, 'Answers'].values[0]
    gender = data.loc[
        data['Questions'] == GENDER_QUESTION, 'Answers'].values[0]
    return age, gender


def extract_group(subjects):
    """Build the table of age and gender for a list of subjects."""
    ages, genders = [], []
    for subject in subjects:
        age, gender = read_demographics(subject)
        ages.append(age)
        genders.append(gender)

    df = pd.DataFrame({'subject': subjects, 'age': ages, 'gender': genders})
    # Ages as integers; a missing age stays missing instead of failing
    df['age'] = pd.to_numeric(df['age'], errors='raise').astype('Int64')
    return df


def summarize_group(df):
    """Return a one-line summary of gender counts and age statistics.

    The SD is the sample SD (ddof=1, i.e. divided by N - 1), the usual
    estimate when the participants are a sample of a wider population.
    """
    ages = df['age'].dropna().astype(float)
    genders = df['gender'].value_counts(dropna=False)
    counts = ', '.join('%s: %d' % (g if pd.notna(g) else MISSING, n)
                       for g, n in genders.items())
    summary = ('N = %d (%s); age range %d - %d years; '
               'mean / SD = %.2f +/- %.2f years'
               % (len(df), counts, ages.min(), ages.max(),
                  ages.mean(), ages.std(ddof=1)))
    n_missing_age = df['age'].isna().sum()
    if n_missing_age:
        summary += ' (age missing for %d subjects)' % n_missing_age
    return summary


# %%
# ============================ RUN =====================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for output_fname, subjects in GROUPS.items():
        df = extract_group(subjects)
        output_path = os.path.join(OUTPUT_DIR, output_fname + '.tsv')
        df.to_csv(output_path, sep='\t', index=False, na_rep=MISSING)
        print('Saved %s (%d subjects)' % (output_path, len(df)))
        print('    %s' % summarize_group(df))

    # All batches together; the batch column tells them apart
    batch_dfs = []
    for batch, subjects in ALL_BATCHES.items():
        df = extract_group(subjects)
        df.insert(1, 'batch', batch)
        batch_dfs.append(df)
    df_all = pd.concat(batch_dfs, ignore_index=True)
    output_path = os.path.join(OUTPUT_DIR, ALL_BATCHES_FNAME + '.tsv')
    df_all.to_csv(output_path, sep='\t', index=False, na_rep=MISSING)
    print('Saved %s (%d subjects)' % (output_path, len(df_all)))
    print('    %s' % summarize_group(df_all))
