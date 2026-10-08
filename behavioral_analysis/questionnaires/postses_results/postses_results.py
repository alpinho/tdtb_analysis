"""
Extract the answers to the post-session questionnaires (behavioral and
imaging sessions) and correct the answers given on a reversed scale.

Questions 1-6 are answered on a 1-6 scale. Some participants answered
some questions as if the scale were reversed (the evidence is in
suspected_reversed_scales.tsv, at the top of the private folder of the
project, tdtb_private). The answers listed
in REVERSED_SCALES are recoded as 7 - answer; each entry lists one or
more questions of a session. To stop or start correcting answers, edit
REVERSED_SCALES.

Free-text answers (description of the strategy, comments) are not
exported.

Author: Ana Luisa Pinho
e-mail: alpinho@uwo.ca

Created: October 2026
Last update: October 2026

Compatibility: Python 3.10.14
"""

import os
import sys

import pandas as pd

# The subject lists are in the parent folder (questionnaires)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from subject_lists import (BEHAV_SUBJECTS, IMG_SUBJECTS,  # noqa: E402
                           GOOD_SB_SUBJECTS, GOOD_TB_SUBJECTS)

# %%
# =========================== INPUTS ===================================
# Subject lists per batch are shared with the other questionnaire
# scripts, in ../subject_lists.py

# Outputs are written next to this script. The curated sheets (sub-XX
# folders) contain participant data and are kept outside the
# repository, in the private OneDrive folder of the project; change
# INPUT_DIR if your copy of that folder is elsewhere.
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.expanduser('~/OneDrive/tdtb_private/forms')

# Sessions of each kind: behavioral sessions 1-3 and imaging sessions 1-2
BEHAV_SESSIONS = ['behav_ses-01', 'behav_ses-02', 'behav_ses-03']
IMG_SESSIONS = ['mri_ses-01', 'mri_ses-02']

# Per batch, one output table for the behavioral sessions, one for the
# imaging sessions and one for all sessions; batches without imaging
# sessions only have the first. Each table is (subjects, sessions) and is
# written next to this script.
GROUPS = {
    # First batch
    'postses_results_behavioral_sessions_fb': (BEHAV_SUBJECTS,
                                               BEHAV_SESSIONS),
    'postses_results_imaging_sessions_fb': (IMG_SUBJECTS, IMG_SESSIONS),
    'postses_results_all_sessions_fb': (BEHAV_SUBJECTS,
                                        BEHAV_SESSIONS + IMG_SESSIONS),
    # Second batch (behavioral sessions only)
    'postses_results_behavioral_sessions_sb': (GOOD_SB_SUBJECTS,
                                               BEHAV_SESSIONS),
    # Third batch (behavioral sessions only)
    'postses_results_behavioral_sessions_tb': (GOOD_TB_SUBJECTS,
                                               BEHAV_SESSIONS)}

# Behavioral sessions of all batches together, with a column for the batch
ALL_BATCHES = {'fb': BEHAV_SUBJECTS, 'sb': GOOD_SB_SUBJECTS,
               'tb': GOOD_TB_SUBJECTS}
ALL_BATCHES_FNAME = 'postses_results_behavioral_sessions_all'

# Sessions: behavioral sessions 1-3 and imaging sessions 1-2. A session
# without a questionnaire (e.g. not done yet) is skipped.
SESSIONS = {'behav_ses-01': 'behav_postexp_sub-%02d_ses-01.tsv',
            'behav_ses-02': 'behav_postexp_sub-%02d_ses-02.tsv',
            'behav_ses-03': 'behav_postexp_sub-%02d_ses-03.tsv',
            'mri_ses-01': 'mri_postexp_sub-%02d_ses-01.tsv',
            'mri_ses-02': 'mri_postexp_sub-%02d_ses-02.tsv'}

# Questions exported, in the order of the questionnaire, and the name of
# their column in the outputs
QUESTIONS = {
    'Q1': ('How well did you understand the task?',
           'q1_understanding'),          # 1 = very well, 6 = not at all
    'Q2': ('How difficult did you find the task?',
           'q2_difficulty'),             # 1 = very easy, 6 = very difficult
    'Q3': ('How strongly did you concentrate on the task?',
           'q3_concentration'),          # 1 = highly, 6 = not concentrated
    'Q4': ('Did your concentration on the task change throughout the '
           'experiment?',
           'q4_concentration_change'),   # 1 = no change, 6 = strong change
    'Q4a': ('If your concentration changed, in what direction did it '
            'change?',
            'q4a_concentration_direction'),  # 1 = improved, 6 = declined
    'Q5': ('Were there occasions where you guessed when responding?',
           'q5_guessing'),               # 1 = almost never, 6 = almost always
    'Q6': ('How motivated were you for the experiment?',
           'q6_motivation'),             # 1 = not motivated, 6 = highly
    'Q7': ('Did you use/develop any specific strategies during the '
           'experiment to solve the task?',
           'q7_strategy')}               # Yes / No

SCALE_MAX = 6

# ********************** Reversed scales ******************************
# Answers given on a reversed scale, as (subject, session, questions),
# where questions is one question ('Q1') or several (('Q1', 'Q3')).
# Each line is one session: comment it out to keep the answers as given,
# remove a question from it to keep only that answer as given, or add a
# line to correct other answers. The likelihood comes from
# suspected_reversed_scales.tsv (private folder); "possible" cases
# are listed but commented out.
REVERSED_SCALES = [
    # (4, 'behav_ses-01', 'Q6'),             # possible (single item)
    (8, 'behav_ses-02', ('Q1', 'Q3')),       # likely
    (8, 'behav_ses-03', ('Q1', 'Q3')),       # likely
    (10, 'behav_ses-01', 'Q1'),              # very likely
    (10, 'behav_ses-02', 'Q1'),              # very likely
    (10, 'behav_ses-03', 'Q1'),              # very likely
    # (10, 'behav_ses-01', 'Q3'),            # possible
    # (10, 'behav_ses-02', 'Q3'),            # possible
    # (10, 'behav_ses-03', 'Q3'),            # possible
    (14, 'behav_ses-01', ('Q1', 'Q3')),      # very likely
    (14, 'behav_ses-02', ('Q1', 'Q3')),      # very likely
    (15, 'behav_ses-01', 'Q1'),              # likely
    (15, 'behav_ses-02', 'Q1'),              # likely
    (18, 'behav_ses-01', ('Q1', 'Q3')),      # likely
    (18, 'behav_ses-02', ('Q1', 'Q3')),      # likely
    (18, 'behav_ses-03', ('Q1', 'Q3')),      # likely
    (18, 'mri_ses-01', ('Q1', 'Q3')),        # likely
    (18, 'mri_ses-02', ('Q1', 'Q3')),        # likely
    # (25, 'behav_ses-01', 'Q3'),            # possible
    (26, 'behav_ses-01', 'Q1'),              # likely
    # (32, 'behav_ses-03', 'Q6'),            # possible (single item)
    (35, 'behav_ses-02', 'Q1'),              # likely
    (40, 'behav_ses-01', ('Q1', 'Q3')),      # likely
    (40, 'behav_ses-02', ('Q1', 'Q3')),      # likely
    # (45, 'behav_ses-01', 'Q1'),            # possible
    # (46, 'mri_ses-02', 'Q6'),              # possible (single item)
    (48, 'behav_ses-02', 'Q1'),              # likely
    (50, 'behav_ses-01', ('Q1', 'Q3')),      # likely
    (55, 'behav_ses-01', 'Q1'),              # likely
    # (56, 'behav_ses-01', ('Q1', 'Q3')),    # possible
    # (56, 'behav_ses-02', ('Q1', 'Q3')),    # possible
    (62, 'behav_ses-02', 'Q1'),              # likely
    (71, 'behav_ses-01', 'Q1'),              # likely (knew the task)
    (71, 'behav_ses-02', 'Q1'),              # likely (knew the task)
    (72, 'behav_ses-01', ('Q1', 'Q3')),      # likely
    (77, 'behav_ses-01', ('Q1')),            # very likely
    (77, 'behav_ses-02', ('Q1')),            # very likely
]

# Missing values follow the BIDS convention
MISSING = 'n/a'

# %%
# ========================== FUNCTIONS =================================


def as_tuple(questions):
    """Return the questions of a REVERSED_SCALES entry as a tuple."""
    return (questions,) if isinstance(questions, str) else tuple(questions)


def check_reversed_scales():
    """Check that every corrected answer refers to an existing answer."""
    reversible = [q for q in QUESTIONS if q != 'Q7']
    seen = set()
    for subject, session, questions in REVERSED_SCALES:
        if session not in SESSIONS:
            raise ValueError('Unknown session "%s" in REVERSED_SCALES.'
                             % session)
        if (subject, session) in seen:
            raise ValueError('sub-%02d, %s is listed more than once in '
                             'REVERSED_SCALES.' % (subject, session))
        seen.add((subject, session))
        for question in as_tuple(questions):
            if question not in reversible:
                raise ValueError('Question "%s" cannot be reversed '
                                 '(allowed: %s).' % (question, reversible))
        path = os.path.join(INPUT_DIR, 'sub-%02d' % subject,
                            SESSIONS[session] % subject)
        if not os.path.exists(path):
            raise ValueError('No questionnaire for sub-%02d, %s, listed in '
                             'REVERSED_SCALES.' % (subject, session))


def read_session(subject, session):
    """Return the answers of one questionnaire, or None if there is none."""
    path = os.path.join(INPUT_DIR, 'sub-%02d' % subject,
                        SESSIONS[session] % subject)
    if not os.path.exists(path):
        return None
    # Keep every answer as a string; "n/a" is read as missing
    data = pd.read_csv(path, delimiter='\t', names=['Questions', 'Answers'],
                       dtype=str, keep_default_na=False,
                       na_values=[MISSING])
    answers = dict(zip(data['Questions'], data['Answers']))

    row = {'subject': subject, 'session': session}
    for question, (text, column) in QUESTIONS.items():
        if text not in answers:
            raise ValueError('Question "%s" not found in %s.'
                             % (text, path))
        answer = answers[text]
        if question != 'Q7' and pd.notna(answer):
            answer = int(answer)
            if not 1 <= answer <= SCALE_MAX:
                raise ValueError('Answer %d out of range for %s in %s.'
                                 % (answer, question, path))
        row[column] = answer
    return row


def correct_reversed(row):
    """Recode the answers listed in REVERSED_SCALES as 7 - answer."""
    reversed_questions = []
    for subject, session, questions in REVERSED_SCALES:
        if subject == row['subject'] and session == row['session']:
            for question in as_tuple(questions):
                column = QUESTIONS[question][1]
                if pd.notna(row[column]):
                    row[column] = SCALE_MAX + 1 - row[column]
                    reversed_questions.append(question)
    row['reversed'] = (';'.join(reversed_questions) if reversed_questions
                       else pd.NA)
    return row


def extract_group(subjects, sessions):
    """Build the table of answers of a group, one row per session."""
    rows = []
    for subject in subjects:
        for session in sessions:
            row = read_session(subject, session)
            if row is not None:
                rows.append(correct_reversed(row))
    df = pd.DataFrame(rows)
    for question, (_, column) in QUESTIONS.items():
        if question != 'Q7':
            df[column] = df[column].astype('Int64')
    return df


def save(df, fname):
    """Save a table next to this script."""
    output_path = os.path.join(OUTPUT_DIR, fname + '.tsv')
    df.to_csv(output_path, sep='\t', index=False, na_rep=MISSING)
    print('Saved %s (%d subjects, %d sessions, %d corrected answers)'
          % (output_path, df['subject'].nunique(), len(df),
             df['reversed'].dropna().str.split(';').str.len().sum()))


# %%
# ============================ RUN =====================================

if __name__ == "__main__":
    check_reversed_scales()

    for fname, (subjects, sessions) in GROUPS.items():
        save(extract_group(subjects, sessions), fname)

    # Behavioral sessions of all batches; the batch column tells them apart
    batch_dfs = []
    for batch, subjects in ALL_BATCHES.items():
        df = extract_group(subjects, BEHAV_SESSIONS)
        df.insert(1, 'batch', batch)
        batch_dfs.append(df)
    save(pd.concat(batch_dfs, ignore_index=True), ALL_BATCHES_FNAME)
