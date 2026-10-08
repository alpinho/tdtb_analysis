"""
Extract GOLD-MSI questionnaire answers and compute subscale scores for
each subject.

Each subscale score is the sum of its item scores (1-7), with the
reverse-coded items reversed. Missing data (answers left blank, "n/a")
are handled per subscale:
    - no item missing: the score is the sum of the items;
    - one item missing: the score is prorated, i.e. the mean of the
      answered items multiplied by the number of items in the subscale
      (the missing item is assumed to score like the participant's other
      answers); prorated scores are not rounded;
    - two or more items missing: the score is missing ("n/a").
A missing item is never counted as 0, since 0 is not a valid score.

Author: Anmar Alsibaie and Ana Luisa Pinho
e-mails: aalsibai@uwo.ca, alpinho@uwo.ca

Created: July 11th, 2025
Last update: October 1st, 2026

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

# Outputs are written to the goldmsi_scores folder, next to this
# script (created if missing). The curated sheets (sub-XX folders)
# contain participant data and are kept outside the repository, in
# the private OneDrive folder of the project; change INPUT_DIR if
# your copy of that folder is elsewhere.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'goldmsi_scores')
INPUT_DIR = os.path.expanduser('~/OneDrive/tdtb_admin/tdtb_private/forms')

# One output table per group of subjects (written to OUTPUT_DIR)
GROUPS = {'goldmsi_scores_all_fb': ALL_SUBJECTS,              # first batch
          'goldmsi_scores_behavioral_fb': BEHAV_SUBJECTS,     # first batch
          'goldmsi_scores_imaging_fb': IMG_SUBJECTS,          # first batch
          'goldmsi_scores_behavioral_sb': GOOD_SB_SUBJECTS,   # second batch
          'goldmsi_scores_behavioral_tb': GOOD_TB_SUBJECTS}   # third batch

# Table with the participants of all batches, with a column for the batch
ALL_BATCHES = {'fb': BEHAV_SUBJECTS, 'sb': GOOD_SB_SUBJECTS,
               'tb': GOOD_TB_SUBJECTS}
ALL_BATCHES_FNAME = 'goldmsi_scores_behavioral_all'

# Export the item answers of all subjects, for debugging only: the table
# contains information about the participants (dates, free text), so it
# must never be shared
EXPORT_ALL_ANSWERS = False
ALL_ANSWERS_FNAME = 'goldmsi_answers_debug'

# Missing values follow the BIDS convention
MISSING = 'n/a'

# Question (first column of the curated sheets) of each item
ITEM_QUESTIONS = {i: 'question_%02d' % i for i in range(1, 32)}
ITEM_QUESTIONS.update({
    32: 'I engaged in regular, daily practice of a musical instrument '
        '(including voice) for _ years.',
    33: 'At the peak of my interest, I practiced _ hours per day on my '
        'primary instrument.',
    34: 'I have attended _ live music events as an audience member in the '
        'past twelve months.',
    35: 'I have had formal training in music theory for _ years.',
    36: 'I have had _ years of formal training on a musical instrument '
        '(including voice) during my lifetime.',
    37: 'I can play _ musical instruments.',
    38: 'I listen attentively to music for _ per day.'})

# Answer options of items 32-38, as printed on the questionnaire, and
# their scores on the 1-7 scale
ITEM_OPTIONS = {
    32: ['0', '1', '2', '3', '4-5', '6-9', '10 or more'],
    33: ['0', '0.5', '1', '1.5', '2', '3-4', '5 or more'],
    34: ['0', '1', '2', '3', '4-6', '7-10', '11 or more'],
    35: ['0', '0.5', '1', '2', '3', '4-6', '7 or more'],
    36: ['0', '0.5', '1', '2', '3-5', '6-9', '10 or more'],
    37: ['0', '1', '2', '3', '4', '5', '6 or more'],
    38: ['0-15 min', '15-30 min', '30-60 min', '60-90 min', '2 hrs',
         '2-3 hrs', '4 hrs or more']}
ITEM_SCORES = {item: {option: score for score, option
                      in enumerate(options, start=1)}
               for item, options in ITEM_OPTIONS.items()}

# Items of each subscale and those that are reverse coded. As in the
# official R code, items are reversed per subscale only.
SUBSCALES = {
    'active_engagement': ([1, 3, 8, 15, 21, 24, 28, 34, 38], [21]),
    'perceptual_abilities': ([5, 6, 11, 12, 13, 18, 22, 23, 26],
                             [11, 13, 23]),
    'musical_training': ([14, 27, 32, 33, 35, 36, 37], [14, 27]),
    'singing_abilities': ([4, 7, 10, 17, 25, 29, 30], [17, 25]),
    'emotions': ([2, 9, 16, 19, 20, 31], [9]),
    'general_factor': ([1, 3, 4, 7, 10, 12, 14, 15, 17, 19, 23, 24, 25,
                        27, 29, 32, 33, 37], [14, 17, 23, 25, 27])}

LIKERT_MAX = 7

# Largest number of missing items per subscale for which the score is
# prorated; with more missing items the score is missing
MAX_MISSING_ITEMS = 1

# %%
# ========================== FUNCTIONS =================================


def read_answers(subject):
    """Return the answers of one subject as {question: answer}."""
    path = os.path.join(INPUT_DIR, 'sub-%02d' % subject,
                        'gold_msi_sub-%02d.tsv' % subject)
    # Keep every answer as a string; "n/a" is read as missing
    data = pd.read_csv(path, delimiter='\t', names=['Questions', 'Answers'],
                       dtype=str, keep_default_na=False,
                       na_values=[MISSING])
    answers = dict(zip(data['Questions'], data['Answers']))

    missing_questions = [q for q in ITEM_QUESTIONS.values()
                         if q not in answers]
    if missing_questions:
        raise ValueError('Questions not found for subject %02d: %s'
                         % (subject, missing_questions))
    return answers


def score_items(subject, answers):
    """Return the 1-7 score of items 1-38 as {item: score}."""
    scores = {}
    for item, question in ITEM_QUESTIONS.items():
        answer = answers[question]
        if pd.isna(answer):
            scores[item] = pd.NA
        elif item in ITEM_SCORES:
            if answer not in ITEM_SCORES[item]:
                raise ValueError('Unknown answer "%s" to item %d for '
                                 'subject %02d.' % (answer, item, subject))
            scores[item] = ITEM_SCORES[item][answer]
        else:
            score = int(answer)
            if not 1 <= score <= LIKERT_MAX:
                raise ValueError('Score %d out of range in item %d for '
                                 'subject %02d.' % (score, item, subject))
            scores[item] = score
    return scores


def reverse_code(score, likert=LIKERT_MAX):
    """Reverse a score on a 1-likert scale."""
    if pd.isna(score):
        return pd.NA
    return likert + 1 - score


def compute_subscore(items_df, items, reversed_items):
    """Sum the items of a subscale, reversing the reverse-coded ones.

    Missing items are handled as described in the module docstring: one
    missing item is prorated, two or more make the score missing.
    """
    sub_df = items_df[items].copy()
    for item in reversed_items:
        sub_df[item] = sub_df[item].apply(reverse_code)
    sub_df = sub_df.astype('Float64')

    n_missing = sub_df.isna().sum(axis=1)
    prorated = sub_df.mean(axis=1, skipna=True) * len(items)
    score = sub_df.sum(axis=1, skipna=True).where(n_missing == 0, prorated)
    return score.where(n_missing <= MAX_MISSING_ITEMS)


def score_group(subjects):
    """Return the item answers and the subscale scores of a group."""
    all_answers, all_items = [], []
    for subject in subjects:
        answers = read_answers(subject)
        all_answers.append({'subject': subject, **answers})
        all_items.append({'subject': subject,
                          **score_items(subject, answers)})

    items_df = pd.DataFrame(all_items).set_index('subject')
    items_df = items_df.astype('Int64')

    scores_df = pd.DataFrame(index=items_df.index)
    for name, (items, reversed_items) in SUBSCALES.items():
        scores_df[name] = compute_subscore(items_df, items, reversed_items)
    return pd.DataFrame(all_answers), scores_df.reset_index()


def check_public(df):
    """Check that a score table holds only the columns allowed in public.

    Only the subject number, the batch and the GOLD-MSI scores may be
    shared; scores must be numbers or missing.
    """
    allowed = ['subject', 'batch'] + list(SUBSCALES)
    extra = [col for col in df.columns if col not in allowed]
    if extra:
        raise ValueError('Columns not allowed in a public table: %s' % extra)
    for col in SUBSCALES:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError('Non-numeric scores in column "%s".' % col)


def save(df, fname, public=True):
    """Save a table in OUTPUT_DIR."""
    if public:
        check_public(df)
    output_path = os.path.join(OUTPUT_DIR, fname + '.tsv')
    # '%g' writes whole scores without decimals (e.g. 46, 25.875)
    df.to_csv(output_path, sep='\t', index=False, na_rep=MISSING,
              float_format='%g')
    print('Saved %s (%d subjects)' % (output_path, len(df)))


# %%
# ============================ RUN =====================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for fname, subjects in GROUPS.items():
        _, scores_df = score_group(subjects)
        save(scores_df, fname)

    # All batches together; the batch column tells them apart
    batch_answers, batch_scores = [], []
    for batch, subjects in ALL_BATCHES.items():
        answers_df, scores_df = score_group(subjects)
        scores_df.insert(1, 'batch', batch)
        answers_df.insert(1, 'batch', batch)
        batch_scores.append(scores_df)
        batch_answers.append(answers_df)
    save(pd.concat(batch_scores, ignore_index=True), ALL_BATCHES_FNAME)

    if EXPORT_ALL_ANSWERS:
        save(pd.concat(batch_answers, ignore_index=True), ALL_ANSWERS_FNAME,
             public=False)
