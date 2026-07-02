"""
Session-to-session reliability of the behavioural metrics of the TDTB
project.

For each metric, and for each session combination requested, the script
computes the one-way intraclass correlation coefficient (ICC) across the
sessions in that combination, i.e. how consistently the metric ranks
participants from one session to another. It is computed within each
modality (auditory, visual) and across modalities (the mean of the two
modalities per participant and session).

The script lives in the parent directory of the per-task folders
(production/, perception/, ntfd/) and reads their per-session dataframes.
Results are written to reliability_results/, one file per metric, with the
session-combination tag in both the filename and the table.

author: Ana Luisa Pinho
e-mail: agrilopi@uwo.ca

Created: 2nd of July 2026
Updated: July 2026

Compatibility: Python 3.10+
"""

import os
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd

from scipy import stats


# %%
# ======================== RELIABILITY STATISTIC ======================


def icc_oneway(matrix):
    """One-way ICC from a participants-by-sessions matrix.

    Model: a one-way random-effects ANOVA with participants as the random
    factor and the sessions as interchangeable repeats (ICC(1,1) and
    ICC(1,k) in the Shrout and Fleiss scheme). Any systematic difference
    between sessions is absorbed into the error term, so this is the
    conservative choice for a consistency claim. Only participants with a
    value in every session are used (complete cases), keeping the design
    balanced and the mean-square decomposition exact.

    Returns a dict with:
      n            number of participants (complete cases)
      k            number of sessions
      icc_1_1      single-session reliability, (MSB - MSW) / (MSB + (k-1) MSW)
      icc_1_k      reliability of the k-session mean, (MSB - MSW) / MSB
      var_between  between-participant variance component
      var_within   within-participant (error) variance component
      f, p         F = MSB / MSW and its p-value (H0: ICC = 0)
    """
    x = np.asarray(matrix, dtype=float)
    if x.ndim != 2:
        x = x.reshape(len(x), -1)
    x = x[~np.isnan(x).any(axis=1)]  # complete cases only
    n, k = x.shape

    empty = dict(n=int(n), k=int(k), icc_1_1=np.nan, icc_1_k=np.nan,
                 var_between=np.nan, var_within=np.nan, f=np.nan, p=np.nan)
    if n < 2 or k < 2:
        return empty

    grand = x.mean()
    subj_means = x.mean(axis=1)
    ms_between = k * np.sum((subj_means - grand) ** 2) / (n - 1)
    ms_within = np.sum((x - subj_means[:, None]) ** 2) / (n * (k - 1))

    if ms_between + (k - 1) * ms_within == 0:
        return empty

    icc_1_1 = (ms_between - ms_within) / (ms_between + (k - 1) * ms_within)
    icc_1_k = ((ms_between - ms_within) / ms_between
               if ms_between > 0 else np.nan)
    var_between = (ms_between - ms_within) / k
    var_within = ms_within
    f = ms_between / ms_within if ms_within > 0 else np.nan
    p = stats.f.sf(f, n - 1, n * (k - 1)) if np.isfinite(f) else np.nan

    return dict(n=int(n), k=int(k), icc_1_1=icc_1_1, icc_1_k=icc_1_k,
                var_between=var_between, var_within=var_within, f=f, p=p)


# %%
# ======================== DATA LOADING ===============================


def _normalise(series, mapping=None):
    """Lower-case a categorical column and optionally remap its levels."""
    out = series.astype(str).str.strip().str.lower()
    if mapping:
        out = out.replace(mapping)
    return out


def per_session_values(metric, task_cfg, batch, sessions, subjects):
    """Per-session, per-cell metric value for one metric.

    Returns {session: DataFrame[subject, modality, condition, value]},
    where 'value' is the metric aggregated (mean or SD) over standards,
    runs and trials within each participant x modality x condition cell of
    that single session.
    """
    out = {}
    for s in sessions:
        path = os.path.join(MAIN_DIR, task_cfg['path'](batch, s))
        df = pd.read_csv(path, sep='\t')

        cols = {
            'subject': pd.to_numeric(
                df[task_cfg['subject']].astype(str).str.extract(r'(\d+)')[0],
                errors='coerce').astype('Int64'),
            'modality': _normalise(df[task_cfg['modality']],
                                   {'audio': 'auditory'}),
            'condition': _normalise(df[task_cfg['condition']]),
            'value': pd.to_numeric(df[metric['value']], errors='coerce'),
        }
        std_col = task_cfg.get('standard')
        if std_col and std_col in df.columns:
            cols['standard'] = df[std_col].values
        tidy = pd.DataFrame(cols)

        # Perception: keep only valid psychometric fits, if flagged.
        valid_col = task_cfg.get('valid_col')
        if valid_col and valid_col in df.columns:
            keep = df[valid_col].astype(str).str.lower().isin(
                ['true', '1', '1.0', 'yes'])
            tidy = tidy[keep.values]

        tidy = tidy[tidy['subject'].isin(subjects)]
        tidy = tidy.dropna(subset=['value'])

        keys = ['subject', 'modality', 'condition']
        group_extra = metric.get('group_extra', [])
        if group_extra and all(c in tidy.columns for c in group_extra):
            # Two-step: aggregate within each standard, then average across
            # standards (matches ffx_dvar in the task scripts, so the SD is
            # the within-standard SD rather than one pooled across standards).
            step1 = (tidy.groupby(keys + group_extra)['value']
                     .agg(metric['agg']).reset_index())
            agg = step1.groupby(keys)['value'].mean().reset_index()
        else:
            agg = (tidy.groupby(keys)['value']
                   .agg(metric['agg']).reset_index())
        out[s] = agg
    return out


def cell_matrix(session_frames, sessions, condition, modality_level):
    """Participants-by-sessions matrix for one condition and modality level.

    modality_level is 'auditory', 'visual', or 'across'. For 'across', the
    per-participant value in each session is the mean of the auditory and
    visual values, defined only where both are present.
    """
    columns = {}
    for s in sessions:
        frame = session_frames[s]
        frame = frame[frame['condition'] == condition]
        if modality_level in ('auditory', 'visual'):
            col = (frame[frame['modality'] == modality_level]
                   .set_index('subject')['value'])
        else:
            wide = frame.pivot_table(index='subject', columns='modality',
                                     values='value')
            if {'auditory', 'visual'}.issubset(wide.columns):
                col = wide[['auditory', 'visual']].dropna().mean(axis=1)
            else:
                col = pd.Series(dtype=float)
        columns[s] = col
    return pd.DataFrame(columns)


# %%
# ======================== DRIVER =====================================


def reliability_table(metric, task_cfg, batch, tag, sessions, subjects):
    """One tidy reliability table for a metric and session combination."""
    session_frames = per_session_values(
        metric, task_cfg, batch, sessions, subjects)

    conditions = sorted(set().union(
        *[set(f['condition']) for f in session_frames.values()]))

    rows = []
    for condition in conditions:
        for modality_level in ('auditory', 'visual', 'across'):
            mat = cell_matrix(
                session_frames, sessions, condition, modality_level)
            res = icc_oneway(mat)
            rows.append({
                'metric': metric['name'],
                'batch': batch,
                'sessions_tag': tag,
                'sessions': ','.join(str(s) for s in sessions),
                'modality': modality_level,
                'condition': condition,
                **res,
            })
    return pd.DataFrame(rows)


def run(batch, tag):
    """Compute and save reliability tables for every metric, one file each."""
    sessions = SESSIONS[tag]
    if len(sessions) < 2:
        print(f'  [{tag}] only {len(sessions)} session(s); ICC needs >= 2. '
              f'Skipping.')
        return

    subjects = SUBJECTS[batch]
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    for metric in METRICS:
        task_cfg = TASKS[metric['task']]
        table = reliability_table(
            metric, task_cfg, batch, tag, sessions, subjects)
        out_path = os.path.join(
            RESULTS_FOLDER,
            f"reliability_{metric['name']}_{batch}_{tag}.tsv")
        table.to_csv(out_path, index=False, sep='\t', na_rep='')
        print(f"  [{tag}] {metric['name']:26s} -> "
              f"{os.path.basename(out_path)}")


# %%
# ============================ INPUTS =================================

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FOLDER = os.path.join(MAIN_DIR, 'reliability_results')

# Production latency offset used in the per-session filenames. Reliability
# (a rank statistic) is invariant to a constant latency shift, so this only
# selects which files are read.
PROD_OFFSET = '0_0_0'

# Path tokens per batch.
_BK = {'first': 'fb', 'second': 'sb'}            # production filename token
_BATCHDIR = {'first': 'first_batch',             # perception/ntfd folder token
             'second': 'second_batch'}

# #### Subjects' lists (mirror the task scripts) ####
GOOD_SUBJECTS = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                 21, 22, 23, 24, 25, 26, 27, 28, 29, 32, 34, 35, 38, 39, 40,
                 41, 42, 43, 44, 45, 46, 47]
GOOD_SB_SUBJECTS = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61]

SUBJECTS = {'first': GOOD_SUBJECTS, 'second': GOOD_SB_SUBJECTS}

# #### Session combinations (tag -> list of session numbers) ####
# Complete-case handling inside icc_oneway restricts each ICC to the
# participants who have all sessions of the requested combination.
SESSIONS = {
    'allses': [1, 2, 3, 4, 5],
    'behavses': [1, 2, 3],
    'imgses': [4, 5],
    'ses-01': [1],
    'ses-02': [2],
    'ses-03': [3],
    'ses-04': [4],
    'ses-05': [5],
    'behav12': [1, 2],
    'behav13': [1, 3],
    'behav23': [2, 3],
}

# #### Metrics ####
METRICS = [
    dict(name='production_mean_asynchrony', task='production',
         value='signed_asynchrony', agg='mean', group_extra=['standard']),
    dict(name='production_sd_asynchrony', task='production',
         value='signed_asynchrony', agg='std', group_extra=['standard']),
    dict(name='ntfd_reaction_time', task='ntfd',
         value='reaction_time', agg='mean'),
    dict(name='ntfd_success', task='ntfd',
         value='score', agg='mean'),
    dict(name='perception_dl', task='perception',
         value='DL', agg='mean'),
    dict(name='perception_pse', task='perception',
         value='PSE', agg='mean'),
]

# #### Per-task file layout and column names ####
TASKS = {
    'production': dict(
        path=lambda batch, s: os.path.join(
            'production', 'production_results', 'dataframes',
            f'df_production_{_BK[batch]}_{PROD_OFFSET}_ses-{s:02d}.tsv'),
        subject='subject', modality='modality', condition='condition',
        standard='standard'),
    'ntfd': dict(
        path=lambda batch, s: os.path.join(
            'ntfd', f'ntfd_results_{_BATCHDIR[batch]}', 'dataframes',
            f'df_ntfd_ses-{s:02d}.tsv'),
        subject='subject', modality='modality', condition='condition',
        standard='standard'),
    'perception': dict(
        path=lambda batch, s: os.path.join(
            'perception', f'perception_results_{_BATCHDIR[batch]}', 'anovas',
            f'df_perception_postfit_ses-{s:02d}.tsv'),
        subject='Subject', modality='Modality', condition='Condition',
        standard='Standard', valid_col='FitValid'),
}

# #### What to run ####
# Session combinations to process (each is one "input"). Single-session
# tags are skipped automatically (ICC needs >= 2 sessions).
BATCHES_TO_RUN = ['first']
TAGS_TO_RUN = ['behavses', 'allses', 'imgses', 'behav12', 'behav13', 'behav23']


# %%
# ============================ RUN ====================================

if __name__ == '__main__':
    for batch_tag in BATCHES_TO_RUN:
        print(f'\nBatch: {batch_tag}')
        for session_tag in TAGS_TO_RUN:
            run(batch_tag, session_tag)
