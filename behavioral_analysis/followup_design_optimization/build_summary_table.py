"""
Build the per-participant summary tables used to plan the follow-up cohorts.

Reads the per-session dataframes written by the task pipelines
(production_df.py, perception_analysis.py, ntfd_df.py) for every batch and
writes two long tables:

    session_level.tsv   one row per batch, subject, task, modality, session
                        and condition (beat, interval)
    run_level.tsv       the same for Production and NTFD, one row per run;
                        Perception is fitted per session only, so it has no
                        run level
    standard_level.tsv  Production and NTFD, one row per session, condition
                        and Standard, with the trial mean and count, so that
                        any set of sessions can be pooled at the trial level
                        before averaging across Standards, as
                        cross_cohort_behaviour.py does

Columns common to both: batch, instruction, software, imaging, subject,
task, modality, session, condition, value, sd, n_trials. The session table
adds n_runs and, for Perception, n_valid_standards.

Metric conventions follow cross_cohort_behaviour.py (the manuscript's
Supplementary Note 4):

    Production  mean signed asynchrony, no latency correction (*_0_0_0
                files); Standards averaged first, then across Standards
    Perception  difference limen from the post-fit files, averaged over the
                Standards with a valid fit
    NTFD        mean reaction time, latency-corrected per batch and modality;
                Beat and Interval trials of NTFD and NTFD-Rand runs pooled;
                no bad-trial filter unless NTFD_FILTER_BAD_TRIALS

'sd' is the standard deviation across trials of the same cell (Production,
NTFD) or across the Standards' DLs (Perception). 'n_trials' counts the
trials with a response.
"""

import os
import re
import pandas as pd

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
BEHAV_ROOT = os.path.join(MAIN_DIR, '..')
OUTPUT_DIR = os.path.join(MAIN_DIR, 'summary_tables')

PROD_LATENCY = '0_0_0'
NTFD_FILTER_BAD_TRIALS = False
NTFD_LATENCY = {
    'auditory': {'fb': 133 + 20, 'sb': 63 + 20, 'tb': 63 + 20},
    'visual':   {'fb': 35 + 20,  'sb': 35 + 20, 'tb': 35 + 20},
}

BATCHES = {
    'fb': {'word': 'first',  'instruction': 'implicit', 'software': 'expyriment',
           'sessions': ['ses-01', 'ses-02', 'ses-03', 'ses-04', 'ses-05']},
    'sb': {'word': 'second', 'instruction': 'implicit', 'software': 'psychopy',
           'sessions': ['ses-01', 'ses-02']},
    'tb': {'word': 'third',  'instruction': 'explicit', 'software': 'psychopy',
           'sessions': ['ses-01', 'ses-02']},
}

IMG_SUBJECTS = {3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26,
                28, 29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47}


def _subj_int(x):
    return int(re.search(r'(\d+)', str(x)).group(1))


def _prod_path(batch, tag):
    return os.path.join(BEHAV_ROOT, 'production', 'production_results',
                        'dataframes',
                        f'df_production_{batch}_{PROD_LATENCY}_{tag}.tsv')


def _perc_path(batch, tag):
    return os.path.join(BEHAV_ROOT, 'perception',
                        f'perception_results_{BATCHES[batch]["word"]}_batch',
                        'anovas', f'df_perception_postfit_{tag}.tsv')


def _ntfd_path(batch, tag):
    return os.path.join(BEHAV_ROOT, 'ntfd',
                        f'ntfd_results_{BATCHES[batch]["word"]}_batch',
                        'dataframes', f'df_ntfd_{tag}.tsv')


def _summarise(d, keys):
    """Mean over Standards of the per-Standard trial means, trial SD, n."""
    per_std = (d.groupby(keys + ['standard'])['value']
               .mean().reset_index())
    out = per_std.groupby(keys)['value'].mean().reset_index()
    out['sd'] = d.groupby(keys)['value'].std().values
    out['n_trials'] = d.groupby(keys)['value'].count().values
    return out


def load_trials(task, batch, tag):
    if task == 'Production':
        d = pd.read_csv(_prod_path(batch, tag), sep='\t')
        d = d.rename(columns={'signed_asynchrony': 'value'})
    else:
        d = pd.read_csv(_ntfd_path(batch, tag), sep='\t')
        d = d[d['condition'].isin(['beat', 'interval'])].copy()
        d = d.dropna(subset=['reaction_time'])
        d['value'] = d['reaction_time'].astype(float)
        for mod in ('auditory', 'visual'):
            d.loc[d['modality'] == mod, 'value'] -= NTFD_LATENCY[mod][batch]
        if NTFD_FILTER_BAD_TRIALS:
            d = d[(d['value'] >= 100) & (d['value'] <= 700) & (d['score'] == 1)]
    d['subject'] = d['subject'].apply(_subj_int)
    return d[['subject', 'session', 'run', 'modality', 'condition',
              'standard', 'value']]


def load_perception(batch, tag):
    d = pd.read_csv(_perc_path(batch, tag), sep='\t')
    d = d.rename(columns={'Subject': 'subject', 'Condition': 'condition',
                          'Standard': 'standard', 'Modality': 'modality',
                          'Session': 'session'})
    d['subject'] = d['subject'].apply(_subj_int)
    d['session'] = d['session'].apply(_subj_int)
    d['modality'] = d['modality'].replace({'audio': 'auditory'})
    keys = ['subject', 'session', 'modality', 'condition']
    g = d.groupby(keys)['DL']
    out = g.mean().reset_index().rename(columns={'DL': 'value'})
    out['sd'] = g.std().values
    out['n_valid_standards'] = g.count().values
    out['n_trials'] = pd.NA
    out['n_runs'] = pd.NA
    return out


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    session_rows, run_rows, std_rows = [], [], []
    for batch, info in BATCHES.items():
        for tag in info['sessions']:
            for task in ('Production', 'NTFD'):
                d = load_trials(task, batch, tag)
                keys = ['subject', 'session', 'modality', 'condition']
                s = _summarise(d, keys)
                s['n_runs'] = (d.groupby(keys)['run'].nunique().values)
                s['task'] = task
                session_rows.append(s.assign(batch=batch))
                r = _summarise(d, keys[:1] + ['session', 'run'] + keys[2:])
                r['task'] = task
                run_rows.append(r.assign(batch=batch))
                g = d.groupby(keys + ['standard'])['value']
                st = g.mean().reset_index()
                st['n_trials'] = g.count().values
                st['task'] = task
                std_rows.append(st.assign(batch=batch))
            p = load_perception(batch, tag)
            p['task'] = 'Perception'
            session_rows.append(p.assign(batch=batch))

    def finish(df):
        df['instruction'] = df['batch'].map(
            {b: v['instruction'] for b, v in BATCHES.items()})
        df['software'] = df['batch'].map(
            {b: v['software'] for b, v in BATCHES.items()})
        df['imaging'] = (df['batch'].eq('fb')
                         & df['subject'].isin(IMG_SUBJECTS))
        return df

    ses = finish(pd.concat(session_rows, ignore_index=True))
    ses = ses[['batch', 'instruction', 'software', 'imaging', 'subject',
               'task', 'modality', 'session', 'condition', 'value', 'sd',
               'n_trials', 'n_runs', 'n_valid_standards']]
    ses = ses.sort_values(['batch', 'subject', 'task', 'modality', 'session',
                           'condition']).reset_index(drop=True)
    ses.to_csv(os.path.join(OUTPUT_DIR, 'session_level.tsv'),
               sep='\t', index=False)

    run = finish(pd.concat(run_rows, ignore_index=True))
    run = run[['batch', 'instruction', 'software', 'imaging', 'subject',
               'task', 'modality', 'session', 'run', 'condition', 'value',
               'sd', 'n_trials']]
    run = run.sort_values(['batch', 'subject', 'task', 'modality', 'session',
                           'run', 'condition']).reset_index(drop=True)
    run.to_csv(os.path.join(OUTPUT_DIR, 'run_level.tsv'),
               sep='\t', index=False)

    std = finish(pd.concat(std_rows, ignore_index=True))
    std = std[['batch', 'instruction', 'software', 'imaging', 'subject',
               'task', 'modality', 'session', 'condition', 'standard',
               'value', 'n_trials']]
    std = std.sort_values(['batch', 'subject', 'task', 'modality', 'session',
                           'condition', 'standard']).reset_index(drop=True)
    std.to_csv(os.path.join(OUTPUT_DIR, 'standard_level.tsv'),
               sep='\t', index=False)

    print(ses.groupby(['batch', 'task', 'session'])['subject']
          .nunique().unstack('session'))


if __name__ == '__main__':
    main()
