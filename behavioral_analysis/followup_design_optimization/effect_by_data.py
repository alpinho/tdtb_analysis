"""
How much data is needed to see the Beat-Interval effect, and does it
change with practice?  Reads the tables written by build_summary_table.py
and writes, in results/:

    effect_by_session.tsv   Interval - Beat effect for every cohort, task,
                            modality and data subset (single sessions,
                            cumulative sessions, behavioural, imaging, all)
    effect_by_runs.tsv      the same for the first k behavioural runs
                            (Production and NTFD only; Perception is fitted
                            per session)
    learning.tsv            Session x Condition RM ANOVA per cohort, task and
                            modality, on participants complete for the
                            sessions listed, plus the per-session effects
    retest.tsv              test-retest correlation between pairs of
                            sessions of the per-participant effect
                            (Interval - Beat) and of their level (the mean
                            of the two conditions)
    cohort_comparison.tsv   Explicit (third batch) vs implicit cohorts:
                            Welch test of the Cohort x Condition interaction
                            with Cohen's d and BF01, as in cross_cohort_
                            behaviour.py
    power.tsv               participants per cohort needed to detect a
                            between-cohort difference in the effect equal
                            to a fraction of the first batch's effect, from
                            the SD of the per-participant effect after 1, 2
                            or 3 behavioural sessions (80% power, two-sided
                            alpha 0.05); and the total N if blocking is
                            manipulated within participants instead. Since
                            the test-retest table shows the two effects of a
                            participant to be uncorrelated, that N equals
                            one between-cohort group, so the within design
                            halves the total sample and nothing more.

The effect is Interval - Beat: positive when Beat sequences give a smaller
signed asynchrony, a lower DL or a faster reaction time. Its confidence
interval is the within-subject one, since the effect is a paired difference.

Per-participant values follow cross_cohort_behaviour.py exactly: Production
and NTFD pool the trials of the subset's sessions, average within each
Standard and then across Standards (standard_level.tsv); Perception takes
the post-fit file of the session tag that matches the subset, whose DLs
are already session-averaged fits, and averages the valid Standards. The
learning and test-retest analyses use the per-session values instead
(session_level.tsv), because there the session is the unit.
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
import pingouin as pg

warnings.simplefilter("ignore", FutureWarning)
from scipy import stats
from statsmodels.stats.power import TTestIndPower, TTestPower

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MAIN_DIR, '..'))
from cross_cohort_behaviour import interaction  # noqa: E402

TABLES = os.path.join(MAIN_DIR, 'summary_tables')
RESULTS = os.path.join(MAIN_DIR, 'results')

TASKS = ['Production', 'Perception', 'NTFD']
MODALITIES = ['auditory', 'visual']

# Analysis samples: batch(es) and optional restriction to imaging subjects.
IMG_SUBJECTS = {3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26,
                28, 29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47}

COHORTS = {
    'fb':     {'batches': ['fb'], 'imaging_only': False},
    'fb_img': {'batches': ['fb'], 'imaging_only': True},
    'sb':     {'batches': ['sb'], 'imaging_only': False},
    'fb+sb':  {'batches': ['fb', 'sb'], 'imaging_only': False},
    'tb':     {'batches': ['tb'], 'imaging_only': False},
}

# Data subsets, as lists of sessions. Only cohorts that have every session
# of a subset are evaluated on it.
SUBSETS = {
    'ses1': [1], 'ses2': [2], 'ses3': [3], 'ses4': [4], 'ses5': [5],
    'ses1+2': [1, 2], 'ses1+2+3': [1, 2, 3],
    'img': [4, 5], 'all': [1, 2, 3, 4, 5],
}

# Session tag of the task pipelines that corresponds to each subset, for the
# Perception post-fit files.
SUBSET_TAG = {'ses1': 'ses-01', 'ses2': 'ses-02', 'ses3': 'ses-03',
              'ses4': 'ses-04', 'ses5': 'ses-05', 'img': 'imgses',
              'all': 'allses', 'ses1+2': {'fb': 'behav12', 'sb': 'behavses',
                                          'tb': 'behavses'},
              'ses1+2+3': 'behavses'}
BATCH_WORD = {'fb': 'first', 'sb': 'second', 'tb': 'third'}

# Cumulative behavioural runs, in chronological order across sessions.
RUN_COUNTS = [1, 2, 3, 4, 6, 8, 12]

SESSION_PAIRS = [(1, 2), (2, 3), (1, 3), (4, 5)]

POWER_FRACTIONS = [0.5, 0.75, 1.0]
POWER_SUBSETS = ['ses1', 'ses1+2', 'ses1+2+3']

COMPARISONS = [('tb', 'sb'), ('tb', 'fb+sb'), ('tb', 'fb'), ('sb', 'fb'),
               ('sb', 'fb_img')]


def perception_postfit(batch, subset):
    tag = SUBSET_TAG[subset]
    if isinstance(tag, dict):
        tag = tag[batch]
    path = os.path.join(MAIN_DIR, '..', 'perception',
                        f'perception_results_{BATCH_WORD[batch]}_batch',
                        'anovas', f'df_perception_postfit_{tag}.tsv')
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path, sep='\t')
    d = d.rename(columns={'Subject': 'subject', 'Condition': 'condition',
                          'Standard': 'standard', 'Modality': 'modality'})
    d['subject'] = d['subject'].str.extract(r'(\d+)').astype(int)
    d['modality'] = d['modality'].replace({'audio': 'auditory'})
    d['batch'] = batch
    return d[['batch', 'subject', 'modality', 'condition', 'standard', 'DL']]


def perception_means(cohort, subset, modality):
    """Subject x {beat, interval} mean DL over Standards, from the post-fit
    file of the subset's tag, restricted to the cohort's participants."""
    frames = [perception_postfit(b, subset) for b in COHORTS[cohort]['batches']]
    if any(f is None for f in frames):
        return None
    d = pd.concat(frames)
    if COHORTS[cohort]['imaging_only']:
        d = d[d['subject'].isin(IMG_SUBJECTS)]
    d = d[d['modality'] == modality]
    g = d.groupby(['subject', 'condition'])['DL'].mean()
    return g.unstack('condition').dropna()


def pooled_means(std, sessions, complete=True):
    """Subject x {beat, interval} mean over the given sessions: trials
    pooled within each Standard, then averaged across Standards."""
    d = std[std['session'].isin(sessions)]
    if complete:
        n_ses = (d.groupby(['subject', 'condition'])['session'].nunique()
                 .unstack('condition'))
        keep = n_ses.index[(n_ses == len(sessions)).all(axis=1)]
        d = d[d['subject'].isin(keep)]
    w = d.assign(_wv=d['value'] * d['n_trials'])
    cell = w.groupby(['subject', 'condition', 'standard'])
    cell = (cell['_wv'].sum() / cell['n_trials'].sum()).reset_index(name='v')
    g = cell.groupby(['subject', 'condition'])['v'].mean()
    return g.unstack('condition').dropna()


def select(df, cohort):
    spec = COHORTS[cohort]
    d = df[df['batch'].isin(spec['batches'])]
    if spec['imaging_only']:
        d = d[d['imaging']]
    return d


def paired_stats(wide):
    """Effect statistics from a subject x {beat, interval} frame."""
    diff = (wide['interval'] - wide['beat']).to_numpy(dtype=float)
    n = len(diff)
    out = dict(n=n, mean_beat=wide['beat'].mean(),
               mean_interval=wide['interval'].mean(),
               effect=diff.mean(), sd_diff=diff.std(ddof=1),
               se=diff.std(ddof=1) / np.sqrt(n))
    half = out['se'] * stats.t.ppf(0.975, n - 1)
    out['ci_low'] = out['effect'] - half
    out['ci_high'] = out['effect'] + half
    out['d_z'] = out['effect'] / out['sd_diff']
    out['t'], out['p'] = stats.ttest_rel(wide['interval'], wide['beat'])
    return out


def weighted_mean(d, keys):
    """Trial-count-weighted mean of 'value' per group."""
    w = d.assign(_wv=d['value'] * d['n_trials']).groupby(keys)
    return w['_wv'].sum() / w['n_trials'].sum()


def subject_means(d, sessions, weights=None):
    """Per-subject, per-condition mean over the given sessions, complete
    cases only. Perception averages session DLs; trial-level tasks weight
    sessions by trial count."""
    d = d[d['session'].isin(sessions)].dropna(subset=['value'])
    complete = (d.groupby(['subject', 'condition'])['session'].nunique()
                .unstack('condition'))
    keep = complete.index[(complete == len(sessions)).all(axis=1)]
    d = d[d['subject'].isin(keep)]
    if weights is None or d['n_trials'].isna().any():
        g = d.groupby(['subject', 'condition'])['value'].mean()
    else:
        g = weighted_mean(d, ['subject', 'condition'])
    return g.unstack('condition').dropna()


def effect_by_session(std, ses):
    rows = []
    for cohort in COHORTS:
        s0 = select(std, cohort)
        for task in TASKS:
            for mod in MODALITIES:
                if task == 'Perception':
                    have = set(select(ses, cohort)['session'].unique())
                else:
                    d = s0[(s0['task'] == task) & (s0['modality'] == mod)]
                    have = set(d['session'].unique())
                for name, sessions in SUBSETS.items():
                    if not set(sessions) <= have:
                        continue
                    if task == 'Perception':
                        wide = perception_means(cohort, name, mod)
                        if wide is None:
                            continue
                    else:
                        wide = pooled_means(d, sessions)
                    if len(wide) < 3:
                        continue
                    rows.append(dict(cohort=cohort, task=task, modality=mod,
                                     subset=name, n_sessions=len(sessions),
                                     **paired_stats(wide)))
    return pd.DataFrame(rows)


def effect_by_runs(run):
    run = run[run['session'] <= 3].copy()
    run = run.sort_values(['subject', 'session', 'run'])
    run['k'] = (run.groupby(['batch', 'subject', 'task', 'modality',
                             'condition']).cumcount() + 1)
    rows = []
    for cohort in COHORTS:
        d0 = select(run, cohort)
        for task in ('Production', 'NTFD'):
            for mod in MODALITIES:
                d = d0[(d0['task'] == task) & (d0['modality'] == mod)]
                for k in RUN_COUNTS:
                    dk = d[d['k'] <= k]
                    counts = (dk.groupby(['subject', 'condition'])['k']
                              .max().unstack('condition'))
                    keep = counts.index[(counts == k).all(axis=1)]
                    dk = dk[dk['subject'].isin(keep)]
                    if dk['subject'].nunique() < 3:
                        continue
                    wide = (weighted_mean(dk, ['subject', 'condition'])
                            .unstack('condition').dropna())
                    trials = int(dk.groupby(['subject', 'condition'])
                                 ['n_trials'].sum().mean().round())
                    rows.append(dict(cohort=cohort, task=task, modality=mod,
                                     n_runs=k, trials_per_condition=trials,
                                     **paired_stats(wide)))
    return pd.DataFrame(rows)


def learning(ses):
    rows = []
    for cohort in COHORTS:
        d0 = select(ses, cohort)
        for task in TASKS:
            for mod in MODALITIES:
                d = d0[(d0['task'] == task) & (d0['modality'] == mod)]
                d = d.dropna(subset=['value'])
                have = sorted(d['session'].unique())
                for sessions in ([s for s in have if s <= 3],
                                 [s for s in have if s >= 4]):
                    if len(sessions) < 2:
                        continue
                    dd = d[d['session'].isin(sessions)]
                    n_ses = (dd.groupby(['subject', 'condition'])['session']
                             .nunique().unstack('condition'))
                    keep = n_ses.index[(n_ses == len(sessions)).all(axis=1)]
                    dd = dd[dd['subject'].isin(keep)]
                    if len(keep) < 3:
                        continue
                    a = pg.rm_anova(data=dd, dv='value',
                                    within=['session', 'condition'],
                                    subject='subject', detailed=True)
                    a = a.set_index('Source')
                    per_ses = {}
                    for s in sessions:
                        w = (dd[dd['session'] == s]
                             .pivot(index='subject', columns='condition',
                                    values='value'))
                        per_ses[f'effect_ses{s}'] = (
                            w['interval'] - w['beat']).mean()
                        per_ses[f'mean_ses{s}'] = w.stack().mean()
                    rows.append(dict(
                        cohort=cohort, task=task, modality=mod,
                        sessions='+'.join(map(str, sessions)), n=len(keep),
                        F_session=a.loc['session', 'F'],
                        p_session=a.loc['session', 'p-unc'],
                        F_condition=a.loc['condition', 'F'],
                        p_condition=a.loc['condition', 'p-unc'],
                        F_interaction=a.loc['session * condition', 'F'],
                        p_interaction=a.loc['session * condition', 'p-unc'],
                        **per_ses))
    return pd.DataFrame(rows)


def retest(ses):
    rows = []
    for cohort in COHORTS:
        d0 = select(ses, cohort)
        for task in TASKS:
            for mod in MODALITIES:
                d = d0[(d0['task'] == task) & (d0['modality'] == mod)]
                d = d.dropna(subset=['value'])
                wide = (d.pivot_table(index=['subject', 'session'],
                                      columns='condition', values='value')
                        .dropna())
                measures = {
                    'effect': wide['interval'] - wide['beat'],
                    'level': (wide['interval'] + wide['beat']) / 2,
                }
                for measure, series in measures.items():
                    eff = series.unstack('session')
                    for s1, s2 in SESSION_PAIRS:
                        if s1 not in eff or s2 not in eff:
                            continue
                        pair = eff[[s1, s2]].dropna()
                        if len(pair) < 5:
                            continue
                        r, p = stats.pearsonr(pair[s1], pair[s2])
                        icc = pg.intraclass_corr(
                            data=pair.stack().rename('v').reset_index(),
                            targets='subject', raters='session', ratings='v')
                        icc = icc.set_index('Type').loc['ICC3', 'ICC']
                        rows.append(dict(cohort=cohort, task=task,
                                         modality=mod, measure=measure,
                                         sessions=f'{s1}-{s2}', n=len(pair),
                                         r=r, p=p, icc3=icc))
    return pd.DataFrame(rows)


def cohort_comparison(std):
    """Cohort x Condition interaction on behavioural sessions 1-2 only (the
    first batch's third and imaging sessions are never used). 'ses1+2_
    completed' pools each participant's completed sessions, as in the
    manuscript; 'ses1' uses session 1 alone, which every participant has."""
    rows = []
    for label, sessions, subset in (('ses1+2_completed', [1, 2], 'ses1+2'),
                                    ('ses1', [1], 'ses1')):
        for a, b in COMPARISONS:
            for task in TASKS:
                for mod in MODALITIES:
                    diffs = []
                    for cohort in (a, b):
                        if task == 'Perception':
                            w = perception_means(cohort, subset, mod)
                        else:
                            d = select(std, cohort)
                            d = d[(d['task'] == task) & (d['modality'] == mod)]
                            w = pooled_means(d, sessions, complete=False)
                        diffs.append(w['interval'] - w['beat'])
                    res = interaction(diffs[0], diffs[1])
                    rows.append(dict(
                        data=label, cohort_a=a, cohort_b=b, task=task,
                        modality=mod, n_a=len(diffs[0]), n_b=len(diffs[1]),
                        effect_a=diffs[0].mean(), effect_b=diffs[1].mean(),
                        **res))
    return pd.DataFrame(rows)


def power(effects):
    ref = effects[(effects['cohort'] == 'fb')
                  & effects['subset'].isin(POWER_SUBSETS)]
    full = ref[ref['subset'] == 'ses1+2+3'].set_index(['task', 'modality'])
    rows = []
    for _, r in ref.iterrows():
        base = full.loc[(r['task'], r['modality']), 'effect']
        for frac in POWER_FRACTIONS:
            delta = frac * base
            d = delta / r['sd_diff']
            n_between = TTestIndPower().solve_power(
                effect_size=abs(d), alpha=0.05, power=0.8)
            n_within = TTestPower().solve_power(
                effect_size=abs(d) / np.sqrt(2), alpha=0.05, power=0.8)
            rows.append(dict(task=r['task'], modality=r['modality'],
                             sessions=r['subset'], sd_diff=r['sd_diff'],
                             reference_effect=base, fraction=frac,
                             delta=delta, d=d,
                             n_per_cohort_between=int(np.ceil(n_between)),
                             n_within=int(np.ceil(n_within))))
    return pd.DataFrame(rows)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    ses = pd.read_csv(os.path.join(TABLES, 'session_level.tsv'), sep='\t')
    run = pd.read_csv(os.path.join(TABLES, 'run_level.tsv'), sep='\t')
    std = pd.read_csv(os.path.join(TABLES, 'standard_level.tsv'), sep='\t')
    for name, fn, args in (('effect_by_session', effect_by_session,
                            (std, ses)),
                           ('effect_by_runs', effect_by_runs, (run,)),
                           ('learning', learning, (ses,)),
                           ('retest', retest, (ses,)),
                           ('cohort_comparison', cohort_comparison, (std,))):
        out = fn(*args)
        out.to_csv(os.path.join(RESULTS, name + '.tsv'), sep='\t',
                   index=False, float_format='%.5g')
        print(f'{name}: {len(out)} rows')
        if name == 'effect_by_session':
            pw = power(out)
            pw.to_csv(os.path.join(RESULTS, 'power.tsv'), sep='\t',
                      index=False, float_format='%.5g')
            print(f'power: {len(pw)} rows')


if __name__ == '__main__':
    main()
