"""
Figures for the follow-up design-optimization analysis, from the tables written by
build_summary_table.py and effect_by_data.py. Written to figures/ as PDF
(for Beamer) and PNG.

    effect_by_session_<task>.pdf
                                Interval - Beat effect with its 95% CI for
                                every data subset and modality; the three
                                batches side by side
    condition_means.pdf         Beat and Interval means per session with
                                within-subject (Cousineau-Morey) error bars
    precision_by_runs.pdf       standard error of the effect against trials
                                per condition, with the 1/sqrt(n) reference
    retest.pdf                  per-participant effect in session 1 against
                                session 2 (auditory and visual Production)
    cohort_comparison.pdf       explicit against implicit cohorts
    power.pdf                   participants per cohort against sessions
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(MAIN_DIR, 'summary_tables')
RESULTS = os.path.join(MAIN_DIR, 'results')
FIGURES = os.path.join(MAIN_DIR, 'figures')

COL = {'beat': '#2a78d6', 'interval': '#eb6834',
       'fb': '#2a78d6', 'sb': '#eb6834', 'tb': '#1baf7a'}
COHORT_LABEL = {'fb': 'Implicit, Expyriment (batch 1)',
                'sb': 'Implicit, PsychoPy (batch 2)',
                'tb': 'Explicit, PsychoPy (batch 3)'}
TASKS = ['Production', 'Perception', 'NTFD']
MODALITIES = ['auditory', 'visual']
UNIT = {'Production': 'Interval $-$ Beat, mean signed asynchrony (prop. of S)',
        'Perception': 'Interval $-$ Beat, difference limen (prop. of S)',
        'NTFD': 'Interval $-$ Beat, reaction time (ms)'}
SUBSETS = ['ses1', 'ses2', 'ses3', 'ses1+2', 'ses1+2+3', 'ses4', 'ses5', 'img']
SUBSET_LABEL = {'ses1': 'S1', 'ses2': 'S2', 'ses3': 'S3', 'ses1+2': 'S1+2',
                'ses1+2+3': 'S1+2+3', 'ses4': 'fMRI1', 'ses5': 'fMRI2',
                'img': 'fMRI1+2'}

plt.rcParams.update({
    'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': '#e6e5e0', 'grid.linewidth': 0.6,
    'axes.axisbelow': True, 'legend.frameon': False,
    'figure.facecolor': 'white', 'savefig.dpi': 200,
})


def save(fig, name):
    fig.savefig(os.path.join(FIGURES, name + '.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIGURES, name + '.png'), bbox_inches='tight')
    plt.close(fig)


def stars(p):
    return ('***' if p < 0.001 else '**' if p < 0.01
            else '*' if p < 0.05 else '')


def effect_by_session(eff):
    """One figure per task: auditory and visual panels, the three batches
    side by side, batch-1 statistics in a text row under the axis."""
    for task in TASKS:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
        for j, mod in enumerate(MODALITIES):
            ax = axes[j]
            d = eff[(eff['task'] == task) & (eff['modality'] == mod)]
            ax.axhline(0, color='#9a9a94', lw=0.8, zorder=1)
            ax.axvspan(4.5, 7.5, color='#f2f1ec', zorder=0)
            for k, cohort in enumerate(['fb', 'sb', 'tb']):
                dc = d[d['cohort'] == cohort].set_index('subset')
                xs, ys, lo, hi = [], [], [], []
                for s, subset in enumerate(SUBSETS):
                    if subset not in dc.index:
                        continue
                    r = dc.loc[subset]
                    xs.append(s + (k - 1) * 0.24)
                    ys.append(r['effect'])
                    lo.append(r['effect'] - r['ci_low'])
                    hi.append(r['ci_high'] - r['effect'])
                ax.errorbar(xs, ys, yerr=[lo, hi], fmt='o', ms=4.5,
                            color=COL[cohort], ecolor=COL[cohort],
                            elinewidth=1.4, capsize=2.5, zorder=3,
                            label=COHORT_LABEL[cohort] if j == 0 else None)
            fb = d[d['cohort'] == 'fb'].set_index('subset')
            for s, subset in enumerate(SUBSETS):
                if subset in fb.index:
                    r = fb.loc[subset]
                    ax.annotate(f"$d_z$ {r['d_z']:.2f}{stars(r['p'])}\n"
                                f"p {r['p']:.2g}",
                                (s, 0), xycoords=('data', 'axes fraction'),
                                xytext=(0, -30), textcoords='offset points',
                                ha='center', va='top', fontsize=6.5,
                                color='#52514e', annotation_clip=False)
            ax.set_title(f'{task} — {mod}', loc='left', fontsize=10,
                         fontweight='bold')
            ax.set_xticks(range(len(SUBSETS)))
            ax.set_xticklabels([SUBSET_LABEL[s] for s in SUBSETS],
                               fontsize=8)
            ax.set_xlim(-0.6, len(SUBSETS) - 0.4)
            ax.set_ylabel(UNIT[task].replace(', ', ',\n'), fontsize=8)
        axes[0].legend(loc='best', fontsize=7)
        fig.text(0.5, -0.1, 'S = behavioural session; fMRI = imaging '
                 'session (batch 1 only, shaded). Bars: 95% CI of the paired '
                 'difference. Text row: batch 1, $d_z$ and uncorrected p '
                 '(* <.05, ** <.01, *** <.001).',
                 ha='center', fontsize=7.5, color='#52514e')
        fig.tight_layout()
        save(fig, 'effect_by_session_' + task.lower())


def cousineau_morey(wide):
    """Within-subject SEM per column (Cousineau 2005, Morey 2008)."""
    n, k = wide.shape
    norm = wide.sub(wide.mean(axis=1), axis=0) + wide.values.mean()
    return norm.std(ddof=1) / np.sqrt(n) * np.sqrt(k / (k - 1))


def condition_means(ses):
    fb = ses[ses['batch'] == 'fb']
    fig, axes = plt.subplots(3, 2, figsize=(9, 8.5), sharex=True)
    for i, task in enumerate(TASKS):
        for j, mod in enumerate(MODALITIES):
            ax = axes[i, j]
            d = fb[(fb['task'] == task) & (fb['modality'] == mod)]
            d = d.dropna(subset=['value'])
            for cond in ('beat', 'interval'):
                xs, ys, es = [], [], []
                for sessions in ([1, 2, 3], [4, 5]):
                    dd = d[d['session'].isin(sessions)]
                    wide = dd.pivot_table(index='subject',
                                          columns=['session', 'condition'],
                                          values='value').dropna()
                    sem = cousineau_morey(wide)
                    for s in sessions:
                        xs.append(s)
                        ys.append(wide[(s, cond)].mean())
                        es.append(sem[(s, cond)])
                ax.errorbar(xs, ys, yerr=es, fmt='-o', ms=4.5, lw=1.6,
                            color=COL[cond], capsize=2.5,
                            label=cond.capitalize())
            ax.axvspan(3.5, 5.5, color='#f2f1ec', zorder=0)
            ax.set_title(f'{task} — {mod}', loc='left', fontsize=10,
                         fontweight='bold')
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.set_xticklabels(['S1', 'S2', 'S3', 'fMRI 1', 'fMRI 2'])
            if task == 'Production':
                ax.axhline(0, color='#9a9a94', lw=0.8)
            if j == 0:
                ax.set_ylabel({'Production': 'Mean signed asynchrony\n'
                               '(prop. of S)',
                               'Perception': 'Difference limen\n(prop. of S)',
                               'NTFD': 'Reaction time (ms)'}[task],
                              fontsize=8)
    axes[0, 0].legend(loc='upper right', fontsize=8)
    fig.text(0.5, 0.005, 'Batch 1 (N = 39 behavioural, 31 imaging). Bars: '
             'within-subject SEM (Cousineau–Morey), computed separately for '
             'the behavioural and the imaging sessions.',
             ha='center', fontsize=7.5, color='#52514e')
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save(fig, 'condition_means')


def precision_by_runs(runs):
    fb = runs[runs['cohort'] == 'fb']
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, task in zip(axes, ['Production', 'NTFD']):
        for mod, marker in (('auditory', 'o'), ('visual', 's')):
            d = fb[(fb['task'] == task) & (fb['modality'] == mod)]
            d = d.sort_values('trials_per_condition')
            ax.plot(d['trials_per_condition'], d['se'], marker=marker,
                    ms=5, lw=1.6, color=COL['beat'] if mod == 'auditory'
                    else COL['interval'], label=f'{mod}: observed SE')
            ref = d['se'].iloc[0] * np.sqrt(
                d['trials_per_condition'].iloc[0] / d['trials_per_condition'])
            ax.plot(d['trials_per_condition'], ref, ls='--', lw=1,
                    color=COL['beat'] if mod == 'auditory'
                    else COL['interval'], alpha=0.6,
                    label=f'{mod}: 1/$\\sqrt{{n}}$ from first run')
            for _, r in d.iterrows():
                ax.annotate(f"{r['d_z']:.2f}",
                            (r['trials_per_condition'], r['se']),
                            xytext=(0, 5), textcoords='offset points',
                            ha='center', fontsize=6.5, color='#52514e')
        aud = fb[(fb['task'] == task) & (fb['modality'] == 'auditory')]
        aud = aud.set_index('n_runs')['trials_per_condition']
        for k, lab in ((4, '1 session'), (8, '2 sessions'),
                       (12, '3 sessions')):
            x = aud[k]
            ax.axvline(x, color='#c9c8c2', lw=0.8, ls=':')
            ax.text(x, ax.get_ylim()[1] * 0.98, lab, rotation=90,
                    va='top', ha='right', fontsize=7, color='#52514e')
        ax.set_title(task, loc='left', fontsize=10, fontweight='bold')
        ax.set_xlabel('Trials per condition per participant')
        ax.set_ylabel('SE of the Interval $-$ Beat effect')
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7)
    fig.text(0.5, -0.02, 'Batch 1, cumulative behavioural runs in '
             'chronological order (N = 39). Point labels: $d_z$.',
             ha='center', fontsize=7.5, color='#52514e')
    fig.tight_layout()
    save(fig, 'precision_by_runs')


def retest(ses, ret):
    fb = ses[(ses['batch'] == 'fb') & (ses['task'] == 'Production')]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    for ax, mod in zip(axes, MODALITIES):
        d = fb[fb['modality'] == mod]
        e = d.pivot_table(index=['subject', 'session'], columns='condition',
                          values='value')
        e = (e['interval'] - e['beat']).unstack('session')[[1, 2]].dropna()
        lim = np.abs(e.values).max() * 1.05
        ax.plot([-lim, lim], [-lim, lim], color='#c9c8c2', lw=0.8, ls='--')
        ax.axhline(0, color='#9a9a94', lw=0.8)
        ax.axvline(0, color='#9a9a94', lw=0.8)
        ax.scatter(e[1], e[2], s=28, color=COL['fb'], edgecolor='white',
                   lw=0.8, zorder=3)
        r = ret[(ret['cohort'] == 'fb') & (ret['task'] == 'Production')
                & (ret['modality'] == mod) & (ret['sessions'] == '1-2')]
        ax.set_title(f"Production — {mod}   r = {r['r'].iloc[0]:.2f}, "
                     f"p = {r['p'].iloc[0]:.2f}", loc='left', fontsize=10,
                     fontweight='bold')
        ax.set_xlabel('Interval $-$ Beat, session 1')
        ax.set_ylabel('Interval $-$ Beat, session 2')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
    r23 = ret[(ret['cohort'] == 'fb') & (ret['task'] == 'Production')
              & (ret['sessions'] == '2-3')].set_index('modality')['r']
    fig.text(0.5, -0.02, 'Batch 1 (N = 39): each point is one participant. '
             'The group mean is positive in every session, but the '
             'individual effect is not stable: between sessions 2 and 3, '
             f'r = {r23["auditory"]:.2f} (auditory) and '
             f'{r23["visual"]:.2f} (visual).',
             ha='center', fontsize=7.5, color='#52514e')
    fig.tight_layout()
    save(fig, 'retest')


def cohort_comparison(eff, comp):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.9))
    ns = {c: int(eff[(eff['cohort'] == c) & (eff['subset'] == 'ses1')
                     & (eff['task'] == 'Production')]['n'].iloc[0])
          for c in ['fb', 'sb', 'tb']}
    for ax, task in zip(axes, TASKS):
        d = eff[(eff['task'] == task) & (eff['subset'] == 'ses1')]
        ax.axhline(0, color='#9a9a94', lw=0.8)
        for j, mod in enumerate(MODALITIES):
            for k, cohort in enumerate(['fb', 'sb', 'tb']):
                r = d[(d['modality'] == mod) & (d['cohort'] == cohort)]
                if r.empty:
                    continue
                r = r.iloc[0]
                ax.errorbar([j * 4 + k], [r['effect']],
                            yerr=[[r['effect'] - r['ci_low']],
                                  [r['ci_high'] - r['effect']]],
                            fmt='o', ms=5, color=COL[cohort], capsize=3,
                            elinewidth=1.4,
                            label=f'{COHORT_LABEL[cohort]}, n = {ns[cohort]}'
                            if (task, j) == ('Production', 0) else None)
            c = comp[(comp['task'] == task) & (comp['modality'] == mod)
                     & (comp['data'] == 'ses1')
                     & (comp['cohort_a'] == 'tb')
                     & (comp['cohort_b'] == 'fb+sb')].iloc[0]
            ax.annotate(f"d = {c['d']:.2f}, p = {c['p']:.2f}\n"
                        f"BF$_{{01}}$ = {c['bf01']:.1f}",
                        (j * 4 + 1, 0), xycoords=('data', 'axes fraction'),
                        xytext=(0, -30), textcoords='offset points',
                        ha='center', va='top', fontsize=7, color='#52514e',
                        annotation_clip=False)
        ax.set_xticks([1, 5])
        ax.set_xticklabels(MODALITIES)
        ax.set_xlim(-1, 7)
        ax.set_title(task, loc='left', fontsize=10, fontweight='bold')
        ax.set_ylabel(UNIT[task].replace(', ', ',\n'), fontsize=8)
    fig.legend(loc='lower center', ncol=3, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.13))
    fig.text(0.5, -0.2, 'Session 1 only, every participant. Bars: 95% CI '
             'of the paired difference. Text row: explicit (batch 3) vs '
             'implicit (batches 1 + 2), Welch t on the per-participant '
             'effects; BF$_{01}$ favours no difference.',
             ha='center', fontsize=7.5, color='#52514e')
    fig.tight_layout()
    save(fig, 'cohort_comparison')


def power(pw):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    d0 = pw[pw['task'] == 'Production']
    for ax, mod in zip(axes, MODALITIES):
        d = d0[d0['modality'] == mod]
        for frac, ls in ((0.5, '-'), (0.75, '--'), (1.0, ':')):
            dd = d[d['fraction'] == frac].set_index('sessions')
            dd = dd.loc[['ses1', 'ses1+2', 'ses1+2+3']]
            ax.plot([1, 2, 3], dd['n_per_cohort_between'], marker='o', ms=5,
                    lw=1.6, ls=ls, color=COL['beat'],
                    label=f'difference = {int(frac * 100)}% of effect '
                          f'({dd["delta"].iloc[0]:.3f})')
            for x, y in zip([1, 2, 3], dd['n_per_cohort_between']):
                ax.annotate(str(int(y)), (x, y), xytext=(4, 3),
                            textcoords='offset points', fontsize=7,
                            color='#52514e')
        ax.set_title(f'Production — {mod}', loc='left', fontsize=10,
                     fontweight='bold')
        ax.set_xticks([1, 2, 3])
        ax.set_xlabel('Behavioural sessions per participant')
        ax.set_yscale('log')
        ax.set_yticks([10, 20, 50, 100, 200])
        ax.set_yticklabels(['10', '20', '50', '100', '200'])
        ax.legend(fontsize=7, title='Between-cohort difference to detect',
                  title_fontsize=7)
    axes[0].set_ylabel('Participants per cohort (80% power, α = .05)')
    fig.text(0.5, -0.03, 'From the SD of the per-participant effect in '
             'batch 1 after 1, 2 or 3 sessions. A within-participant '
             'blocking factor needs the same N in total (the two effects '
             'of a participant are uncorrelated).',
             ha='center', fontsize=7.5, color='#52514e')
    fig.tight_layout()
    save(fig, 'power')


def main():
    os.makedirs(FIGURES, exist_ok=True)
    rd = lambda n: pd.read_csv(os.path.join(RESULTS, n + '.tsv'), sep='\t')
    ses = pd.read_csv(os.path.join(TABLES, 'session_level.tsv'), sep='\t')
    eff, runs, ret, comp, pw = (rd('effect_by_session'), rd('effect_by_runs'),
                                rd('retest'), rd('cohort_comparison'),
                                rd('power'))
    effect_by_session(eff)
    condition_means(ses)
    precision_by_runs(runs)
    retest(ses, ret)
    cohort_comparison(eff, comp)
    power(pw)


if __name__ == '__main__':
    main()
