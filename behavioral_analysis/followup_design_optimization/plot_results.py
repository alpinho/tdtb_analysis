"""
Figures for the follow-up design-optimization analysis, from the tables written by
build_summary_table.py and effect_by_data.py. Written to figures/ as PDF
(for Beamer) and PNG.

    effect_by_session_<task>.pdf
                                Interval - Beat effect with its 95% CI for
                                every data subset and modality; the three
                                batches side by side
    condition_means.pdf         Beat and Interval means per session with
                                within-subject (Cousineau-Morey) 95% CIs
    precision_by_runs.pdf       standard error of the effect against trials
                                per condition, with the 1/sqrt(n) reference
    retest.pdf                  per-participant effect in session 1 against
                                session 2 (auditory and visual Production)
    cohort_comparison.pdf       explicit against implicit cohorts
    explicit_cohort.pdf         the explicit cohort's effects (sessions 1-2)
                                beside the implicit cohorts, same layout as
                                the effect row of imaging_cohort_behaviour
    imaging_cohort_behaviour.pdf
                                the imaging participants' behavioural
                                effects (sessions 1-2) beside the
                                jitter-free second cohort
    power.pdf                   participants per cohort against sessions
"""

import os
import numpy as np
import pandas as pd
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MAIN_DIR)
sys.path.insert(0, os.path.join(MAIN_DIR, '..'))
TABLES = os.path.join(MAIN_DIR, 'summary_tables')
RESULTS = os.path.join(MAIN_DIR, 'results')
FIGURES = os.path.join(MAIN_DIR, 'figures')

COL = {'beat': '#2a78d6', 'interval': '#eb6834', 'fb_img': '#4a3aa7',
       'fb': '#4a3aa7', 'sb': '#d55181', 'tb': '#8c6d1f'}
COHORT_LABEL = {'fb': 'Implicit, Expyriment (cohort 1)',
                'sb': 'Implicit, PsychoPy (cohort 2)',
                'tb': 'Explicit, PsychoPy (cohort 3)'}
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
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2) if task == 'Production'
                                 else (9, 3.6))
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
                            label=COHORT_LABEL[cohort] if j == 1 else None)
            from statsmodels.stats.multitest import multipletests
            for s, subset in enumerate(SUBSETS):
                rows = [(k, cohort, d[(d['cohort'] == cohort)
                                      & (d['subset'] == subset)])
                        for k, cohort in enumerate(['fb', 'sb', 'tb'])]
                rows = [(k, c, dc.iloc[0]) for k, c, dc in rows
                        if not dc.empty]
                if not rows:
                    continue
                p_holm = multipletests([r['p'] for _, _, r in rows],
                                       method='holm')[1]
                for (k, cohort, r), ph in zip(rows, p_holm):
                    sig = stars(r['p']) if ph < 0.05 else ''
                    sign = '\u2212' if r['d_z'] < 0 else '\u2002'
                    ax.annotate(f"{sign}{abs(r['d_z']):.2f}{sig}",
                                (s - 0.45, 0),
                                xycoords=('data', 'axes fraction'),
                                xytext=(0, -30 - 11 * k),
                                textcoords='offset points', ha='left',
                                va='top', fontsize=8, color=COL[cohort],
                                annotation_clip=False)
            ax.annotate('$d_z$', (-0.6, 0), xycoords=('data', 'axes fraction'),
                        xytext=(0, -30), textcoords='offset points',
                        ha='center', va='top', fontsize=8,
                        color='#52514e', annotation_clip=False)
            ax.set_title(f'{mod.capitalize()} {task}', loc='center', fontsize=10,
                         fontweight='bold')
            ax.set_xticks(range(len(SUBSETS)))
            ax.set_xticklabels([SUBSET_LABEL[s] for s in SUBSETS],
                               fontsize=8)
            ax.set_xlim(-0.6, len(SUBSETS) - 0.4)
            ax.set_ylabel(UNIT[task].replace(', ', ',\n'), fontsize=8)
        if task == 'Production':
            axes[1].legend(loc='lower right', fontsize=7)
            fig.text(0.5, -0.14, 'Paired t-test of Interval vs Beat per '
                     'cohort: bars, 95% CI of the paired difference; $d_z$ per '
                     'cohort (colour).\nStars: each cohort\'s uncorrected p from '
                     'its paired t-test, shown only where its Holm-corrected p '
                     'is < .05 (* <.05, ** <.01, *** <.001).',
                     ha='center', fontsize=9, color='#52514e')
        fig.tight_layout()
        save(fig, 'effect_by_session_' + task.lower())


def cousineau_morey(wide):
    """Within-subject 95% CI half-width per column (Cousineau 2005, Morey
    2008): the normalised SEM scaled by the t critical value."""
    from scipy import stats
    n, k = wide.shape
    norm = wide.sub(wide.mean(axis=1), axis=0) + wide.values.mean()
    sem = norm.std(ddof=1) / np.sqrt(n) * np.sqrt(k / (k - 1))
    return sem * stats.t.ppf(0.975, n - 1)


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
            ax.set_title(f'{mod.capitalize()} {task}', loc='center', fontsize=10,
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
    fig.text(0.5, 0.005, 'Cohort 1 (N = 39 behavioural, 31 imaging). Bars: '
             'within-subject 95% CI (Cousineau–Morey),\ncomputed separately '
             'for the behavioural and the imaging sessions.',
             ha='center', fontsize=14, color='#52514e')
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, 'condition_means')


def precision_by_runs(runs):
    fb = runs[runs['cohort'] == 'fb']
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, task in zip(axes, ['Production', 'NTFD']):
        for mod, marker in (('auditory', 'o'), ('visual', 's')):
            d = fb[(fb['task'] == task) & (fb['modality'] == mod)]
            d = d.sort_values('trials_per_condition')
            mcol = '#1baf7a' if mod == 'auditory' else '#eda100'
            ax.plot(d['trials_per_condition'], d['se'], marker=marker,
                    ms=5, lw=1.6, color=mcol,
                    label=f'{mod.capitalize()}: observed SE')
            ref = d['se'].iloc[0] * np.sqrt(
                d['trials_per_condition'].iloc[0] / d['trials_per_condition'])
            ax.plot(d['trials_per_condition'], ref, ls='--', lw=1,
                    color=mcol, alpha=0.7,
                    label=f'{mod.capitalize()}: predicted, $SE_1\\sqrt{{n_1/n}}$')
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
            ax.text(x, 0.02, lab, rotation=90, va='bottom', ha='right',
                    fontsize=8, color='#52514e',
                    transform=ax.get_xaxis_transform())
        ax.set_title(task, loc='center', fontsize=10, fontweight='bold')
        ax.set_xlabel('Trials per condition per participant')
        ax.set_ylabel('SE of the Interval $-$ Beat effect')
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout()
    fig.subplots_adjust(left=0.3)
    xc = (axes[0].get_position().x0 + axes[1].get_position().x1) / 2
    fig.text(xc, -0.03, 'Cohort 1, cumulative behavioural runs in '
             'chronological order (N = 39). Point labels: $d_z$.',
             ha='center', fontsize=9, color='#52514e')
    fig.text(0.005, 0.55,
             'For the first $k$ runs of\neach participant $i$\n'
             '($n$ trials per condition):\n\n'
             '$\\Delta_i = \\overline{Interval}_i - \\overline{Beat}_i$\n\n'
             '$SE = \\dfrac{SD(\\Delta_1, \\ldots, \\Delta_N)}{\\sqrt{N}}$\n\n'
             'Dashed: precision expected\nif only trial noise shrinks,\n'
             '$SE_1\\sqrt{n_1/n}$ with $n_1 = 15$.',
             ha='left', va='center', fontsize=11, color='#52514e')
    save(fig, 'precision_by_runs')


def retest(ses, ret):
    """Session 1 against session 2 per participant, cohort 1 Production:
    top row the level (mean of Beat and Interval), bottom row the Beat
    advantage (Interval - Beat)."""
    fb = ses[(ses['batch'] == 'fb') & (ses['task'] == 'Production')]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.2))
    for c, mod in enumerate(MODALITIES):
        d = fb[fb['modality'] == mod]
        w = d.pivot_table(index=['subject', 'session'], columns='condition',
                          values='value')
        for r, (measure, series, label) in enumerate((
                ('level', (w['interval'] + w['beat']) / 2,
                 'Level: mean across Temporal Structure'),
                ('effect', w['interval'] - w['beat'],
                 'Beat advantage: Interval $-$ Beat'))):
            ax = axes[r, c]
            e = series.unstack('session')[[1, 2]].dropna()
            lim_lo, lim_hi = e.values.min(), e.values.max()
            pad = (lim_hi - lim_lo) * 0.08
            lo, hi = lim_lo - pad, lim_hi + pad
            ax.plot([lo, hi], [lo, hi], color='#c9c8c2', lw=0.8, ls='--')
            ax.axhline(0, color='#9a9a94', lw=0.8)
            ax.axvline(0, color='#9a9a94', lw=0.8)
            ax.scatter(e[1], e[2], s=28, color=COL['fb'], edgecolor='white',
                       lw=0.8, zorder=3)
            rr = ret[(ret['cohort'] == 'fb') & (ret['task'] == 'Production')
                     & (ret['modality'] == mod) & (ret['measure'] == measure)
                     & (ret['sessions'] == '1-2')].iloc[0]
            ax.set_title(f"{mod.capitalize()} Production\n{label}\n"
                         f"r = {rr['r']:.2f}, p = {rr['p']:.2g}",
                         loc='center', fontsize=9.5, fontweight='bold')
            ax.set_xlabel('Session 1')
            ax.set_ylabel('Session 2')
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect('equal')
    fig.text(0.5, -0.04, 'Cohort 1 (N = 39): each point is one participant.\n'
             'The level is stable across sessions; the Beat advantage is '
             'not.', ha='center', va='top', fontsize=13, color='#52514e')
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
               bbox_to_anchor=(0.5, -0.24))
    fig.text(0.5, -0.2, 'Session 1 only, every participant. Bars: 95% CI '
             'of the paired difference. Text row: explicit (cohort 3) vs '
             'implicit (cohorts 1 + 2), Welch t on the per-participant '
             'effects; BF$_{01}$ favours no difference.',
             ha='center', fontsize=7.5, color='#52514e')
    fig.tight_layout()
    save(fig, 'cohort_comparison')


MCOL = {'auditory': '#1baf7a', 'visual': '#eda100'}


def cohort_means(std, cohort, task, mod):
    """Subject x {beat, interval} means over behavioural sessions 1-2, each
    participant's completed sessions pooled (Supplementary Note 4)."""
    import effect_by_data as e
    if task == 'Perception':
        return e.perception_means(cohort, 'ses1+2', mod)
    d = e.select(std, cohort)
    d = d[(d['task'] == task) & (d['modality'] == mod)]
    return e.pooled_means(d, [1, 2], complete=False)


def sessions_label(std, cohort):
    import effect_by_data as e
    d = e.select(std, cohort)
    d = d[(d['task'] == 'Production') & (d['session'].isin([1, 2]))]
    n_ses = d.groupby('subject')['session'].nunique()
    n2 = int((n_ses >= 2).sum()); n1 = int((n_ses == 1).sum())
    return (f"{n1 + n2} ({n2} with 2 sessions, {n1} with 1)"
            if n1 else f"{n2}, 2 sessions each")


def effect_panel(bot, std, task, cohorts, compare, legend, style,
                 fs=12.5, step=3.4):
    """Interval - Beat per cohort and modality with 95% CI, the d_z and d
    rows (stars where Holm-corrected p < .05 over the tests of the panel)
    and the Welch cohort comparison. `cohorts` is a list of (cohort,
    label); `compare` = (a, b) the cohorts whose interaction is reported,
    d > 0 meaning a larger effect in a; `style(cohort, mod)` gives the
    marker (color, face color)."""
    import effect_by_data as e
    from cross_cohort_behaviour import interaction
    from statsmodels.stats.multitest import multipletests
    n = len(cohorts)
    group = n * step + 0.6
    centre = 0.25 + (n - 1) * step / 2
    bot.axhline(0, color='#9a9a94', lw=0.8)
    stats_task = []
    for j, mod in enumerate(MODALITIES):
        for k, (cohort, label) in enumerate(cohorts):
            wide = cohort_means(std, cohort, task, mod)
            x = j * group + k * step + 0.25
            color, mfc = style(cohort, mod)
            st = e.paired_stats(wide)
            stats_task.append((x, st, wide))
            bot.errorbar([x], [st['effect']],
                         yerr=[[st['effect'] - st['ci_low']],
                               [st['ci_high'] - st['effect']]],
                         fmt='o', ms=8, color=color, mfc=mfc,
                         mew=1.8, capsize=4, elinewidth=1.8)
            if legend and j == 0:
                bot.plot([], [], 'o', ms=8, color=color, mfc=mfc, mew=1.4,
                         label=f"{label}: n = {sessions_label(std, cohort)}")
        diffs = [cohort_means(std, c, task, mod) for c in compare]
        res = interaction(*[w['interval'] - w['beat'] for w in diffs])
        bot.annotate(f"cohorts:\nd = {res['d']:.2f}\np = {res['p']:.2f}"
                     f"\nBF$_{{01}}$ = {res['bf01']:.1f}",
                     (j * group + centre, 0),
                     xycoords=('data', 'axes fraction'),
                     xytext=(0, -76), textcoords='offset points',
                     ha='center', va='top', fontsize=fs,
                     color='#52514e', annotation_clip=False)
    p_holm = multipletests([st['p'] for _, st, _ in stats_task],
                           method='holm')[1]
    for tag, dy in (('$d_z$', -34), ('$d$', -52)):
        bot.annotate(tag, (-0.05, 0), xycoords='axes fraction',
                     xytext=(0, dy), textcoords='offset points',
                     ha='right', va='top', fontsize=fs,
                     color='#52514e', annotation_clip=False)
    for (x, st, wide), ph in zip(stats_task, p_holm):
        sig = stars(st['p']) if ph < 0.05 else ''
        sd_pooled = np.sqrt((wide['beat'].var(ddof=1)
                             + wide['interval'].var(ddof=1)) / 2)
        d_pooled = st['effect'] / sd_pooled
        for val, dy in ((st['d_z'], -34), (d_pooled, -52)):
            bot.annotate(f"{val:.2f}{sig}", (x, 0),
                         xycoords=('data', 'axes fraction'),
                         xytext=(0, dy), textcoords='offset points',
                         ha='center', va='top', fontsize=fs,
                         color='#52514e', annotation_clip=False)
    ticks = [centre, group + centre]
    bot.set_xticks(ticks)
    bot.set_xticklabels([m.capitalize() for m in MODALITIES])
    bot.set_xlim(-1.4, group + 2 * centre + 1.4)
    metric = {'Production': 'Mean Signed Asynchrony',
              'Perception': 'Difference Limen',
              'NTFD': 'Reaction Time (ms)'}[task]
    bot.set_ylabel('Interval $-$ Beat,\n' + metric, fontsize=12)
    return ticks, metric


def imaging_cohort_behaviour(std, ses):
    """Two rows per task: Beat and Interval means per cohort (top) and the
    Interval - Beat effect per cohort with the cohort comparison (bottom).
    Behavioural sessions 1-2, each participant's completed sessions pooled,
    as in Supplementary Note 4 of the manuscript."""
    cohorts = [('fb_img', 'Imaging cohort (cohort 1)'),
               ('sb', 'Second cohort (cohort 2, no jitter)')]
    style = lambda cohort, mod: (MCOL[mod],
                                 MCOL[mod] if cohort == 'fb_img' else 'white')
    plt.rcParams.update({'font.size': 13})
    fig, axes = plt.subplots(2, 3, figsize=(16, 6.4),
                             gridspec_kw={'height_ratios': [1, 1.2]})
    for c, task in enumerate(TASKS):
        top, bot = axes[0, c], axes[1, c]
        for j, mod in enumerate(MODALITIES):
            for k, (cohort, label) in enumerate(cohorts):
                wide = cohort_means(std, cohort, task, mod)
                sem = cousineau_morey(wide[['beat', 'interval']])
                x0 = j * 7.4 + k * 3.4
                mfc = style(cohort, mod)[1]
                top.errorbar([x0, x0 + 0.5],
                             [wide['beat'].mean(), wide['interval'].mean()],
                             yerr=[sem['beat'], sem['interval']],
                             fmt='-o', ms=7, lw=1.8, color=MCOL[mod],
                             mfc=mfc, mew=1.8, capsize=3)
                top.annotate('B', (x0, wide['beat'].mean()), xytext=(-7, -4),
                             textcoords='offset points', ha='right',
                             va='top', fontsize=11, fontweight='bold',
                             color='#52514e')
                top.annotate('I', (x0 + 0.5, wide['interval'].mean()),
                             xytext=(7, -4), textcoords='offset points',
                             ha='left', va='top', fontsize=11,
                             fontweight='bold', color='#52514e')
        ticks, metric = effect_panel(bot, std, task, cohorts,
                                     ('fb_img', 'sb'), style=style,
                                     legend=(task == 'Production'))
        top.margins(y=0.25)
        top.set_xticks(ticks)
        top.set_xticklabels([m.capitalize() for m in MODALITIES])
        top.set_xlim(bot.get_xlim())
        top.set_title(task, loc='center', fontsize=14, fontweight='bold')
        top.set_ylabel(metric, fontsize=12)
        if task == 'Production':
            top.axhline(0, color='#9a9a94', lw=0.8)
    fig.legend(loc='lower center', ncol=2, fontsize=14,
               bbox_to_anchor=(0.5, -0.17))
    fig.tight_layout(h_pad=2.0)
    fig.subplots_adjust(left=0.23, wspace=0.5)
    row_text = {
        0: 'Beat (B) and Interval (I)\nmeans, within-subject 95% CI',
        1: 'Interval $-$ Beat, 95% CI\n\n'
           '$d_z$ = mean difference /\nSD of the per-person differences\n'
           '(how consistently does each\nperson show the effect?)\n'
           '$d$ = mean difference /\npooled SD of Beat and Interval\n'
           '(how large is the effect relative\nto the spread across participants?)\n\n'
           'stars: the paired t-test of the\neffect, which both d\'s describe;\n'
           'uncorrected p, shown where\nHolm-corrected p < .05\n'
           '(four tests per task)\n* <.05, ** <.01, *** <.001\n\n'
           '"cohorts": Welch\'s t-test of\nCohort $\\times$ Temporal Structure\n'
           '(d > 0: larger effect in the\nimaging cohort)',
    }
    for r in (0, 1):
        pos = axes[r, 0].get_position()
        y = (pos.y0 + pos.y1) / 2 - (0.15 if r else 0)
        fig.text(-0.02, y, row_text[r], ha='left', va='center',
                 fontsize=12, color='#52514e')
    save(fig, 'imaging_cohort_behaviour')
    plt.rcParams.update({'font.size': 9})


def explicit_cohort(std):
    """One row: the three cohorts' Interval - Beat effects side by side,
    behavioural sessions 1-2, with the same statistics as the effect row
    of imaging_cohort_behaviour; the cohort comparison is explicit
    (cohort 3) against implicit (cohorts 1 + 2), d > 0 meaning a larger
    effect in the explicit cohort."""
    cohorts = [(c, COHORT_LABEL[c]) for c in ['fb', 'sb', 'tb']]
    style = lambda cohort, mod: (COL[cohort], COL[cohort])
    plt.rcParams.update({'font.size': 13})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for c, task in enumerate(TASKS):
        effect_panel(axes[c], std, task, cohorts, ('tb', 'fb+sb'),
                     legend=(task == "Production"), style=style, fs=10.5,
                     step=4.4)
        axes[c].set_title(task, loc='center', fontsize=14,
                          fontweight='bold')
    fig.legend(loc="lower center", ncol=1, fontsize=12.5,
               bbox_to_anchor=(0.5, -0.24))
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.45)
    save(fig, 'explicit_cohort')
    plt.rcParams.update({'font.size': 9})


def power(pw):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.8), sharey=True)
    d0 = pw[pw['task'] == 'Production']
    for ax, mod in zip(axes, MODALITIES):
        d = d0[d0['modality'] == mod]
        for frac, ls, keep in ((0.5, '-', '½'), (0.75, '--', '¼'),
                               (1.0, ':', '0')):
            dd = d[d['fraction'] == frac].set_index('sessions')
            dd = dd.loc[['ses1', 'ses1+2', 'ses1+2+3']]
            ax.plot([1, 2, 3], dd['n_per_cohort_between'], marker='o', ms=5,
                    lw=1.6, ls=ls, color=COL['beat'],
                    label=f'{keep} of the effect: '
                          f'{dd["reference_effect"].iloc[0] - dd["delta"].iloc[0]:.3f} '
                          f'(difference {dd["delta"].iloc[0]:.3f})')
            for x, y in zip([1, 2, 3], dd['n_per_cohort_between']):
                ax.annotate(str(int(y)), (x, y), xytext=(4, 3),
                            textcoords='offset points', fontsize=7,
                            color='#52514e')
        ax.set_title(f'{mod.capitalize()} Production', loc='center',
                     fontsize=10, fontweight='bold')
        ax.set_xticks([1, 2, 3])
        ax.set_xlabel('Behavioural sessions per participant')
        ax.set_yscale('log')
        ax.set_yticks([10, 20, 50, 100, 200])
        ax.set_yticklabels(['10', '20', '50', '100', '200'])
        ax.legend(fontsize=7, title='The other cohort keeps',
                  title_fontsize=7)
    axes[0].set_ylabel('Participants per cohort (80% power, α = .05)')
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
    std = pd.read_csv(os.path.join(TABLES, 'standard_level.tsv'), sep='\t')
    imaging_cohort_behaviour(std, ses)
    explicit_cohort(std)
    power(pw)


if __name__ == '__main__':
    main()
