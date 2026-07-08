"""
Analysis of behavioral data for the NTFD Tasks of the TDTB project

author: Ana Luisa Pinho
e-mail: agrilopi@uwo.ca

Created: May 5, 2024
Last update: July 2026

Compatibility: Python 3.10.14
"""

import os
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import seaborn as sns

from scipy import stats
from matplotlib import pyplot as plt
from statannotations.Annotator import Annotator


# %%
# ======================== MAIN FUNCTIONS ==============================

def ffx_dvar(df):
    # Fixed Effects within subjects, averaged across subjects
    df_ffx = df.drop(['session'], axis=1)
    df_ffx = df_ffx.groupby([
        'condition', 'modality', 'subject']).mean(
            numeric_only=True).reset_index()

    return df_ffx


def get_ylim(data, pad_ratio=.15, default=(0., 1.), bounds=None):
    """Return padded y-limits from finite data values."""
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]

    if data.size == 0:
        return default

    ymin = np.min(data)
    ymax = np.max(data)
    pad = (ymax - ymin) * pad_ratio if ymax != ymin else 1.

    ylim_b = ymin - pad
    ylim_t = ymax + pad

    if bounds is not None:
        lower, upper = bounds
        if lower is not None:
            ylim_b = max(lower, ylim_b)
        if upper is not None:
            ylim_t = min(upper, ylim_t)

    return ylim_b, ylim_t


def plot_pttest(data_audio, data_visual,
                y, title, output_dir, fname,
                pval_audio_bi, pval_visual_bi,
                pval_audio_br=None, pval_audio_ir=None,
                pval_visual_br=None, pval_visual_ir=None,
                norand=False, loc='inside',
                annotation_text_format='star',
                hide_non_significant=True,
                y_bounds=None):

    allowed_formats = ['star', 'simple', 'full']
    if annotation_text_format not in allowed_formats:
        raise ValueError(
            'annotation_text_format must be one of: ' +
            ', '.join(allowed_formats))

    modalities = ['audio', 'visual']
    fig, ax = plt.subplots(1, len(modalities))

    plt.subplots_adjust(left=.15, bottom=.15, wspace=.25, top=.8)

    x = 'Conditions'

    if norand:
        pval_audio = [pval_audio_bi]
        pval_visual = [pval_visual_bi]
    else:
        pval_audio = [pval_audio_bi, pval_audio_br, pval_audio_ir]
        pval_visual = [pval_visual_bi, pval_visual_br, pval_visual_ir]

    ylim_b, ylim_t = get_ylim(
        data_audio + data_visual,
        bounds=y_bounds)

    for m, modality in enumerate(modalities):
        if modality == 'audio':
            data_list = data_audio
            pvalue = pval_audio
        else:
            assert modality == 'visual'
            data_list = data_visual
            pvalue = pval_visual

        if norand:
            conditions = \
                np.repeat('Beat', len(data_list) / 2).tolist() + \
                np.repeat('Interval', len(data_list) / 2).tolist()
        else:
            conditions = \
                np.repeat('Beat', len(data_list) / 3).tolist() + \
                np.repeat('Interval', len(data_list) / 3).tolist() + \
                np.repeat('Random', len(data_list) / 3).tolist()

        d = {x: conditions, y: data_list}
        df = pd.DataFrame(data=d)

        colors = ['tab:blue', 'tab:orange', 'tab:pink']

        sns.boxplot(
            ax=ax[m],
            x=x,
            y=y,
            data=df,
            palette=colors,
            medianprops={"color": "k", "linewidth": 0.},
            notch=True,
            meanline=True,
            showmeans=True,
            meanprops=dict(color="tab:brown", linewidth=1.5),
            **{'boxprops': {'alpha': 0.5, 'edgecolor': 'black'}})

        if norand:
            pairs = [('Beat', 'Interval')]
        else:
            pairs = [('Beat', 'Interval'),
                     ('Beat', 'Random'),
                     ('Interval', 'Random')]

        annotator = Annotator(ax[m], pairs, data=df, x=x, y=y)
        annotator.configure(test=None,
                            text_format=annotation_text_format,
                            fontsize=10.,
                            hide_non_significant=hide_non_significant,
                            loc=loc,
                            line_offset_to_group=1.,
                            line_height=0.05)
        
        annotator.set_pvalues(pvalue)
        annotator.annotate()

        if annotation_text_format == 'full':
            for text in ax[m].texts:
                current = text.get_text()
                if current.startswith('None '):
                    text.set_text(current.replace('None ', ''))

        ax[m].set_ylim(bottom=ylim_b, top=ylim_t)

        if m == 1:
            ax[m].axes.get_yaxis().set_visible(False)
            ax[m].spines['left'].set_visible(False)
            ax[m].set_xlabel('Visual Conditions', fontweight='semibold',
                             labelpad=14, fontsize=12)
        else:
            assert m == 0
            ax[m].set_xlabel('Auditory Conditions', fontweight='semibold',
                             labelpad=14, fontsize=12)
            ax[m].set_ylabel(y, fontsize=13, labelpad=10)
            ax[m].tick_params(axis='y', labelsize=11)

        ax[m].tick_params(axis='x', labelsize=11)

        ax[m].spines['right'].set_visible(False)
        ax[m].spines['top'].set_visible(False)

    plt.savefig(
        os.path.join(output_dir, fname + '.png'),
        dpi=300,
        bbox_inches='tight')


def condition_arrays(df, modality, conditions, value_col):
    """Return subject-aligned arrays for one modality and value column."""
    sub_df = df[
        (df['modality'] == modality) &
        (df['condition'].isin(conditions))
    ].copy()

    wide = sub_df.pivot(
        index='subject',
        columns='condition',
        values=value_col,
    )

    available = [cond for cond in conditions if cond in wide.columns]
    if len(available) != len(conditions):
        return [np.array([]) for _ in conditions]

    wide = wide.dropna(subset=conditions).sort_index()
    return [wide[cond].to_numpy() for cond in conditions]


def paired_pvalue(values_a, values_b):
    """Return paired t-test p-value, or NaN when n < 2."""
    if len(values_a) < 2 or len(values_b) < 2:
        return np.nan

    _, p_value = stats.ttest_rel(
        values_a,
        values_b,
        alternative='two-sided',
    )
    return p_value


def flatten_arrays(arrays):
    """Flatten a list of condition arrays preserving condition order."""
    return [value for array in arrays for value in array.tolist()]


def has_random_condition(df):
    """Return True when random trials exist for both modalities."""
    random_df = df[df['condition'] == 'random']
    modalities = set(random_df['modality'].unique())
    return {'auditory', 'visual'}.issubset(modalities)

# %%
# ========================== INPUTS ====================================

# All good subjects including img pilot (sub-04)
GOOD_SUBJECTS = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 32,
                 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

# Excludes subjects 4, 5, and 9 from GOOD_SUBJECTS.
RAND_SUBJECTS = [3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 32, 34,
                 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

# Imaging subjects only, without the pilot subject.
IMG_SUBJECTS = [3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21,
                22, 23, 26, 28, 29, 32, 34, 35, 38, 39, 40, 41,
                42, 43, 44, 45, 46, 47]

# Subjects who completed behavioral sessions with the random condition.
BEHAV_RAND_SUBJECTS = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                       26, 27, 28, 29, 32, 34, 35, 38, 39, 40,
                       41, 42, 43, 44, 45, 46, 47]

# Second batch
ALL_SB_SUBJECTS = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]

GOOD_SB_SUBJECTS = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61, 62]

SB2_SUBJECTS = [50, 51, 52, 55, 57, 59]

SB3_SUBJECTS = []

# #####################################################################

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))

# #### First Batch ####

fb_sessions_dic = {
    'allses': 'All Sessions',
    'behavses': 'Behavioral Sessions',
    'imgses': 'Imaging Sessions',
    'ses-01': 'Session 1',
    'ses-02': 'Session 2',
    'ses-03': 'Session 3',
    'ses-04': 'Session 4',
    'ses-05': 'Session 5',
    'behav12': 'Behavioral Sessions 1 and 2',
    'behav13': 'Behavioral Sessions 1 and 3',
    'behav23': 'Behavioral Sessions 2 and 3',
}

fb_subjects_dic = {
    'allses': RAND_SUBJECTS,
    'behavses': BEHAV_RAND_SUBJECTS,
    'imgses': IMG_SUBJECTS,
    'ses-01': BEHAV_RAND_SUBJECTS,
    'ses-02': BEHAV_RAND_SUBJECTS,
    'ses-03': BEHAV_RAND_SUBJECTS,
    'ses-04': IMG_SUBJECTS,
    'ses-05': IMG_SUBJECTS,
    'behav12': BEHAV_RAND_SUBJECTS,
    'behav13': BEHAV_RAND_SUBJECTS,
    'behav23': BEHAV_RAND_SUBJECTS,
}

# #### Second Batch ####

sb_sessions_dic = {
    'behavses': 'Behavioral Sessions',
    'ses-01': 'Session 1',
    'ses-02': 'Session 2',
}

sb_subjects_dic = {
    'behavses': GOOD_SB_SUBJECTS,
    'ses-01': GOOD_SB_SUBJECTS,
    'ses-02': SB2_SUBJECTS,
}

# #### Map tag -> integer session list ####

sessions_list_dic = {
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

# #### Batch configurations ####

# Reaction times are always latency-corrected in this script.
# First batch: Expyriment auditory latency.
# Second batch: PsychoPy auditory latency.

fb_audio_latency = 133
sb_audio_latency = 63
visual_latency = 35
button_press = 20

# Keep this list explicit so each batch can be run one at a time
# by commenting out any entry if needed.
# BATCHES_TO_RUN = ['first', 'second']
BATCHES_TO_RUN = ['second']

batch_dic = {
    'first': {
        'sessions': fb_sessions_dic,
        'subjects': fb_subjects_dic,
        'audio_latency': fb_audio_latency,
        'visual_latency': visual_latency,
        'button_press': button_press,
        'results_folder': os.path.join(MAIN_DIR, 'ntfd_results_first_batch'),
    },
    'second': {
        'sessions': sb_sessions_dic,
        'subjects': sb_subjects_dic,
        'audio_latency': sb_audio_latency,
        'visual_latency': visual_latency,
        'button_press': button_press,
        'results_folder': os.path.join(MAIN_DIR, 'ntfd_results_second_batch'),
    },
}

# Plot annotation options.
# Use 'star' for significance stars, 'simple' for p-values, or
# 'full' for the verbose statannotations format.
ANNOTATION_TEXT_FORMAT = 'full'
HIDE_NON_SIGNIFICANT = False


# %%
# ============================ RUN =====================================

if __name__ == "__main__":

    for batch_tag in BATCHES_TO_RUN:
        batch_info = batch_dic[batch_tag]
        sessions_dic = batch_info['sessions']
        subjects_dic = batch_info['subjects']
        audio_latency = batch_info['audio_latency']
        visual_latency = batch_info['visual_latency']
        button_press = batch_info['button_press']
        results_folder = batch_info['results_folder']
        dataframes_folder = os.path.join(results_folder, 'dataframes')
        plots_folder = os.path.join(results_folder, 'rt_and_success')

        print('\n' + '=' * 60)
        print(f'Batch: {batch_tag}')
        print(
            'Latencies: '
            f'audio={audio_latency}, visual={visual_latency}, '
            f'button={button_press}')
        print(f'Results folder: {results_folder}')
        print('=' * 60)

        for key, value in sessions_dic.items():
            sessions_list = sessions_list_dic[key]

            print(f'\nSession tag: {key}  |  {value}')

            db_path = os.path.join(dataframes_folder, f'df_ntfd_{key}.tsv')
            if not os.path.exists(db_path):
                raise FileNotFoundError(
                    'Could not find NTFD dataframe for tag ' + key +
                    ' in ' + dataframes_folder)

            db = pd.read_csv(db_path, sep='\t')

            df_subfiltered = db[db['subject'].isin(subjects_dic[key])]
            df = df_subfiltered[
                df_subfiltered['session'].isin(sessions_list)]

            df = df.dropna(subset=['reaction_time'])
            df = df.drop(columns=['answer'])

            df['reaction_time'] = df['reaction_time'].astype(float)
            df.loc[df['modality'] == 'auditory', 'reaction_time'] -= (
                audio_latency + button_press)
            df.loc[df['modality'] == 'visual', 'reaction_time'] -= (
                visual_latency + button_press)

            df_ffx = ffx_dvar(df)
            has_random = has_random_condition(df_ffx)
            conditions = ['beat', 'interval']
            if has_random:
                conditions.append('random')

            rt_audio_arrays = condition_arrays(
                df_ffx, 'auditory', conditions, 'reaction_time')
            rt_visual_arrays = condition_arrays(
                df_ffx, 'visual', conditions, 'reaction_time')
            score_audio_arrays = condition_arrays(
                df_ffx, 'auditory', conditions, 'score')
            score_visual_arrays = condition_arrays(
                df_ffx, 'visual', conditions, 'score')

            n_audio = len(rt_audio_arrays[0])
            n_visual = len(rt_visual_arrays[0])
            if n_audio < 2 or n_visual < 2:
                print(
                    'Skipping group statistics and plotting for ' +
                    key + ': at least 2 complete subjects are required; '
                    f'found auditory={n_audio}, visual={n_visual}.')
                continue

            rt_ab, rt_ai = rt_audio_arrays[0], rt_audio_arrays[1]
            rt_vb, rt_vi = rt_visual_arrays[0], rt_visual_arrays[1]
            score_ab = score_audio_arrays[0]
            score_ai = score_audio_arrays[1]
            score_vb = score_visual_arrays[0]
            score_vi = score_visual_arrays[1]

            rt_audio = flatten_arrays(rt_audio_arrays)
            rt_visual = flatten_arrays(rt_visual_arrays)
            score_audio = flatten_arrays(score_audio_arrays)
            score_visual = flatten_arrays(score_visual_arrays)

            pval_rt_abi = paired_pvalue(rt_ab, rt_ai)
            pval_rt_vbi = paired_pvalue(rt_vb, rt_vi)
            pval_score_abi = paired_pvalue(score_ab, score_ai)
            pval_score_vbi = paired_pvalue(score_vb, score_vi)

            if has_random:
                rt_ar = rt_audio_arrays[2]
                rt_vr = rt_visual_arrays[2]
                score_ar = score_audio_arrays[2]
                score_vr = score_visual_arrays[2]

                pval_rt_abr = paired_pvalue(rt_ab, rt_ar)
                pval_rt_air = paired_pvalue(rt_ai, rt_ar)
                pval_rt_vbr = paired_pvalue(rt_vb, rt_vr)
                pval_rt_vir = paired_pvalue(rt_vi, rt_vr)

                pval_score_abr = paired_pvalue(score_ab, score_ar)
                pval_score_air = paired_pvalue(score_ai, score_ar)
                pval_score_vbr = paired_pvalue(score_vb, score_vr)
                pval_score_vir = paired_pvalue(score_vi, score_vr)

            if not os.path.exists(plots_folder):
                os.makedirs(plots_folder)

            rt_title = (
                'Group Mean of Reaction Time for the NTFD tasks: ' +
                value)
            rt_fname = 'group_rt_ntfd_' + key
            score_title = (
                'Group Mean of the Success Rate for the NTFD tasks: ' +
                value)
            score_fname = 'group_scores_ntfd_' + key

            if not has_random:
                plot_pttest(
                    rt_audio,
                    rt_visual,
                    'Reaction Time (ms)',
                    rt_title, plots_folder, rt_fname,
                    pval_rt_abi,
                    pval_rt_vbi,
                    norand=True, loc='inside',
                    annotation_text_format=ANNOTATION_TEXT_FORMAT,
                    hide_non_significant=HIDE_NON_SIGNIFICANT)
                plot_pttest(
                    np.multiply(score_audio, 100).tolist(),
                    np.multiply(score_visual, 100).tolist(),
                    'Group Mean Score (%)',
                    score_title, plots_folder, score_fname,
                    pval_score_abi,
                    pval_score_vbi,
                    norand=True, loc='outside',
                    annotation_text_format=ANNOTATION_TEXT_FORMAT,
                    hide_non_significant=HIDE_NON_SIGNIFICANT,
                    y_bounds=(0, 100))
            else:
                plot_pttest(
                    rt_audio,
                    rt_visual,
                    'Reaction Time (ms)',
                    rt_title, plots_folder, rt_fname,
                    pval_rt_abi, pval_rt_vbi,
                    pval_audio_br=pval_rt_abr,
                    pval_audio_ir=pval_rt_air,
                    pval_visual_br=pval_rt_vbr,
                    pval_visual_ir=pval_rt_vir, loc='inside',
                    annotation_text_format=ANNOTATION_TEXT_FORMAT,
                    hide_non_significant=HIDE_NON_SIGNIFICANT)
                plot_pttest(
                    np.multiply(score_audio, 100).tolist(),
                    np.multiply(score_visual, 100).tolist(),
                    'Group Mean Score (%)',
                    score_title, plots_folder, score_fname,
                    pval_score_abi, pval_score_vbi,
                    pval_audio_br=pval_score_abr,
                    pval_audio_ir=pval_score_air,
                    pval_visual_br=pval_score_vbr,
                    pval_visual_ir=pval_score_vir, loc='outside',
                    annotation_text_format=ANNOTATION_TEXT_FORMAT,
                    hide_non_significant=HIDE_NON_SIGNIFICANT,
                    y_bounds=(70, 100))