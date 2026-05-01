"""
Analysis of behavioral data for the NTFD Tasks of the Music-SDTB project

author: Ana Luisa Pinho
e-mail: agrilopi@uwo.ca

Created: May 5, 2024
Last update: April, 2026

Compatibility: Python 3.10.14
"""

import os
import sys
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


def plot_pttest(data_audio, data_visual,
                y, ylim_b, ylim_t, title, output_dir, fname,
                pval_audio_bi, pval_visual_bi,
                pval_audio_br=None, pval_audio_ir=None,
                pval_visual_br=None, pval_visual_ir=None,
                norand=False, loc='inside'):

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
                            text_format="star",
                            fontsize=10.,
                            hide_non_significant=True,
                            loc=loc,
                            line_offset_to_group=1.,
                            line_height=0.05)
        annotator.set_pvalues(pvalue)
        annotator.annotate()

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

    plt.savefig(os.path.join(output_dir, fname + '.pdf'))


# %%
# ========================== INPUTS ====================================

SUBJECTS = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
            18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 32,
            34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

RAND_SUBJECTS = [3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 32, 34,
                 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

IMG_SUBJECTS = [3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21,
                22, 23, 26, 28, 29, 32, 34, 35, 38, 39, 40, 41,
                42, 43, 44, 45, 46, 47]

BEHAV_RAND_SUBJECTS = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                       26, 27, 28, 29, 32, 34, 35, 38, 39, 40,
                       41, 42, 43, 44, 45, 46, 47]

BEHAVIMG_RAND_SUBJECTS = [16, 18, 20, 21, 22, 23, 26, 28, 29, 32,
                          34, 35, 38, 39, 40, 41, 42, 43, 44, 45,
                          46, 47]

audio_latency = 133
visual_latency = 35
button_press = 20

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FOLDER = os.path.join(MAIN_DIR, 'ntfd_results')
DATAFRAMES_FOLDER = os.path.join(RESULTS_FOLDER, 'dataframes')
PLOTS_FOLDER = os.path.join(RESULTS_FOLDER, 'rt_and_success')

sessions_dic = {'allses': 'All Sessions',
                'ses-01': 'Session 1',
                'ses-02': 'Session 2',
                'ses-03': 'Session 3',
                'ses-04': 'Session 4',
                'ses-05': 'Session 5',
                'behavses': 'Behavioral Sessions',
                'behavses12': 'Behavioral Sessions',
                'behavses13': 'Behavioral Sessions',
                'behavses23': 'Behavioral Sessions',
                'imgses': 'Imaging Sessions',
                'imgses_ntfd': 'Imaging Sessions'}

subjects_dic = {'allses': BEHAVIMG_RAND_SUBJECTS,
                'ses-01': BEHAV_RAND_SUBJECTS,
                'ses-02': BEHAV_RAND_SUBJECTS,
                'ses-03': BEHAV_RAND_SUBJECTS,
                'ses-04': IMG_SUBJECTS,
                'ses-05': IMG_SUBJECTS,
                'behavses': BEHAVIMG_RAND_SUBJECTS,
                'behavses12': BEHAVIMG_RAND_SUBJECTS,
                'behavses13': BEHAVIMG_RAND_SUBJECTS,
                'behavses23': BEHAVIMG_RAND_SUBJECTS,
                'imgses': BEHAVIMG_RAND_SUBJECTS,
                'imgses_ntfd': BEHAVIMG_RAND_SUBJECTS}

# %%
# ============================ RUN =====================================

if __name__ == "__main__":

    for key, value in sessions_dic.items():

        if key == 'allses':
            sessions_list = [1, 2, 3, 4, 5]
        elif key == 'ses-01':
            sessions_list = [1]
        elif key == 'ses-02':
            sessions_list = [2]
        elif key == 'ses-03':
            sessions_list = [3]
        elif key == 'ses-04':
            sessions_list = [4]
        elif key == 'ses-05':
            sessions_list = [5]
        elif key == 'behavses':
            sessions_list = [1, 2, 3]
        elif key == 'behavses12':
            sessions_list = [1, 2]
        elif key == 'behavses13':
            sessions_list = [1, 3]
        elif key == 'behavses23':
            sessions_list = [2, 3]
        else:
            assert key in ['imgses', 'imgses_ntfd']
            sessions_list = [4, 5]

        db_path = os.path.join(DATAFRAMES_FOLDER, 'df_ntfd.tsv')
        db = pd.read_csv(db_path, sep='\t')

        df_subfiltered = db[db['subject'].isin(subjects_dic[key])]
        df = df_subfiltered[df_subfiltered['session'].isin(sessions_list)]

        if key == 'imgses_ntfd':
            df = df[df['task'] == 'NTFD']

        df = df.dropna(subset=['reaction_time'])
        df = df.drop(columns=['answer'])

        df['reaction_time'] = df['reaction_time'].astype(float)
        df.loc[df['modality'] == 'auditory', 'reaction_time'] -= (
            audio_latency + button_press)
        df.loc[df['modality'] == 'visual', 'reaction_time'] -= (
            visual_latency + button_press)

        df_ffx = ffx_dvar(df)

        rt_ab = (
            df_ffx[df_ffx.modality == 'auditory']
                  [df_ffx.condition == 'beat']
            .reaction_time
            .values
        )
        rt_ai = (
            df_ffx[df_ffx.modality == 'auditory']
                  [df_ffx.condition == 'interval']
            .reaction_time
            .values
        )
        rt_ar = (
            df_ffx[df_ffx.modality == 'auditory']
                  [df_ffx.condition == 'random']
            .reaction_time
            .values
        )
        rt_vb = (
            df_ffx[df_ffx.modality == 'visual']
                  [df_ffx.condition == 'beat']
            .reaction_time
            .values
        )
        rt_vi = (
            df_ffx[df_ffx.modality == 'visual']
                  [df_ffx.condition == 'interval']
            .reaction_time
            .values
        )
        rt_vr = (
            df_ffx[df_ffx.modality == 'visual']
                  [df_ffx.condition == 'random']
            .reaction_time
            .values
        )

        score_ab = (
            df_ffx[df_ffx.modality == 'auditory']
                  [df_ffx.condition == 'beat']
            .score
            .values
        )
        score_ai = (
            df_ffx[df_ffx.modality == 'auditory']
                  [df_ffx.condition == 'interval']
            .score
            .values
        )
        score_ar = (
            df_ffx[df_ffx.modality == 'auditory']
                  [df_ffx.condition == 'random']
            .score
            .values
        )
        score_vb = (
            df_ffx[df_ffx.modality == 'visual']
                  [df_ffx.condition == 'beat']
            .score
            .values
        )
        score_vi = (
            df_ffx[df_ffx.modality == 'visual']
                  [df_ffx.condition == 'interval']
            .score
            .values
        )
        score_vr = (
            df_ffx[df_ffx.modality == 'visual']
                  [df_ffx.condition == 'random']
            .score
            .values
        )

        if key in ['ses-04', 'imgses_ntfd']:
            rt_audio = rt_ab.tolist() + rt_ai.tolist()
            rt_visual = rt_vb.tolist() + rt_vi.tolist()

            score_audio = score_ab.tolist() + score_ai.tolist()
            score_visual = score_vb.tolist() + score_vi.tolist()
        else:
            rt_audio = rt_ab.tolist() + rt_ai.tolist() + rt_ar.tolist()
            rt_visual = rt_vb.tolist() + rt_vi.tolist() + rt_vr.tolist()

            score_audio = score_ab.tolist() + score_ai.tolist() + \
                score_ar.tolist()
            score_visual = score_vb.tolist() + score_vi.tolist() + \
                score_vr.tolist()

        _, pval_rt_abi = stats.ttest_rel(rt_ab, rt_ai,
                                         alternative='two-sided')
        _, pval_rt_vbi = stats.ttest_rel(rt_vb, rt_vi,
                                         alternative='two-sided')

        _, pval_score_abi = stats.ttest_rel(score_ab, score_ai,
                                            alternative='two-sided')
        _, pval_score_vbi = stats.ttest_rel(score_vb, score_vi,
                                            alternative='two-sided')

        if key not in ['ses-04', 'imgses_ntfd']:
            _, pval_rt_abr = stats.ttest_rel(rt_ab, rt_ar,
                                             alternative='two-sided')
            _, pval_rt_air = stats.ttest_rel(rt_ai, rt_ar,
                                             alternative='two-sided')
            _, pval_rt_vbr = stats.ttest_rel(rt_vb, rt_vr,
                                             alternative='two-sided')
            _, pval_rt_vir = stats.ttest_rel(rt_vi, rt_vr,
                                             alternative='two-sided')

            _, pval_score_abr = stats.ttest_rel(score_ab, score_ar,
                                                alternative='two-sided')
            _, pval_score_air = stats.ttest_rel(score_ai, score_ar,
                                                alternative='two-sided')
            _, pval_score_vbr = stats.ttest_rel(score_vb, score_vr,
                                                alternative='two-sided')
            _, pval_score_vir = stats.ttest_rel(score_vi, score_vr,
                                                alternative='two-sided')

        if not os.path.exists(PLOTS_FOLDER):
            os.mkdir(PLOTS_FOLDER)

        rt_title = 'Group Mean of Reaction Time for the NTFD tasks: ' + value
        rt_fname = 'group_rt_ntfd_' + key
        score_title = \
            'Group Mean of the Success Rate for the NTFD tasks: ' + value
        score_fname = 'group_scores_ntfd_' + key

        if key in ['ses-04', 'imgses_ntfd']:
            plot_pttest(rt_audio,
                        rt_visual,
                        'Reaction Time (ms)', 100., 700.,
                        rt_title, PLOTS_FOLDER, rt_fname,
                        pval_rt_abi,
                        pval_rt_vbi,
                        norand=True, loc='inside')
            plot_pttest(np.multiply(score_audio, 100).tolist(),
                        np.multiply(score_visual, 100).tolist(),
                        'Group Mean Score (%)', 0, 100,
                        score_title, PLOTS_FOLDER, score_fname,
                        pval_score_abi,
                        pval_score_vbi,
                        norand=True, loc='outside')
        else:
            plot_pttest(rt_audio,
                        rt_visual,
                        'Reaction Time (ms)', 100., 700.,
                        rt_title, PLOTS_FOLDER, rt_fname,
                        pval_rt_abi, pval_rt_vbi,
                        pval_audio_br=pval_rt_abr,
                        pval_audio_ir=pval_rt_air,
                        pval_visual_br=pval_rt_vbr,
                        pval_visual_ir=pval_rt_vir, loc='inside')
            plot_pttest(np.multiply(score_audio, 100).tolist(),
                        np.multiply(score_visual, 100).tolist(),
                        'Group Mean Score (%)', 70, 100,
                        score_title, PLOTS_FOLDER, score_fname,
                        pval_score_abi, pval_score_vbi,
                        pval_audio_br=pval_score_abr,
                        pval_audio_ir=pval_score_air,
                        pval_visual_br=pval_score_vbr,
                        pval_visual_ir=pval_score_vir, loc='outside')