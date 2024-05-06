"""
Analysis of behavioral data for the NTFD Tasks of the Music-SDTB project

author: Ana Luisa Pinho
e-mail: agrilopi@uwo.ca

Created: May 5, 2024
Last update: May 2024

Compatibility: Python 3.10.4
"""

import sys
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
    df_ffx = df.drop(['Session'], axis=1)
    df_ffx = df_ffx.groupby([
        'Condition', 'Modality', 'Subject']).mean().reset_index()

    return df_ffx


def plot_pttest(data_audio, data_visual,
                y, ylim_b, ylim_t, yshift, title, output_dir, fname,
                pval_audio_bi, pval_visual_bi, 
                pval_audio_br=None, pval_audio_ir=None,
                pval_visual_br=None, pval_visual_ir=None,
                norand=False):

    modalities = ['audio', 'visual']
    fig, ax = plt.subplots(1, len(modalities))

    # left   # the left side of the subplots of the figure
    # right  # the right side of the subplots of the figure
    # bottom # the bottom of the subplots of the figure
    # top    # the top of the subplots of the figure
    # wspace # the amount of width reserved for blank space between subplots
    # hspace # the amount of height reserved for white space between subplots
    plt.subplots_adjust(left=.15, bottom=.15, wspace=.25)

    # Define subplot of bar charts and its position in the fig
    # plt.axes([left, bottom, width, height])
    # ax = plt.axes([.225, .145, .65, .65])

    # Prepare the data
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
            x_label = 'Auditory Conditions'
        else:
            assert modality == 'visual'
            data_list = data_visual
            pvalue = pval_visual
            x_label = 'Visual Conditions'

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
        # Create boxplot
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
            meanprops = dict(color="tab:brown",linewidth=1.5),
            **{'boxprops': {'alpha': 0.5, 'edgecolor': 'black'}})

        # Annotate
        if norand:
            pairs = [('Beat', 'Interval')]
        else:
            pairs = [('Beat', 'Interval'),
                     ('Beat', 'Random'),
                     ('Interval', 'Random')]
        
        annotator = Annotator(ax[m], pairs, data=df, x=x, y=y)
        annotator.configure(test=None,
                            text_format="star",
                            # test_short_name="pttest",
                            fontsize=10.,
                            hide_non_significant=False)
        annotator.set_pvalues(pvalue)
        annotator.annotate()

        # Set limits of y-axis
        ax[m].set_ylim(bottom=ylim_b, top=ylim_t)

        if m ==1:
            # Remove labels and ticks
            ax[m].axes.get_yaxis().set_visible(False)
            # Remove y frame
            ax[m].spines['left'].set_visible(False)
            # Change x label
            ax[m].set_xlabel('Visual Conditions', fontweight='semibold',
                             labelpad=15)
        else:
            assert m == 0
            ax[m].set_xlabel('Auditory Conditions', fontweight='semibold',
                             labelpad=15)

        # Hide the right and top spines
        ax[m].spines['right'].set_visible(False)
        ax[m].spines['top'].set_visible(False)

    # Title
    plt.suptitle(title, size=10, y=.96)
    # plt.title('95% CI for the Mean', size=8, x=-.15)

    # Save figure
    plt.savefig(os.path.join(output_dir, fname + '.pdf'))



# %%
# ========================== INPUTS ====================================

RAND_SUBJECTS = [3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                 23, 24, 25, 26, 27, 28, 29, 32, 34, 35, 38, 39, 40, 41, 42,
                 43, 44, 45, 46, 47]

BEHAV_RAND_SUBJECTS = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                       32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

IMG_SUBJECTS = [3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26,
                28, 29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FOLDER = os.path.join(MAIN_DIR, 'ntfd_results')
DATAFRAMES_FOLDER = os.path.join(RESULTS_FOLDER, 'dataframes')
PLOTS_FOLDER = os.path.join(RESULTS_FOLDER, 'rt')

sessions_dic = {'allses': 'All Sessions',
                'ses-01': 'Session 1',
                'ses-02': 'Session 2',
                'ses-03': 'Session 3',
                'ses-04': 'Session 4',
                'ses-05': 'Session 5'}

subjects_dic = {'allses': RAND_SUBJECTS,
                'ses-01': BEHAV_RAND_SUBJECTS,
                'ses-02': BEHAV_RAND_SUBJECTS,
                'ses-03': BEHAV_RAND_SUBJECTS,
                'ses-04': IMG_SUBJECTS,
                'ses-05': IMG_SUBJECTS}

# %%
# ============================ RUN =====================================

if __name__ == "__main__":

    for key, value in sessions_dic.items():
        # Open dataframe
        db_path = os.path.join(DATAFRAMES_FOLDER,
                                'df_ntfd_' + key + '.tsv')
        db = pd.read_csv(db_path, sep='\t')
        db['RT'] = db['RT'].astype('str')

        # Remove rows with 'n/a' entries
        na = db['RT'].str.contains('n/a')
        filtered_db = db[~na]

        # Remove rows with nan's entries
        nans = filtered_db['RT'].str.contains('nan')
        filtered_db = filtered_db[~nans]

        # Filter subjects
        filtered_db = filtered_db[
            filtered_db['Subject'].isin(subjects_dic[key])]

        # Convert RT to numbers
        filtered_db['RT'] = filtered_db['RT'].apply(pd.to_numeric)

        # Extract dependant variable
        db_ffx = ffx_dvar(filtered_db)

        # Concatenate dependent variable
        rt_ab = db_ffx[
            db_ffx.Modality=='audio'][db_ffx.Condition=='beat'].RT.values
        rt_ai = db_ffx[
            db_ffx.Modality=='audio'][db_ffx.Condition=='interval'].RT.values
        rt_ar = db_ffx[
            db_ffx.Modality=='audio'][db_ffx.Condition=='random'].RT.values
        rt_vb = db_ffx[
            db_ffx.Modality=='visual'][db_ffx.Condition=='beat'].RT.values
        rt_vi = db_ffx[
            db_ffx.Modality=='visual'][db_ffx.Condition=='interval'].RT.values
        rt_vr = db_ffx[
            db_ffx.Modality=='visual'][db_ffx.Condition=='random'].RT.values

        if key == 'ses-04':
            rt_audio = rt_ab.tolist() + rt_ai.tolist()
            rt_visual = rt_vb.tolist() + rt_vi.tolist()
        else:
            rt_audio = rt_ab.tolist() + rt_ai.tolist() + rt_ar.tolist()
            rt_visual = rt_vb.tolist() + rt_vi.tolist() + rt_vr.tolist()

        # Compute paired t-tests
        _, pval_abi = stats.ttest_rel(rt_ab, rt_ai, alternative='two-sided')
        _, pval_vbi = stats.ttest_rel(rt_vb, rt_vi, alternative='two-sided')

        if key != 'ses-04':
            _, pval_abr = stats.ttest_rel(rt_ab, rt_ar, alternative='two-sided')
            _, pval_air = stats.ttest_rel(rt_ai, rt_ar, alternative='two-sided')
            _, pval_vbr = stats.ttest_rel(rt_vb, rt_vr, alternative='two-sided')
            _, pval_vir = stats.ttest_rel(rt_vi, rt_vr, alternative='two-sided')

        # Plot
        if not os.path.exists(PLOTS_FOLDER):
            os.mkdir(PLOTS_FOLDER)

        title = 'Group Mean of Reaction Time for the NTFD tasks: ' + value
        fname = 'group_rt_ntfd_' + key

        if key == 'ses-04':
            plot_pttest(rt_audio, rt_visual,
                        'Reaction Time (ms)', 300., 900., -100.,
                        title, PLOTS_FOLDER, fname,
                        pval_abi, pval_vbi,
                        norand=True)
        else:
            plot_pttest(rt_audio, rt_visual,
                        'Reaction Time (ms)', 300., 900., -100.,
                        title, PLOTS_FOLDER, fname,
                        pval_abi, pval_vbi, 
                        pval_audio_br=pval_abr, pval_audio_ir=pval_air,
                        pval_visual_br=pval_vbr, pval_visual_ir=pval_vir)
