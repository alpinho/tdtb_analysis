"""
This script performs several ANOVA analysis using ROIS extracted from contrasts
 of the Music-SDTB Project.

Author: Ana Luisa Pinho

Created: October 2024
Last update: October 2024

Compatibility: Python 3.10.8

"""

import os
import sys
import numpy as np
import pandas as pd

from scipy.stats import ttest_rel

import seaborn as sns
import pingouin as pg
from statannotations.Annotator import Annotator
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import MultiComparison
from matplotlib import pyplot as plt
from itertools import chain

# ############################ FUNCTIONS ################################

def dataframe(data, hemispheres, tasks, contrasts, n_subjects, outpath):
    # input data shape: (hemisphere, tasks, contrasts, subjects)
    # ## Open npy file
    data = np.load(data)

    subjects = ['sub-%02d' % s for s in n_subjects]
    category = [[contrast[s+1:] for s, char in enumerate(contrast[:-1])
                 if char == ' '][0] for contrast in contrasts]
    modality = [[contrast[:s] for s, char in enumerate(contrast[:-1])
                 if char == ' '][0] for contrast in contrasts]

    # ## Subjects column
    subjects_col = np.tile(
        subjects,
        data.shape[2] * data.shape[1] * data.shape[0])
    # ## Contrasts column
    contrasts_rep = np.repeat(contrasts, len(subjects))
    contrasts_col = np.tile(
        contrasts_rep,
        data.shape[1] * data.shape[0])
    # ## Category column
    category_rep = np.repeat(category, len(subjects))
    category_col = np.tile(
        category_rep, data.shape[1] * data.shape[0])
    # ## Modality column
    modality_rep = np.repeat(modality, len(subjects))
    modality_col = np.tile(
        modality_rep, data.shape[1] * data.shape[0])
    # ## Tasks column
    tasks_rep = np.repeat(tasks, len(modality_rep))
    tasks_col = np.tile(tasks_rep, data.shape[0])
    # ## Hemispheres column
    hem_col = np.repeat(hemispheres, len(tasks_rep))

    # ## Data column
    data_col = np.ravel(data)
    table = np.vstack((data_col, subjects_col, contrasts_col,
                       category_col, modality_col,
                       tasks_col, hem_col)).T

    # ## Build dataframe
    df = pd.DataFrame(table,
                      columns=['PSC', 'Subject', 'Contrast',
                               'Category', 'Modality', 'Task',
                               'Hemisphere'])

    # Create outdir, if it does not exist
    outdir = os.path.dirname(outpath)
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # Save dataframe
    df.to_csv(outpath, index=False, sep='\t')

    return df


def threeway_rmanova(df, output_dir, prefix, roi, hems=['lh', 'rh', 'bh']):
    """
    Compute 2 X 2 X 3 RM-ANOVA
    """
    # Open dataframe
    if isinstance(df, str):
        df = pd.read_csv(df, sep='\t')

    # Remove 'All Tasks' rows from Dataframe
    df = df[df.Task != 'All Tasks']

    # Convert PSC entries to numeric type
    df['PSC'] = df['PSC'].apply(pd.to_numeric)

    # For each hemisphere:
    for hem in hems:
        db = pd.DataFrame()
        db = df[df.Hemisphere == hem]

        # Create AnovaRM object
        model = AnovaRM(data=db, depvar='PSC', subject='Subject',
                        within=['Category', 'Modality', 'Task'])

        # Run the 3-way repeated measures ANOVA
        results = model.fit()

        # Perform pairwise t-tests corrected w/ Holm's procedure
        mccat = MultiComparison(db['PSC'], db['Category'])
        phoc_category = mccat.allpairtest(ttest_rel, method='Holm')[0]

        mcmod = MultiComparison(db['PSC'], db['Modality'])
        phoc_modality = mcmod.allpairtest(ttest_rel, method='Holm')[0]

        mctask = MultiComparison(db['PSC'], db['Task'])
        phoc_task = mctask.allpairtest(ttest_rel, method='Holm')[0]

        mccatmod = MultiComparison(db['PSC'], db['Contrast'])
        phoc_catmod = mccatmod.allpairtest(ttest_rel, method='Holm')[0]

        # Create output_dir, if it does not exist
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        # Save results in a TSV file...
        flabel = prefix + '_' + roi + '_' + hem + '_3w_'

        # ... for ANOVA
        results.anova_table.to_csv(
            os.path.join(output_dir, flabel + 'anova.tsv'), sep='\t')

        # ... and for posthoc
        phoc_flabel = flabel + 'posthoc_'

        pd.DataFrame(phoc_category).to_csv(
            os.path.join(output_dir, phoc_flabel + 'category.tsv'),
            index=False, header=False, sep='\t')

        pd.DataFrame(phoc_modality).to_csv(
            os.path.join(output_dir, phoc_flabel + 'modality.tsv'),
            index=False, header=False, sep='\t')

        pd.DataFrame(phoc_task).to_csv(
            os.path.join(output_dir, phoc_flabel + 'task.tsv'),
            index=False, header=False, sep='\t')

        pd.DataFrame(phoc_catmod).to_csv(
            os.path.join(output_dir, phoc_flabel + 'catmod.tsv'),
            index=False, header=False, sep='\t')


def twoway_rmanova_task(df, tasks_dic, output_dir, prefix, roi,
                        alternative='two-sided', hems=['lh', 'rh', 'bh']):
    """
    Compute 2 X 2 ANOVA per task
    """
    # Open dataframe
    if isinstance(df, str):
        df = pd.read_csv(df, sep='\t')

    # Remove Column of Contrasts
    df = df.drop(['Contrast'], axis=1)

    # Convert PSC entries to numeric type
    df['PSC'] = df['PSC'].apply(pd.to_numeric)

    # Tasks
    ttags = list(tasks_dic.keys())
    tasks_list = list(tasks_dic.values())

    # For each task:
    for ttag, task in zip(ttags, tasks_list):
        # For each hemisphere:
        for hem in hems:
            db = pd.DataFrame()
            db = df[df.Task == task][df.Hemisphere == hem]

            # Run the 2-way repeated measures ANOVA
            anova_results = pg.rm_anova(
                data=db, dv='PSC', within=['Modality', 'Category'],
                subject='Subject', detailed=True)

            # Perform pairwise t-tests corrected w/ Holm's procedure
            posthoc_results = pg.pairwise_tests(
                data=db, dv='PSC', within=['Category', 'Modality'],
                subject='Subject', alternative=alternative, return_desc=True,
                padjust='holm', effsize='eta-square')

            # Create output_dir, if it does not exist
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)

            # Save results in a TSV file...
            flabel = prefix + '_' + roi + '_' + hem + '_2w-' + ttag + '_'

            # ... for ANOVA
            anova_results.to_csv(
                os.path.join(output_dir, flabel + 'anova.tsv'), sep='\t',
                index=False)

            # ... and for posthoc
            posthoc_results.to_csv(
                os.path.join(output_dir, flabel + 'posthoc.tsv'), sep='\t',
                index=False)


def twoway_rmanova_gtasks(df, output_dir, prefix, roi,
                          hems=['lh', 'rh', 'bh']):
    """
    Compute 2 X 2 RM-ANOVA across all tasks
    """
    # Open dataframe
    if isinstance(df, str):
        df = pd.read_csv(df, sep='\t')

    # Remove 'All Tasks' rows from Dataframe
    df = df[df.Task != 'All Tasks']

    # Remove Column of Tasks and Contrasts
    df = df.drop(['Task'], axis=1)
    df = df.drop(['Contrast'], axis=1)

    # Convert PSC entries to numeric type
    df['PSC'] = df['PSC'].apply(pd.to_numeric)

    # For each hemisphere:
    for hem in hems:
        db = pd.DataFrame()
        db = df[df.Hemisphere == hem]

        # Averaged PSC across Tasks, i.e. grouped by Category and Modality ...
        # ... and averaged afterwards
        db = db.drop(['Hemisphere'], axis=1)
        db = db.groupby([
            'Category', 'Modality', 'Subject']).mean().reset_index()

        # Run the 2-way repeated measures ANOVA
        anova_results = pg.rm_anova(
            data=db, dv='PSC', within=['Modality', 'Category'],
            subject='Subject', detailed=True)

        # Perform pairwise t-tests corrected w/ Holm's procedure
        posthoc_results = pg.pairwise_tests(
            data=db, dv='PSC', within=['Category', 'Modality'],
            subject='Subject', return_desc=True,
            padjust='holm', effsize='eta-square')

        # Create output_dir, if it does not exist
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        # Save results in a TSV file...
        flabel = prefix + '_' + roi + '_' + hem + '_2w-taskavg_'

        # ... for ANOVA
        anova_results.to_csv(
            os.path.join(output_dir, flabel + 'anova.tsv'), sep='\t',
            index=False)

        # ... and for posthoc
        posthoc_results.to_csv(
            os.path.join(output_dir, flabel + 'posthoc.tsv'), sep='\t',
            index=False)


def oneway_rmanova(df, tasks_dic, output_dir, prefix, roi,
                   hems=['lh', 'rh', 'bh']):
    """
    Compute one-way ANOVA: one categorical variable, i.e. category,
    with two levels (equivalent to pptest; function for sanity check)
    """
    # Open dataframe
    if isinstance(df, str):
        df = pd.read_csv(df, sep='\t')

    # Remove Column of Contrasts
    df = df.drop(['Contrast'], axis=1)

    # Convert PSC entries to numeric type
    df['PSC'] = df['PSC'].apply(pd.to_numeric)

    # Tasks
    ttags = list(tasks_dic.keys())
    tasks_list = list(tasks_dic.values())

    # For each task:
    for ttag, task in zip(ttags, tasks_list):
        # For each modality:
        for modality in ['Auditory', 'Visual']:
            # For each hemisphere:
            for hem in hems:
                db = pd.DataFrame()
                db = df[df.Task == task][df.Modality == modality][
                    df.Hemisphere == hem]

                # Run the 2-way repeated measures ANOVA
                anova_results = pg.rm_anova(
                    data=db, dv='PSC', within='Category', subject='Subject',
                    detailed=True)

                # Perform pairwise t-tests corrected w/ Holm's procedure
                posthoc_results = pg.pairwise_tests(
                    data=db, dv='PSC', within='Category', subject='Subject',
                    return_desc=True, padjust='holm', effsize='eta-square')

                # Create output_dir, if it does not exist
                if not os.path.exists(output_dir):
                    os.mkdir(output_dir)

                # Save results in a TSV file...
                flabel = prefix + '_' + roi + '_' + hem + '_1w-' + \
                    ttag + '_' + modality.lower() + '_'

                # ... for ANOVA
                anova_results.to_csv(
                    os.path.join(output_dir, flabel + 'anova.tsv'), sep='\t',
                    index=False)

                # ... and for posthoc
                posthoc_results.to_csv(
                    os.path.join(output_dir, flabel + 'posthoc.tsv'), sep='\t',
                    index=False)


def twoway_rmanova_catroi(df, tasks_dic, output_dir, prefix,
                          alternative='two-sided', modality=None,
                          hems=['lh', 'rh', 'bh']):
    """
    Compute 2 X 2 ANOVA per task
    """
    # Open dataframe
    if isinstance(df, str):
        df = pd.read_csv(df, sep='\t')

    # Remove Column of Contrasts
    df = df.drop(['Contrast'], axis=1)

    # Convert PSC entries to numeric type
    df['PSC'] = df['PSC'].apply(pd.to_numeric)

    if modality is None:
        df = df.drop(['Modality'], axis=1)
        # Averaged PSC across Modalities, i.e. grouped by Category and Task ...
        # ... and averaged afterwards
        df = df.groupby(['Category', 'Task', 'Subject', 'ROI',
                         'Hemisphere']).mean().reset_index()
    elif modality == 'auditory':
        df = df[df.Modality == 'Auditory']
        df = df.drop(['Modality'], axis=1)
    else:
        assert modality == 'visual'
        df = df[df.Modality == 'Visual']
        df = df.drop(['Modality'], axis=1)

    # Tasks
    ttags = list(tasks_dic.keys())
    tasks_list = list(tasks_dic.values())

    # For each task:
    for ttag, task in zip(ttags, tasks_list):
        # For each hemisphere:
        for hem in hems:
            db = pd.DataFrame()
            db = df[df.Task == task][df.Hemisphere == hem]

            # Run the 2-way repeated measures ANOVA
            anova_results = pg.rm_anova(
                data=db, dv='PSC', within=['ROI', 'Category'],
                subject='Subject', detailed=True)

            # Perform pairwise t-tests corrected w/ Holm's procedure
            posthoc_results = pg.pairwise_tests(
                data=db, dv='PSC', within=['ROI', 'Category'],
                subject='Subject', alternative=alternative, return_desc=True,
                padjust='holm', effsize='eta-square')

            # Create output_dir, if it does not exist
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)

            # Save results in a TSV file...
            flabel = prefix + '_' + hem + '_2w-' + ttag + '_'

            # ... for ANOVA
            anova_results.to_csv(
                os.path.join(output_dir, flabel + 'anova.tsv'), sep='\t',
                index=False)

            # ... and for posthoc
            posthoc_results.to_csv(
                os.path.join(output_dir, flabel + 'posthoc.tsv'), sep='\t',
                index=False)


def twoway_rmanova_timingroi(df, output_dir, prefix, alternative='two-sided',
                             modality=None, hems=['lh', 'rh', 'bh']):
    """
    Compute 2 X 2 ANOVA per task
    """
    # Open dataframe
    if isinstance(df, str):
        df = pd.read_csv(df, sep='\t')

    # Remove Column of Contrasts
    df = df.drop(['Contrast'], axis=1)

    # Convert PSC entries to numeric type
    df['PSC'] = df['PSC'].apply(pd.to_numeric)

    if modality is None:
        df = df.drop(['Modality'], axis=1)
        # Averaged PSC across Modalities, i.e. grouped by Category and Task ...
        # ... and averaged afterwards
        df = df.groupby(['Category', 'Task', 'Subject', 'ROI',
                         'Hemisphere']).mean().reset_index()
    elif modality == 'auditory':
        df = df[df.Modality == 'Auditory']
        df = df.drop(['Modality'], axis=1)
    else:
        assert modality == 'visual'
        df = df[df.Modality == 'Visual']
        df = df.drop(['Modality'], axis=1)

    # Add explicit/implicit timing column
    # df['Timing'] = np.where(df['Task']== 'Production', 'Explicit', 'Implicit')

    # Remove Column of Task and Category
    # df = df.drop(['Task'], axis=1)
    df = df.drop(['Category'], axis=1)

    # Remove 'All Tasks' rows from Dataframe
    df = df[df.Task != 'All Tasks']

    # For each hemisphere:
    for hem in hems:
        db = pd.DataFrame()
        db = df[df.Hemisphere == hem]

        # Run the 2-way repeated measures ANOVA
        anova_results = pg.rm_anova(
            data=db, dv='PSC', within=['ROI', 'Task'],
            subject='Subject', detailed=True)

        # Perform pairwise t-tests corrected w/ Holm's procedure
        posthoc_results = pg.pairwise_tests(
            data=db, dv='PSC', within=['ROI', 'Task'],
            subject='Subject', alternative=alternative, return_desc=True,
            padjust='holm', effsize='eta-square')

        # Create output_dir, if it does not exist
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        # Save results in a TSV file...
        flabel = prefix + '_' + hem + '_2w_'

        # ... for ANOVA
        anova_results.to_csv(
            os.path.join(output_dir, flabel + 'anova.tsv'), sep='\t',
            index=False)

        # ... and for posthoc
        posthoc_results.to_csv(
            os.path.join(output_dir, flabel + 'posthoc.tsv'), sep='\t',
            index=False)


def pval_label_converter(pvalues):
    # * For "star" text_format: `[[1e-4, "****"], [1e-3, "***"],
    #                         [1e-2, "**"], [0.05, "*"],
    #                         [1, "ns"]]`.
    pval_labels = []
    for pval in pvalues:
        if pval <= .0001:
            pval_labels.append('****')
        elif pval > .0001 and pval <= .001:
            pval_labels.append('***')
        elif pval > .001 and pval <= .01:
            pval_labels.append('**')
        elif pval > .01 and pval <= .05:
            pval_labels.append('*')
        else:
            pval_labels.append('ns')

    return pval_labels


def plot_roi_vertical(arr_conmean, region, roi, atlas, ianalysis, effect_type,
                      prefix, hypothesis='greater'):
    # input shape: (hemisphere, tasks, contrasts, subjects)
    if isinstance(arr_conmean, str):
        # ## Open npy files and plot
        arr_conmean = np.load(arr_conmean).tolist()

    # Names of Tasks
    tnames = list(tasks.values())
    n_tasks = len(tnames)

    # Names of Contrasts
    cnames = list(filtered_contrasts.values())
    n_pairs = len(np.arange(len(cnames))[::2])

    for h, hem in enumerate(['Left Hemisphere', 'Right Hemisphere']):
        for t, tname in enumerate(tnames):
            if h==0 and t == 0:
                fig = plt.figure(figsize=(12, 12))

            for c, cidx in enumerate(np.arange(len(cnames))[::2]):

                # Define subplot of bar charts and its position in the fig
                # plt.axes([left, bottom, width, height])
                ax = plt.axes([.07 + h*.49 + c*.11, .675 - t*.2, .1, .15])

                con1 = arr_conmean[h][t][cidx]
                con2 = arr_conmean[h][t][cidx+1]
                data_list = np.append(con1, con2).tolist()

                cname1 = cnames[cidx]
                cname2 = cnames[cidx+1]
                cname = np.append(np.repeat(cname1, len(con1)),
                                  np.repeat(cname2, len(con2))).tolist()

                x = 'Contrasts Names'
                y = 'Mean of %BOLD change'
                # Long data frame
                d = {x: cname,
                     y: data_list}
                df = pd.DataFrame(data=d)
                # Create bar plot
                b = sns.barplot(ax=ax,
                                x=x,
                                y=y,
                                data=df,
                                palette=[sns.color_palette("colorblind")[2],
                                         sns.color_palette("colorblind")[8]],
                                estimator=np.mean,
                                ci=95, # errorbar=('ci', 95), # 1.96 * standard error (95% confidence interval)
                                errcolor="black", errwidth=1.5, capsize = 0.2, alpha=0.5)

                # Compute p-value
                _, pvalue = ttest_rel(con1, con2, alternative=hypothesis)
                print(pvalue)

                # Annotate
                pair = tuple([[(cname1), (cname2)]])
                annotator = Annotator(ax, pair, data=df, x=x, y=y)
                annotator.configure(test=None,
                                    text_format="star", # text_format="simple"
                                    # test_short_name="pttest", # if former is "simple"
                                    fontsize=10.)

                annotator.set_pvalues([pvalue])
                annotator.annotate()

                # Remove x-label of axis
                b.set(xlabel=None)

                # Rotate xtick labels
                ax.set_xticklabels(ax.get_xticklabels(), rotation=20,
                                   ha='right', fontsize=8)

                # Hide the right and top spines
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)

                if t != len(tnames)-1:
                    # ... remove x labels but keep ticks
                    plt.gca().set_xticklabels([])

                if c > 0:
                    # ... remove y labels and y ticks
                    ax.axes.get_yaxis().set_visible(False)
                    # ... remove y frame
                    ax.spines['left'].set_visible(False)
                else:
                    # Title
                    plt.title(tname, size=12, x=2., fontweight='bold')
                    # Customize label of y-axis
                    if (h == 0 and t != 2) or h > 0:
                        # Remove y-label of axis
                        b.set(ylabel=None)
                    else:
                        b.yaxis.set_label_coords(-.4, 1.2)

                    if t==0:
                        # Title of figure
                        plt.text(1.8 + h*.09, 1.15, hem, fontsize=14,
                                 fontweight='bold')

                # Set limits of ticks in y axis
                plt.ylim([0., .8])

        # Title of figure
        plt.suptitle(roi.capitalize(), x=.5, y=.97, size=18, linespacing=.75,
                     fontweight='bold')

        output_folder = os.path.join(msdtb_dir, region, atlas, ianalysis)
        fname = prefix + '_' + roi + '_' + effect_type + '_' + hypothesis
        # Save figure
        plt.savefig(os.path.join(output_folder, fname + '.pdf'))


def posthoc_catroi(df, tasks_dic, output_folder, prefix, n_rois, order_list,
                   modality=None, hems=['lh', 'rh', 'bh']):
    """
    Plot posthoc 2w-ANOVA per task
    """
    # Open dataframe
    if isinstance(df, str):
        df = pd.read_csv(df, sep='\t')

    # Remove Column of Contrasts
    df = df.drop(['Contrast'], axis=1)

    # Convert PSC entries to numeric type
    df['PSC'] = df['PSC'].apply(pd.to_numeric)

    if modality is None:
        df = df.drop(['Modality'], axis=1)
        # Averaged PSC across Modalities, i.e. grouped by Category and Task ...
        # ... and averaged afterwards
        df = df.groupby(['Category', 'Task', 'Subject', 'ROI',
                         'Hemisphere']).mean().reset_index()
    elif modality == 'auditory':
        df = df[df.Modality == 'Auditory']
        df = df.drop(['Modality'], axis=1)
    else:
        assert modality == 'visual'
        df = df[df.Modality == 'Visual']
        df = df.drop(['Modality'], axis=1)

    # # Replace strings with names of ROIs
    df['ROI'] = df['ROI'].str.replace('dstr', 'Dorsal Striatum')
    df['ROI'] = df['ROI'].str.replace('cereb', 'Cerebellum')
    order_list = [s.replace('dstr', 'Dorsal Striatum') for s in order_list]
    order_list = [s.replace('cereb', 'Cerebellum') for s in order_list]

    # Tasks
    ttags = list(tasks_dic.keys())
    tasks_list = list(tasks_dic.values())

    fig = plt.figure(figsize=(12, 12))

    # For each hemisphere:
    for h, hem in enumerate(hems):

        # For each task:
        for t, (ttag, task) in enumerate(zip(ttags, tasks_list)):

            # Define subplot of bar charts and its position in the fig
            # plt.axes([left, bottom, width, height])
            ax = plt.axes([.07 + h*.3, .7825 - t*.2425, .23, .15])

            db = pd.DataFrame()
            db = df[df.Task == task][df.Hemisphere == hem]

            # Create bar plot
            s = sns.barplot(
                ax=ax,
                x='ROI',
                y='PSC',
                hue='Category',
                data=db,
                estimator=np.mean,
                ci=95, # 1.96 * standard error (95% confidence interval)
                errcolor="darkgray", errwidth=1.5, capsize = 0.2, alpha=0.5,
                order=order_list
            )

            if hem == 'bh' and task == 'All Tasks':

                # Annotate
                # rois = np.flip(np.unique(df.ROI.values))
                # pairs = tuple([[(str(roi), 'Beat'), (str(roi), 'Interval')]
                #                for roi in rois])
                # annotator = Annotator(ax, pairs, data=db, x='ROI', y='PSC',
                #                       hue='Category')
                # annotator.configure(test=None,
                #                     text_format="star", # text_format="simple"
                #                     # test_short_name="pttest", # if former is "simple"
                #                     fontsize=10., hide_non_significant=True)

                # annotator.set_pvalues([0.0471665164707565, 0.471495843530365])
                # annotator.annotate()

                ax.text(-.05, .3, '95% CI for the Mean of PSC', size=8)

                # Remove frame of legend
                ax.legend(frameon=False)

                # Task
                plt.title('Audio Tasks', size=12, x=.5, y=1.1,
                          fontweight='bold', color='mediumaquamarine')
            else:
                # ... remove legend
                ax.legend([],[], frameon=False)

                # Task
                plt.title(task, size=12, x=.5, y=1.1, fontweight='bold')

            # Add values inside bars
            for i in s.containers:
                ax.bar_label(i, padding=-10, fontsize=6, fmt='%.3f')

            # Hide the right and top spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            # Set limits of ticks in y axis
            plt.ylim([0., .3])

            # Add units in ylabel
            plt.ylabel('Percent Signal Change (%)', labelpad=7)

            # Rotate x_labels
            if n_rois == 6:
                # Rotating X-axis labels
                plt.xticks(rotation = 30, fontsize=10)

            # Hemisphere
            if t == 0:
                plt.text(1.15, .39, hem.capitalize(), size=18,
                         linespacing=.75, fontweight='bold')

            if hem == 'lh' and task == 'Production':
                if modality:
                    plt.text(-1., .39, modality.capitalize(), size=12,
                             linespacing=.75, fontweight='bold')
                else:
                    plt.text(-1., .39, 'Both Mod.', size=12,
                             linespacing=.75, fontweight='bold')
                plt.text(-1., .36, prefix, size=12,
                         linespacing=.75, fontweight='bold')

    # Save figure
    if modality:
        fname = prefix + '_' + str(n_rois) + '-rois_2w_posthoc_' + modality
    else:
        fname = prefix + '_' + str(n_rois) + '-rois_2w_posthoc_both-modalitites'
    plt.savefig(os.path.join(output_folder, fname + '.pdf'))


def posthoc_timingroi(df, output_folder, prefix, n_rois, order_list,
                      modality=None, hems=['lh', 'rh', 'bh']):
    """
    Plot posthoc 2w-ANOVA
    """
    # Open dataframe
    if isinstance(df, str):
        df = pd.read_csv(df, sep='\t')

    # Remove Column of Contrasts
    df = df.drop(['Contrast'], axis=1)

    # Convert PSC entries to numeric type
    df['PSC'] = df['PSC'].apply(pd.to_numeric)

    if modality is None:
        df = df.drop(['Modality'], axis=1)
        # Averaged PSC across Modalities, i.e. grouped by Category and Task ...
        # ... and averaged afterwards
        df = df.groupby(['Category', 'Task', 'Subject', 'ROI',
                         'Hemisphere']).mean().reset_index()
    elif modality == 'auditory':
        df = df[df.Modality == 'Auditory']
        df = df.drop(['Modality'], axis=1)
    else:
        assert modality == 'visual'
        df = df[df.Modality == 'Visual']
        df = df.drop(['Modality'], axis=1)

    # Add explicit/implicit timing column
    # df['Timing'] = np.where(df['Task']== 'Production', 'Explicit', 'Implicit')

    # Remove Column of Task
    # df = df.drop(['Task'], axis=1)

    # Remove 'All Tasks' rows from Dataframe
    df = df[df.Task != 'All Tasks']

    # # Replace strings with names of ROIs
    df['ROI'] = df['ROI'].str.replace('dstr', 'Dorsal Striatum')
    df['ROI'] = df['ROI'].str.replace('cereb', 'Cerebellum')
    order_list = [s.replace('dstr', 'Dorsal Striatum') for s in order_list]
    order_list = [s.replace('cereb', 'Cerebellum') for s in order_list]

    fig = plt.figure(figsize=(12, 4))

    # For each hemisphere:
    for h, hem in enumerate(hems):

        # Define subplot of bar charts and its position in the fig
        # plt.axes([left, bottom, width, height])
        ax = plt.axes([.07 + h*.3, .15, .23, .65])

        db = pd.DataFrame()
        db = df[df.Hemisphere == hem]

        # Create bar plot
        s = sns.barplot(
            ax=ax,
            x='ROI',
            y='PSC',
            hue='Task',
            data=db,
            estimator=np.mean,
            ci=95,  # 1.96 * standard error (95% confidence interval)
            errcolor="darkgray", errwidth=1.5, capsize=0.2, alpha=0.5,
            order=order_list,
            hue_order=['Production', 'Perception', 'NTFD'],
            palette=['indigo', 'm', 'salmon']
        )

        ax.text(.4, .33, '95% CI for the Mean of PSC', size=7)

        if hem == 'bh':

            # Annotate
            # rois = np.flip(np.unique(df.ROI.values))
            # pairs = [[[(str(roi), 'Production'), (str(roi), 'Perception')],
            #           [(str(roi), 'Production'), (str(roi), 'NTFD')],
            #           [(str(roi), 'Perception'), (str(roi), 'NTFD')]]
            #          for roi in rois]

            # pairs = list(chain.from_iterable(pairs))

            # annotator = Annotator(ax, pairs, data=db, x='ROI', y='PSC',
            #                       hue='Task')

            # annotator.configure(
            #     test=None,
            #     text_format="star", # text_format="simple"
            #     # test_short_name="pttest", # if former is "simple"
            #     fontsize=10., hide_non_significant=True)

            # annotator.set_pvalues([0.207254325877246, 0.00751591163809919,
            #                        0.0939423736958822, 0.0306539507348838,
            #                        0.00000001279586027571, 0.00000000050751689771])

            # annotator.annotate()

            # Remove frame of legend
            ax.legend(frameon=False, loc=(.5, .8))

            # Task
            # plt.title('Audio Tasks', size=12, x=.5, y=1.1,
            #           fontweight='bold', color='mediumaquamarine')
        else:
            # ... remove legend
            ax.legend([],[], frameon=False)

        # Add values inside bars
        for i in s.containers:
            ax.bar_label(i, padding=-10, fontsize=6, fmt='%.3f')

        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Set limits of ticks in y axis
        plt.ylim([0., .3])

        # Add units in ylabel
        plt.ylabel('Percent Signal Change (%)', labelpad=7)

        # Rotate x_labels
        if n_rois == 6:
            # Rotating X-axis labels
            plt.xticks(rotation=30, fontsize=10)

        # Hemisphere
        # plt.text(.4, .35, hem.capitalize(), size=18,
        #          linespacing=.75, fontweight='bold')

        if hem == 'lh':
            if modality:
                plt.text(-1., .37, modality.capitalize(), size=12,
                         linespacing=.75, fontweight='bold')
            else:
                plt.text(-1., .37, 'Both Mod.', size=12,
                         linespacing=.75, fontweight='bold')
            plt.text(-1., .35, prefix, size=12,
                     linespacing=.75, fontweight='bold')

    # Save figure
    if modality:
        fname = prefix + '_' + str(n_rois) + '-rois_2w_posthoc_' + modality
    else:
        fname = prefix + '_' + str(n_rois) + '-rois_2w_posthoc_both-modalitites'
    plt.savefig(os.path.join(output_folder, fname + '.pdf'))


# ############################# INPUTS ##################################

# Subjects w/ pilot
# SUBJECTS = [3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21,
#             22, 23, 26, 28, 29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 
#             44, 45, 46, 47]

# Subjects without pilot
SUBJECTS = [3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26, 28,
            29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

tasks = {'prod': 'Production', 'percep': 'Perception', 'ntfd': 'NTFD',
         'allmain_tasks': 'All Tasks'}

filtered_contrasts = {10: 'Auditory Beat',
                      11: 'Auditory Interval',
                      14: 'Visual Beat',
                      15: 'Visual Interval'}

working_dir = os.path.dirname(os.path.abspath(__file__))

atlases_dir = os.path.join(working_dir, 'atlases')
fsl_dir = os.path.join(atlases_dir, 'fsl_atlases')
atag_dir = os.path.join(atlases_dir, 'atag_atlas')
ntk_dir = os.path.join(atlases_dir, 'nettekoven_atlas')
hmat_dir = os.path.join(atlases_dir, 'hmat_atlas')

msdtb_dir = os.path.join(working_dir, 'roi_analyses')

# ### Define number of ROIs of the analysis ###
# All ROIs: 7 ROIs
atlas_dirnames7 = [fsl_dir, ntk_dir, ntk_dir, ntk_dir,
                   hmat_dir, hmat_dir, hmat_dir]
atlas_names7 = ['hos', 'ntk_symmni128', 'ntk_symmni128', 'ntk_symmni128',
                'hmat', 'hmat', 'hmat']
region_names7 = ['dorsal_striatum', 'cerebellum', 'cerebellum', 'cerebellum',
                 'motor_area', 'motor_area', 'motor_area']
roi_names7 = ['dstr', 'cereb-s', 'cereb-i', 'cereb',
              'pmd', 'sma', 'presma']

# 6 ROIs
atlas_dirnames6 = [fsl_dir, ntk_dir, ntk_dir,
                   hmat_dir, hmat_dir, hmat_dir]
atlas_names6 = ['hos', 'ntk_symmni128', 'ntk_symmni128',
                'hmat', 'hmat', 'hmat']
region_names6 = ['dorsal_striatum', 'cerebellum', 'cerebellum',
                 'motor_area', 'motor_area', 'motor_area']
roi_names6 = ['dstr', 'cereb-s', 'cereb-i',
              'pmd', 'sma', 'presma']

# 3 ROIs
atlas_dirnames3 = [fsl_dir, ntk_dir, ntk_dir]
atlas_names3 = ['hos', 'ntk_symmni128', 'ntk_symmni128']
region_names3 = ['dorsal_striatum', 'cerebellum', 'cerebellum']
roi_names3 = ['dstr', 'cereb-s', 'cereb-i']

# 2 ROIs
atlas_dirnames2 = [fsl_dir, ntk_dir]
atlas_names2 = ['hos', 'ntk_symmni128']
region_names2 = ['dorsal_striatum', 'cerebellum']
roi_names2 = ['dstr', 'cereb']

tags = ['i', 'a', 'g']

# Tuple: (individual_weight, average_weight, group_weight)
weights_list = [(1., 0.), (.5, .5), (0., 1.)]


# ############################## RUN ####################################

if __name__ == '__main__':

    # ========= SET COMMAND-LINE ARGUMENTS TO BE PASSED TO THE SCRIPT ====
    assert(len(sys.argv) > 1), "No arg was introduced. " + \
                               "You must pass a valid arg to the script."

    n_rois = int(sys.argv[1])

    if n_rois == 7:
        atlas_dirnames = atlas_dirnames7
        atlas_names = atlas_names7
        region_names = region_names7
        roi_names = roi_names7
    elif n_rois == 6:
        atlas_dirnames = atlas_dirnames6
        atlas_names = atlas_names6
        region_names = region_names6
        roi_names = roi_names6
    elif n_rois == 3:
        atlas_dirnames = atlas_dirnames3
        atlas_names = atlas_names3
        region_names = region_names3
        roi_names = roi_names3
    elif n_rois == 2:
        atlas_dirnames = atlas_dirnames2
        atlas_names = atlas_names2
        region_names = region_names2
        roi_names = roi_names2
    else:
        raise ValueError("The number of ROIs must be 7, 6, 3 or 2.")

    # ====================== COMPUTE STATS ===============================
    for tag, wpair in zip(tags, weights_list):
        dfrois = pd.DataFrame()
        for atlas_dirname, atlas_name, region_name, roi_name in zip(
                atlas_dirnames, atlas_names, region_names, roi_names):

            # Define output-dir path
            if region_name == 'dorsal_striatum':
                outdir = os.path.join(msdtb_dir, region_name, atlas_name)
            else:
                outdir = os.path.join(msdtb_dir, region_name, atlas_name,
                                      roi_name)
            # Open ROI file and create paths
            rois_path = os.path.join(
                outdir, 'rois_extraction', tag + '_' + roi_name + '_psc.npy')
            anovas_dir = os.path.join(outdir, 'anovas')
            df_path = os.path.join(
                anovas_dir, tag + '_' + roi_name + '_df.tsv')

            # Create dataframe per ROI
            dfroi = dataframe(rois_path,
                              ['lh', 'rh', 'bh'],
                              list(tasks.values()),
                              list(filtered_contrasts.values()),
                              SUBJECTS,
                              df_path)

            # Add roi column to dataframe
            roi_arr = np.repeat(roi_name, len(dfroi.index))
            dfroi['ROI'] = roi_arr
            # Append dataframe
            dfrois = pd.concat([dfrois, dfroi], ignore_index=True, sort=False)

            # # ############## Run ANOVAs per ROI #########################

            if n_rois == 7:
                # 3-way RM-ANOVA
                three_anova_dir = os.path.join(anovas_dir, '3way-anova')
                threeway_rmanova(df_path, three_anova_dir, tag, roi_name)

                # 2-way RM-ANOVA per task
                twoway_anova_task_dir = os.path.join(
                    anovas_dir, '2way-anova_task')
                twoway_rmanova_task(
                    df_path, tasks, twoway_anova_task_dir, tag, roi_name)

                # 2-way RM-ANOVA collapsed across tasks
                twoway_anova_taskavg_dir = os.path.join(
                    anovas_dir, '2way-anova_grouped-tasks')
                twoway_rmanova_gtasks(
                    df_path, twoway_anova_taskavg_dir, tag, roi_name)

                # 1-way RM-ANOVA for beat/interval
                oneway_anova_task_dir = os.path.join(
                    anovas_dir, '1way-anova')
                oneway_rmanova(
                    df_path, tasks, oneway_anova_task_dir, tag, roi_name)


        # ################# CATROI ANALYSES #############################
        # ##################### 6 ROIs ##################################

        if n_rois == 6:
            # 2-way RM-ANOVA for roi and category for both modalities
            twoway_anova_catroi_dir = os.path.join(
                msdtb_dir, '2way-anova_cat6rois')
            twoway_rmanova_catroi(dfrois, tasks, twoway_anova_catroi_dir, tag)
            posthoc_catroi(dfrois, tasks, twoway_anova_catroi_dir, tag, 6,
                           roi_names)

            # 2-way RM-ANOVA for roi and category for auditory tasks
            twoway_anova_catroi_dir = os.path.join(
                msdtb_dir, '2way-anova_cat6rois_auditory')
            twoway_rmanova_catroi(dfrois, tasks, twoway_anova_catroi_dir, tag,
                                  modality='auditory')
            posthoc_catroi(dfrois, tasks, twoway_anova_catroi_dir, tag, 6,
                           roi_names, modality='auditory')

            # 2-way RM-ANOVA for roi and category for vision tasks
            twoway_anova_catroi_dir = os.path.join(
                msdtb_dir, '2way-anova_cat6rois_visual')
            twoway_rmanova_catroi(dfrois, tasks, twoway_anova_catroi_dir, tag,
                                  modality='visual')
            posthoc_catroi(dfrois, tasks, twoway_anova_catroi_dir, tag, 6,
                           roi_names, modality='visual')

        # ##################### 3 ROIs ##################################

        if n_rois == 3:
            # 2-way RM-ANOVA for roi and category for both modalities
            twoway_anova_catroi_dir = os.path.join(
                msdtb_dir, '2way-anova_cat3rois')
            twoway_rmanova_catroi(dfrois, tasks, twoway_anova_catroi_dir, tag)
            posthoc_catroi(dfrois, tasks, twoway_anova_catroi_dir, tag, 3,
                           roi_names)

            # 2-way RM-ANOVA for roi and category for auditory tasks
            twoway_anova_catroi_dir = os.path.join(
                msdtb_dir, '2way-anova_cat3rois_auditory')
            twoway_rmanova_catroi(dfrois, tasks, twoway_anova_catroi_dir, tag,
                                  modality='auditory')
            posthoc_catroi(dfrois, tasks, twoway_anova_catroi_dir, tag, 3,
                           roi_names, modality='auditory')

            # 2-way RM-ANOVA for roi and category for vision tasks
            twoway_anova_catroi_dir = os.path.join(
                msdtb_dir, '2way-anova_cat3rois_visual')
            twoway_rmanova_catroi(dfrois, tasks, twoway_anova_catroi_dir, tag,
                                  modality='visual')
            posthoc_catroi(dfrois, tasks, twoway_anova_catroi_dir, tag, 3,
                           roi_names, modality='visual')

        # ##################### 2 ROIs ##################################

        if n_rois == 2:
            # 2-way RM-ANOVA for roi and category for both modalities
            twoway_anova_catroi_dir = os.path.join(
                msdtb_dir, '2way-anova_cat2rois')
            twoway_rmanova_catroi(dfrois, tasks, twoway_anova_catroi_dir, tag)
            posthoc_catroi(dfrois, tasks, twoway_anova_catroi_dir, tag, 2,
                           roi_names)

            # 2-way RM-ANOVA for roi and category for auditory tasks
            twoway_anova_catroi_dir = os.path.join(
                msdtb_dir, '2way-anova_cat2rois_auditory')
            twoway_rmanova_catroi(dfrois, tasks, twoway_anova_catroi_dir, tag,
                                  modality='auditory')
            posthoc_catroi(dfrois, tasks, twoway_anova_catroi_dir, tag, 2,
                           roi_names, modality='auditory')

            # 2-way RM-ANOVA for roi and category for vision tasks
            twoway_anova_catroi_dir = os.path.join(
                msdtb_dir, '2way-anova_cat2rois_visual')
            twoway_rmanova_catroi(dfrois, tasks, twoway_anova_catroi_dir, tag,
                                  modality='visual')
            posthoc_catroi(dfrois, tasks, twoway_anova_catroi_dir, tag, 2,
                           roi_names, modality='visual')

            # ###### EXPLICIT/IMPLICIT TIMING ROI ANALYSES ###############
            # ##################### 2 ROIs ###############################

            # 2-way RM-ANOVA for roi and timing type tasks for both modalities
            twoway_anova_timingroi_dir = os.path.join(
                msdtb_dir, '2way-anova_timing2rois')
            twoway_rmanova_timingroi(dfrois, twoway_anova_timingroi_dir, tag)
            posthoc_timingroi(dfrois, twoway_anova_timingroi_dir, tag, 2,
                              roi_names)

            # 2-way RM-ANOVA for roi and timing type tasks for auditory tasks
            twoway_anova_timingroi_dir = os.path.join(
                msdtb_dir, '2way-anova_timing2rois_auditory')
            twoway_rmanova_timingroi(dfrois, twoway_anova_timingroi_dir, tag,
                                     modality='auditory')
            posthoc_timingroi(dfrois, twoway_anova_timingroi_dir, tag, 2,
                              roi_names, modality='auditory')


            # 2-way RM-ANOVA for roi and timing type tasks for visual tasks
            twoway_anova_timingroi_dir = os.path.join(
                msdtb_dir, '2way-anova_timing2rois_visual')
            twoway_rmanova_timingroi(dfrois, twoway_anova_timingroi_dir, tag,
                modality='visual')
            posthoc_timingroi(dfrois, twoway_anova_timingroi_dir, tag, 2,
                roi_names, modality='visual')
