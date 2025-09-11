"""
This script performs several ANOVA analysis using ROIS extracted from 
contrasts of the Music-SDTB Project.

Author: Ana Luisa Pinho

Created: October 2024
Last update: September 2025

Compatibility: Python 3.10.14

How to run the script:
python roi_anova_msdtb.py <n_rois> <encoding_type>
Example:
python roi_anova_msdtb.py 2 all

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
                   hems=['lh', 'rh', 'bh'], modalities=['Auditory', 'Visual']):
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
        for modality in modalities:
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
                padjust='holm', effsize='cohen')

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
    Posthoc 2w-ANOVA plotting with:
      • Shared global y-scale across PDFs,
      • Wider canvas for many ROIs,
      • Diagonal ROI tick labels (many ROIs),
      • Raised x-axis label + extra bottom margin,
      • Thicker bars when single series; overlap-safe width when multiple,
      • Centered LH/RH/BH headers,
      • Tight bbox saving to avoid clipping.
    """
    # --- Load / standardize ---
    if isinstance(df, str):
        df_full = pd.read_csv(df, sep='\t')
    else:
        df_full = df.copy()
    if 'Contrast' in df_full.columns:
        df_full = df_full.drop(['Contrast'], axis=1)
    df_full['PSC'] = pd.to_numeric(df_full['PSC'], errors='coerce')

    # --- Global y-limits from mean ± 1.96*SE ---
    def _ci_extents(frame, by_cols):
        if frame.empty:
            return np.nan, np.nan
        g = (frame.groupby(by_cols)['PSC']
                  .agg(['mean', 'std', 'count'])
                  .reset_index())
        g['se'] = g['std'] / np.sqrt(g['count'].clip(lower=1))
        upper = g['mean'] + 1.96 * g['se']
        lower = g['mean'] - 1.96 * g['se']
        return np.nanmin(lower.values), np.nanmax(upper.values)

    base_group = ['ROI', 'Category', 'Task', 'Hemisphere']
    ymin_list, ymax_list = [], []
    if 'Modality' in df_full.columns:
        ylo1, yhi1 = _ci_extents(df_full, base_group + ['Modality'])
        ymin_list.append(ylo1); ymax_list.append(yhi1)
        df_collapsed = (df_full.drop(columns=['Modality'])
                        .groupby(base_group + ['Subject'], as_index=False)
                        .mean())
        ylo2, yhi2 = _ci_extents(df_collapsed, base_group)
        ymin_list.append(ylo2); ymax_list.append(yhi2)
    else:
        ylo, yhi = _ci_extents(df_full, base_group)
        ymin_list.append(ylo); ymax_list.append(yhi)

    ymin = float(np.nanmin(ymin_list)) if len(ymin_list) else 0.0
    ymax = float(np.nanmax(ymax_list)) if len(ymax_list) else 0.0
    rng = max(ymax - ymin, 1e-6)
    pad = 0.14 * rng
    global_bottom = min(0.0, ymin - pad)
    global_top = ymax + pad

    # --- Modality handling (match plotting behavior) ---
    df = df_full.copy()
    if modality is None:
        if 'Modality' in df.columns:
            df = df.drop(['Modality'], axis=1)
            df = (df.groupby(['Category', 'Task', 'Subject',
                              'ROI', 'Hemisphere'], as_index=False)
                    .mean())
    elif modality == 'auditory':
        df = df[df.Modality == 'Auditory'].drop(['Modality'], axis=1)
    else:
        assert modality == 'visual'
        df = df[df.Modality == 'Visual'].drop(['Modality'], axis=1)

    # --- ROI naming + order sync ---
    df['ROI'] = df['ROI'].str.replace('dstr', 'Dorsal Striatum')
    df['ROI'] = df['ROI'].str.replace('cereb', 'Cerebellum')
    order_list = [s.replace('dstr', 'Dorsal Striatum') for s in order_list]
    order_list = [s.replace('cereb', 'Cerebellum') for s in order_list]

    # Tasks
    ttags = list(tasks_dic.keys())
    tasks_list = list(tasks_dic.values())

    # --- Wider figure for many ROIs + more bottom margin for xlabel ---
    if n_rois <= 6:
        fig_w = 12
    elif n_rois <= 8:
        fig_w = 22
    else:
        fig_w = 24
    fig = plt.figure(figsize=(fig_w, 12))
    fig.subplots_adjust(bottom=0.16)

    # Figure-level labels (inside canvas)
    top_label = modality.capitalize() if modality else 'Both Mod.'
    fig.text(0.01, 0.985, top_label, ha='left', va='top',
             fontsize=12, fontweight='bold')
    fig.text(0.01, 0.958, prefix, ha='left', va='top',
             fontsize=12, fontweight='bold')

    # Track top row axes for centered column headers
    top_row_axes = []

    # --- Plot grid ---
    for h, hem in enumerate(hems):
        for t, (ttag, task) in enumerate(zip(ttags, tasks_list)):
            ax = plt.axes([.07 + h*.3, .7825 - t*.2425, .23, .15])
            if t == 0:
                top_row_axes.append((hem, ax))

            db = df[(df.Task == task) & (df.Hemisphere == hem)].copy()

            s = sns.barplot(
                ax=ax,
                x='ROI', y='PSC', hue='Category', data=db,
                estimator=np.mean, ci=95,
                errcolor="darkgray", errwidth=1.5,
                capsize=0.2, alpha=0.5,
                order=order_list
            )

            # --- Overlap-safe bar width adjustment ---
            nhue = db['Category'].nunique()
            if nhue <= 1:
                scale = 1.25 if n_rois >= 8 else 1.10
            else:
                scale = 0.96  # slightly slimmer to keep a gap between hues
            for p in s.patches:
                w = p.get_width()
                new_w = w * scale
                dx = (new_w - w) / 2.0
                p.set_x(p.get_x() - dx)
                p.set_width(new_w)

            # Value labels (slightly smaller for many ROIs)
            lbl_fs = 6 if n_rois <= 6 else 5
            lbl_pad = -8 if n_rois >= 8 else -10
            for container in s.containers:
                ax.bar_label(container, padding=lbl_pad,
                             fontsize=lbl_fs, fmt='%.3f', clip_on=False)

            # Shared y-scale
            ax.set_ylim(global_bottom, global_top)

            # Cosmetics
            ax.legend([], [], frameon=False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_ylabel('Percent Signal Change (%)', labelpad=7)
            ax.margins(x=0.05)

            # Diagonal ROI tick labels when many ROIs
            if n_rois >= 8:
                ax.set_xticklabels(ax.get_xticklabels(),
                                   rotation=45, ha='right', fontsize=9)
            elif n_rois == 6:
                ax.set_xticklabels(ax.get_xticklabels(),
                                   rotation=30, fontsize=10)

            # Raise x-axis label slightly; bottom margin already increased
            ax.set_xlabel('ROI', labelpad=8)

    # --- Centered hemisphere headers above each column ---
    for hem, ax0 in top_row_axes:
        pos = ax0.get_position()
        x_center = 0.5 * (pos.x0 + pos.x1)
        fig.text(x_center, 0.97, hem.upper(),
                 ha='center', va='top', fontsize=14, fontweight='bold')

    # --- Save ---
    fname = (f"{prefix}_{n_rois}-rois_2w_posthoc_{modality}"
             if modality else f"{prefix}_{n_rois}-rois_2w_posthoc_both-modalities")
    plt.savefig(os.path.join(output_folder, fname + '.pdf'),
                bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def posthoc_timingroi(df, output_folder, prefix, n_rois, order_list,
                      modality=None, hems=['lh', 'rh', 'bh']):
    """
    Posthoc (timing x ROI) barplots per hemisphere.

    Fixes:
      • CI-aware global y-limits with headroom so error bars never clip.
      • Extra left margin so 'PSC (%)' ylabel doesn't touch/cut at page edge
        (especially for n_rois=2).
      • Mean values shown INSIDE bars (centered), not on top.
      • Wider canvas when many ROIs; diagonal ROI tick labels when needed.
      • Tight bbox save.
    """
    # --- Load / coerce ---
    if isinstance(df, str):
        df = pd.read_csv(df, sep='\t')
    if 'Contrast' in df.columns:
        df = df.drop(columns=['Contrast'])
    df['PSC'] = pd.to_numeric(df['PSC'], errors='coerce')

    # --- Modality handling ---
    if modality is None and 'Modality' in df.columns:
        df_plot = (
            df.drop(columns=['Modality'])
              .groupby(['Category', 'Task', 'Subject', 'ROI', 'Hemisphere'],
                       as_index=False)
              .mean()
        )
    elif modality == 'auditory':
        df_plot = df.loc[df.Modality == 'Auditory'].drop(columns=['Modality'])
    elif modality == 'visual':
        df_plot = df.loc[df.Modality == 'Visual'].drop(columns=['Modality'])
    else:
        df_plot = df.copy()

    # Remove synthetic line if present
    df_plot = df_plot.loc[df_plot.Task != 'All Tasks'].copy()

    # --- ROI name normalization + order sync ---
    rep = {
        'dstr': 'Dorsal Striatum', 'cereb': 'Cerebellum',
        'pmd': 'PMD', 'pmv': 'PMV', 'presma': 'PreSMA',
        'sma': 'SMA', 'heschl': 'Heschl', 'occipital': 'Occipital'
    }
    for k, v in rep.items():
        df_plot['ROI'] = df_plot['ROI'].str.replace(k, v)
    order_list = [rep.get(s, s) for s in order_list]

    # --- Global y-limits from mean ± 1.96*SE across ALL plotted data ---
    def _ci_extents(frame, by_cols):
        if frame.empty:
            return 0.0, 0.0
        g = (frame.groupby(by_cols)['PSC']
                  .agg(['mean', 'std', 'count'])
                  .reset_index())
        g['se'] = g['std'] / np.sqrt(g['count'].clip(lower=1))
        upper = g['mean'] + 1.96 * g['se']
        lower = g['mean'] - 1.96 * g['se']
        return float(np.nanmin(lower.values)), float(np.nanmax(upper.values))

    ylo, yhi = _ci_extents(df_plot, ['ROI', 'Task', 'Hemisphere'])
    rng = max(yhi - ylo, 1e-6)
    pad = 0.14 * rng
    y_min = min(0.0, ylo - pad)      # include 0 if all positive
    y_max = yhi + pad

    # --- Figure sizing & margins ---
    # Wider canvas for many ROIs; more left margin when n_rois is 
    # small so ylabel isn't cut
    if n_rois <= 2:
        fig_w = 10
        left_margin = 0.12
    elif n_rois <= 6:
        fig_w = 12
        left_margin = 0.10
    elif n_rois <= 8:
        fig_w = 20
        left_margin = 0.09
    else:
        fig_w = 24
        left_margin = 0.09
    n_hems = len(hems)
    fig_h = 4.5 * n_hems

    fig, axes = plt.subplots(n_hems, 1, figsize=(fig_w, fig_h), sharey=True)
    if n_hems == 1:
        axes = [axes]

    # Titles / header
    top_label = modality.capitalize() if modality else 'Both Mod.'
    fig.text(0.01, 0.985, top_label, ha='left', va='top',
             fontsize=12, fontweight='bold')
    fig.text(0.01, 0.958, prefix, ha='left', va='top',
             fontsize=12, fontweight='bold')

    hue_order = ['Production', 'Perception', 'NTFD']
    hue_order = [t for t in hue_order if t in df_plot['Task'].unique()]

    # --- Draw per-hemisphere rows ---
    for i, hem in enumerate(hems):
        ax = axes[i]
        db = df_plot.loc[df_plot.Hemisphere == hem].copy()

        s = sns.barplot(
            ax=ax, x='ROI', y='PSC', hue='Task', data=db,
            estimator=np.mean, ci=95,
            errcolor="darkgray", errwidth=1.5, capsize=0.2, alpha=0.6,
            order=order_list, hue_order=hue_order,
            palette=['indigo', 'm', 'salmon'][:len(hue_order)]
        )

        # Inside-bar mean labels (centered)
        # Use smaller text if many ROIs to avoid crowding
        lbl_fs = 7 if n_rois <= 6 else 6
        for container in s.containers:
            ax.bar_label(container, fmt='%.3f', label_type='center',
                         fontsize=lbl_fs, color='black', clip_on=False)

        # Global y-limits (include error-bar headroom)
        ax.set_ylim(y_min, y_max)

        # Cosmetics
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel('PSC (%)', fontsize=12, labelpad=12)
        ax.set_title(f"Hemisphere: {hem.upper()}",
             fontsize=13, fontweight='bold',
             pad=(16 if n_rois >= 8 else 12))

        # ROI tick labels: rotate when many ROIs
        if n_rois >= 8:
            ax.set_xticklabels(ax.get_xticklabels(),
                               rotation=45, ha='right', fontsize=10)
        elif n_rois == 6:
            ax.set_xticklabels(ax.get_xticklabels(),
                               rotation=30, ha='right', fontsize=11)
        else:
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=11)

        # Lower x-axis label a bit (but margins keep it off the page edge)
        ax.set_xlabel('ROI', fontsize=12, labelpad=(4 if n_rois >= 8 else 10))

        # De-clutter legend: keep only on the top subplot
        if i == 0:
            ax.legend(frameon=False, loc='upper right', title=None)
        else:
            ax.legend([], [], frameon=False)

    # Layout: extra bottom margin for xlabels, extra left for ylabel
    hspace = 0.9 if n_rois >= 8 else 0.6
    plt.subplots_adjust(hspace=hspace, bottom=0.14, top=0.92, left=left_margin)

    # Save (tight bbox so caps/labels aren’t cropped)
    modality_suffix = modality if modality else 'both-modalities'
    fname = f"{prefix}_{n_rois}-rois_2w_posthoc_{modality_suffix}"
    plt.savefig(os.path.join(output_folder, fname + '.pdf'),
                bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def threeway_rmanova_timing(df, output_dir, prefix, hems=['lh','rh','bh']):
    """
    3-way RM-ANOVA (ROI × Task × Modality) via statsmodels.AnovaRM,
    then Holm-corrected paired t-tests:
     • mains: ROI, Task, Modality
     • ROI×Modality: only Aud vs Vis within each ROI
     • ROI×Task:     only each Task-pair within each ROI
     • Modality×Task: only each Task-pair within each Modality
     • 3-way:        only each Task-pair within each (ROI,Modality) cell
    All posthocs in one TSV per hemisphere, with the same columns
    as your 2-way posthoc files.
    """
    if isinstance(df, str):
        df = pd.read_csv(df, sep='\t')

    # drop “All Tasks” and coerce
    df = df.loc[df.Task!='All Tasks'].copy()
    df['PSC'] = pd.to_numeric(df['PSC'])

    for hem in hems:
        sub = df.loc[df.Hemisphere==hem]
        agg = (
            sub
            .groupby(['Subject','ROI','Modality','Task'], as_index=False)
            ['PSC']
            .mean()
        )

        # 1) omnibus 3-way ANOVA
        model = AnovaRM(agg, depvar='PSC', subject='Subject',
                        within=['ROI','Modality','Task'])
        res3  = model.fit()

        os.makedirs(output_dir, exist_ok=True)
        base = f"{prefix}_{hem}_3way"

        # save ANOVA
        res3.anova_table.to_csv(
            os.path.join(output_dir, base + '_anova.tsv'),
            sep='\t'
        )

        # 2) post-hocs
        rows = []

        # — mains —
        for factor in ['ROI','Modality','Task']:
            mc = MultiComparison(agg['PSC'], agg[factor])
            ph = mc.allpairtest(ttest_rel, method='Holm')[0]
            df_ph = pd.DataFrame(ph)
            df_ph.insert(0, 'Contrast', factor)
            rows.append(df_ph)

        # — ROI × Modality (Aud vs Vis within each ROI) —
        for roi in agg['ROI'].unique():
            sub_roi = agg.loc[agg.ROI==roi]
            mc = MultiComparison(sub_roi['PSC'], sub_roi['Modality'])
            ph = mc.allpairtest(ttest_rel, method='Holm')[0]
            df_ph = pd.DataFrame(ph)
            df_ph.insert(0, 'Contrast', 'ROI:Modality')
            df_ph.insert(1, 'ROI', roi)
            rows.append(df_ph)

        # — ROI × Task (all 3 Task pairs within each ROI) —
        for roi in agg['ROI'].unique():
            sub_roi = agg.loc[agg.ROI==roi]
            mc = MultiComparison(sub_roi['PSC'], sub_roi['Task'])
            ph = mc.allpairtest(ttest_rel, method='Holm')[0]
            df_ph = pd.DataFrame(ph)
            df_ph.insert(0, 'Contrast', 'ROI:Task')
            df_ph.insert(1, 'ROI', roi)
            rows.append(df_ph)

        # — Modality × Task (all 3 Task pairs within each Modality) —
        for mod in agg['Modality'].unique():
            sub_mod = agg.loc[agg.Modality==mod]
            mc = MultiComparison(sub_mod['PSC'], sub_mod['Task'])
            ph = mc.allpairtest(ttest_rel, method='Holm')[0]
            df_ph = pd.DataFrame(ph)
            df_ph.insert(0, 'Contrast', 'Modality:Task')
            df_ph.insert(1, 'Modality', mod)
            rows.append(df_ph)

        # — 3-way ROI × Modality × Task —
        #    (only Task-pairs within each (ROI,Modality) cell)
        for roi in agg['ROI'].unique():
            for mod in agg['Modality'].unique():
                sub_cell = agg[(agg.ROI==roi)&(agg.Modality==mod)]
                mc = MultiComparison(sub_cell['PSC'], sub_cell['Task'])
                ph = mc.allpairtest(ttest_rel, method='Holm')[0]
                df_ph = pd.DataFrame(ph)
                df_ph.insert(0, 'Contrast', 'ROI:Modality:Task')
                df_ph.insert(1, 'ROI', roi)
                df_ph.insert(2, 'Modality', mod)
                rows.append(df_ph)

        # concat & save
        posthoc_all = pd.concat(rows, ignore_index=True, sort=False)

        # **Reorder columns:** Contrast, ROI, Modality, then the rest
        cols = list(posthoc_all.columns)
        front = ['Contrast', 'ROI', 'Modality']
        rest  = [c for c in cols if c not in front]
        posthoc_all = posthoc_all[ front + rest ]

        posthoc_all.to_csv(
            os.path.join(output_dir, base + '_posthoc.tsv'),
            sep='\t', index=False
        )


# ############################# INPUTS ##################################

# Subjects w/ pilot
# SUBJECTS = [3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21,
#             22, 23, 26, 28, 29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 
#             44, 45, 46, 47]

# Subjects without pilot
SUBJECTS = [3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26, 28,
            29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

# #############

# tasks = {
#     'prod': 'Production', 
#     'percep': 'Perception', 
#     'ntfd': 'NTFD',
#     'allmain_tasks': 'All Tasks'
# }
# selected_contrasts = {
#     10: 'Auditory Beat',
#     11: 'Auditory Interval',
#     14: 'Visual Beat',
#     15: 'Visual Interval'
# }
# task_roidef_id = 'allmain_tasks'
# folder_name = 'main_tasks's

tasks = {
    'rand_ntfd': 'NTFD Random'
}
selected_contrasts = {
    18: 'Auditory Beat',
    19: 'Auditory Interval',
    21: 'Auditory Random',
    30: 'Visual Beat',
    31: 'Visual Interval',
    33: 'Visual Random'
}
task_roidef_id = 'allmain_tasks'
folder_name = 'main_tasks'

# #############

tags = ['i', 'i9a', 'i8a', 'i7a', 'i6a', 'a', 'a4g', 'a3g', 'a2g', 'a1g', 'g']

# #############

# Tuple: (individual_weight, average_weight, group_weight)
weights_list = [(1., 0.), (.9, .1), (.8, .2), (.7, .3), (.6, .4), (.5, .5),
                (.4, .6), (.3, .7), (.2, .8), (.1, .9), (0., 1.)]

# ========================= PARAMETERS =================================

working_dir = os.path.dirname(os.path.abspath(__file__))
atlases_dir = os.path.join(working_dir, 'atlases')
fsl_dir = os.path.join(atlases_dir, 'fsl_atlases')
atag_dir = os.path.join(atlases_dir, 'atag_atlas')
ntk_dir = os.path.join(atlases_dir, 'nettekoven_atlas')
hmat_dir = os.path.join(atlases_dir, 'hmat_atlas')

model = 'rwls' # 'rwls'; or 'standard' (no rwls)
masking = 'wb' # 'wb' for whole-brain; 'gm' for grey matter

hrf_cutoff = 'hrf128' # 'hrf128' or 'hrf42'
# hrf_cutoff = 'hrf128_timederiv'
# hrf_cutoff = 'hrf128_timedispderiv'

roi_dir = os.path.join(
    working_dir,
    'roi_analyses_' 
    + model + '_' 
    + hrf_cutoff + '_' 
    + masking + 
    '_puncorr_unsmoothed')

# ### Define number of ROIs of the analysis ###
# All ROIs: 10 ROIs
atlas_dirnames10 = [fsl_dir, 
                    ntk_dir, ntk_dir, ntk_dir,
                    hmat_dir, hmat_dir, hmat_dir, hmat_dir,
                    fsl_dir, 
                    fsl_dir]
atlas_names10 = ['hos', 
                 'ntk_symmni128', 'ntk_symmni128', 'ntk_symmni128',
                 'hmat', 'hmat', 'hmat', 'hmat',
                 'hos', 
                 'hos']
region_names10 = ['dorsal_striatum', 
                  'cerebellum', 'cerebellum', 'cerebellum',
                  'motor_area', 'motor_area', 'motor_area', 'motor_area', 
                  'heschl_gyrus', 
                  'occipital_lobe']
roi_names10 = ['dstr', 
               'cereb-s', 'cereb-i', 'cereb',
               'pmd', 'pmv', 'sma', 'presma',
               'heschl',
               'occipital']

# 8 ROIs
atlas_dirnames8 = [fsl_dir, 
                   ntk_dir,
                   hmat_dir, hmat_dir, hmat_dir, hmat_dir,
                   fsl_dir,
                   fsl_dir]
atlas_names8 = ['hos', 
                'ntk_symmni128', 
                'hmat', 'hmat', 'hmat', 'hmat',
                'hos',
                'hos']
region_names8 = ['dorsal_striatum', 
                 'cerebellum', 
                 'motor_area', 'motor_area', 'motor_area', 'motor_area',
                 'heschl_gyrus',
                 'occipital_lobe']
roi_names8 = ['dstr',
              'cereb',
              'pmd', 'pmv', 'sma', 'presma',
              'heschl',
              'occipital']

# 2 ROIs
atlas_dirnames2 = [fsl_dir, ntk_dir]
atlas_names2 = ['hos', 'ntk_symmni128']
region_names2 = ['dorsal_striatum', 'cerebellum']
roi_names2 = ['dstr', 'cereb']


# ############################## RUN ####################################

if __name__ == '__main__':

    # ========= SET COMMAND-LINE ARGUMENTS TO BE PASSED TO THE SCRIPT ====
    assert(len(sys.argv) > 2), "No arg was introduced. " + \
                               "You must pass two valid args to the script."

    n_rois = int(sys.argv[1])

    if n_rois == 10:
        atlas_dirnames = atlas_dirnames10
        atlas_names = atlas_names10
        region_names = region_names10
        roi_names = roi_names10
    elif n_rois == 8:
        atlas_dirnames = atlas_dirnames8
        atlas_names = atlas_names8
        region_names = region_names8
        roi_names = roi_names8
    elif n_rois == 2:
        atlas_dirnames = atlas_dirnames2
        atlas_names = atlas_names2
        region_names = region_names2
        roi_names = roi_names2
    else:
        raise ValueError("The number of ROIs must be 10, 8 or 2.")

    encoding_type = sys.argv[2]
    msdtb_dir = os.path.join(roi_dir, encoding_type + '_' + task_roidef_id, 
                             folder_name)
    keys = list(selected_contrasts.keys())
    if encoding_type == 'bothmod':
        filtered_contrasts = selected_contrasts
    elif encoding_type == 'auditory':
        auditory_keys = keys[:len(keys)//2]
        filtered_contrasts = {
            key: selected_contrasts[key]
            for key in auditory_keys if key in selected_contrasts}
    elif encoding_type == 'visual':
        visual_keys = keys[len(keys)//2:]
        filtered_contrasts = {
            key: selected_contrasts[key]
            for key in visual_keys if key in selected_contrasts}
    else:
        raise ValueError(
            "The argument must be 'bothmod', 'auditory' or 'visual'.")

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

            if n_rois == 10:
                if encoding_type == 'all':
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
                    
                elif encoding_type == 'auditory':
                    # 1-way RM-ANOVA for beat/interval
                    oneway_anova_task_dir = os.path.join(
                        anovas_dir, '1way-anova')
                    oneway_rmanova(
                        df_path, tasks, oneway_anova_task_dir, tag, roi_name,
                        modalities=['Auditory'])

                else:
                    assert encoding_type == 'visual'
                    # 1-way RM-ANOVA for beat/interval
                    oneway_anova_task_dir = os.path.join(
                        anovas_dir, '1way-anova')
                    oneway_rmanova(
                        df_path, tasks, oneway_anova_task_dir, tag, roi_name,
                        modalities=['Visual'])
                    
        # Save dataframe with all ROIs
        dfrois.to_csv(
            os.path.join(
                msdtb_dir, 'dfrois_' + tag + '_' + str(n_rois) + '-rois.tsv'),
            sep='\t', index=False)
                
        # ##################### 8 ROIs ################################
        if n_rois == 8:
            if encoding_type == 'all':

                # ################# CATROI ANALYSES ###################
                # 2-way RM-ANOVA for roi and category for both modalities
                twoway_anova_catroi_dir = os.path.join(
                    msdtb_dir, '2way-anova_cat8rois')
                twoway_rmanova_catroi(dfrois, tasks, twoway_anova_catroi_dir,
                                      tag)
                posthoc_catroi(dfrois, tasks, twoway_anova_catroi_dir, tag, 8,
                               roi_names)
                
                # ###### EXPLICIT/IMPLICIT TIMING ROI ANALYSES ########
                # 2-way RM-ANOVA for roi and timing type tasks for ...
                # ...both modalities
                twoway_anova_timingroi_dir = os.path.join(
                    msdtb_dir, '2way-anova_timing8rois')
                twoway_rmanova_timingroi(
                    dfrois, twoway_anova_timingroi_dir, tag)
                posthoc_timingroi(
                    dfrois, twoway_anova_timingroi_dir, tag, 8, roi_names)

                # ######## 3-WAY ROI × TASK × MODALITY ANOVA ########
                threeway_anova_roi_task_modality_dir = os.path.join(
                    msdtb_dir, '3way-anova_timing8rois')

                threeway_rmanova_timing(
                    dfrois, threeway_anova_roi_task_modality_dir, tag)


            if encoding_type in ['all', 'auditory']:

                # ################# CATROI ANALYSES ###################
                # 2-way RM-ANOVA for roi and category for auditory tasks
                twoway_anova_catroi_dir = os.path.join(
                    msdtb_dir, '2way-anova_cat8rois_auditory')
                twoway_rmanova_catroi(dfrois, tasks, twoway_anova_catroi_dir,
                                      tag, modality='auditory')
                posthoc_catroi(dfrois, tasks, twoway_anova_catroi_dir, tag, 8,
                               roi_names, modality='auditory')
                
                # ###### EXPLICIT/IMPLICIT TIMING ROI ANALYSES ########
                # 2-way RM-ANOVA for roi and timing type tasks for...
                # ... auditory tasks
                twoway_anova_timingroi_dir = os.path.join(
                    msdtb_dir, '2way-anova_timing8rois_auditory')
                twoway_rmanova_timingroi(
                    dfrois, twoway_anova_timingroi_dir, tag, 
                    modality='auditory')
                posthoc_timingroi(
                    dfrois, twoway_anova_timingroi_dir, tag, 8, roi_names, 
                    modality='auditory')

            if encoding_type in ['all', 'visual']:

                # ################# CATROI ANALYSES ###################
                # 2-way RM-ANOVA for roi and category for vision tasks
                twoway_anova_catroi_dir = os.path.join(
                    msdtb_dir, '2way-anova_cat8rois_visual')
                twoway_rmanova_catroi(dfrois, tasks, twoway_anova_catroi_dir,
                                      tag, modality='visual')
                posthoc_catroi(dfrois, tasks, twoway_anova_catroi_dir, tag, 8,
                               roi_names, modality='visual')
                
                # ###### EXPLICIT/IMPLICIT TIMING ROI ANALYSES ########
                # 2-way RM-ANOVA for roi and timing type tasks for 
                # visual tasks
                twoway_anova_timingroi_dir = os.path.join(
                    msdtb_dir, '2way-anova_timing8rois_visual')
                twoway_rmanova_timingroi(
                    dfrois, twoway_anova_timingroi_dir, tag, 
                    modality='visual')
                posthoc_timingroi(
                    dfrois, twoway_anova_timingroi_dir, tag, 8, roi_names, 
                    modality='visual')

        # ##################### 2 ROIs ##################################

        if n_rois == 2:
            if encoding_type == 'all':

                # ################# CATROI ANALYSES ###################
                # 2-way RM-ANOVA for roi and category for both modalities
                twoway_anova_catroi_dir = os.path.join(
                    msdtb_dir, '2way-anova_cat2rois')
                twoway_rmanova_catroi(dfrois, tasks, twoway_anova_catroi_dir,
                                      tag)
                posthoc_catroi(dfrois, tasks, twoway_anova_catroi_dir, tag, 2,
                               roi_names)
                
                # ###### EXPLICIT/IMPLICIT TIMING ROI ANALYSES ########
                # 2-way RM-ANOVA for roi and timing type tasks for ...
                # ...both modalities
                twoway_anova_timingroi_dir = os.path.join(
                    msdtb_dir, '2way-anova_timing2rois')
                twoway_rmanova_timingroi(dfrois, twoway_anova_timingroi_dir,
                                         tag)
                posthoc_timingroi(
                    dfrois, twoway_anova_timingroi_dir, tag, 2, roi_names)

            if encoding_type in ['all', 'auditory']:

                # ################# CATROI ANALYSES ###################
                # 2-way RM-ANOVA for roi and category for auditory tasks
                twoway_anova_catroi_dir = os.path.join(
                    msdtb_dir, '2way-anova_cat2rois_auditory')
                twoway_rmanova_catroi(dfrois, tasks, twoway_anova_catroi_dir,
                                      tag, modality='auditory')
                posthoc_catroi(dfrois, tasks, twoway_anova_catroi_dir, tag, 2,
                               roi_names, modality='auditory')
                
                # ###### EXPLICIT/IMPLICIT TIMING ROI ANALYSES ########
                # 2-way RM-ANOVA for roi and timing type tasks for...
                # ... auditory tasks
                twoway_anova_timingroi_dir = os.path.join(
                    msdtb_dir, '2way-anova_timing2rois_auditory')
                twoway_rmanova_timingroi(
                    dfrois, twoway_anova_timingroi_dir, tag,
                    modality='auditory')
                posthoc_timingroi(
                    dfrois, twoway_anova_timingroi_dir, tag, 2, roi_names, 
                    modality='auditory')

            if encoding_type in ['all', 'visual']:

                # ################# CATROI ANALYSES ###################
                # 2-way RM-ANOVA for roi and category for vision tasks
                twoway_anova_catroi_dir = os.path.join(
                    msdtb_dir, '2way-anova_cat2rois_visual')
                twoway_rmanova_catroi(dfrois, tasks, twoway_anova_catroi_dir,
                                      tag, modality='visual')
                posthoc_catroi(dfrois, tasks, twoway_anova_catroi_dir, tag, 2,
                               roi_names, modality='visual')
                
                # ###### EXPLICIT/IMPLICIT TIMING ROI ANALYSES ########
                # 2-way RM-ANOVA for roi and timing type tasks for visual
                # tasks
                twoway_anova_timingroi_dir = os.path.join(
                    msdtb_dir, '2way-anova_timing2rois_visual')
                twoway_rmanova_timingroi(
                    dfrois, twoway_anova_timingroi_dir, tag,
                    modality='visual')
                posthoc_timingroi(
                    dfrois, twoway_anova_timingroi_dir, tag, 2, roi_names, 
                    modality='visual')
