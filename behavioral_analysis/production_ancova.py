"""
ANCOVA analyses of behavioral data of Production Tasks of the Music-SDTB project

author: Ana Luisa Pinho
e-mail: agrilopi@uwo.ca

Created: May 5, 2024
Last update: May 2024

Compatibility: Python 3.10.8
"""

import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

# %%
# ======================== MAIN FUNCTIONS ==============================

def ffx_dvar(df):
    # Fixed Effects within subjects
    df_ffx = df.drop(['Session'], axis=1)
    df_ffx = df_ffx.groupby([
        'Condition', 'Modality', 'Standard', 'Subject']).mean().reset_index()

    return df_ffx


def group_dvar(df_ffx, estimator='mean'):
    # Group effect for plotting
    df_group = df_ffx.drop(['Subject'], axis=1)
    if estimator == 'mean':
        df_group = df_group.groupby([
            'Condition', 'Modality', 'Standard']).mean().reset_index()
    else:
        assert estimator == 'std'
        df_group = df_group.groupby([
            'Condition', 'Modality', 'Standard']).std().reset_index()

    async_ab = df_group[df_group.Modality=='audio'][
        df_group.Condition=='beat'].Asynchronies.values
    async_ai = df_group[df_group.Modality=='audio'][
        df_group.Condition=='interval'].Asynchronies.values
    async_vb = df_group[df_group.Modality=='visual'][
        df_group.Condition=='beat'].Asynchronies.values
    async_vi = df_group[df_group.Modality=='visual'][
        df_group.Condition=='interval'].Asynchronies.values

    return [[async_ab] + [async_ai]] + [[async_vb] + [async_vi]]


def wide_dataframe(df, output_folder, sesstag):
    wdf = pd.pivot(df, values='Asynchronies', index=['Subject', 'Standard'],
                   columns=['Condition', 'Modality'])

    # Create output_folder, if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save dataframe
    wdf_outpath = os.path.join(
        output_folder, 'wide_df_production_' + sesstag + '.tsv')
    wdf.to_csv(wdf_outpath, index=True, sep='\t')


def plot_ancova(x, y, y_values, yaxis_name, yname_pos, title,
                output_folder, fname, legend_loc='lower left',
                hline_legend=None, hline_yloc=[.4275, .435]):
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    # left   # the left side of the subplots of the figure
    # right  # the right side of the subplots of the figure
    # bottom # the bottom of the subplots of the figure
    # top    # the top of the subplots of the figure
    # wspace # the amount of width reserved for blank space between subplots
    # hspace # the amount of height reserved for white space between subplots
    plt.subplots_adjust(left=.095, bottom=.11, right=.98, wspace=.175)

    colors = ['tab:blue', 'tab:orange']
    legend_labels = ['Beat', 'Interval']

    for m, modality_y in enumerate(y):
        for c, condition_y in enumerate(modality_y):
            # Linear fit
            a, b = np.polyfit(x, condition_y, deg=1)
            y_est = a * x + b
            # y_err = x.std() * \
            #     np.sqrt(1/len(x) + (x - x.mean())**2 / np.sum((x - x.mean())**2))

            # Plot the linear fit
            ax[m].plot(x, y_est, '-', color=colors[c], linewidth=12,
                       label=legend_labels[c], alpha=.5)
            # ax[0].fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2)
            ax[m].plot(x, condition_y, 'bo', color=colors[c], markersize=16,
                       alpha=.5)
            # Hide the right and top spines
            ax[m].spines['right'].set_visible(False)
            ax[m].spines['top'].set_visible(False)
            # Set x axis
            x_labels = [str(xl) for xl in x]
            ax[m].set_xticks(x, x_labels, fontsize=24)
            # Set limits of y-axis
            y_labels = [str(yl) for yl in y_values]
            ax[m].set_yticks(y_values, y_labels, fontsize=24)
            # Add horizontal dashed line at y = 0.5
            if hline_legend:
                ax[m].axhline(0., linestyle='--', color='grey', linewidth=12,
                              alpha=.5)

        # Add legend
        if m == 0:
            ax[m].set_title('Auditory Production', weight='bold', pad=0,
                            fontsize=24)
            ax[m].legend(loc=legend_loc, frameon=False, prop={'size': 24})
        else:
            assert m == 1
            ax[m].set_title('Visual Production', weight='bold', pad=0,
                            fontsize=24)

        # Name of x-axis
        fig.text(.465, .018, 'Standards (ms)', fontsize=24)
        # Name of y-axis
        fig.text(.005, yname_pos, yaxis_name, fontsize=26, rotation=90)
        # Legends for horizontal dashed lines
        if hline_legend:
            fig.text(.355, hline_yloc[0], hline_legend, fontsize=24,
                     color='dimgrey')
            fig.text(.825, hline_yloc[1], hline_legend, fontsize=24,
                     color='dimgrey')

    # Title
    plt.suptitle(title, x=.5, y=.98, size=24, linespacing=.75)

    # Create output_folder, if it does not exist
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Save figure
    plt.savefig(os.path.join(output_folder, fname + '.pdf'))

# %%
# ========================== INPUTS ====================================

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FOLDER = os.path.join(MAIN_DIR, 'production_results')
DATAFRAMES_FOLDER = os.path.join(RESULTS_FOLDER, 'dataframes')
JASP_FOLDER = os.path.join(RESULTS_FOLDER, 'ancova', 'jasp')
PLOTS_FOLDER = os.path.join(RESULTS_FOLDER, 'ancova', 'plots')

sessions_dic = {'allses': 'All Sessions',
                'ses-01': 'Session 1',
                'ses-02': 'Session 2',
                'ses-03': 'Session 3',
                'ses-04': 'Session 4',
                'ses-05': 'Session 5'}

# %%
# ============================ RUN =====================================

if __name__ == "__main__":

    for key, value in sessions_dic.items():
        # Open dataframe
        db_path = os.path.join(DATAFRAMES_FOLDER,
                               'df_production_' + key + '.tsv')
        db = pd.read_csv(db_path, sep='\t')

        # Extract covariate
        standards = np.unique(db['Standard'])

        # Extract dependent variable
        db_ffx = ffx_dvar(db)
        mean_async = group_dvar(db_ffx, estimator='mean')
        std_async = group_dvar(db_ffx, estimator='std')

        # Convert dataframe in the wide format for ancova analyses with JASP
        wide_dataframe(db_ffx, JASP_FOLDER, key)

        # Plot ANCOVA
        plot_ancova(
            standards, mean_async, np.around(np.arange(-.1, .2, .05), 2),
            'Mean of Signed Asynchrony', .225,
            'Mean of Signed Asynchrony for every Standard: ' + value,
            PLOTS_FOLDER, 'mean_ancova_production_' + key,
            hline_legend=r'$RT=Standard$', hline_yloc=[.41, .41],
            legend_loc='upper right')

        if key in ['ses-01', 'ses-02', 'ses-03']:
            ylimits = np.around(np.arange(.06, .19, .02), 3)
        elif key == 'ses-04':
            ylimits = np.around(np.arange(.06, .26, .02), 3)
        elif key == 'ses-05':
            ylimits = np.around(np.arange(.06, .40, .04), 3)
        else:
            ylimits = np.around(np.arange(.06, .14, .02), 3)
        plot_ancova(
            standards, std_async, ylimits,
            'SD of Signed Asynchrony', .225,
            'Standard Deviation (SD) of Signed Asynchrony ' + \
            'for every Standard: ' + value, PLOTS_FOLDER,
            'std_ancova_production_' + key, legend_loc='upper right')
