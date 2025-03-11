"""
Script to plot brain activity for the contrasts Beat vs Rest and
 Interval vs. Rest across subjects for both Auditory and Visual.

Each circle represents the contribution of one participant.
 Error bars represent 95% confidence intervals.

Author: Ana Luisa Pinho
Email: agrilopi@uwo.ca

Creation: 10th of March 2025
Last Update: March 2025

Compatibility: Python 3.10.14
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ========================== FUNCTIONS =================================

def create_df(data, con_mask, subjects, output_dir='./', fname='dataframe'):

    if con_mask == 'Auditory Encoding':
        cols = ['auditory_beat', 'auditory_interval']
    elif con_mask == 'Visual Encoding':
        cols = ['visual_beat', 'visual_interval']
    else:
        assert con_mask == 'Encoding'
        cols = ['auditory_beat', 'auditory_interval',
                'visual_beat', 'visual_interval']

    # Create DataFrame
    df = pd.DataFrame(data, columns=cols)
    # Insert Subject column
    df.insert(0, 'subject', subjects)
    # Save as TSV file
    df.to_csv(os.path.join(output_dir, fname + '.tsv'), sep='\t', index=False) 


def plot_boxplots(data, y_label='Percent Signal Change (%)',
                  output_dir='./', fname='boxplot',
                  subplot_titles=None):
    """
    Parameters
    ----------
    data : numpy.ndarray
        A (31,2) or (31,4) array representing data for two or four conditions.
    y_label : str, optional
        Label for the y-axis (default is 'Percent Signal Change (%)').
    output_dir : str, optional
        Directory where the plot will be saved (default is './').
    fname : str, optional
        Name of the output file without extension (default is 'boxplot').
    subplot_titles : list of str, optional
        Custom titles for the subplots. Should match the number of subplots.
        Default is ['Auditory Conditions', 'Visual Conditions'].

    Returns
    -------
    None
        Displays the plot and saves it as a PDF.
    """

    # Determine the number of subplots
    num_conditions = data.shape[1]
    num_subplots = 1 if num_conditions == 2 else 2

    # Default subplot titles if not provided
    if subplot_titles is None:
        subplot_titles = [
            'Auditory Conditions', 'Visual Conditions'][:num_subplots]

    # Set up figure with one or two subplots
    fig, ax = plt.subplots(1, num_subplots, figsize=(4 * num_subplots, 6),
                           sharey=True)

    # Adjust subplot spacing
    fig.subplots_adjust(left=0.275, right=0.95, bottom=0.15,
                        top=0.85, wspace=0.1)

    if num_subplots == 1:
        ax = [ax]  # Convert to list for consistency

    # Condition labels and their original colors
    condition_labels = ['Beat', 'Interval']
    condition_colors = ['tab:blue', 'tab:orange']

    for i, ax_i in enumerate(ax):
        # Select appropriate data columns
        conditions = data[:, i * 2: (i + 1) * 2]

        # Compute and print means
        means = np.mean(conditions, axis=0)
        print(f"Means for subplot '{subplot_titles[i]}':")
        for cond, mean in zip(condition_labels, means):
            print(f"{cond}: {mean:.4f}")

        # Prepare data for Seaborn
        datum = {
            'Conditions': np.repeat(condition_labels, data.shape[0]),
            y_label: np.concatenate((conditions[:, 0], conditions[:, 1]))
        }
        df = pd.DataFrame(data=datum)

        # Create boxplot with reduced width and no dodge
        sns.boxplot(
            ax=ax_i,
            x='Conditions',
            y=y_label,
            data=df,
            width=0.5,  # Keep width at 0.4
            notch=True,
            dodge=False,  # Prevents extra spacing between boxes
            showmeans=True,
            meanline=True,
            meanprops={'color': 'black', 'linewidth': 1.5},
            medianprops={'visible': False},  # Removes black median line
            boxprops={'facecolor': "none", "edgecolor": "black"},
            whiskerprops={'color': 'black'},
            capprops={'color': 'black'}
        )

        # Overlay individual data points as spheres with colored contours
        x_positions = [0, 1]  # Categorical x-axis positions
        for j, (condition, color) in enumerate(zip(condition_labels,
                                                   condition_colors)):
            y_values = df[df['Conditions'] == condition][y_label].values
            ax_i.scatter(
                np.full_like(y_values, x_positions[j], dtype=float),
                y_values,
                facecolors='none',  # No fill
                edgecolors=color,  # Use original condition colors for contour
                s=80,  # Size of spheres
                linewidth=1.5,  # Thickness of the contour
                marker='o'  # Ensure circular markers
            )

        # Add dashed gray line at y=0
        ax_i.axhline(0, color='gray', linestyle='dashed', linewidth=1.5)

        # Set labels and titles
        ax_i.set_xlabel(subplot_titles[i], fontweight='bold', labelpad=16,
                        fontsize=16)
        if i == 0:
            ax_i.set_ylabel(y_label, fontsize=16, labelpad=5)
            ax_i.tick_params(axis='y', labelsize=14)
        else:
            ax_i.axes.get_yaxis().set_visible(False)
            ax_i.spines['left'].set_visible(False)

        ax_i.tick_params(axis='x', labelsize=14)

        # Hide unnecessary spines
        ax_i.spines['right'].set_visible(False)
        ax_i.spines['top'].set_visible(False)

        # Reduce white space inside each subplot
        ax_i.set_xlim(-0.6, 1.6)  # Adjust the x-axis to remove extra space

    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, fname + '.pdf'))
    # plt.show()


# =========================== INPUTS ===================================

# Subjects without pilot
SUBJECTS = [3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26, 28,
            29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

# Relative path for output folders
contrasts_folder = 'control_contrasts'

task_tag = 'All Tasks'
contrast_mask = 'Auditory Encoding'
roi = 'dstr' # dstr or cerebellum

# ========================= PARAMETERS =================================

# Parent directories
home = os.path.expanduser('~')
rois_folder = os.path.join(
    home, os.path.dirname(os.path.abspath(__file__)),
    'roi_analyses_rwls_hrf128')

if contrast_mask == 'Auditory Encoding':
    contrast_mask_folder = os.path.join(rois_folder, 'auditory')
    x_label = ['Auditory Conditions']
elif contrast_mask == 'Visual Encoding':
    contrast_mask_folder = os.path.join(rois_folder, 'visual')
    x_label = ['Visual Conditions']
else:
    assert contrast_mask == 'Encoding'
    contrast_mask_folder = os.path.join(rois_folder, 'all')
    x_label = ['Auditory Conditions', 'Visual Conditions']

dstr_folder = os.path.join(contrast_mask_folder, 'dorsal_striatum', 'hos',
                           'rois_extraction')
cerebellum_folder = os.path.join(contrast_mask_folder, 'cerebellum',
                                 'ntk_symmni128', 'cereb', 'rois_extraction')

dstr_file = os.path.join(dstr_folder, 'i_dstr_psc.npy')
cerebellum_file = os.path.join(cerebellum_folder, 'i_cereb_psc.npy')

# ######################################################################

tasks = {'prod': 'Production', 'percep': 'Perception', 'ntfd': 'NTFD',
         'allmain_tasks': 'All Tasks'}

all_contrasts = {1: 'Encoding',
                 2: 'Auditory Encoding',
                 3: 'Visual Encoding',
                 4: 'Auditory vs Visual Encoding',
                 5: 'Visual vs Auditory Encoding',
                 6: 'Beat',
                 7: 'Interval',
                 8: 'Beat vs Interval',
                 9: 'Interval vs Beat',
                 10: 'Auditory Beat',
                 11: 'Auditory Interval',
                 12: 'Auditory Beat vs Auditory Interval',
                 13: 'Auditory Interval vs Auditory Beat',
                 14: 'Visual Beat',
                 15: 'Visual Interval',
                 16: 'Visual Beat vs Visual Interval',
                 17: 'Visual Interval vs Visual Beat',
                 18: 'Decision'}

task_id = {v: k for k, v in tasks.items()}.get(task_tag)
position = list(tasks.keys()).index(task_id)
contrast_id = {v: k for k, v in all_contrasts.items()}.get(contrast_mask)


# ============================ RUN =====================================

if __name__ == '__main__':

    # Load individual PSC's for a certain ROI
    # Shape (hemisphere, tasks, contrasts, subjects)
    # hemisphere: lh, rh, bh
    # tasks: prod, percep, ntfd, allmain_tasks
    # contrasts: Auditory Beat, Auditory Interval, Visual Beat, Visual Interval
    # subjects: list of subjects' ids
    if roi == 'dstr':
        roi_data = np.load(dstr_file)
    else:
        assert roi == 'cerebellum'
        roi_data = np.load(cerebellum_file)

    # Extract only bh data and task data specified in task_tag
    roi_data = roi_data[-1, position, :, :]

    # Swap axes in order to have shape: (subjects, contrasts)
    roi_data = np.swapaxes(roi_data, 0, 1)

    # Save dataframe
    create_df(
        roi_data, contrast_mask, SUBJECTS, output_dir=contrasts_folder,
        fname='ipscs_' + roi + '_' + contrast_mask.lower().replace(' ', '-'))

    # Plot
    plot_boxplots(
        roi_data, output_dir=contrasts_folder,
        fname='ipscs_' + roi + '_' + contrast_mask.lower().replace(' ', '-'),
        subplot_titles=x_label)
