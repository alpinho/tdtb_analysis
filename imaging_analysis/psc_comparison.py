"""
Script to plot brain activity for the contrasts Beat vs Rest and
 Interval vs. Rest across subjects for both Auditory and Visual.

Each circle represents the contribution of one participant.
 Error bars represent 95% confidence intervals.

Author: Ana Luisa Pinho
Email: agrilopi@uwo.ca

Creation: 10th of March 2025
Last Update: June 2025

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


def plot_boxplots_mod(data, roi_name, y_label='Percent Signal Change (%)',
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
        print(
            f"Means for subplot '{subplot_titles[i]}'"
            f"of '{roi_name}':"
        )
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
            width=.5,  # Keep width at 0.4
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

    plt.savefig(os.path.join(output_dir, fname + '.png'), dpi=300)
    # plt.show()


def plot_boxplots_rois(rois_data, modality='both',
                       y_label='Percent Signal Change (%)',
                       output_dir='./', fname='roi_comparison',
                       annotate=False):
    """
    Plots two subplots, each for a different ROI
    (dorsal striatum and cerebellum),
    for Auditory conditions, Visual conditions, or both.

    Parameters
    ----------
    rois_data : numpy.ndarray
        Data array of shape (2, subjects, contrasts), where
        rois_data[0] corresponds to the dorsal striatum and
        rois_data[1] corresponds to the cerebellum.
    modality : str, optional
        Choose from 'Auditory', 'Visual', or 'both' (default: 'both').
    y_label : str, optional
        Label for the y-axis (default is 'Percent Signal Change (%)').
    output_dir : str, optional
        Directory where the plot will be saved (default is './').
    fname : str, optional
        Name of the output file without extension
        (default is 'roi_comparison').

    Returns
    -------
    None
        Displays the plot and saves it as a PDF.
    """

    # Define modality
    if modality == 'auditory':
        indices = [0, 1]
        condition_labels = ['Auditory Beat', 'Auditory Interval']
    elif modality == 'visual':
        if rois_data.shape[2] == 2:
            indices = [0, 1]
        else:
            assert rois_data.shape[2] == 4
            indices = [2, 3]
        condition_labels = ['Visual Beat', 'Visual Interval']
    else:
        assert modality == 'both'
        indices = [0, 1, 2, 3]
        condition_labels = ['Beat', 'Interval']

    # Apply line break for better formatting
    condition_labels = [label.replace(' ', '\n') for label in condition_labels]

    # Extract the relevant conditions
    dstr_data = rois_data[0][:, indices]
    cerebellum_data = rois_data[1][:, indices]

    # If both conditions are selected, merge auditory and visual in...
    # ... the same boxplot
    if modality == 'both':
        dstr_data = np.vstack((
            np.concatenate((dstr_data[:, 0], dstr_data[:, 2])),
            np.concatenate((dstr_data[:, 1], dstr_data[:, 3])))).T
        cerebellum_data = np.vstack((
            np.concatenate((cerebellum_data[:, 0], cerebellum_data[:, 2])),
            np.concatenate((cerebellum_data[:, 1], cerebellum_data[:, 3])))).T

    # Setup figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(8, 6), sharey=True)

    # Adjust subplot spacing
    # fig.subplots_adjust(left=0.275, right=0.95, bottom=0.15,
    #                     top=0.85, wspace=0.1)
    fig.subplots_adjust(bottom=0.)

    # Titles and text info
    # fig.suptitle(
    #     ('Both Modalities' if modality == 'both'
    #      else f'{modality.capitalize()} Tasks'),
    #     fontsize=16,
    #     fontweight='bold',
    #     y=.98
    # )
    fig.text(0.79, .94, "95% CI for the Mean of PSC", fontsize=16, ha='center')

    # Compute and print mean values for each ROI and modality
    dstr_means = np.mean(dstr_data, axis=0)
    cerebellum_means = np.mean(cerebellum_data, axis=0)
    print(f"Dorsal Striatum ({modality.capitalize()}):")
    print(f"  Beat: {dstr_means[0]:.4f}, Interval: {dstr_means[1]:.4f}")
    print(f"Cerebellum ({modality.capitalize()}):")
    print(f"  Beat: {cerebellum_means[0]:.4f},"
          f"  Interval: {cerebellum_means[1]:.4f}"
          )

    for i, (roi_data, ax, title) in enumerate(
            zip([dstr_data, cerebellum_data], axes,
                ['Dorsal Striatum', 'Cerebellum'])):

        # Prepare data for Seaborn
        datum = {
            'Conditions': np.repeat(condition_labels, roi_data.shape[0]),
            y_label: np.concatenate((roi_data[:, 0], roi_data[:, 1]))
        }
        df = pd.DataFrame(data=datum)

        # Create boxplot
        sns.boxplot(
            ax=ax,
            x='Conditions',
            y=y_label,
            data=df,
            width=.5,
            notch=True,
            showmeans=True,
            meanline=True,
            meanprops={'color': 'black', 'linewidth': 2.},
            medianprops={'visible': False},
            boxprops={'facecolor': "none", "edgecolor": "black",
                      'linewidth': 2.},
            whiskerprops={'color': 'black', 'linewidth': 2.},
            capprops={'color': 'black'}
        )

        # Overlay individual data points
        x_positions = [0, 1]
        for j, condition in enumerate(condition_labels):
            y_values = df[df['Conditions'] == condition][y_label].values
            ax.scatter(
                np.full_like(y_values, x_positions[j], dtype=float),
                y_values, facecolors='none',
                edgecolors='tab:blue' if j == 0 else 'tab:orange', s=80,
                linewidth=2., marker='o'
            )

        # Formatting
        ax.axhline(0, color='gray', linestyle='dashed', linewidth=2.5)
        ax.set_xlabel(title, fontweight='bold', fontsize=22, labelpad=22)
        if i == 0:
            ax.set_ylabel(y_label, fontsize=20, labelpad=5)
            ax.tick_params(axis='y', labelsize=20, width=2.)
        else:
            ax.set_ylabel('')
            ax.tick_params(axis='y', left=False, labelleft=False)
            ax.spines['left'].set_visible(False)
        ax.tick_params(axis='x', labelsize=20, pad=10, width=2.)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_linewidth(2.)  # X-axis
        ax.spines['left'].set_linewidth(2.)    # Y-axis

        # Add Annotation
        if annotate and modality == 'auditory' and title == 'Dorsal Striatum':
            y_max = np.max(roi_data) * 1.1
            ax.plot(x_positions, [y_max, y_max], color='k', linestyle='-',
                    linewidth=2.5)
            ax.vlines(x_positions, y_max, y_max * 0.975, color='k',
                      linewidth=2.5)
            ax.text(np.mean(x_positions), y_max * 1.01, f'*',
                    ha='center', va='bottom', fontsize=28)

    # Adjust layout to fit the title
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
    plt.savefig(os.path.join(output_dir, f"{fname}_modality-{modality}.png"),
                dpi=300)

    # plt.show()


# =========================== INPUTS ===================================

# Subjects without pilot
SUBJECTS = [3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26, 28,
            29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

# Relative path for output folders
output_folder = 'results/ipscs'

task_tag = 'All Tasks'
contrast_mask = 'Encoding' # 'Encoding', 'Auditory Encoding', or
                           # 'Visual Encoding'

# ========================= PARAMETERS =================================

# Parent directories
home = os.path.expanduser('~')
rois_folder = os.path.join(
    home, os.path.dirname(os.path.abspath(__file__)),
    'roi_analyses_rwls_hrf128_wb_puncorr')

if contrast_mask == 'Auditory Encoding':
    contrast_mask_folder = os.path.join(rois_folder, 'auditory')
    x_label = ['Auditory Conditions']
    n_conditions = 2
    modalities = ['auditory']
elif contrast_mask == 'Visual Encoding':
    contrast_mask_folder = os.path.join(rois_folder, 'visual')
    x_label = ['Visual Conditions']
    n_conditions = 2
    modalities = ['visual']
else:
    assert contrast_mask == 'Encoding'
    contrast_mask_folder = os.path.join(rois_folder, 'all')
    x_label = ['Auditory Conditions', 'Visual Conditions']
    n_conditions = 4
    modalities = ['auditory', 'visual', 'both']

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

    rois_data_list = []
    for roi in ['dstr', 'cerebellum']:
        # Load individual PSC's for a certain ROI
        # Shape (hemisphere, tasks, contrasts, subjects)
        # hemisphere: lh, rh, bh
        # tasks: prod, percep, ntfd, allmain_tasks
        # contrasts: Auditory Beat, Auditory Interval, Visual Beat,
        #            Visual Interval
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

        # Append to rois_data
        rois_data_list.append(roi_data)

        # Name of output files
        outname_roi = 'ipscs_roi-' + roi + '_task-' + \
            task_id.replace('_', '-') + '_mask-' + \
            contrast_mask.lower().replace(' ', '-')
        
        # Save dataframe
        create_df(roi_data, contrast_mask, SUBJECTS, output_dir=output_folder,
                  fname=outname_roi)

        # Plot
        plot_boxplots_mod(roi_data, roi, output_dir=output_folder,
                          fname=outname_roi, subplot_titles=x_label)

    # Final shape: (rois, subjects, contrasts)
    rois_data = np.stack(rois_data_list, axis=0)
    outname_rois = 'ipscs_two-rois_task-' + \
        task_id.replace('_', '-') + '_mask-' + \
        contrast_mask.lower().replace(' ', '-')
    for tmod in modalities:
        if contrast_mask == 'Encoding':
            plot_boxplots_rois(rois_data, modality=tmod,
                               y_label='Percent Signal Change (%)',
                               output_dir=output_folder, fname=outname_rois,
                               annotate=True)
        else:
            plot_boxplots_rois(rois_data, modality=tmod,
                               y_label='Percent Signal Change (%)',
                               output_dir=output_folder, fname=outname_rois)
