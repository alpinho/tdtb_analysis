"""
Script to quantify “profile similarity” between two ROIs across tasks
using repeated-measures correlation

Author: Ana Luisa Pinho
Email: agrilopi@uwo.ca

Creation: 27th of June 2025
Last Update: October 2025

Compatibility: Python 3.10.16
"""

import os

import numpy as np
import pandas as pd
import pingouin as pg

from scipy.spatial.distance import cosine

import matplotlib.pyplot as plt


# ############################# INPUTS ################################

n_rois = 8  # number of ROIs in the set (2, 4, 6, or 8)
individualization = 'i8a'  # 'i', 'i9a', 'i8a', 'i7a', 'i6a', 'a', 'a4g', 'a3g', 'a2g', 'a1g', or 'g'
roi1, roi2 = 'dstr', 'sma'  # the two ROIs to compare

loc_leg = 'upper right'  # legend location

# ########################### PARAMETERS ##############################

working_dir = os.path.dirname(os.path.abspath(__file__))
model = 'rwls'         # 'rwls' or 'standard'
masking = 'wb'         # 'wb' or 'gm'
hrf_cutoff = 'hrf128'  # 'hrf128' or 'hrf42'

rois_dir = os.path.join(
    working_dir,
    f"roi_analyses_{model}_{hrf_cutoff}_{masking}_puncorr_unsmoothed",
    'bothmod_allmain_tasks', 'main_tasks'
)
df_dir = os.path.join(rois_dir, 'df_rois_volume')
df_path = os.path.join(
    df_dir, 
    f"dfrois_{individualization}_{str(n_rois)}-rois.tsv"
)

# ——— SET of ROIs ———
rois = [roi1, roi2]
roi_labels = {
    'dstr': 'Dorsal Striatum',
    'cereb': 'Cerebellum',
    'pmv': 'PMV',
    'pmd': 'PMD',
    'presma': 'preSMA',
    'sma': 'SMA',
    'heschl': 'Heschl Gyrus',
    'occipital_lobe': 'Occipital Lobe'
}

hemis = ['bh']   # hemisphere labels
tasks = ['Production', 'Perception', 'NTFD']  # exact Task strings

# Text‐box coordinates
anno_x, anno_y = .05, .1

# ############################## RUN ####################################

# Read and clean
df_all = pd.read_csv(
    df_path,
    sep='\t',
    dtype={
        'Subject':    str,
        'Task':       str,
        'ROI':        str,
        'Hemisphere': str,
        'PSC':        float
    }
)
df_all = df_all[df_all['Task'] != 'All Tasks']  # drop summary rows

for hemi in hemis:

    # --- Repeated-measures correlation on Subject×Task PSCs ---

    # reshape so each row is one Subject×Task, columns are PSCs for...
    # ... roi1 and roi2
    wide = (
        df_all
        .query("Hemisphere == @hemi and ROI in @rois")
        .pivot_table(
            index=['Subject', 'Task'],
            columns='ROI',
            values='PSC'
        )
        .reset_index()
    )

    # keep only the tasks we care about and drop any incomplete rows
    wide = wide[wide['Task'].isin(tasks)].dropna(subset=rois)

    # compute repeated-measures correlation
    rmc = pg.rm_corr(data=wide, x=roi1, y=roi2, subject='Subject')
    r_val = rmc['r'].iloc[0]
    p_val = rmc['pval'].iloc[0]
    print(f"{hemi} repeated-measures r_rm = {r_val:.3f}, p = {p_val:.3f}")

    # --- Group-mean profiles for plotting ---
    grp = (
        df_all
        .groupby(['Hemisphere', 'ROI', 'Task'])['PSC']
        .mean()
        .reset_index()
    )
    sub_grp = grp[(grp['Hemisphere'] == hemi) & (grp['ROI'].isin(rois))]
    mat = sub_grp.pivot(index='Task', columns='ROI', values='PSC').loc[tasks]

    # --- Plot with annotation ---
    plt.figure(figsize=(5, 4))
    plt.plot(tasks, mat[roi1], marker='o', label=roi_labels.get(roi1, roi1))
    plt.plot(tasks, mat[roi2], marker='s', label=roi_labels.get(roi2, roi2))
    plt.title(f'{hemi} ROI Profiles Comparison')
    plt.xlabel('Task')
    plt.ylabel('PSC (%)')
    plt.legend()

    # Define the axes for the legend and annotation
    ax = plt.gca()

    # remove top and right axes spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # draw the ROI legend without a frame
    leg = ax.legend(frameon=False, loc=loc_leg)

    ax.text(
        anno_x, anno_y,
        rf"$r_{{rm}} = {r_val:.3f},\ p = {p_val:.3f}$",
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle="round,pad=.3", fc="white", ec="gray", alpha=0.7)
    )

    plt.tight_layout()

    # --- Save figure ---
    out_dir = os.path.join(rois_dir, 'rmcorr_profile_similarity')
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(
        out_dir,
        f"pearson_{individualization}_{str(n_rois)}-rois_{roi1}-"
        f"{roi2}_{hemi}.png"
    )
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {fname}")

    plt.close()