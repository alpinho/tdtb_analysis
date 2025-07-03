"""
Script to quantify “profile similarity” between two ROIs across tasks
using repeated‐measures correlation

Author: Ana Luisa Pinho
Email: agrilopi@uwo.ca

Creation: 27th of June 2025
Last Update: July 2025

Compatibility: Python 3.10.16
"""

import os

import numpy as np
import pandas as pd
import pingouin as pg

import matplotlib.pyplot as plt

# ############################# INPUTS ##################################

working_dir = os.path.dirname(os.path.abspath(__file__))
model      = 'rwls'      # 'rwls' or 'standard'
masking    = 'wb'        # 'wb' or 'gm'
hrf_cutoff = 'hrf128'    # 'hrf128' or 'hrf42'

roi_dir   = os.path.join(
    working_dir,
    f"roi_analyses_{model}_{hrf_cutoff}_{masking}_puncorr_unsmoothed"
)
msdtb_dir = os.path.join(roi_dir, 'all')
df_path   = os.path.join(msdtb_dir, 'dfrois_i8a_8-rois.tsv')

# ——— SET of ROIs ———
roi1, roi2 = 'cereb', 'pmv'
rois = [roi1, roi2]
roi_labels = {
    'dstr': 'Dorsal Striatum',
    'sma':  'SMA',
    'cereb': 'Cerebellum',
    'pmv': 'PMV',
    # add more mappings for other ROIs
}

hemis = ['bh']   # hemisphere labels
tasks = ['Production', 'Perception', 'NTFD']  # exact Task strings

# Text‐box coordinates
anno_x, anno_y = 0.05, 0.1

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
        .groupby(['Hemisphere','ROI','Task'])['PSC']
        .mean()
        .reset_index()
    )
    sub_grp = grp[(grp['Hemisphere']==hemi) & (grp['ROI'].isin(rois))]
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
    leg = ax.legend(frameon=False, loc='best')

    ax.text(
        anno_x, anno_y,
        rf"$r_{{rm}} = {r_val:.3f},\ p = {p_val:.3f}$",
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7)
    )

    plt.tight_layout()

    # --- Save figure ---
    out_dir = os.path.join(msdtb_dir, 'rmcorr_profile_similarity')
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(
        out_dir,
        f'profile_similarity_i8a_8-rois_{roi1}-{roi2}_{hemi}.png'
    )
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {fname}")

    plt.close()