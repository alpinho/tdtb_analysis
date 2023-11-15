"""
This script computes the mean activity in ROIS
for a given set of contrasts of the Music-SDTB Project.

Author: Ana Luisa Pinho

Created: October 2023
Last update: November 2023

Compatibility: Python 3.10.10

"""

import os
import numpy as np
import pandas as pd

from nilearn.image import load_img, new_img_like, resample_to_img
from nilearn.input_data import NiftiLabelsMasker

import seaborn as sns
from matplotlib import pyplot as plt


# ############################ FUNCTIONS ################################

def nonan_map(con_path):
    # Load Encoding Map
    con = load_img(con_path)

    # Remove NaN's
    con_val = con.get_fdata()
    con_val[np.isnan(con_val)] = 0
    con_map = new_img_like(con, con_val)

    return con_val, con_map


def threshold_map(con_val, thresh_min, thresh_max=None):

    # Threshold
    thresholded_con_val = con_val
    thresholded_con_val[thresholded_con_val < thresh_min] = 0
    if thresh_max is not None:
        thresholded_con_val[thresholded_con_val > thresh_max] = 0

    return thresholded_con_val


def create_group_roimask(con_path, con_thresh_min, atlas_maskpath,
                         msdtb_maskpath, con_thresh_max=None):

    # Remove NaNs from contrast map
    con_val, con_map = nonan_map(con_path)

    # Threshold contrast map
    if con_thresh_max is None:
        thresholded_con_val = threshold_map(con_val, con_thresh_min)
    else:
        thresholded_con_val = threshold_map(con_val, con_thresh_min,
                                            thresh_max=con_thresh_max)

    # Binarize contrast map
    bin_con_val = (thresholded_con_val != 0)

    # Load masks generated from a selected atlas
    atlas_mask = load_img(atlas_maskpath)

    # Resample atlas mask
    atlas_rmask = resample_to_img(atlas_mask, con_map,
                                  interpolation='nearest')

    # Get data from atlas mask
    atlas_val = atlas_rmask.get_fdata()

    # Intersection of contrast-of-interest w/ atlas mask
    msdtb_val = np.logical_and(
        bin_con_val.astype(bool), atlas_val.astype(bool)).astype(int)

    # Create msdtb-Putamen mask
    msdtb_mask = new_img_like(atlas_rmask, msdtb_val)

    # Save msdtb-Putamen mask
    msdtb_mask.to_filename(msdtb_maskpath)

    return np.count_nonzero(msdtb_val)


def create_individual_roimask(individual_con_path, con_thresh_min,
                              atlas_maskpath, individual_roi_maskpath,
                              n_voxels, con_thresh_max=None):

    # Remove NaNs from contrast map
    con_val, con_map = nonan_map(individual_con_path)

    # Threshold contrast map
    if con_thresh_max is None:
        thresholded_con_val = threshold_map(con_val, con_thresh_min)
    else:
        thresholded_con_val = threshold_map(con_val, con_thresh_min,
                                            thresh_max=con_thresh_max)

    # Binarize contrast map
    bin_con_val = (thresholded_con_val != 0)

    # Load masks generated from a selected atlas
    atlas_mask = load_img(atlas_maskpath)

    # Resample atlas mask
    atlas_rmask = resample_to_img(atlas_mask, con_map,
                                  interpolation='nearest')

    # Get data from atlas mask
    atlas_val = atlas_rmask.get_fdata()

    # Intersection of contrast-of-interest w/ atlas mask
    msdtb_val = np.logical_and(
        bin_con_val.astype(bool), atlas_val.astype(bool)).astype(int)

    # Retain an ROI with the same size as the one found at the group level
    individual_roi_val = np.multiply(thresholded_con_val, msdtb_val)
    individual_thresh = np.sort(np.ravel(individual_roi_val))[n_voxels - 1]
    individual_roi_val[individual_roi_val < individual_thresh] = 0

    # Binarize individual roi
    bin_individual_roi_val = (individual_roi_val != 0)

    # Create individual roi mask
    individual_roi_mask = new_img_like(atlas_rmask, bin_individual_roi_val)

    # Save individual roi mask
    individual_roi_mask.to_filename(individual_roi_maskpath)

    return individual_roi_mask


def extract_roi(rmask, contrasts, subject_estimates_dir, filetype):

    # # For each task design
    allcontrasts_subject = []
    for t, (tk, task) in enumerate(tasks.items()):

        task_contrasts = []
        # # For every contrast
        for key in list(contrasts.keys())[1:]:

            contrast_fname = filetype + '_%04d_desc-sm8wbmasked.nii' % key

            masker = NiftiLabelsMasker(labels_img=rmask)
            masker.fit()

            masked_con = os.path.join(subject_estimates_dir, tk,
                                      'masked_derivatives_rwls',
                                      contrast_fname)
            print(np.array(masked_con))

            # Extract mean average of contrasts effect-size in ROI...
            # ... for a certain participant
            mask_data = masker.transform(masked_con)[0][0]

            task_contrasts.append(mask_data)

        # all designs
        allcontrasts_subject.append(task_contrasts)

    return allcontrasts_subject


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


def plot_roi_horizontal(arr_conmean, arr_conpval, roi_ref, output_file):
    # ## Open npy files and plot
    allcontrasts_mean = np.load(arr_conmean)
    allpvalues = np.load(arr_conpval)

    fig = plt.figure(figsize=(30, 25))
    # For each hemisphere
    for r, roi in enumerate(allcontrasts_mean):
        # For each task
        for c, cmean in enumerate(roi):
            # plt.axes([left, bottom, width, height])
            ax = plt.axes([.16 + r*.475, .73 - c*.23, .325, .18])
            cnames = list(contrasts.values())
            y_pos = np.arange(len(cnames))
            pval_labels = pval_label_converter(allpvalues[r][c])
            rects = ax.barh(y_pos, cmean, align='center')
            ax.bar_label(rects, labels=pval_labels, padding=3)
            ax.set_yticks(y_pos, labels=cnames, fontsize=16)
            plt.xticks(fontsize=16)
            # Hide the right and top spines
            ax.spines[['right', 'top']].set_visible(False)
            plt.title(list(tasks.values())[c], size=20, x=.5,
                      fontweight='semibold')
        if r == 0:
            column_title = 'Left Hemisphere'
        else:
            column_title = 'Right Hemisphere'
        fig.text(.25 + r*.5, .9425, column_title, ha='center',
                 fontsize=24, weight='bold')
    plt.suptitle(roi_ref, size=28, weight='bold', linespacing=.75)
    fig.savefig(output_file, dpi=300)


def plot_roi_vertical(arr_conmean):
    # ## Open npy files and plot
    allcontrasts_mean = np.load(arr_conmean).tolist()

    # fig = plt.figure(figsize=(30, 35))
    fig, ax = plt.subplots(len(allcontrasts_mean[0]), len(allcontrasts_mean))

    # left   # the left side of the subplots of the figure
    # right  # the right side of the subplots of the figure
    # bottom # the bottom of the subplots of the figure
    # top    # the top of the subplots of the figure
    # wspace # the amount of width reserved for blank space between subplots
    # hspace # the amount of height reserved for white space between subplots
    plt.subplots_adjust(left=.12, right=.99, bottom=.15, wspace=.075)

    # For each hemisphere
    for r, roi in enumerate(allcontrasts_mean):
        # For each task
        for c, cmean in enumerate(roi):

            # plt.axes([left, bottom, width, height])
            ax = plt.axes([.16 + r*.475, .73 - c*.2, .1, .1])

            # Names of Contrasts
            cnames = list(filtered_contrasts.values())



            d = {x: np.ravel(standard),
            y: np.ravel(data_list),
            z: np.ravel(conditions)}
            df = pd.DataFrame(data=d)
            # Create bar plot
            sns.barplot(ax=ax[m][c],
                x=x,
                y=y,
                hue=z,
                data=df,
                estimator=np.mean,
                ci=95, # 1.96 * standard error (95% confidence interval)
                errcolor="black", errwidth=1.5, capsize = 0.2, alpha=0.5)




            if display_plabels:
                pval_labels = pval_label_converter(allpvalues[r][c])
                ax.bar_label(rects, labels=pval_labels, padding=3)

            ax.set_xticks(x_pos, labels=cnames, fontsize=16,
                          fontweight='semibold', rotation=45, ha='right')
            ax.xaxis.set_tick_params(width=10.)
            plt.yticks(fontsize=16, fontweight='semibold')
            ax.set_ylim([-.05, .06])
            ax.set_ylabel('Effect Size', fontweight='semibold', fontsize=20)
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(2)
            # Hide the right and top spines
            ax.spines[['right', 'top']].set_visible(False)
            plt.title(list(tasks.values())[c], size=30, x=.5,
                      fontweight='semibold')


        if r == 0:
            column_title = 'Left Hemisphere'
        else:
            column_title = 'Right Hemisphere'
        fig.text(.25 + r*.5, .9425, column_title, ha='center',
                 fontsize=30, weight='bold')
    plt.suptitle(roi_ref, size=28, weight='bold', linespacing=.75)
    fig.savefig(output_file, dpi=300)


# ############################# INPUTS ##################################

if os.path.isdir('/home/analu/diedrichsen_data/data'):
    base_dir = '/home/analu/diedrichsen_data/data'
else:
    base_dir = '/srv/diedrichsen/data'

data_dir = os.path.join(base_dir, 'Cerebellum/music-sdtb/derivatives')
mask_wb = os.path.join(data_dir, 'group/anat/group_mask_noskull.nii')
mask_gm = os.path.join(data_dir, 'group/anat/group_mask_gray.nii')

# Subjects w/ pilot
# SUBJECTS = [3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21,
#             22, 23, 26, 28, 29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 
#             44, 45, 46, 47]

# Subjects without pilot
SUBJECTS = [3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26, 28,
            29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

tasks = {'prod': 'Production', 'percep': 'Perception', 'ntfd': 'NTFD',
         'allmain_tasks': 'All Tasks'}

all_contrasts = {1: 'Encoding',
                 2: 'Auditory Encoding',
                 3: 'Visual Encoding',
                 4: 'Auditory vs. Visual Encoding',
                 5: 'Visual vs. Auditory Encoding',
                 6: 'Beat',
                 7: 'Interval',
                 8: 'Beat vs. Interval',
                 9: 'Auditory Beat vs. Auditory Interval',
                 10: 'Visual Beat vs. Visual Interval',
                 11: 'Interval vs. Beat',
                 12: 'Auditory Interval vs. Auditory Beat',
                 13: 'Visual Interval vs. Visual Beat',
                 14: 'Decision'}

# filtered_contrasts = {
#              9: 'AB vs. AI',
#              10: 'VB vs. VI'}

# filtered_contrasts = {1: 'Encoding',
#                       4: 'Auditory vs. Visual Encoding',
#                       5: 'Visual vs. Auditory Encoding',
#                       8: 'Beat vs. Interval',
#                       9: 'Auditory Beat vs. Auditory Interval',
#                       10: 'Visual Beat vs. Visual Interval',
#                       11: 'Interval vs. Beat',
#                       12: 'Auditory Interval vs. Auditory Beat',
#                       13: 'Visual Interval vs. Visual Beat',
#                       14: 'Decision'}

filtered_contrasts = {1: 'Encoding',
                      2: 'Auditory Encoding',
                      3: 'Visual Encoding',
                      6: 'Beat',
                      7: 'Interval',
                      14: 'Decision'}

wb_masking = 'wb'
gm_masking = 'gm'

group_relative_path = 'group/allmain_tasks/rfx_onesample_t_rwls_'+ \
    wb_masking + '/con_01_Encoding/'

gencoding_relative_path = group_relative_path + 'con_0001.nii'
gencoding_path = os.path.join(data_dir, gencoding_relative_path)
con_thresh_min = 0
con_thresh_max = 1.

group_tmap_relative_path = group_relative_path  + 'spmT_0001.nii'
group_tmap_path = os.path.join(data_dir, group_tmap_relative_path)
# tmap_thresh_min = 3.385

##########################################################

working_dir = os.path.dirname(os.path.abspath(__file__))
atlases_dir = os.path.join(working_dir, 'atlases')
fsl_dir = os.path.join(atlases_dir, 'fsl_atlases')
nettekoven_dir = os.path.join(atlases_dir, 'nettekoven')
msdtb_dir = os.path.join(atlases_dir, 'msdtb')
group_rois_dir = os.path.join(msdtb_dir, 'group_rois')
individual_rois_dir = os.path.join(msdtb_dir, 'individual_rois')

mniflirt_cereb6_lh_maskpath = os.path.join(
    fsl_dir, 'mniflirt_cereb6_lh_mask.nii.gz')
mniflirt_cereb6_rh_maskpath = os.path.join(
    fsl_dir, 'mniflirt_cereb6_rh_mask.nii.gz')

mniflirt_crus1_lh_maskpath = os.path.join(
    fsl_dir, 'mniflirt_crus1_lh_mask.nii.gz')
mniflirt_crus1_rh_maskpath = os.path.join(
    fsl_dir, 'mniflirt_crus1_rh_mask.nii.gz')

mniflirt_cereb7b8a_lh_maskpath = os.path.join(
    fsl_dir, 'mniflirt_cereb7b8a_lh_mask.nii.gz')
mniflirt_cereb7b8a_rh_maskpath = os.path.join(
    fsl_dir, 'mniflirt_cereb7b8a_rh_mask.nii.gz')

atl_cerebd3s_lh_maskpath = os.path.join(
    nettekoven_dir, 'd3ls_atl128_symmni_mask.nii.gz')
atl_cerebd3s_rh_maskpath = os.path.join(
    nettekoven_dir, 'd3rs_atl128_symmni_mask.nii.gz')
atl_cerebd3i_lh_maskpath = os.path.join(
    nettekoven_dir, 'd3li_atl128_symmni_mask.nii.gz')
atl_cerebd3i_rh_maskpath = os.path.join(
    nettekoven_dir, 'd3ri_atl128_symmni_mask.nii.gz')

msdtb_putamen_conmean = os.path.join(
    msdtb_dir, 'msdtb_putamen_conmean.npy')
msdtb_putamen_conpval = os.path.join(
    msdtb_dir, 'msdtb_putamen_conpval.npy')
msdtb_putamen_pscmean = os.path.join(
    msdtb_dir, 'msdtb_putamen_pscmean.npy')
msdtb_putamen_pscpval = os.path.join(
    msdtb_dir, 'msdtb_putamen_pscpval.npy')
msdtb_putamen_con_roih = os.path.join(
    msdtb_dir, 'msdtb_putamen_roi_con_horizontalbarplot.png')
msdtb_putamen_psc_roih = os.path.join(
    msdtb_dir, 'msdtb_putamen_roi_psc_horizontalbarplot.png')
msdtb_putamen_roiv = os.path.join(
    msdtb_dir, 'msdtb_putamen_roi_verticalbarplot.png')

msdtb_cereb6_lh_maskpath = os.path.join(msdtb_dir,
                                        'msdtb_cereb6_mask_lh.nii.gz')
msdtb_cereb6_rh_maskpath = os.path.join(msdtb_dir,
                                        'msdtb_cereb6_mask_rh.nii.gz')
msdtb_cereb6_conmean = os.path.join(msdtb_dir,
                                    'msdtb_cereb6_conmean.npy')
msdtb_cereb6_conpval = os.path.join(msdtb_dir,
                                    'msdtb_cereb6_conpval.npy')
msdtb_cereb6_pscmean = os.path.join(msdtb_dir,
                                    'msdtb_cereb6_pscmean.npy')
msdtb_cereb6_pscpval = os.path.join(msdtb_dir,
                                    'msdtb_cereb6_pscpval.npy')
msdtb_cereb6_con_roih = os.path.join(
    msdtb_dir, 'msdtb_cereb6_roi_con_horizontalbarplot.png')
msdtb_cereb6_psc_roih = os.path.join(
    msdtb_dir, 'msdtb_cereb6_roi_psc_horizontalbarplot.png')
msdtb_cereb6_roiv = os.path.join(
    msdtb_dir, 'msdtb_cereb6_roi_verticalbarplot.png')

msdtb_crus1_lh_maskpath = os.path.join(msdtb_dir,
                                        'msdtb_crus1_mask_lh.nii.gz')
msdtb_crus1_rh_maskpath = os.path.join(msdtb_dir,
                                        'msdtb_crus1_mask_rh.nii.gz')
msdtb_crus1_conmean = os.path.join(msdtb_dir,
                                    'msdtb_crus1_conmean.npy')
msdtb_crus1_conpval = os.path.join(msdtb_dir,
                                    'msdtb_crus1_conpval.npy')
msdtb_crus1_pscmean = os.path.join(msdtb_dir,
                                    'msdtb_crus1_pscmean.npy')
msdtb_crus1_pscpval = os.path.join(msdtb_dir,
                                    'msdtb_crus1_pscpval.npy')
msdtb_crus1_con_roih = os.path.join(
    msdtb_dir, 'msdtb_crus1_roi_con_horizontalbarplot.png')
msdtb_crus1_psc_roih = os.path.join(
    msdtb_dir, 'msdtb_crus1_roi_psc_horizontalbarplot.png')
msdtb_crus1_roiv = os.path.join(
    msdtb_dir, 'msdtb_crus1_roi_verticalbarplot.png')

msdtb_cereb7b8a_lh_maskpath = os.path.join(msdtb_dir,
                                           'msdtb_cereb7b8a_mask_lh.nii.gz')
msdtb_cereb7b8a_rh_maskpath = os.path.join(msdtb_dir,
                                           'msdtb_cereb7b8a_mask_rh.nii.gz')
msdtb_cereb7b8a_conmean = os.path.join(msdtb_dir,
                                       'msdtb_cereb7b8a_conmean.npy')
msdtb_cereb7b8a_conpval = os.path.join(msdtb_dir,
                                       'msdtb_cereb7b8a_conpval.npy')
msdtb_cereb7b8a_pscmean = os.path.join(msdtb_dir,
                                       'msdtb_cereb7b8a_pscmean.npy')
msdtb_cereb7b8a_pscpval = os.path.join(msdtb_dir,
                                       'msdtb_cereb7b8a_pscpval.npy')
msdtb_cereb7b8a_con_roih = os.path.join(
    msdtb_dir, 'msdtb_cereb7b8a_roi_con_horizontalbarplot.png')
msdtb_cereb7b8a_psc_roih = os.path.join(
    msdtb_dir, 'msdtb_cereb7b8a_roi_psc_horizontalbarplot.png')
msdtb_cereb7b8a_roiv = os.path.join(
    msdtb_dir, 'msdtb_cereb7b8a_roi_verticalbarplot.png')

msdtb_cerebd3s_lh_maskpath = os.path.join(msdtb_dir,
                                          'msdtb_cerebd3s_mask_lh.nii.gz')
msdtb_cerebd3s_rh_maskpath = os.path.join(msdtb_dir,
                                          'msdtb_cerebd3s_mask_rh.nii.gz')
msdtb_cerebd3s_conmean = os.path.join(msdtb_dir,
                                      'msdtb_cerebd3s_conmean.npy')
msdtb_cerebd3s_conpval = os.path.join(msdtb_dir,
                                      'msdtb_cerebd3s_conpval.npy')
msdtb_cerebd3s_pscmean = os.path.join(msdtb_dir,
                                      'msdtb_cerebd3s_pscmean.npy')
msdtb_cerebd3s_pscpval = os.path.join(msdtb_dir,
                                      'msdtb_cerebd3s_pscpval.npy')
msdtb_cerebd3s_con_roih = os.path.join(
    msdtb_dir, 'msdtb_cerebd3s_roi_con_horizontalbarplot.png')
msdtb_cerebd3s_psc_roih = os.path.join(
    msdtb_dir, 'msdtb_cerebd3s_roi_psc_horizontalbarplot.png')
msdtb_cerebd3s_roiv = os.path.join(
    msdtb_dir, 'msdtb_cerebd3s_roi_verticalbarplot.png')

msdtb_cerebd3i_lh_maskpath = os.path.join(msdtb_dir,
                                          'msdtb_cerebd3i_mask_lh.nii.gz')
msdtb_cerebd3i_rh_maskpath = os.path.join(msdtb_dir,
                                          'msdtb_cerebd3i_mask_rh.nii.gz')
msdtb_cerebd3i_conmean = os.path.join(msdtb_dir,
                                      'msdtb_cerebd3i_conmean.npy')
msdtb_cerebd3i_conpval = os.path.join(msdtb_dir,
                                      'msdtb_cerebd3i_conpval.npy')
msdtb_cerebd3i_pscmean = os.path.join(msdtb_dir,
                                      'msdtb_cerebd3i_pscmean.npy')
msdtb_cerebd3i_pscpval = os.path.join(msdtb_dir,
                                      'msdtb_cerebd3i_pscpval.npy')
msdtb_cerebd3i_con_roih = os.path.join(
    msdtb_dir, 'msdtb_cerebd3i_roi_con_horizontalbarplot.png')
msdtb_cerebd3i_psc_roih = os.path.join(
    msdtb_dir, 'msdtb_cerebd3i_roi_psc_horizontalbarplot.png')
msdtb_cerebd3i_roiv = os.path.join(
    msdtb_dir, 'msdtb_cerebd3i_roi_verticalbarplot.png')

ls_atl32_symmni_maskpath = os.path.join(
    nettekoven_dir, 'ls_atl32_symmni_mask.nii.gz')
rs_atl32_symmni_maskpath = os.path.join(
    nettekoven_dir, 'rs_atl32_symmni_mask.nii.gz')
li_atl32_symmni_maskpath = os.path.join(
    nettekoven_dir, 'li_atl32_symmni_mask.nii.gz')
ri_atl32_symmni_maskpath = os.path.join(
    nettekoven_dir, 'ri_atl32_symmni_mask.nii.gz')


# ############################## RUN ####################################

if __name__ == '__main__':

    # # # ######################## PUTAMEN ##################################
    # For each hemisphere
    contrasts_means_hems = []
    for hem in ['lh', 'rh']:
        hos_putamen_maskpath = os.path.join(
            fsl_dir, 'hos_putamen_' + hem + '_mask.nii.gz')
        msdtb_putamen_maskpath = os.path.join(
            group_rois_dir, 'putamen',
            'msdtb_putamen_mask_' + hem + '.nii.gz')
        cluster_size = create_group_roimask(
            group_tmap_path, 3.385, hos_putamen_maskpath,
            msdtb_putamen_maskpath)
        # # Create individual ROIs
        subjects_contrasts = []
        for subject in SUBJECTS:
            subject_dir = os.path.join(data_dir, 'sub-%02d') % subject
            estimates_dir = os.path.join(subject_dir, 'estimates')
            subject_encoding_tmap = os.path.join(
                estimates_dir,
                'allmain_tasks',
                'masked_derivatives_rwls',
                'wspmT_0001_desc-sm8wbmasked.nii')
            individual_putamen_maskpath = os.path.join(
                individual_rois_dir, 'putamen',
                'putamen_mask_sub-%02d_' + hem + '.nii.gz') % subject
            irmask = create_individual_roimask(subject_encoding_tmap, 0.,
                                               hos_putamen_maskpath,
                                               individual_putamen_maskpath,
                                               cluster_size)
            subject_estimates_dir = os.path.join(data_dir, 'sub-%02d' % subject,
                                                 'estimates')
            subject_contrasts = extract_roi(irmask, filtered_contrasts,
                                            subject_estimates_dir, 'wpsc')

            # shape (subjects, tasks, contrasts)
            subjects_contrasts.append(subject_contrasts)

        contrasts_means = np.mean(subjects_contrasts, axis=0)

    contrasts_means_hems.append(contrasts_means)

    ## Save
    np.save(msdtb_putamen_conmean, contrasts_means_hems, allow_pickle=False)

    # Plot
    plot_roi_vertical(msdtb_putamen_conmean)


    # # # # ###################### CEREBELLUM VI ############################

    # # # Create Music-SDTB ROIs
    # # create_msdtb_roi(tmap_path, tmap_thresh_min,
    # #                  mniflirt_cereb6_lh_maskpath, mniflirt_cereb6_rh_maskpath,
    # #                  msdtb_cereb6_lh_maskpath, msdtb_cereb6_rh_maskpath,
    # #                  map_thresh_max=None)

    # # ## Extract data from ROIs in both hemispheres
    # msdtb_cereb6_lh_mask = load_img(msdtb_cereb6_lh_maskpath)
    # msdtb_cereb6_rh_mask = load_img(msdtb_cereb6_rh_maskpath)
    # cereb6_masks = [msdtb_cereb6_lh_mask, msdtb_cereb6_rh_mask]
    # # compute_rois(cereb6_masks, mask_wb, contrasts,
    # #              msdtb_cereb6_conmean, msdtb_cereb6_conpval, 'wcon')
    # compute_rois(cereb6_masks, mask_wb, contrasts,
    #              msdtb_cereb6_pscmean, msdtb_cereb6_pscpval, 'wpsc')

    # # # Plot
    # # plot_roi_horizontal(msdtb_cereb6_conmean, msdtb_cereb6_conpval,
    # #                     'Cerebellar Lobule VI', msdtb_cereb6_con_roih)
    # plot_roi_horizontal(msdtb_cereb6_pscmean, msdtb_cereb6_pscpval,
    #                     'Cerebellar Lobule VI', msdtb_cereb6_psc_roih)


    # # ##################### CEREBELLUM CRUS I #############################

    # # Create Music-SDTB ROIs
    # # create_msdtb_roi(tmap_path, 3.,
    # #                  mniflirt_crus1_lh_maskpath, mniflirt_crus1_rh_maskpath,
    # #                  msdtb_crus1_lh_maskpath, msdtb_crus1_rh_maskpath,
    # #                  map_thresh_max=None)

    # ## Extract data from ROIs in both hemispheres
    # msdtb_crus1_lh_mask = load_img(msdtb_crus1_lh_maskpath)
    # msdtb_crus1_rh_mask = load_img(msdtb_crus1_rh_maskpath)
    # crus1_masks = [msdtb_crus1_lh_mask, msdtb_crus1_rh_mask]
    # # compute_rois(crus1_masks, mask_wb, contrasts,
    # #              msdtb_crus1_conmean, msdtb_crus1_conpval, 'wcon')
    # compute_rois(crus1_masks, mask_wb, contrasts,
    #              msdtb_crus1_pscmean, msdtb_crus1_pscpval, 'wpsc')

    # # Plot
    # # plot_roi_horizontal(msdtb_crus1_conmean, msdtb_crus1_conpval, 'Crus I',
    # #                     msdtb_crus1_con_roih)
    # plot_roi_horizontal(msdtb_crus1_pscmean, msdtb_crus1_pscpval, 'Crus I',
    #                     msdtb_crus1_psc_roih)

    # # ##################### CEREBELLUM 7b-8a #############################

    # # Create Music-SDTB ROIs
    # create_msdtb_roi(tmap_path, tmap_thresh_min,
    #                  mniflirt_cereb7b8a_lh_maskpath,
    #                  mniflirt_cereb7b8a_rh_maskpath,
    #                  msdtb_cereb7b8a_lh_maskpath,
    #                  msdtb_cereb7b8a_rh_maskpath,
    #                  map_thresh_max=None)

    # ## Extract data from ROIs in both hemispheres
    # msdtb_cereb7b8a_lh_mask = load_img(msdtb_cereb7b8a_lh_maskpath)
    # msdtb_cereb7b8a_rh_mask = load_img(msdtb_cereb7b8a_rh_maskpath)
    # cereb7b8a_masks = [msdtb_cereb7b8a_lh_mask, msdtb_cereb7b8a_rh_mask]
    # compute_rois(cereb7b8a_masks, mask_wb, contrasts,
    #              msdtb_cereb7b8a_conmean, msdtb_cereb7b8a_conpval, 'wcon')
    # compute_rois(cereb7b8a_masks, mask_wb, contrasts,
    #              msdtb_cereb7b8a_pscmean, msdtb_cereb7b8a_pscpval, 'wpsc')

    # # Plot
    # plot_roi_horizontal(msdtb_cereb7b8a_conmean, msdtb_cereb7b8a_conpval,
    #                     'Cerebellar Lobules VIIb-VIIIa',
    #                     msdtb_cereb7b8a_con_roih)
    # plot_roi_horizontal(msdtb_cereb7b8a_pscmean, msdtb_cereb7b8a_pscpval,
    #                     'Cerebellar Lobules VIIb-VIIIa',
    #                     msdtb_cereb7b8a_psc_roih)

    # # ##################### CEREBELLUM D3s #############################

    # # Create Music-SDTB ROIs
    # create_msdtb_roi(tmap_path, tmap_thresh_min,
    #                  atl_cerebd3s_lh_maskpath,
    #                  atl_cerebd3s_rh_maskpath,
    #                  msdtb_cerebd3s_lh_maskpath,
    #                  msdtb_cerebd3s_rh_maskpath,
    #                  map_thresh_max=None)

    # ## Extract data from ROIs in both hemispheres
    # msdtb_cerebd3s_lh_mask = load_img(msdtb_cerebd3s_lh_maskpath)
    # msdtb_cerebd3s_rh_mask = load_img(msdtb_cerebd3s_rh_maskpath)
    # cerebd3s_masks = [msdtb_cerebd3s_lh_mask, msdtb_cerebd3s_rh_mask]
    # compute_rois(cerebd3s_masks, mask_wb, contrasts,
    #              msdtb_cerebd3s_conmean, msdtb_cerebd3s_conpval, 'wcon')
    # compute_rois(cerebd3s_masks, mask_wb, contrasts,
    #              msdtb_cerebd3s_pscmean, msdtb_cerebd3s_pscpval, 'wpsc')

    # # Plot
    # plot_roi_horizontal(msdtb_cerebd3s_conmean, msdtb_cerebd3s_conpval,
    #                     'Cerebellum D3 Superior',
    #                     msdtb_cerebd3s_con_roih)
    # plot_roi_horizontal(msdtb_cerebd3s_pscmean, msdtb_cerebd3s_pscpval,
    #                     'Cerebellum D3 Superior',
    #                     msdtb_cerebd3s_psc_roih)

    # # ##################### CEREBELLUM D3i #############################

    # # Create Music-SDTB ROIs
    # create_msdtb_roi(tmap_path, tmap_thresh_min,
    #                  atl_cerebd3i_lh_maskpath,
    #                  atl_cerebd3i_rh_maskpath,
    #                  msdtb_cerebd3i_lh_maskpath,
    #                  msdtb_cerebd3i_rh_maskpath,
    #                  map_thresh_max=None)

    # ## Extract data from ROIs in both hemispheres
    # msdtb_cerebd3i_lh_mask = load_img(msdtb_cerebd3i_lh_maskpath)
    # msdtb_cerebd3i_rh_mask = load_img(msdtb_cerebd3i_rh_maskpath)
    # cerebd3i_masks = [msdtb_cerebd3i_lh_mask, msdtb_cerebd3i_rh_mask]
    # compute_rois(cerebd3i_masks, mask_wb, contrasts,
    #              msdtb_cerebd3i_conmean, msdtb_cerebd3i_conpval, 'wcon')
    # compute_rois(cerebd3i_masks, mask_wb, contrasts,
    #              msdtb_cerebd3i_pscmean, msdtb_cerebd3i_pscpval, 'wpsc')

    # # Plot
    # plot_roi_horizontal(msdtb_cerebd3i_conmean, msdtb_cerebd3i_conpval,
    #                     'Cerebellum D3 Inferior',
    #                     msdtb_cerebd3i_con_roih)
    # plot_roi_horizontal(msdtb_cerebd3i_pscmean, msdtb_cerebd3i_pscpval,
    #                     'Cerebellum D3 Inferior',
    #                     msdtb_cerebd3i_psc_roih)
