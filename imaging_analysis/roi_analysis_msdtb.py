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

from scipy import ndimage, stats

from nilearn.plotting import plot_glass_brain
from nilearn.image import load_img, new_img_like, resample_to_img
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker

from matplotlib import pyplot as plt


# ############################ FUNCTIONS ################################


def binarize_map(con_path, thresh_min, thresh_max=None):
    # Load Encoding Map
    con = load_img(con_path)

    # Remove NaN's
    con_val = con.get_fdata()
    con_val[np.isnan(con_val)] = 0
    new_con = new_img_like(con, con_val)

    # Threshold
    thresholded_con_val = con_val
    thresholded_con_val[thresholded_con_val < thresh_min] = 0
    if thresh_max is not None:
        thresholded_con_val[thresholded_con_val > thresh_max] = 0

    # Binarization
    bin_con_val = (thresholded_con_val != 0)

    return new_con, bin_con_val


def create_msdtb_roi(map_path, map_thresh_min,
                     atlas_lh_maskpath, atlas_rh_maskpath,
                     msdtb_lh_maskpath, msdtb_rh_maskpath,
                     map_thresh_max=None):

    # Load contrast-of-interest
    if map_thresh_max is None:
        new_map, bin_map_val = binarize_map(map_path, map_thresh_min)
    else:
        new_map, bin_map_val = binarize_map(map_path, map_thresh_min,
                                            thresh_max=map_thresh_max)

    # Load masks generated from a selected atlas
    atlas_lh_mask = load_img(atlas_lh_maskpath)
    atlas_rh_mask = load_img(atlas_rh_maskpath)

    # Resample atlas masks
    atlas_lh_rmask = resample_to_img(atlas_lh_mask, new_map,
                                     interpolation='nearest')
    atlas_rh_rmask = resample_to_img(atlas_rh_mask, new_map,
                                     interpolation='nearest')

    # Get data from atlas masks
    atlas_lh_val = atlas_lh_rmask.get_fdata()
    atlas_rh_val = atlas_rh_rmask.get_fdata()

    # Intersection of contrast-of-interest w/ atlas masks
    msdtb_lh_val = np.logical_and(
        bin_map_val.astype(bool), atlas_lh_val.astype(bool)).astype(int)
    msdtb_rh_val = np.logical_and(
        bin_map_val.astype(bool), atlas_rh_val.astype(bool)).astype(int)

    # Create msdtb-Putamen masks
    msdtb_lh_mask = new_img_like(atlas_lh_rmask, msdtb_lh_val)
    msdtb_rh_mask = new_img_like(atlas_rh_rmask, msdtb_rh_val)

    # Save msdtb-Putamen masks
    msdtb_lh_mask.to_filename(msdtb_lh_maskpath)
    msdtb_rh_mask.to_filename(msdtb_rh_maskpath)


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


def compute_rois(rmasks, mask_type, contrasts, arr_conmean, arr_conpval,
                 filetype):
    hemrois_contrasts_mean = []
    hemrois_allpvalues = []
    # # For each hemisphere
    for rmask in rmasks:
        masker = NiftiLabelsMasker(labels_img=rmask, mask_img=mask_type)
        masker.fit()

        # # For each task design
        allcontrasts_mean = []
        allpvalues = []
        for t, (tk, task) in enumerate(tasks.items()):
            contrasts_mean = []
            pvalues = []

            # # For every contrast
            for key in contrasts.keys():
                contrast_fname = filetype + '_%04d_desc-sm8wbmasked.nii' % key
                print(contrast_fname)
                masked_con = [os.path.join(estimate_dir, tk,
                                           'masked_derivatives_rwls',
                                           contrast_fname)
                              for estimate_dir in estimates_dir]
                print(np.array(masked_con))

                # Extract mean average of contrasts effect-size in ROI...
                # ... for every participant
                mask_data = [masker.transform(mcon)[0][0]
                             for mcon in masked_con]

                # Compute mean
                mean_roi = np.mean(mask_data)

                # Compute stat
                _, pval = stats.ttest_1samp(mask_data, popmean=0.,
                                            alternative='greater')

                contrasts_mean.append(mean_roi)
                pvalues.append(pval)

            allcontrasts_mean.append(contrasts_mean)
            allpvalues.append(pvalues)

        hemrois_contrasts_mean.append(allcontrasts_mean)
        hemrois_allpvalues.append(allpvalues)

    # ## Save
    np.save(arr_conmean, hemrois_contrasts_mean, allow_pickle=False)
    np.save(arr_conpval, hemrois_allpvalues, allow_pickle=False)


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


def plot_roi_vertical(arr_conmean, arr_conpval, roi_ref, output_file,
                      display_plabels=False):
    # ## Open npy files and plot
    allcontrasts_mean = np.load(arr_conmean).tolist()
    allpvalues = np.load(arr_conpval).tolist()

    print(allcontrasts_mean)
    print(allpvalues)

    fig = plt.figure(figsize=(30, 35))
    # For each hemisphere
    for r, roi in enumerate(allcontrasts_mean):
        # For each task
        for c, cmean in enumerate(roi):

            # Filter conditions
            # filtered_idx = list(filtered_contrasts.keys())
            # filtered_cmean = [cmean[i-1] for i in filtered_idx]
            # filtered_pvalues = [allpvalues[r][c][i] for i in filtered_idx]

            # plt.axes([left, bottom, width, height])
            ax = plt.axes([.16 + r*.475, .73 - c*.2, .1, .1])
            cnames = list(filtered_contrasts.values())
            # x_pos = [.2, .35, .65, .8]
            x_pos = [.495, .505]
            color_code = ['mediumseagreen', 'goldenrod']
            colors = color_code * (len(cmean)//len(color_code))
            rects = ax.bar(x_pos, cmean, align='center', width=.009,
                           color=colors)

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

data_dir = '/home/analu/diedrichsen_data/data/Cerebellum/music-sdtb/derivatives'
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

contrasts = {1: 'Encoding',
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

filtered_contrasts = {1: 'Encoding',
                      4: 'Auditory vs. Visual Encoding',
                      5: 'Visual vs. Auditory Encoding',
                      8: 'Beat vs. Interval',
                      9: 'Auditory Beat vs. Auditory Interval',
                      10: 'Visual Beat vs. Visual Interval',
                      11: 'Interval vs. Beat',
                      12: 'Auditory Interval vs. Auditory Beat',
                      13: 'Visual Interval vs. Visual Beat',
                      14: 'Decision'}

subjects_dir = [os.path.join(data_dir, 'sub-%02d') % sbj for sbj in SUBJECTS]
estimates_dir = [os.path.join(subject_dir, 'estimates')
                 for subject_dir in subjects_dir]

wb_masking = 'wb'
gm_masking = 'gm'

relative_path = 'group/allmain_tasks/rfx_onesample_t_rwls_'+ wb_masking + \
    '/con_01_Encoding/'

con_relative_path = relative_path + 'con_0001.nii'
con_path = os.path.join(data_dir, con_relative_path)
con_thresh_min = 0
con_thresh_max = 1.

tmap_relative_path = relative_path + 'spmT_0001.nii'
tmap_path = os.path.join(data_dir, tmap_relative_path)
tmap_thresh_min = 3.385

##########################################################

working_dir = os.path.dirname(os.path.abspath(__file__))
atlases_dir = os.path.join(working_dir, 'atlases')
fsl_dir = os.path.join(atlases_dir, 'fsl_atlases')
msdtb_dir = os.path.join(atlases_dir, 'msdtb')

hos_putamen_lh_maskpath = os.path.join(
    fsl_dir, 'hos_putamen_lh_mask.nii.gz')
hos_putamen_rh_maskpath = os.path.join(
    fsl_dir, 'hos_putamen_rh_mask.nii.gz')

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

msdtb_putamen_lh_maskpath = os.path.join(msdtb_dir,
                                         'msdtb_putamen_mask_lh.nii.gz')
msdtb_putamen_rh_maskpath = os.path.join(msdtb_dir,
                                         'msdtb_putamen_mask_rh.nii.gz')
msdtb_putamen_conmean = os.path.join(msdtb_dir,
                                     'msdtb_putamen_conmean.npy')
msdtb_putamen_conpval = os.path.join(msdtb_dir,
                                     'msdtb_putamen_conpval.npy')
msdtb_putamen_pscmean = os.path.join(msdtb_dir,
                                     'msdtb_putamen_pscmean.npy')
msdtb_putamen_pscpval = os.path.join(msdtb_dir,
                                     'msdtb_putamen_pscpval.npy')
msdtb_putamen_con_roih = os.path.join(msdtb_dir,
                                  'msdtb_putamen_roi_con_horizontalbarplot.png')
msdtb_putamen_psc_roih = os.path.join(msdtb_dir,
                                  'msdtb_putamen_roi_psc_horizontalbarplot.png')
msdtb_putamen_roiv = os.path.join(msdtb_dir,
                                  'msdtb_putamen_roi_verticalbarplot.png')

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


# ############################## RUN ####################################
  
if __name__ == '__main__':

    # # # ######################## PUTAMEN ##################################
    # # Create Music-SDTB ROIs
    # create_msdtb_roi(tmap_path, tmap_thresh_min,
    #                  hos_putamen_lh_maskpath, hos_putamen_rh_maskpath,
    #                  msdtb_putamen_lh_maskpath, msdtb_putamen_rh_maskpath,
    #                  map_thresh_max=None)

    # # ## Extract data from ROIs in both hemispheres
    # msdtb_putamen_lh_mask = load_img(msdtb_putamen_lh_maskpath)
    # msdtb_putamen_rh_mask = load_img(msdtb_putamen_rh_maskpath)
    # putamen_masks = [msdtb_putamen_lh_mask, msdtb_putamen_rh_mask]
    # compute_rois(putamen_masks, mask_wb, contrasts,
    #              msdtb_putamen_conmean, msdtb_putamen_conpval, 'wcon')
    # compute_rois(putamen_masks, mask_wb, contrasts,
    #              msdtb_putamen_pscmean, msdtb_putamen_pscpval, 'wpsc')

    # ## Plot
    # plot_roi_horizontal(msdtb_putamen_conmean, msdtb_putamen_conpval,
    #                     'Putamen', msdtb_putamen_con_roih)
    # plot_roi_horizontal(msdtb_putamen_pscmean, msdtb_putamen_pscpval,
    #                     'Putamen', msdtb_putamen_psc_roih)


    # # # ###################### CEREBELLUM VI ############################

    # # Create Music-SDTB ROIs
    # create_msdtb_roi(tmap_path, tmap_thresh_min,
    #                  mniflirt_cereb6_lh_maskpath, mniflirt_cereb6_rh_maskpath,
    #                  msdtb_cereb6_lh_maskpath, msdtb_cereb6_rh_maskpath,
    #                  map_thresh_max=None)

    # ## Extract data from ROIs in both hemispheres
    # msdtb_cereb6_lh_mask = load_img(msdtb_cereb6_lh_maskpath)
    # msdtb_cereb6_rh_mask = load_img(msdtb_cereb6_rh_maskpath)
    # cereb6_masks = [msdtb_cereb6_lh_mask, msdtb_cereb6_rh_mask]
    # compute_rois(cereb6_masks, mask_wb, contrasts,
    #              msdtb_cereb6_conmean, msdtb_cereb6_conpval, 'wcon')
    # compute_rois(cereb6_masks, mask_wb, contrasts,
    #              msdtb_cereb6_pscmean, msdtb_cereb6_pscpval, 'wpsc')

    # # Plot
    # plot_roi_horizontal(msdtb_cereb6_conmean, msdtb_cereb6_conpval,
    #                     'Cerebellar Lobule VI', msdtb_cereb6_con_roih)
    # plot_roi_horizontal(msdtb_cereb6_pscmean, msdtb_cereb6_pscpval,
    #                     'Cerebellar Lobule VI', msdtb_cereb6_psc_roih)


    # ##################### CEREBELLUM CRUS I #############################

    # Create Music-SDTB ROIs
    # create_msdtb_roi(tmap_path, 3.,
    #                  mniflirt_crus1_lh_maskpath, mniflirt_crus1_rh_maskpath,
    #                  msdtb_crus1_lh_maskpath, msdtb_crus1_rh_maskpath,
    #                  map_thresh_max=None)

    ## Extract data from ROIs in both hemispheres
    # msdtb_crus1_lh_mask = load_img(msdtb_crus1_lh_maskpath)
    # msdtb_crus1_rh_mask = load_img(msdtb_crus1_rh_maskpath)
    # crus1_masks = [msdtb_crus1_lh_mask, msdtb_crus1_rh_mask]
    # compute_rois(crus1_masks, mask_wb, contrasts,
    #              msdtb_crus1_conmean, msdtb_crus1_conpval, 'wcon')
    # compute_rois(crus1_masks, mask_wb, contrasts,
    #              msdtb_crus1_pscmean, msdtb_crus1_pscpval, 'wpsc')

    # Plot
    # plot_roi_horizontal(msdtb_crus1_conmean, msdtb_crus1_conpval, 'Crus I',
    #                     msdtb_crus1_con_roih)
    # plot_roi_horizontal(msdtb_crus1_pscmean, msdtb_crus1_pscpval, 'Crus I',
    #                     msdtb_crus1_psc_roih)

    # ##################### CEREBELLUM 7b-8a #############################

    # Create Music-SDTB ROIs
    create_msdtb_roi(tmap_path, tmap_thresh_min,
                     mniflirt_cereb7b8a_lh_maskpath,
                     mniflirt_cereb7b8a_rh_maskpath,
                     msdtb_cereb7b8a_lh_maskpath,
                     msdtb_cereb7b8a_rh_maskpath,
                     map_thresh_max=None)

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
