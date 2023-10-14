"""
This script computes the mean activity in ROIS
for a given set of contrasts of the Music-SDTB Project.

Author: Ana Luisa Pinho

Created: October 2023
Last update: October 2023

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

def plot_mask(mask1, mask_description, output_file, mask2=None, cb=True,
              color_map='viridis'):

    fig = plt.figure(figsize=(6, 2.75))
    # left, bottom, width, height
    axes = plt.axes([0., 0., 1., 1.])

    display = plot_glass_brain(None, display_mode='lzr', black_bg=False,
                               alpha=1., axes=axes, title=mask_description,
                               vmin=0., vmax=1., symmetric_cbar=False)

    cmap = plt.get_cmap(color_map)
    display.add_overlay(mask1, cmap=cmap, colorbar=cb)
    if mask2 is not None:
        display.add_overlay(mask2, cmap=cmap)

    fig.savefig(output_file, dpi=600)


def binarize_equal(mask_path, label):

    # Load
    mask = load_img(mask_path)

    # Threshold
    thresholded_mask_val = mask.get_fdata()
    thresholded_mask_val[thresholded_mask_val != label] = 0

    # Binarization
    bin_mask_val = (thresholded_mask_val != 0)

    # Dilation
    dil_bin_mask_val = ndimage.binary_dilation(bin_mask_val)
    dil_bin_mask_val = dil_bin_mask_val.astype(int)
    dil_bin_mask = new_img_like(mask, dil_bin_mask_val)

    return dil_bin_mask


def binarize_bigger(mask_path, threshold = .33):

    # Load
    mask = load_img(mask_path)

    # Threshold
    thresholded_mask_val = mask.get_fdata()
    thresholded_mask_val[thresholded_mask_val < threshold] = 0

    # Binarization
    bin_mask_val = (thresholded_mask_val != 0)

    # Dilation
    dil_bin_mask_val = ndimage.binary_dilation(bin_mask_val)
    dil_bin_mask_val = dil_bin_mask_val.astype(int)
    dil_bin_mask = new_img_like(mask, dil_bin_mask_val)

    return dil_bin_mask


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


def compute_rois(rmasks, arr_conmean, arr_conpval):
    hemrois_contrasts_mean = []
    hemrois_allpvalues = []
    # # For each hemisphere
    for rmask in rmasks:
        masker = NiftiLabelsMasker(labels_img=rmask, mask_img=mask_gm)
        masker.fit()

        # # For each task design
        allcontrasts_mean = []
        allpvalues = []
        for t, (tk, task) in enumerate(tasks.items()):
            contrasts_mean = []
            pvalues = []

            # # For every contrast
            for key in contrasts.keys():
                contrast_fname = 'wcon_%04d_desc-sm8wbmasked.nii' % key
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
                tstat, pval = stats.ttest_1samp(mask_data, popmean=0.,
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
    for r, roi in enumerate(allcontrasts_mean):
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


def plot_roi_vertical(arr_conmean, arr_conpval, roi_ref, output_file):
    # ## Open npy files and plot
    allcontrasts_mean = np.load(arr_conmean).tolist()
    allpvalues = np.load(arr_conpval).tolist()

    fig = plt.figure(figsize=(30, 35))
    # For each hemisphere
    for r, roi in enumerate(allcontrasts_mean):
        # For each task
        for c, cmean in enumerate(roi):
            # Filter conditions
            filtered_idx = list(filtered_contrasts.keys())
            filtered_cmean = [cmean[i] for i in filtered_idx]
            # plt.axes([left, bottom, width, height])
            ax = plt.axes([.16 + r*.475, .73 - c*.2, .15, .1])
            cnames = list(filtered_contrasts.values())
            x_pos = [.2, .35, .65, .8]
            filtered_pvalues = [allpvalues[r][c][i] for i in filtered_idx]
            pval_labels = pval_label_converter(filtered_pvalues)
            rects = ax.bar(x_pos, filtered_cmean, align='center', width=.1,
                           color=['mediumseagreen', 'gold', 'mediumseagreen', 'gold'])
            # ax.bar_label(rects, labels=pval_labels, padding=3)
            ax.set_xticks(x_pos, labels=cnames, fontsize=16, fontweight='semibold', rotation=45, ha='right')
            ax.xaxis.set_tick_params(width=10.)
            plt.yticks(fontsize=16, fontweight='semibold')
            ax.set_ylim([-.24, .24])
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

working_dir = os.path.dirname(os.path.abspath(__file__))
atlases_dir = os.path.join(working_dir, 'atlases')
data_dir = '/home/analu/diedrichsen_data/data/Cerebellum/music-sdtb/derivatives'
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

filtered_contrasts = {
             9: 'AB vs. AI',
             10: 'VB vs. VI'}

subjects_dir = [os.path.join(data_dir, 'sub-%02d') % sbj for sbj in SUBJECTS]
estimates_dir = [os.path.join(subject_dir, 'estimates')
                 for subject_dir in subjects_dir]

# ### Inputs ###
wb_masking = 'wb'
gm_masking = 'gm'

con_path = os.path.join(data_dir, )
con_relative_path = 'group/allmain_tasks/rfx_onesample_t_rwls_'gm/' + \
    'con_01_Encoding/con_0001.nii'
con_path = os.path.join(data_dir, con_relative_path)

# ############################## RUN ####################################
  
if __name__ == '__main__':

    ## Plot AAL3 mask for Putamen
    plot_mask(aal3_2mm, 'AAL3 2mm', aal3_plot)

    # ######################### Putamen ##################################

    # # ## Binarize masks
    # putamen_aal3_lh_bin = binarize_equal(aal3_2mm, 77.)
    # putamen_aal3_rh_bin = binarize_equal(aal3_2mm, 78.)

    # # ## Resample masks
    # resampled_putamen_aal3_lh_bin = resample_to_img(putamen_aal3_lh_bin,
    #                                                 mask_gm)
    # resampled_putamen_aal3_rh_bin = resample_to_img(putamen_aal3_rh_bin,
    #                                                 mask_gm)

    # # ## Binarize again
    # rr_putamen_aal3_lh_bin = binarize_bigger(resampled_putamen_aal3_lh_bin)
    # rr_putamen_aal3_rh_bin = binarize_bigger(resampled_putamen_aal3_rh_bin)

    # # Plot
    # plot_mask(rr_putamen_aal3_lh_bin,
    #           'Putamen: AAL3',
    #           putamen_aal3_resampled_bin_plot,
    #           rr_putamen_aal3_rh_bin,
    #           cb=False, color_map='viridis_r')

    # # ## Save maks
    # rr_putamen_aal3_lh_bin.to_filename(os.path.join(aal3_plots,
    #                                                 'aal3_putamen_lh.nii.gz'))
    # rr_putamen_aal3_rh_bin.to_filename(os.path.join(aal3_plots,
    #                                                 'aal3_putamen_rh.nii.gz'))

    # ## Extract data from ROIs in both hemispheres
    # rmasks = [rr_putamen_aal3_lh_bin, rr_putamen_aal3_rh_bin]
    # compute_rois(rmasks, putamen_aal3_conmean, putamen_aal3_conpval)

    # Plot
    # plot_roi_horizontal(putamen_aal3_conmean, putamen_aal3_conpval,
    #                     'Putamen: AAL3', putamen_aal3_roi)

    plot_roi_vertical(putamen_aal3_conmean, putamen_aal3_conpval,
                      'Putamen: AAL3', putamen_aal3_roiv)

    # ################## Cerebellum Crus I ##############################

    # # ## Binarize masks
    # crus1_aal3_lh_bin = binarize_equal(aal3_2mm, 95.)
    # crus1_aal3_rh_bin = binarize_equal(aal3_2mm, 96.)

    # # ## Resample masks
    # resampled_crus1_aal3_lh_bin = resample_to_img(crus1_aal3_lh_bin, mask_gm)
    # resampled_crus1_aal3_rh_bin = resample_to_img(crus1_aal3_rh_bin, mask_gm)

    # # ## Binarize again
    # rr_crus1_aal3_lh_bin = binarize_bigger(resampled_crus1_aal3_lh_bin,
    #                                        threshold = 1.)
    # rr_crus1_aal3_rh_bin = binarize_bigger(resampled_crus1_aal3_rh_bin,
    #                                        threshold = 1.)

    # # Plot
    # plot_mask(rr_crus1_aal3_lh_bin,
    #           'Cerebellum Crus I: AAL3',
    #           crus1_aal3_resampled_bin_plot,
    #           rr_crus1_aal3_rh_bin,
    #           cb=False, color_map='viridis_r')

    # # ## Save maks
    # rr_crus1_aal3_lh_bin.to_filename(os.path.join(aal3_plots,
    #                                               'aal3_crus1_lh.nii.gz'))
    # rr_crus1_aal3_rh_bin.to_filename(os.path.join(aal3_plots,
    #                                               'aal3_crus1_rh.nii.gz'))

    # ## Extract data from ROIs in both hemispheres
    # rmasks = [rr_crus1_aal3_lh_bin, rr_crus1_aal3_rh_bin]
    # compute_rois(rmasks, crus1_aal3_conmean, crus1_aal3_conpval)

    # Plot
    plot_roi_horizontal(crus1_aal3_conmean, crus1_aal3_conpval,
                        'Cerebellum Crus I: AAL3', crus1_aal3_roi)

    plot_roi_vertical(crus1_aal3_conmean, crus1_aal3_conpval,
                      'Crus I: AAL3', crus1_aal3_roiv)

    # ################## Cerebellum VI ###############################

    # # ## Binarize masks
    # cereb6_aal3_lh_bin = binarize_equal(aal3_2mm, 103.)
    # cereb6_aal3_rh_bin = binarize_equal(aal3_2mm, 104.)

    # # ## Resample masks
    # resampled_cereb6_aal3_lh_bin = resample_to_img(cereb6_aal3_lh_bin, mask_gm)
    # resampled_cereb6_aal3_rh_bin = resample_to_img(cereb6_aal3_rh_bin, mask_gm)

    # # ## Binarize again
    # rr_cereb6_aal3_lh_bin = binarize_bigger(resampled_cereb6_aal3_lh_bin,
    #                                         threshold = 1.)
    # rr_cereb6_aal3_rh_bin = binarize_bigger(resampled_cereb6_aal3_rh_bin,
    #                                         threshold = 1.)

    # # Plot
    # plot_mask(rr_cereb6_aal3_lh_bin,
    #           'Cerebellum VI: AAL3',
    #           cereb6_aal3_resampled_bin_plot,
    #           rr_cereb6_aal3_rh_bin,
    #           cb=False, color_map='viridis_r')

    # # ## Save maks
    # rr_cereb6_aal3_lh_bin.to_filename(os.path.join(aal3_plots,
    #                                                'aal3_cereb6_lh.nii.gz'))
    # rr_cereb6_aal3_rh_bin.to_filename(os.path.join(aal3_plots,
    #                                                'aal3_cereb6_rh.nii.gz'))

    # ## Extract data from ROIs in both hemispheres
    # rmasks = [rr_cereb6_aal3_lh_bin, rr_cereb6_aal3_rh_bin]
    # compute_rois(rmasks, cereb6_aal3_conmean, cereb6_aal3_conpval)

    # Plot
    plot_roi_horizontal(cereb6_aal3_conmean, cereb6_aal3_conpval,
                        'Cerebellum VI: AAL3', cereb6_aal3_roi)

    plot_roi_vertical(cereb6_aal3_conmean, cereb6_aal3_conpval,
                      'Cerebellum VI: AAL3', cereb6_aal3_roiv)
