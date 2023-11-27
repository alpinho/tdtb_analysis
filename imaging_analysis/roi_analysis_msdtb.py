"""
This script computes the mean activity in ROIS
for a given set of contrasts of the Music-SDTB Project.

Author: Ana Luisa Pinho

Created: October 2023
Last update: November 2023

Compatibility: Python 3.10.8

"""

import os
import numpy as np
import pandas as pd

from scipy.ndimage import binary_dilation, binary_erosion
from scipy.stats import ttest_rel

from nilearn.image import load_img, new_img_like, resample_to_img
from nilearn.input_data import NiftiLabelsMasker

import seaborn as sns
from statannotations.Annotator import Annotator
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


def binary_dilation_with_limit(image, target_count, gmask):
    s1, s2, s3 = np.random.choice(
        np.random.permutation(np.random.permutation(np.arange(1, 10))), 3)
    dilated_image = binary_dilation(image, mask=gmask,
                                    structure=np.ones((s1, s2, s3)))

    current_count = np.count_nonzero(dilated_image)

    while current_count != target_count:
        s1, s2, s3 = np.random.choice(
            np.random.permutation(np.random.permutation(np.arange(1, 10))), 3)
        if current_count < target_count:
            dilated_image = binary_dilation(dilated_image, mask=gmask,
                                            structure=np.ones((s1, s2, s3)))
        elif current_count > target_count:
            dilated_image = binary_erosion(dilated_image, mask=gmask,
                                           structure=np.ones((s1, s2, s3)))
        else:
            pass
        print(current_count)
        current_count = np.count_nonzero(dilated_image)

    return dilated_image


def create_group_roimask(con_path, atlas_maskpath, msdtb_maskpath,
                         con_thresh_min=3.385, con_thresh_max=None):

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

    return msdtb_mask, np.count_nonzero(msdtb_val)


def create_individual_roimask(individual_con_path, atlas_maskpath,
                              gmask, n_voxels, individual_roi_maskpath):

    print(individual_con_path)

    # Remove NaNs from contrast map
    con_val, con_map = nonan_map(individual_con_path)

    # Load masks generated from a selected atlas
    atlas_mask = load_img(atlas_maskpath)

    # Resample atlas mask
    atlas_rmask = resample_to_img(atlas_mask, con_map,
                                  interpolation='nearest')

    # Get data from atlas mask
    atlas_val = atlas_rmask.get_fdata()

    # Intersection of contrast map w/ atlas mask
    bin_msdtb_val = np.logical_and(
        con_val.astype(bool), atlas_val.astype(bool)).astype(int)

    # Retain an ROI with the same size as the one found at the group level
    individual_roi_val = np.where(bin_msdtb_val, con_val, 0)
    individual_thresh = np.sort(
        np.ravel(individual_roi_val))[::-1][n_voxels - 1]
    individual_roi_val[individual_roi_val < individual_thresh] = 0

    # Binarize individual ROI
    bin_individual_roi_val = (individual_roi_val != 0)

    # Test whether binarized roi is empty
    if not np.count_nonzero(bin_individual_roi_val):
        print(np.count_nonzero(bin_individual_roi_val))
        individual_roi_mask = gmask
    else:
        # Test whether individual has equal or bigger size than group mask
        # If not,
        if not individual_thresh:
            # Do dilation restricted to n_voxels
            gmask_val = gmask.get_fdata()
            dilated_mask_val = binary_dilation_with_limit(
                bin_individual_roi_val, n_voxels, gmask_val)
            print('Dilation performed! ', np.count_nonzero(dilated_mask_val))
            individual_roi_mask = new_img_like(atlas_rmask, dilated_mask_val)
        else:
            print(np.count_nonzero(bin_individual_roi_val))
            individual_roi_mask = new_img_like(atlas_rmask,
                                               bin_individual_roi_val)

    # Save individual ROI mask
    individual_roi_mask.to_filename(individual_roi_maskpath)

    return individual_roi_mask


def extract_roi(rmask, task, contrasts, subject_estimates_dir, filetype):

    # # For every contrast
    task_contrasts = []
    # Do not count w/ Encoding contrast
    for key in list(contrasts.keys())[1:]:

        contrast_fname = filetype + '_%04d_desc-sm8wbmasked.nii' % key

        masker = NiftiLabelsMasker(labels_img=rmask)
        masker.fit()

        masked_con = os.path.join(subject_estimates_dir, task,
                                  'masked_derivatives_rwls',
                                  contrast_fname)
        # print(np.array(masked_con))

        # Extract mean average of contrasts effect-size in ROI...
        # ... for a certain participant
        mask_data = masker.transform(masked_con)[0][0]

        task_contrasts.append(mask_data)

    return task_contrasts


def iroicon_estimation(main_dir, atlas_dir, atlas, region, roi, contrasts_dic,
                       contype):

    roi_dir = os.path.join(main_dir, region, atlas)
    group_roi_dir = os.path.join(roi_dir, 'group_rois')
    iroicon_dir = os.path.join(roi_dir, 'iroi_analysis')
    iroimasks_dir = os.path.join(iroicon_dir, 'individual_rois')

    if not os.path.exists(group_roi_dir):
        os.makedirs(group_roi_dir)

    if not os.path.exists(iroicon_dir):
        os.mkdir(iroicon_dir)

    if not os.path.exists(iroimasks_dir):
        os.mkdir(iroimasks_dir)

    # ### For each hemisphere ###
    roi_hems = []
    for hem in ['lh', 'rh']:
        atlasreg_maskpath = os.path.join(
            atlas_dir, atlas + '_' + region + '_' + hem + '_mask.nii.gz')

        # Intersection of atlas w/ thresholded encoding group tmap
        gencoding_atlasreg_maskpath = os.path.join(
            group_roi_dir,
            'gmsdtb-' + atlas + '_' + roi + '_mask_' + hem + '.nii.gz')

        if os.path.isfile(gencoding_atlasreg_maskpath):
            gmask = load_img(gencoding_atlasreg_maskpath)
            cluster_size = np.count_nonzero(gmask.get_fdata())
        else:
            gmask, cluster_size = create_group_roimask(
                group_tmap_path,
                atlasreg_maskpath,
                gencoding_atlasreg_maskpath)

        # ### For each subject ###
        subjects_alltaskcon = []
        for subject in SUBJECTS:
            subject_dir = os.path.join(data_dir, 'sub-%02d') % subject
            estimates_dir = os.path.join(subject_dir, 'estimates')
            iencoding_atlasreg_maskpath = os.path.join(
                iroimasks_dir,
                roi + '_mask_sub-%02d_' + hem + '.nii.gz') % subject

            # Create individual ROIs
            subject_encoding_tmap = os.path.join(
                estimates_dir, 'allmain_tasks', 'masked_derivatives_rwls',
                'wspmT_0001_desc-sm8wbmasked.nii')
            irmask = create_individual_roimask(
                subject_encoding_tmap, atlasreg_maskpath,
                gmask, cluster_size, iencoding_atlasreg_maskpath)

            # ### For each task ###
            itasks_contrasts = []
            for task in tasks.keys():
                # Extract individual ROIs
                itask_contrasts = extract_roi(
                    irmask, task, contrasts_dic, estimates_dir, contype)
                # ... and append: shape (tasks, contrasts)
                itasks_contrasts.append(itask_contrasts)

            # Append: shape (subjects, tasks, contrasts)
            subjects_alltaskcon.append(itasks_contrasts)

        # Change shape: (tasks, contrasts, subjects)
        tasks_allconsubjects = np.moveaxis(subjects_alltaskcon, 0, -1)
        # ... and append: shape (hemisphere, tasks, contrasts, subjects)
        roi_hems.append(tasks_allconsubjects)

    # Save
    outpath = os.path.join(iroicon_dir, region + '_' + contype[1:] + '.npy')
    if os.path.exists(outpath):
        os.remove(outpath)
    np.save(outpath, roi_hems, allow_pickle=False)

    return roi_hems


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
    # input shape: (hemisphere, tasks, contrasts, subjects)
    if isinstance(arr_conmean, str):
        # ## Open npy files and plot
        arr_conmean = np.load(arr_conmean).tolist()

    # Names of Contrasts
    cnames = list(filtered_contrasts.values())[1:]
    n_pairs = len(np.arange(len(cnames))[::2])

    fig, ax = plt.subplots(1, n_pairs)

    # left   # the left side of the subplots of the figure
    # right  # the right side of the subplots of the figure
    # bottom # the bottom of the subplots of the figure
    # top    # the top of the subplots of the figure
    # wspace # the amount of width reserved for blank space between subplots
    # hspace # the amount of height reserved for white space between subplots
    plt.subplots_adjust(left=.12, right=.99, bottom=.15, wspace=.075)

    for c, cidx in enumerate(np.arange(len(cnames))[::2]):
        con1 = arr_conmean[0][0][cidx]
        con2 = arr_conmean[0][0][cidx+1]
        data_list = con1 +con2

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
        b = sns.barplot(ax=ax[c],
                        x=x,
                        y=y,
                        data=df,
                        palette=[sns.color_palette("colorblind")[2],
                                sns.color_palette("colorblind")[8]],
                        estimator=np.mean,
                        ci=95, # errorbar=('ci', 95), # 1.96 * standard error (95% confidence interval)
                        errcolor="black", errwidth=1.5, capsize = 0.2, alpha=0.5)

        # Compute p-value
        _, pvalue = ttest_rel(con1, con2, alternative='greater')
        # _, pvalue = ttest_rel(con1, con2, alternative='less')
        # _, pvalue = ttest_rel(con1, con2, alternative='two-sided')
        print(pvalue)

        # Annotate
        pair = tuple([[(cname1), (cname2)]])
        annotator = Annotator(ax[c], pair, data=df, x=x, y=y)
        annotator.configure(test=None,
                            text_format="star", # text_format="simple"
                            # test_short_name="pttest", # if former is "simple"
                            fontsize=10.)

        annotator.set_pvalues([pvalue])
        annotator.annotate()

        # Remove x-label
        b.set(xlabel=None)

        # Rotate xtick labels
        ax[c].set_xticklabels(ax[c].get_xticklabels(), rotation=20, ha='right',
                              fontsize=8)

        # Hide the right and top spines
        ax[c].spines['right'].set_visible(False)
        ax[c].spines['top'].set_visible(False)

        if c > 0:
            # ... remove y labels and y ticks
            ax[c].axes.get_yaxis().set_visible(False)
            # ... remove y frame
            ax[c].spines['left'].set_visible(False)

        # Title
        plt.title('Production', size=14, x=-1.25, fontweight='bold')


    output_folder = os.path.join(msdtb_dir, 'putamen/hos/iroi_analysis')
    fname = 'putamen_psc'
    # Save figure
    plt.savefig(os.path.join(output_folder, fname + '.pdf'))

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

# SUBJECTS = [29]

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

filtered_contrasts = {1: 'Encoding',
                      2: 'Auditory Encoding',
                      3: 'Visual Encoding',
                      6: 'Beat',
                      7: 'Interval',
                      10: 'Auditory Beat',
                      11: 'Auditory Interval',
                      14: 'Visual Beat',
                      15: 'Visual Interval'}

wb_masking = 'wb'
gm_masking = 'gm'

group_relative_path = 'group/allmain_tasks/rfx_onesample_t_rwls_'+ \
    wb_masking + '/con_01_Encoding/'

group_con_relative_path = group_relative_path + 'con_0001.nii'
group_con_path = os.path.join(data_dir, group_con_relative_path)

group_tmap_relative_path = group_relative_path  + 'spmT_0001.nii'
group_tmap_path = os.path.join(data_dir, group_tmap_relative_path)

##########################################################

working_dir = os.path.dirname(os.path.abspath(__file__))

atlases_dir = os.path.join(working_dir, 'atlases')
fsl_dir = os.path.join(atlases_dir, 'fsl_atlases')
nettekoven_dir = os.path.join(atlases_dir, 'nettekoven')

msdtb_dir = os.path.join(working_dir, 'roi_analyses')

# msdtb_putamen_con_roih = os.path.join(
#     msdtb_dir, 'msdtb_putamen_roi_con_horizontalbarplot.png')
# msdtb_putamen_psc_roih = os.path.join(
#     msdtb_dir, 'msdtb_putamen_roi_psc_horizontalbarplot.png')
# msdtb_putamen_roiv = os.path.join(
#     msdtb_dir, 'msdtb_putamen_roi_verticalbarplot.png')

# putamen_dic = {'hos': 'putamen'}
# cerebellum_dic = {'mniflirt': 'cereb6', 'mniflirt': 'cereb7b8a', 'mniflirt': 'crus1',
#                   'nettekoven_symmni128': 'd3s', 'nettekoven_symmni128': 'd3i'}


# ############################## RUN ####################################

if __name__ == '__main__':

    # # # ######################## PUTAMEN ##################################

    # ROI extraction using Harvard-Oxford Subcortical atlas
    # putamen_hos_rois = iroicon_estimation(
    #     msdtb_dir, fsl_dir, 'hos', 'putamen', 'putamen', filtered_contrasts,
    #     'wpsc')

    # Plot
    putamen_hos_rois = os.path.join(msdtb_dir,
                                    'putamen/hos/iroi_analysis/putamen_psc.npy')
    plot_roi_vertical(putamen_hos_rois)


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
