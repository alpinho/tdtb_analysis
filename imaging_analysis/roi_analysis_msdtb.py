"""
This script computes the mean activity in ROIS
for a given set of contrasts of the Music-SDTB Project.

Author: Ana Luisa Pinho

Created: October 2023
Last update: December 2023

Compatibility: Python 3.10.8

"""

import os
import glob
import numpy as np
import pandas as pd

from scipy.ndimage import binary_dilation, binary_erosion
from scipy.stats import ttest_rel

from nilearn.image import load_img, new_img_like, resample_to_img
from nilearn.input_data import NiftiLabelsMasker

import seaborn as sns
from statannotations.Annotator import Annotator
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd
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
        np.random.permutation(np.random.permutation(np.arange(1, 30))), 3)
    dilated_image = binary_dilation(image, mask=gmask,
                                    structure=np.ones((s1, s2, s3)))

    current_count = np.count_nonzero(dilated_image)

    n_iter = 0
    flag = 0
    while current_count != target_count:
        n_iter += 1
        s1, s2, s3 = np.random.choice(
            np.random.permutation(np.random.permutation(np.arange(1, 30))), 3)
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

        if n_iter == 100:
            flag += 1
            target_count += 1
            n_iter = 0

        print('Number of iterations: ', flag*100 + n_iter)

    return dilated_image


def create_group_roimask(con_path, atlas_maskpath, msdtb_maskpath,
                         con_thresh_min=3.385, con_thresh_max=None):
    """
    Compute group ROI: intersection of group encoding map
    thresholded to p < .001 with a pre-specified atlas.
    """

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
    n_voxels = np.count_nonzero(msdtb_val)

    if not n_voxels:
        raise ValueError('N_voxels = 0 ! There is no intersection between' + \
                         'thresholded, group encoding map and atlas-roi mask.')

    # Create msdtb-Putamen mask
    msdtb_mask = new_img_like(atlas_rmask, msdtb_val)

    # Save msdtb-Putamen mask
    msdtb_mask.to_filename(msdtb_maskpath)

    return msdtb_mask, np.count_nonzero(msdtb_val)


def create_iroimask(icon_path, atlas_maskpath, gmask, n_voxels,
                    iroi_maskpath, gcon_path=None, weights=None):
    """
    Create individual ROIs: intersection between the
    unthresholded, individual encoding and a pre-specified atlas mask.

    This individual encoding map can be weight-averaged with the
    group encoding map, prior to the intersection

    The cluster size of the intersection has the same number of voxels
    as the group ROI mask produced by :func:`create_group_roimask`.
    To this end, the voxels displaying larger activity are considered.
    If the intersection results in a cluster size smaller than the group,
    dilation towards the group mask is performed.
    """

    print(icon_path)

    # Load and remove NaNs from contrast map
    icon_val, _ = nonan_map(icon_path)
    gcon_val, gcon_map = nonan_map(gcon_path)

    # Load masks generated from a selected atlas
    atlas_mask = load_img(atlas_maskpath)

    # Resample atlas mask
    atlas_rmask = resample_to_img(atlas_mask, gcon_map,
                                  interpolation='nearest')

    # Get data from atlas mask
    atlas_val = atlas_rmask.get_fdata()

    # Average individual contrast with group contrast...
    con_val = np.average(np.array([icon_val, gcon_val]), axis=0,
                         weights=weights)

    # Get only voxels from the encoding map that lie inside the atlas mask
    bin_msdtb_val = np.where(atlas_val, con_val, 0)

    # Retain an ROI with the same size as the one found at the group level
    iroi_val = np.where(bin_msdtb_val, con_val, 0)
    ithresh = np.sort(np.ravel(iroi_val))[::-1][n_voxels - 1]
    iroi_val[iroi_val < ithresh] = 0

    # Binarize individual ROI
    bin_iroi_val = (iroi_val != 0)

    # Test whether binarized roi is empty
    # If it is empty, take group mask
    if not np.count_nonzero(bin_iroi_val):
        print('Binarized ROI is empty. Taking the group mask.')
        iroi_mask = gmask
    # If it is not empty
    else:
        # Test whether individual has equal or bigger size than group mask
        # If not, do dilation
        if not ithresh:
            # Do dilation restricted to n_voxels
            gmask_val = gmask.get_fdata()
            dilated_mask_val = binary_dilation_with_limit(
                bin_iroi_val, n_voxels, gmask_val)
            print('Dilation performed! ', np.count_nonzero(dilated_mask_val))
            iroi_mask = new_img_like(atlas_rmask, dilated_mask_val)
        # Do intersection
        else:
            print(np.count_nonzero(bin_iroi_val))
            iroi_mask = new_img_like(atlas_rmask, bin_iroi_val)

    # Save individual ROI mask
    if weights != (0,1):
        iroi_mask.to_filename(iroi_maskpath)

    return iroi_mask


def overlay_masks(masks_dir, mask_type, roi):

    for hem in ['lh', 'rh']:
        string = mask_type + '_sub-*_' + roi + '_' + hem + '_mask.nii.gz'
        paths = glob.glob(os.path.join(masks_dir, string))
        images = [load_img(path) for path in paths]
        images_val = [image.get_fdata().tolist() for image in images]
        mask_sum_val = np.sum(images_val, axis=0)
        mask_norm_val = mask_sum_val / len(images_val)
        mask_norm = new_img_like(images[0], mask_norm_val)
        mask_name = mask_type + '_' + roi + '_' + hem + '_mask.nii.gz'
        mask_norm.to_filename(os.path.join(masks_dir, mask_name))


def extract_roi(rmask, task, contrasts, subject_estimates_dir, filetype):

    # # For every contrast
    task_contrasts = []
    for key in list(contrasts.keys()):

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
                       contype, prefix, weights=None):

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
            atlas_dir, atlas + '_' + roi + '_' + hem + '_mask.nii.gz')

        # Intersection of atlas w/ thresholded encoding group tmap
        gencoding_atlasreg_maskpath = os.path.join(
            group_roi_dir,
            'g_msdtb_' + atlas + '_' + roi + '_' + hem + '_mask.nii.gz')

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
                prefix + '_sub-%02d_' + roi + '_' + hem +  \
                '_mask.nii.gz') % subject

            # Create individual ROIs
            subject_encoding_tmap = os.path.join(
                estimates_dir, 'allmain_tasks', 'masked_derivatives_rwls',
                'wspmT_0001_desc-sm8wbmasked.nii')
            irmask = create_iroimask(
                subject_encoding_tmap, atlasreg_maskpath, gmask,
                cluster_size, iencoding_atlasreg_maskpath,
                gcon_path=group_tmap_path, weights=weights)

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
        tasks_allconsubjects = np.moveaxis(subjects_alltaskcon, 0, -1).tolist()
        # ... and append: shape (hemisphere, tasks, contrasts, subjects)
        roi_hems.append(tasks_allconsubjects)

    # Save
    outpath = os.path.join(
        iroicon_dir, prefix + '_' + roi + '_' + contype[1:] + '.npy')
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


def dataframe(data, hemispheres, tasks, contrasts, n_subjects, outpath):

    # input shape: (hemisphere, tasks, contrasts, subjects)
    if isinstance(data, str):
        # ## Open npy files and plot
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

    # Save dataframe
    df.to_csv(outpath)

    return df


def threeway_rmanova(df, output_dir, prefix, roi):
    """
    Compute 2 X 2 X 3 RM-ANOVA

    shape of the data: (hemispheres, tasks, contrasts, subjects)
    """
    # If path is given, open dataframe
    if isinstance(df, str):
        df = pd.read_csv(df)

    # Remove 'All Tasks from Dataframe'
    df = df[df.Task != 'All Tasks']

    # Convert PSC entries to numeric type
    df['PSC'] = df['PSC'].apply(pd.to_numeric)

    # For each hemisphere:
    for hem in ['lh', 'rh']:
        db = pd.DataFrame()
        db = df[df.Hemisphere == hem]

        # Create AnovaRM object
        model = AnovaRM(data=db, depvar='PSC', subject='Subject',
                        within=['Category', 'Modality', 'Task'])

        # Run the 3-way repeated measures ANOVA
        results = model.fit()

        # Perform pairwise Tukey HSD tests
        phoc_category = pairwise_tukeyhsd(db['PSC'], db['Category'], alpha=.05)
        phoc_modality = pairwise_tukeyhsd(db['PSC'], db['Modality'], alpha=.05)
        phoc_task = pairwise_tukeyhsd(db['PSC'], db['Task'], alpha=.05)
        phoc_catmod = pairwise_tukeyhsd(db['PSC'], db['Contrast'], alpha=.05)

        # Create output_dir, if it does not exist
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        # Save results in a TSV file...
        flabel = prefix + '_' + roi + '_' + hem + '_'

        # ... for ANOVA
        results.anova_table.to_csv(
            os.path.join(output_dir, flabel + 'anova.tsv'),
            sep='\t')

        # ... and for posthoc
        phoc_flabel = flabel + 'posthoc_'
        with open(os.path.join(
                output_dir, phoc_flabel + 'category.tsv'), 'w') as fc:
            fc.write(phoc_category.summary().as_csv(sep='\t'))

        with open(os.path.join(
                output_dir, phoc_flabel + 'modality.tsv'), 'w') as fm:
            fm.write(phoc_modality.summary().as_csv(sep='\t'))

        with open(os.path.join(
                output_dir, phoc_flabel + 'task.tsv'), 'w') as ft:
            ft.write(phoc_task.summary().as_csv(sep='\t'))

        with open(os.path.join(
                output_dir, phoc_flabel + 'catmod.tsv'), 'w') as fcon:
            fcon.write(phoc_catmod.summary().as_csv(sep='\t'))


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

# SUBJECTS = [26]

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

filtered_contrasts = {10: 'Auditory Beat',
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
atag_dir = os.path.join(atlases_dir, 'atag_atlas')
ntk_dir = os.path.join(atlases_dir, 'nettekoven_atlas')

msdtb_dir = os.path.join(working_dir, 'roi_analyses')

# putamen_dic = {'hos': 'putamen'}
# cerebellum_dic = {'mniflirt': 'cereb6', 'mniflirt': 'cereb7b8a', 'mniflirt': 'crus1',
#                   'ntk_symmni128': 'd3s', 'ntk_symmni128': 'd3i'}

tags = ['i', 'a', 'g']
weights_list = [(1.,0.), (.5,.5), (0.,1.)]

# ############################## RUN ####################################

if __name__ == '__main__':

    # # # #################### STRIATUM #################################

    for tag, wpair in zip(tags, weights_list):

        # Extraction of individual ROIs...
        # ... using Harvard-Oxford Subcortical atlas
        # str_atag_lnorm_rois = iroicon_estimation(
        #     msdtb_dir, atag_dir,
        #     'atag-lnorm', 'striatum', 'str',
        #     filtered_contrasts, 'wpsc', tag, wpair)

        # Overlay Individualized Masks
        if tag != 'g':
            imasks_folder = os.path.join(
                msdtb_dir,
                'striatum/atag-lnorm/iroi_analysis/individual_rois')
            overlay_masks(imasks_folder, tag, 'str')

        # Open ROI file
        str_atag_lnorm_rois = os.path.join(
            msdtb_dir, 'striatum/atag-lnorm/iroi_analysis',
            tag + '_str_psc.npy')

        # Run ANOVA
        str_atag_lnorm_dfpath = os.path.join(
            msdtb_dir, 'striatum/atag-lnorm/iroi_analysis/',
            tag + '_str_df.csv')
        str_atag_lnorm_df = dataframe(
            str_atag_lnorm_rois,
            ['lh', 'rh'],
            list(tasks.values()),
            list(filtered_contrasts.values()),
            SUBJECTS,
            str_atag_lnorm_dfpath)
        str_anova_path = os.path.join(
            msdtb_dir, 'striatum/atag-lnorm/iroi_analysis/anova')

        threeway_rmanova(str_atag_lnorm_df, str_anova_path, tag, 'str')


    # # ##################### CEREBELLUM-s #############################

    # for tag, wpair in zip(tags, weights_list):

    #     # Extraction of individual ROIs using Nettekoven Symmni128 atlas
    #     cs_ntk_symmni128_rois = iroicon_estimation(
    #         msdtb_dir, ntk_dir,
    #         'ntk_symmni128', 'cerebellum', 'cereb-s',
    #         filtered_contrasts, 'wpsc', tag, wpair)

        # # Overlay Individualized Masks
        # if tag != 'g':
        #     imasks_folder = os.path.join(
        #         msdtb_dir,
        #         'cerebellum/nettekoven_symmni128/iroi_analysis/' + \
        #         'individual_rois')
        #     overlay_masks(imasks_folder, 's', tag)

        # # Open ROI file
        # cs_ntk_symmni128_rois = os.path.join(
        #     msdtb_dir, 'cerebellum/nettekoven_symmni128/iroi_analysis',
        #     tag + '_cerebellum-s_psc.npy')

        # # Run ANOVA
        # cs_ntk_symmni128_dfpath = os.path.join(
        #     msdtb_dir, 'cerebellum/nettekoven_symmni128/iroi_analysis',
        #     tag + '_cerebellum-s_df.csv')
        # cs_ntk_symmni128_df = dataframe(
        #     cs_ntk_symmni128_rois,
        #     ['lh', 'rh'],
        #     list(tasks.values()),
        #     list(filtered_contrasts.values()),
        #     SUBJECTS,
        #     cs_ntk_symmni128_dfpath)
        # cs_anova_path = os.path.join(
        #     msdtb_dir, 'cerebellum/nettekoven_symmni128/iroi_analysis/anova')
        # threeway_rmanova(cs_ntk_symmni128_df, cs_anova_path, tag,
        #                  'cerebellum-s')

        # # Plot
        # d3s_ntk_symmni128_rois = os.path.join(
        #     msdtb_dir,
        #     'cerebellum/ntk_dir/iroi_analysis', tag + '_s_psc.npy')
        # plot_roi_vertical(
        #     d3s_ntk_symmni128_rois,
        #     'cerebellum', 'cerebellum-s', 'ntk_symmni128',
        #     'iroi_analysis', 'psc', tag, hypothesis='greater')
        # plot_roi_vertical(
        #     d3s_ntk_symmni128_rois,
        #     'cerebellum', 'cerebellum-s', 'ntk_symmni128',
        #     'iroi_analysis', 'psc', tag, hypothesis='less')
        # plot_roi_vertical(
        #     d3s_ntk_symmni128_rois,
        #     'cerebellum', 'cerebellum-s', 'ntk_symmni128',
        #     'iroi_analysis', 'psc', tag, hypothesis='two-sided')


    # # ##################### CEREBELLUM-i #############################

    # for tag, wpair in zip(tags, weights_list):

    #     # Extraction of individual ROIs using Nettekoven Symmni128 atlas
    #     ci_ntk_symmni128_rois = iroicon_estimation(
    #         msdtb_dir, ntk_dir,
    #         'ntk_symmni128', 'cerebellum', 'cereb-i',
    #         filtered_contrasts, 'wpsc', tag, wpair)

    #     # Overlay Individualized Masks
    #     if tag != 'g':
    #         imasks_folder = os.path.join(
    #             msdtb_dir,
    #             'cerebellum/nettekoven_symmni128/iroi_analysis/' + \
    #             'individual_rois')
    #         overlay_masks(imasks_folder, 'i', tag)

    #     # Open ROI file
    #     ci_ntk_symmni128_rois = os.path.join(
    #         msdtb_dir, 'cerebellum/ntk_symmni128/iroi_analysis',
    #         tag + '_cerebellum-i_psc.npy')

    #     # Run ANOVA
    #     ci_ntk_symmni128_dfpath = os.path.join(
    #         msdtb_dir, 'cerebellum/nettekoven_symmni128/iroi_analysis',
    #         tag + '_cerebellum-i_df.csv')
    #     ci_ntk_symmni128_df = dataframe(
    #         ci_ntk_symmni128_rois,
    #         ['lh', 'rh'],
    #         list(tasks.values()),
    #         list(filtered_contrasts.values()),
    #         SUBJECTS,
    #         ci_ntk_symmni128_dfpath)
    #     ci_anova_path = os.path.join(
    #         msdtb_dir, 'cerebellum/nettekoven_symmni128/iroi_analysis/anova')
    #     threeway_rmanova(ci_ntk_symmni128_df, ci_anova_path, tag,
    #                      'cerebellum-i')

        # # Plot
        # d3i_ntk_symmni128_rois = os.path.join(
        #     msdtb_dir,
        #     'cerebellum/nettekoven_dir/iroi_analysis', tag + '_i_psc.npy')
        # plot_roi_vertical(
        #     d3i_ntk_symmni128_rois,
        #     'cerebellum', 'cerebellum-i', 'ntk_symmni128',
        #     'iroi_analysis', 'psc', tag, hypothesis='greater')
        # plot_roi_vertical(
        #     d3i_ntk_symmni128_rois,
        #     'cerebellum', 'cerebellum-i', 'ntk_symmni128',
        #     'iroi_analysis', 'psc', tag, hypothesis='less')
        # plot_roi_vertical(
        #     d3i_ntk_symmni128_rois,
        #     'cerebellum', 'cerebellum-i', 'ntk_symmni128',
        #     'iroi_analysis', 'psc', tag, hypothesis='two-sided')
