"""
This script computes the mean activity in ROIS
for a given set of contrasts of the Music-SDTB Project.

Author: Ana Luisa Pinho

Created: October 2023
Last update: February 2025

Compatibility: Python 3.10.14

How to run the script:
python roi_extraction_msdtb.py <encoding_type>
Example:
python roi_extraction_msdtb.py auditory

"""

import os
import sys
import glob
import re
import numpy as np

from scipy.ndimage import binary_dilation, binary_erosion

from nilearn.image import load_img, new_img_like, resample_to_img
from nilearn.input_data import NiftiLabelsMasker


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


def combine_masks(maskpath1, maskpath2, combined_maskpath):

    # Load
    mask1 = load_img(maskpath1)
    mask2 = load_img(maskpath2)

    # Get data
    mask1_val = mask1.get_fdata().astype(np.uint8)
    mask2_val = mask2.get_fdata().astype(np.uint8)

    # Merge masks in one single file
    combined_mask_val = mask1_val + mask2_val
    combined_mask_val[combined_mask_val == 2] = 1
    combined_mask = new_img_like(mask1, combined_mask_val)

    # Save file
    combined_mask.to_filename(combined_maskpath)


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

    # Create msdtb mask
    msdtb_mask = new_img_like(atlas_rmask, msdtb_val)

    # Save msdtb mask
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

    iroi_mask.to_filename(iroi_maskpath)

    return iroi_mask


def overlay_masks(pdir, mask_type, roi):

    masks_dir = os.path.join(pdir, 'individual_roi_masks')
    output_dir = os.path.join(pdir, 'overlaid_masks')

    # Create output_dir, if it does not exist
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for hem in ['lh', 'rh']:
        string = mask_type + '_sub-*_' + roi + '_' + hem + '_mask.nii.gz'
        paths = glob.glob(os.path.join(masks_dir, string))
        images = [load_img(path) for path in paths]
        images_val = [image.get_fdata().tolist() for image in images]
        mask_sum_val = np.sum(images_val, axis=0)
        mask_norm_val = mask_sum_val / len(images_val)
        mask_norm = new_img_like(images[0], mask_norm_val)
        mask_name = mask_type + '_' + roi + '_' + hem + '_mask.nii.gz'
        mask_norm.to_filename(os.path.join(output_dir, mask_name))


def extract_roi(rmask, task, contrasts, subject_estimates_dir,
                derivatives_folder, filetype):

    # # For every contrast
    task_contrasts = []
    for key in list(contrasts.keys()):

        contrast_fname = filetype + '_%04d_desc-sm8wbmasked.nii' % key

        masker = NiftiLabelsMasker(labels_img=rmask)
        masker.fit()

        masked_con = os.path.join(subject_estimates_dir, task,
                                  derivatives_folder, contrast_fname)
        # print(np.array(masked_con))

        # Extract mean average of contrasts effect-size in ROI...
        # ... for a certain participant
        mask_data = masker.transform(masked_con)[0][0]

        task_contrasts.append(mask_data)

    return task_contrasts


def iroicon_estimation(main_dir, atlas_dir, atlas, region, roi,
                       group_tmap_path, contrasts_dic, contype, prefix,
                       derivatives_folder, mask, weights=None, subregion=False,
                       hems=['lh', 'rh', 'bh']):

    if subregion:
        roi_dir = os.path.join(main_dir, region, atlas, roi)
    else:
        roi_dir = os.path.join(main_dir, region, atlas)

    groi_dir = os.path.join(roi_dir, 'group_roi_masks')
    iroi_dir = os.path.join(roi_dir, 'individual_roi_masks')
    roiextr_dir = os.path.join(roi_dir, 'rois_extraction')

    if not os.path.exists(groi_dir):
        os.makedirs(groi_dir)

    if not os.path.exists(iroi_dir):
        os.mkdir(iroi_dir)

    if not os.path.exists(roiextr_dir):
        os.mkdir(roiextr_dir)

    # ### For each hemisphere ###
    roi_hems = []
    for hem in hems:
        atlasreg_maskpath = os.path.join(
            atlas_dir, atlas + '_' + roi + '_' + hem + '_mask.nii.gz')

        # Create group ROI mask:
        # Intersection of atlas w/ thresholded encoding group tmap
        gencoding_atlasreg_maskpath = os.path.join(
            groi_dir,
            'g_msdtb_' + atlas + '_' + roi + '_' + hem + '_mask.nii.gz')

        if os.path.isfile(gencoding_atlasreg_maskpath):
            gmask = load_img(gencoding_atlasreg_maskpath)
            if hem in ['lh', 'rh']:
                cluster_size = np.count_nonzero(gmask.get_fdata())
        else:
            if hem in ['lh', 'rh']:
                gmask, cluster_size = create_group_roimask(
                    group_tmap_path,
                    atlasreg_maskpath,
                    gencoding_atlasreg_maskpath)
            else:
                assert hem == 'bh'
                gmask_lh = os.path.join(
                    groi_dir,
                    'g_msdtb_' + atlas + '_' + roi + '_lh_mask.nii.gz')
                gmask_rh = os.path.join(
                    groi_dir,
                    'g_msdtb_' + atlas + '_' + roi + '_rh_mask.nii.gz')
                combine_masks(gmask_lh, gmask_rh, gencoding_atlasreg_maskpath)
                gmask = load_img(gencoding_atlasreg_maskpath)

        # ### For each subject ###
        subjects_alltaskcon = []
        for subject in SUBJECTS:
            subject_dir = os.path.join(data_dir, 'sub-%02d') % subject
            estimates_dir = os.path.join(subject_dir, 'estimates')
            iencoding_atlasreg_maskpath = os.path.join(
                iroi_dir,
                prefix + '_sub-%02d_' + roi + '_' + hem +  \
                '_mask.nii.gz') % subject

            # Create individual ROI masks
            if weights == (0.,1.):
                irmask = gmask
            else:
                idx = [match.end()
                       for match in re.finditer('con_', group_tmap_path)][0]
                con_id = int(group_tmap_path[idx: idx+2])
                subject_encoding_tmap = os.path.join(
                    estimates_dir, 'allmain_tasks', derivatives_folder,
                    'wspmT_%04d' % con_id + '_desc-sm8' + mask + 'masked.nii')

                if hem in ['lh', 'rh']:
                    irmask = create_iroimask(
                        subject_encoding_tmap, atlasreg_maskpath, gmask,
                        cluster_size, iencoding_atlasreg_maskpath,
                        gcon_path=group_tmap_path, weights=weights)
                else:
                    assert hem == 'bh'
                    imask_lh = os.path.join(iroi_dir, prefix + '_sub-%02d_' + \
                                            roi + '_lh_mask.nii.gz') % subject
                    imask_rh = os.path.join(iroi_dir, prefix + '_sub-%02d_' + \
                                            roi + '_rh_mask.nii.gz') % subject
                    combine_masks(imask_lh, imask_rh,
                                  iencoding_atlasreg_maskpath)
                    irmask = load_img(iencoding_atlasreg_maskpath)

            # ### For each task ###
            itasks_contrasts = []
            for task in tasks.keys():
                # Extract individual ROIs
                itask_contrasts = extract_roi(
                    irmask, task, contrasts_dic, estimates_dir,
                    derivatives_folder, contype)
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
        roiextr_dir, prefix + '_' + roi + '_' + contype[1:] + '.npy')
    if os.path.exists(outpath):
        os.remove(outpath)
    np.save(outpath, roi_hems, allow_pickle=False)


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

selected_contrasts = {10: 'Auditory Beat',
                      11: 'Auditory Interval',
                      14: 'Visual Beat',
                      15: 'Visual Interval'}

model = 'rwls' # 'rwls'; or 'standard' (no rwls)
masking = 'wb' # 'wb' for whole-brain; 'gm' for grey matter
design = 'dbb' # 'dbb' if decision and response are modeled together;
               # 'drbb' if otherwise
hrf_cutoff = 'hrf128' # 'hrf128' or 'hrf42'
# hrf_cutoff = 'hrf128_timederiv'
# hrf_cutoff = 'hrf128_timedispderiv'

individual_derivatives_folder = 'masked_derivatives_' + model + '_' + \
    design + '_' + hrf_cutoff

# group_relative_path = 'group/allmain_tasks/rfx_onesample_t_' + model + '_'+ \
#     masking + '_' + design + '_' + hrf_cutoff

group_relative_path = 'group/allmain_tasks/rfx_onesample_t'

group_encoding_folder = 'con_01_Encoding'
gtmap_encoding = os.path.join(data_dir, group_relative_path,
                              group_encoding_folder, 'spmT_0001.nii')
group_audioencoding_folder = 'con_02_Auditory_Encoding'
gtmap_audioencoding = os.path.join(data_dir, group_relative_path,
                                   group_audioencoding_folder, 'spmT_0001.nii')
group_visualencoding_folder = 'con_03_Visual_Encoding'
gtmap_visualencoding = os.path.join(data_dir, group_relative_path,
                                    group_visualencoding_folder, 'spmT_0001.nii')

working_dir = os.path.dirname(os.path.abspath(__file__))

atlases_dir = os.path.join(working_dir, 'atlases')
fsl_dir = os.path.join(atlases_dir, 'fsl_atlases')
atag_dir = os.path.join(atlases_dir, 'atag_atlas')
ntk_dir = os.path.join(atlases_dir, 'nettekoven_atlas')
hmat_dir = os.path.join(atlases_dir, 'hmat_atlas')

# roi_dir = os.path.join(working_dir, 'roi_analyses_' + model + '_' + hrf_cutoff)
roi_dir = os.path.join(working_dir, 'roi_analyses_vdmcorr')

# All ROIs: 7 ROIs
atlas_dirnames = [fsl_dir, ntk_dir, ntk_dir, ntk_dir,
                  hmat_dir, hmat_dir, hmat_dir]
atlas_names = ['hos', 'ntk_symmni128', 'ntk_symmni128', 'ntk_symmni128',
               'hmat', 'hmat', 'hmat']
region_names = ['dorsal_striatum', 'cerebellum', 'cerebellum', 'cerebellum',
                'motor_area', 'motor_area', 'motor_area']
roi_names = ['dstr', 'cereb-s', 'cereb-i', 'cereb',
             'pmd', 'sma', 'presma']

# atlas_dirnames = [fsl_dir, ntk_dir]
# atlas_names = ['hos', 'ntk_symmni128']
# region_names = ['dorsal_striatum', 'cerebellum']
# roi_names = ['dstr', 'cereb']

tags = ['i', 'a', 'g']

# Tuple: (individual_weight, average_weight, group_weight)
weights_list = [(1.,0.), (.5,.5), (0.,1.)]


# ############################## RUN ####################################

if __name__ == '__main__':

    # ========= SET COMMAND-LINE ARGUMENTS TO BE PASSED TO THE SCRIPT ====
    assert(len(sys.argv) > 1), "No arg was introduced. " + \
                               "You must pass a valid arg to the script."

    encoding_type = sys.argv[1]

    if encoding_type == 'all':
        gtmap = gtmap_encoding
        filtered_contrasts = selected_contrasts
        msdtb_dir = os.path.join(roi_dir, 'all')
    elif encoding_type == 'auditory':
        gtmap = gtmap_audioencoding
        filtered_contrasts = {key: selected_contrasts[key]
                              for key in [10, 11] if key in selected_contrasts}
        msdtb_dir = os.path.join(roi_dir, 'auditory')
    elif encoding_type == 'visual':
        gtmap = gtmap_visualencoding
        filtered_contrasts = {key: selected_contrasts[key]
                              for key in [14, 15] if key in selected_contrasts}
        msdtb_dir = os.path.join(roi_dir, 'visual')
    else:
        raise ValueError("The argument must be 'all', 'auditory' or 'visual'.")

    # Create main directory if does not exist
    if not os.path.exists(msdtb_dir):
        os.makedirs(msdtb_dir)

    # ###### Extract ROIs and compute overlay of individual masks ######
    for tag, wpair in zip(tags, weights_list):
        for atlas_dirname, atlas_name, region_name, roi_name in zip(
                atlas_dirnames, atlas_names, region_names, roi_names):

            # Extraction of individual ROIs
            if region_name == 'dorsal_striatum':
                iroicon_estimation(
                    msdtb_dir, atlas_dirname, atlas_name, region_name,
                    roi_name, gtmap, filtered_contrasts, 'wpsc', tag,
                    individual_derivatives_folder, masking, weights=wpair)
            else:
                iroicon_estimation(
                    msdtb_dir, atlas_dirname, atlas_name, region_name,
                    roi_name, gtmap, filtered_contrasts, 'wpsc', tag,
                    individual_derivatives_folder, masking, weights=wpair,
                    subregion=True)

            # Define output-dir path
            if region_name == 'dorsal_striatum':
                outdir = os.path.join(msdtb_dir, region_name, atlas_name)
            else:
                outdir = os.path.join(msdtb_dir, region_name, atlas_name,
                                      roi_name)

            # ##########################################################
            # # Overlay Individualized Masks for each ROI
            if tag != 'g':
                overlay_masks(outdir, tag, roi_name)
