"""
This script creates masks from Harvard-Oxford subcortical atlas.

Author: Ana Luisa Pinho

Created: November 2023
Last update: January 2024

Compatibility: Python 3.10.8

"""

import os
import numpy as np
import csv

from nilearn.image import load_img, new_img_like


# ############################ FUNCTIONS ################################

def binarize(mask_path, label):

    # Load
    mask = load_img(mask_path)
    mask_val = mask.get_fdata().astype(np.uint8)

    # Binarization
    bin_mask_val = (mask_val == label)

    # Create bin mask
    bin_mask = new_img_like(mask, bin_mask_val)

    return bin_mask


def create_cerebellum_quadrants(lutfile, atlas_path, tag, quadrant_path):
    lutlist = [
        line[0].split() for line in csv.reader(open(lutfile), delimiter='\t')]

    atlas = load_img(atlas_path)
    atlas_val = atlas.get_fdata()
    arr_size = atlas_val.shape
    quadrant_val = np.zeros(arr_size)
    for row in lutlist:
        mask_val = np.zeros(arr_size)
        if row[-1][-2:] == tag:
            label = float(row[0])
            print(label)
            mask_val = (atlas_val == label)
            quadrant_val += mask_val

    quadrant = new_img_like(atlas, quadrant_val)
    quadrant.to_filename(quadrant_path)


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


# ############################# INPUTS ##################################

working_dir = os.path.dirname(os.path.abspath(__file__))
atlases_dir = os.path.join(working_dir, 'atlases')
ntk_dir = os.path.join(atlases_dir, 'nettekoven_atlas')

ntk_symmni128_path = os.path.join(
    ntk_dir, 'atl-NettekovenSym128_space-MNI152NLin2009cSymC_dseg.nii')
ntk_symmni128_lutpath = os.path.join(
    ntk_dir, 'atl-NettekovenSym128.lut')

ntk_symmni128_s_lh_maskpath = os.path.join(
    ntk_dir, 'ntk_symmni128_cereb-s_lh_mask.nii.gz')
ntk_symmni128_s_rh_maskpath = os.path.join(
    ntk_dir, 'ntk_symmni128_cereb-s_rh_mask.nii.gz')
ntk_symmni128_i_lh_maskpath = os.path.join(
    ntk_dir, 'ntk_symmni128_cereb-i_lh_mask.nii.gz')
ntk_symmni128_i_rh_maskpath = os.path.join(
    ntk_dir, 'ntk_symmni128_cereb-i_rh_mask.nii.gz')

ntk_symmni128_lh_maskpath = os.path.join(
    ntk_dir, 'ntk_symmni128_cereb_lh_mask.nii.gz')
ntk_symmni128_rh_maskpath = os.path.join(
    ntk_dir, 'ntk_symmni128_cereb_rh_mask.nii.gz')

ntk_symmni128_maskpath = os.path.join(
    ntk_dir, 'ntk_symmni128_cereb_bh_mask.nii.gz')

# ############################## RUN ####################################

if __name__ == '__main__':

    # Masks of Cerebellum quadrants
    create_cerebellum_quadrants(ntk_symmni128_lutpath, ntk_symmni128_path,
                                'Ls', ntk_symmni128_s_lh_maskpath)
    create_cerebellum_quadrants(ntk_symmni128_lutpath, ntk_symmni128_path,
                                'Rs', ntk_symmni128_s_rh_maskpath)
    create_cerebellum_quadrants(ntk_symmni128_lutpath, ntk_symmni128_path,
                                'Li', ntk_symmni128_i_lh_maskpath)
    create_cerebellum_quadrants(ntk_symmni128_lutpath, ntk_symmni128_path,
                                'Ri', ntk_symmni128_i_rh_maskpath)

    # Masks of Cerebellum s+i for each hemisphere
    combine_masks(ntk_symmni128_s_lh_maskpath, ntk_symmni128_i_lh_maskpath,
                  ntk_symmni128_lh_maskpath)
    combine_masks(ntk_symmni128_s_rh_maskpath, ntk_symmni128_i_rh_maskpath,
                  ntk_symmni128_rh_maskpath)

    # Mask of whole Cerebellum (superior+inferior, but vermis and tertiary)
    combine_masks(ntk_symmni128_lh_maskpath, ntk_symmni128_rh_maskpath,
                  ntk_symmni128_maskpath)
