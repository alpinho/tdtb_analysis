"""
This script creates masks from Harvard-Oxford subcortical atlas.

Author: Ana Luisa Pinho

Created: November 2023
Last update: November 2023

Compatibility: Python 3.10.10

"""

import os
import numpy as np
import csv

from nilearn.image import load_img, new_img_like


# ############################ FUNCTIONS ################################

def binarize(mask_path, label):

    # Load
    mask = load_img(mask_path)
    mask_val = mask.get_fdata()

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


# ############################# INPUTS ##################################

working_dir = os.path.dirname(os.path.abspath(__file__))
atlases_dir = os.path.join(working_dir, 'atlases')
nettekoven_dir = os.path.join(atlases_dir, 'nettekoven_atlas')

atl32_symmni_path = os.path.join(
    nettekoven_dir, 'atl-NettekovenSym32_space-MNI152NLin2009cSymC_dseg.nii')
atl128_symmni_path = os.path.join(
    nettekoven_dir, 'atl-NettekovenSym128_space-MNI152NLin2009cSymC_dseg.nii')
atl128_symmni_lutpath = os.path.join(
    nettekoven_dir, 'atl-NettekovenSym128.lut')

d3l_atl32_symmni_maskpath = os.path.join(
    nettekoven_dir, 'nettekoven_symmni128_d3_lh_mask.nii.gz')
d3r_atl32_symmni_maskpath = os.path.join(
    nettekoven_dir, 'nettekoven_symmni128_d3_rh_mask.nii.gz')

d3ls_atl128_symmni_maskpath = os.path.join(
    nettekoven_dir, 'nettekoven_symmni128_d3s_lh_mask.nii.gz')
d3li_atl128_symmni_maskpath = os.path.join(
    nettekoven_dir, 'nettekoven_symmni128_d3i_lh_mask.nii.gz')
d3lt_atl128_symmni_maskpath = os.path.join(
    nettekoven_dir, 'nettekoven_symmni128_d3t_lh_mask.nii.gz')
d3lv_atl128_symmni_maskpath = os.path.join(
    nettekoven_dir, 'nettekoven_symmni128_d3v_lh_mask.nii.gz')
d3rs_atl128_symmni_maskpath = os.path.join(
    nettekoven_dir, 'nettekoven_symmni128_d3s_rh_mask.nii.gz')
d3ri_atl128_symmni_maskpath = os.path.join(
    nettekoven_dir, 'nettekoven_symmni128_d3i_rh_mask.nii.gz')
d3rt_atl128_symmni_maskpath = os.path.join(
    nettekoven_dir, 'nettekoven_symmni128_d3t_rh_mask.nii.gz')
d3rv_atl128_symmni_maskpath = os.path.join(
    nettekoven_dir, 'nettekoven_symmni128_d3v_rh_mask.nii.gz')

ls_atl32_symmni_maskpath = os.path.join(
    nettekoven_dir, 'nettekoven_symmni128_s_lh_mask.nii.gz')
rs_atl32_symmni_maskpath = os.path.join(
    nettekoven_dir, 'nettekoven_symmni128_s_rh_mask.nii.gz')
li_atl32_symmni_maskpath = os.path.join(
    nettekoven_dir, 'nettekoven_symmni128_i_lh_mask.nii.gz')
ri_atl32_symmni_maskpath = os.path.join(
    nettekoven_dir, 'nettekoven_symmni128_i_rh_mask.nii.gz')

# ############################## RUN ####################################

if __name__ == '__main__':

    # D3 - atl32 MNI symmetrical 
    d3l_atl32_symmni = binarize(atl32_symmni_path, 11)
    d3r_atl32_symmni = binarize(atl32_symmni_path, 27)
    d3l_atl32_symmni.to_filename(d3l_atl32_symmni_maskpath)
    d3r_atl32_symmni.to_filename(d3r_atl32_symmni_maskpath)

    # D3 - atl128 MNI symmetrical 
    d3ls_atl128_symmni = binarize(atl128_symmni_path, 41)
    d3li_atl128_symmni = binarize(atl128_symmni_path, 42)
    d3lt_atl128_symmni = binarize(atl128_symmni_path, 43)
    d3lv_atl128_symmni = binarize(atl128_symmni_path, 44)
    d3rs_atl128_symmni = binarize(atl128_symmni_path, 105)
    d3ri_atl128_symmni = binarize(atl128_symmni_path, 106)
    d3rt_atl128_symmni = binarize(atl128_symmni_path, 107)
    d3rv_atl128_symmni = binarize(atl128_symmni_path, 108)
    d3ls_atl128_symmni.to_filename(d3ls_atl128_symmni_maskpath)
    d3li_atl128_symmni.to_filename(d3li_atl128_symmni_maskpath)
    d3lt_atl128_symmni.to_filename(d3lt_atl128_symmni_maskpath)
    d3lv_atl128_symmni.to_filename(d3lv_atl128_symmni_maskpath)
    d3rs_atl128_symmni.to_filename(d3rs_atl128_symmni_maskpath)
    d3ri_atl128_symmni.to_filename(d3ri_atl128_symmni_maskpath)
    d3rt_atl128_symmni.to_filename(d3rt_atl128_symmni_maskpath)
    d3rv_atl128_symmni.to_filename(d3rv_atl128_symmni_maskpath)

    create_cerebellum_quadrants(atl128_symmni_lutpath, atl128_symmni_path,
                                'Ls', ls_atl32_symmni_maskpath)
    create_cerebellum_quadrants(atl128_symmni_lutpath, atl128_symmni_path,
                                'Rs', rs_atl32_symmni_maskpath)
    create_cerebellum_quadrants(atl128_symmni_lutpath, atl128_symmni_path,
                                'Li', li_atl32_symmni_maskpath)
    create_cerebellum_quadrants(atl128_symmni_lutpath, atl128_symmni_path,
                                'Ri', ri_atl32_symmni_maskpath)
