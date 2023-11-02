"""
This script creates masks from Harvard-Oxford subcortical atlas.

Author: Ana Luisa Pinho

Created: November 2023
Last update: November 2023

Compatibility: Python 3.10.10

"""

import os
import numpy as np

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


# ############################# INPUTS ##################################

working_dir = os.path.dirname(os.path.abspath(__file__))
atlases_dir = os.path.join(working_dir, 'atlases')
nettekoven_dir = os.path.join(atlases_dir, 'nettekoven')

atl32_symmni_path = os.path.join(
    nettekoven_dir, 'atl-NettekovenSym32_space-MNI152NLin2009cSymC_dseg.nii')
atl128_symmni_path = os.path.join(
    nettekoven_dir, 'atl-NettekovenSym128_space-MNI152NLin2009cSymC_dseg.nii')

d3l_atl32_symmni_maskpath = os.path.join(
    nettekoven_dir, 'd3l_atl32_symmni_mask.nii.gz')
d3r_atl32_symmni_maskpath = os.path.join(
    nettekoven_dir, 'd3r_atl32_symmni_mask.nii.gz')

d3ls_atl128_symmni_maskpath = os.path.join(
    nettekoven_dir, 'd3ls_atl128_symmni_mask.nii.gz')
d3li_atl128_symmni_maskpath = os.path.join(
    nettekoven_dir, 'd3li_atl128_symmni_mask.nii.gz')
d3lt_atl128_symmni_maskpath = os.path.join(
    nettekoven_dir, 'd3lt_atl128_symmni_mask.nii.gz')
d3lv_atl128_symmni_maskpath = os.path.join(
    nettekoven_dir, 'd3lv_atl128_symmni_mask.nii.gz')
d3rs_atl128_symmni_maskpath = os.path.join(
    nettekoven_dir, 'd3rs_atl128_symmni_mask.nii.gz')
d3ri_atl128_symmni_maskpath = os.path.join(
    nettekoven_dir, 'd3ri_atl128_symmni_mask.nii.gz')
d3rt_atl128_symmni_maskpath = os.path.join(
    nettekoven_dir, 'd3rt_atl128_symmni_mask.nii.gz')
d3rv_atl128_symmni_maskpath = os.path.join(
    nettekoven_dir, 'd3rv_atl128_symmni_mask.nii.gz')

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
