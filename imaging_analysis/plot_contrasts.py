"""
This script plots contrasts in glass brains

Author: Ana Luisa Pinho

Created: September 2023
Last update: October 2023

Compatibility: Python 3.10.10

"""

import os
import numpy as np

from scipy import ndimage, stats

from nilearn.plotting import plot_glass_brain, plot_stat_map
from nilearn.image import load_img, new_img_like
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker

from matplotlib import pyplot as plt


# ############################ FUNCTIONS ################################

def plot_masks(mask1, mask2, direction, output_file, cb=True,
               color_map='Purples'):

    mask1_val = mask1.get_fdata()
    mask2_val = mask2.get_fdata()
    combined_mask = new_img_like(mask1, mask1_val + mask2_val)

    fig = plot_stat_map(combined_mask, display_mode=direction, cut_coords=10,
                        cmap=color_map, colorbar=cb)

    fig.savefig(output_file, dpi=600)


def plot_glassbrain_masks(mask1, direction, output_file, mask2=None, cb=True,
                          color_map='viridis'):

    fig = plt.figure(figsize=(6, 2.75))
    # left, bottom, width, height
    axes = plt.axes([0., 0., 1., 1.])

    display = plot_glass_brain(None, display_mode=direction, black_bg=False,
                               alpha=1., axes=axes)

    cmap = plt.get_cmap(color_map)
    display.add_overlay(mask1, cmap=cmap, colorbar=cb)
    if mask2 is not None:
        display.add_overlay(mask2, cmap=cmap)

    fig.savefig(output_file, dpi=600)


# ############################# INPUTS ##################################

working_dir = os.path.dirname(os.path.abspath(__file__))
atlases_dir = os.path.join(working_dir, 'atlases')
msdtb_dir = os.path.join(atlases_dir, 'msdtb')

msdtb_putamen_lh_maskpath = os.path.join(
    msdtb_dir, 'msdtb_putamen_mask_lh.nii.gz')
msdtb_putamen_rh_maskpath = os.path.join(
    msdtb_dir, 'msdtb_putamen_mask_rh.nii.gz')
msdtb_putamen_maskpath_plot = os.path.join(
    msdtb_dir, 'msdtb_putamen_maskplot.png')
msdtb_putamen_maskpath_glassbrain = os.path.join(
    msdtb_dir, 'msdtb_putamen_maskgb.png')

msdtb_cereb6_lh_maskpath = os.path.join(
    msdtb_dir, 'msdtb_cereb6_mask_lh.nii.gz')
msdtb_cereb6_rh_maskpath = os.path.join(
    msdtb_dir, 'msdtb_cereb6_mask_rh.nii.gz')
msdtb_cereb6_maskpath_plot = os.path.join(
    msdtb_dir, 'msdtb_cereb6_maskplot.png')
msdtb_cereb6_maskpath_glassbrain = os.path.join(
    msdtb_dir, 'msdtb_cereb6_maskgb.png')

data_dir = '/home/analu/diedrichsen_data/data/Cerebellum/music-sdtb/derivatives'

con_relative_path = 'group/allmain_tasks/rfx_onesample_t_rwls_gm/con_01_Encoding/con_0001.nii'
con_path = os.path.join(data_dir, con_relative_path)
thresh_min = .25
thresh_max = 1.

# ############################## RUN ####################################

if __name__ == '__main__':
    # # Load
    # con = load_img(con_path)

    # # Threshold
    # thresholded_con_val = con.get_fdata()
    # thresholded_con_val[thresholded_con_val < thresh_min] = 0
    # thresholded_con_val[thresholded_con_val > thresh_max] = 0
    # thresholded_con = new_img_like(con, thresholded_con_val)

    # # # Plot
    # figy = plot_stat_map(thresholded_con, display_mode='y', cut_coords=10,
    #                      cmap='Purples', colorbar=True)
    # figy.savefig('encoding_vs_rest_y.png', dpi=600)

    # figz = plot_stat_map(thresholded_con, display_mode='z', cut_coords=10,
    #                      cmap='Purples', colorbar=True)
    # figz.savefig('encoding_vs_rest_z.png', dpi=600)
    # plot_mask(thresholded_con, 'encoding_vs_rest.png', cb=True,
    #           color_map='viridis')

    # Load
    msdtb_putamen_lh_mask = load_img(msdtb_putamen_lh_maskpath)
    msdtb_putamen_rh_mask = load_img(msdtb_putamen_rh_maskpath)
    msdtb_cereb6_lh_mask = load_img(msdtb_cereb6_lh_maskpath)
    msdtb_cereb6_rh_mask = load_img(msdtb_cereb6_rh_maskpath)

    plot_masks(msdtb_putamen_lh_mask, msdtb_putamen_rh_mask, 'z',
               msdtb_putamen_maskpath_plot, cb=False)

    plot_masks(msdtb_cereb6_lh_mask, msdtb_cereb6_rh_mask, 'y',
               msdtb_cereb6_maskpath_plot, cb=False)

    plot_glassbrain_masks(msdtb_putamen_lh_mask, 'z',
                          msdtb_putamen_maskpath_glassbrain,
                          mask2=msdtb_putamen_rh_mask, cb=False)

    plot_glassbrain_masks(msdtb_cereb6_lh_mask, 'y',
                          msdtb_cereb6_maskpath_glassbrain,
                          mask2=msdtb_cereb6_rh_mask, cb=False)
