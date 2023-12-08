"""
This script creates masks for several regions using the AAL3 atlas.

Author: Ana Luisa Pinho

Created: September 2023
Last update: December 2023

Compatibility: Python 3.10.10

"""

import os

from nilearn.image import load_img, new_img_like


# ############################ FUNCTIONS ################################


def binarize(mask_path, label):

    # Load
    mask = load_img(mask_path)

    # Threshold
    thresholded_mask_val = mask.get_fdata()
    thresholded_mask_val[thresholded_mask_val != label] = 0

    # Binarization
    bin_mask_val = (thresholded_mask_val != 0)

    # Convert in Niimg file
    bin_mask = new_img_like(mask, bin_mask_val)

    return bin_mask


# ############################# INPUTS ##################################

working_dir = os.path.dirname(os.path.abspath(__file__))
atlases_dir = os.path.join(working_dir, 'atlases')
data_dir = '/home/analu/diedrichsen_data/data/Cerebellum/music-sdtb/derivatives'

# ### AAL3 ###
aal3 = os.path.join(atlases_dir, 'aal3')
aal3_outputs = os.path.join(aal3, 'outputs')

aal3_masks = os.path.join(aal3, 'AAL3')
aal3_2mm = os.path.join(aal3_masks, 'AAL3v1.nii.gz')

# ############################## RUN ####################################
  
if __name__ == '__main__':

    # ######################### Putamen #################################

    # # Binarize masks
    putamen_aal3_lh_bin = binarize(aal3_2mm, 77.)
    putamen_aal3_rh_bin = binarize(aal3_2mm, 78.)

    # # Save maks
    putamen_aal3_lh_bin.to_filename(os.path.join(aal3_outputs,
                                                 'aal3_putamen_lh.nii.gz'))
    putamen_aal3_rh_bin.to_filename(os.path.join(aal3_outputs,
                                                 'aal3_putamen_rh.nii.gz'))

    # ################## Cerebellum Crus I ##############################

    # # Binarize masks
    crus1_aal3_lh_bin = binarize(aal3_2mm, 95.)
    crus1_aal3_rh_bin = binarize(aal3_2mm, 96.)

    # # Save maks
    crus1_aal3_lh_bin.to_filename(os.path.join(aal3_outputs,
                                               'aal3_crus1_lh.nii.gz'))
    crus1_aal3_rh_bin.to_filename(os.path.join(aal3_outputs,
                                               'aal3_crus1_rh.nii.gz'))

    # ################### Cerebellum VI #################################

    # # Binarize masks
    cereb6_aal3_lh_bin = binarize(aal3_2mm, 103.)
    cereb6_aal3_rh_bin = binarize(aal3_2mm, 104.)

    # # Save maks
    cereb6_aal3_lh_bin.to_filename(os.path.join(aal3_outputs,
                                                'aal3_cereb6_lh.nii.gz'))
    cereb6_aal3_rh_bin.to_filename(os.path.join(aal3_outputs,
                                                'aal3_cereb6_rh.nii.gz'))
