"""
This script projects individualized cortical ROIs defined in MNI space
onto fs_LR32k surface space. 

Author: Ana Luisa Pinho

Created: 17th of September 2025
Last update: September 2025

Compatibility: Python 3.10.16

"""

import os
import nibabel as nib
import nitools as nt

from nilearn.image import load_img
from nilearn.surface import vol_to_surf 
from volume_to_surface import get_imeshes


# ########################### FUNCTIONS ###############################

def mask2surf(roi_dir, derivatives_dir, itag, atlas, subjects, roi, 
              surfspace='fslr32k', save='gifti'):

    # Paths of the NON-NORMALIZED individual contrast map for all subjects
    if itag == 'g':
        masks_dir = os.path.join(roi_dir, 'group_roi_masks')
        masks = [os.path.join(
            masks_dir,
            itag + '_msdtb_' + atlas + '_' + roi + '_bh_mask.nii.gz')
        ]
    else:
        masks_dir = os.path.join(roi_dir, 'individual_roi_masks')
        masks = [os.path.join(
            masks_dir,
            itag + '_sub-%02d' % sub + '_' + roi + '_bh_mask.nii.gz')
            for sub in subjects
        ]

    # Paths of individual meshes per hemisphere
    pial_left, pial_right, white_left, white_right = get_imeshes(
        derivatives_dir, subjects, surfspace=surfspace)

    # For each subject...
    for mask, pl, pr, wl, wr, sb in zip(masks, pial_left, pial_right,
                                        white_left, white_right, SUBJECTS):

        # Map individual functional data from  Nifti to the surface of...
        # ... left and right hemispheres
        mask_img = load_img(mask)
        DL = vol_to_surf(mask_img, surf_mesh=pl, inner_mesh=wl)
        DR = vol_to_surf(mask_img, surf_mesh=pr, inner_mesh=wr)
        print(sb)
        print(mask)
        print(DL.shape)
        print(DR.shape)

        # Transform numpy arrays in gifti files
        imask = (itag + '_mask')                    
        GIFTIL = nt.gifti.make_func_gifti(DL, anatomical_struct='CortexLeft',
                                          column_names=[imask])
        GIFTIR = nt.gifti.make_func_gifti(DR, anatomical_struct='CortexRight',
                                          column_names=[imask])
        
        # Create output directory if it does not exist
        output_dir = os.path.join(
            roi_dir, 'individual_roi_' + surfspace + '_masks')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save output
        if save == 'gifti':
            # Save Gifti files
            nib.save(
                GIFTIL,
                os.path.join(
                    output_dir,
                    itag
                    + '_sub-{sb:02d}_'.format(sb=sb)
                    + roi
                    + '_'
                    + surfspace
                    + '.hem-L.func.gii'
                )
            )
            nib.save(
                GIFTIR,
                os.path.join(
                    output_dir,
                    itag
                    + '_sub-{sb:02d}_'.format(sb=sb)
                    + roi
                    + '_'
                    + surfspace
                    + '.hem-R.func.gii'
                )
            )
        else:
            assert save == 'cifti'
            # Create CIFTI
            CIFTI = nt.cifti.join_giftis_to_cifti([GIFTIL, GIFTIR],
                                                  mask=[None, None])
            # Save CIFT file
            nib.save(
                CIFTI,
                os.path.join(
                    output_dir,
                    f'{itag}_sub-{sb:02d}_{roi}_{surfspace}.dscalar.nii'
                )
            )

# ############################ INPUTS #################################

# Subjects without pilot
SUBJECTS = [3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26, 28,
            29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

# Main dir
main_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    'roi_analyses_rwls_hrf128_wb_puncorr_unsmoothed', 
    'bothmod_allmain_tasks', 
    'main_tasks')

# ########################## PARAMETERS ###############################

# Parent directories
if os.path.isdir('/home/analu/diedrichsen_data/data'):
    base_dir = '/home/analu/diedrichsen_data/data'
else:
    base_dir = '/cifs/diedrichsen/data'

music = os.path.join(base_dir, 'Cerebellum', 'music-sdtb')
derivatives_folder = os.path.join(music, 'derivatives')

# All ROIs: 10 ROIs
region_names = [
    'motor_area', 'motor_area', 'motor_area', 'motor_area', 
    'heschl_gyrus', 
    'occipital_lobe'
    ]
atlas_names = [
    'hmat', 'hmat', 'hmat', 'hmat',
    'hos', 
    'hos'
    ]
roi_names = [
    'pmd', 'pmv', 'sma', 'presma',
    'heschl',
    'occipital'
    ]

# region_names = ['motor_area']
# atlas_names = ['hmat']
# roi_names = ['pmd']

tags = ['i', 'i9a', 'i8a', 'i7a', 'i6a', 'a', 'a4g', 'a3g', 'a2g', 'a1g', 'g']
# tags = ['g']

# ############################# RUN ###################################

if __name__ == '__main__':

    for region_name, atlas_name, roi_name in zip(region_names,
                                                 atlas_names,
                                                 roi_names):
        # Define ROI-specific folders
        roi_folder = os.path.join(main_dir, region_name, atlas_name, roi_name)
        
        for tag in tags:
            # Project masks onto surface   
            mask2surf(
                roi_folder, derivatives_folder, tag, atlas_name, SUBJECTS,
                roi_name, surfspace='fslr32k', save='cifti'
            )