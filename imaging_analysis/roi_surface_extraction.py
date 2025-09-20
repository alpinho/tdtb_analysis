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
from nilearn.surface import SurfaceImage, load_surf_data, load_surf_mesh
from nilearn.maskers import SurfaceLabelsMasker 
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
        output_dir = os.path.join(
            roi_dir, 'group_roi_' + surfspace + '_masks')
    else:
        masks_dir = os.path.join(roi_dir, 'individual_roi_masks')
        masks = [os.path.join(
            masks_dir,
            itag + '_sub-%02d' % sub + '_' + roi + '_bh_mask.nii.gz')
            for sub in subjects
        ]
        output_dir = os.path.join(
            roi_dir, 'individual_roi_' + surfspace + '_masks')

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
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save output
        if itag == 'g':
            sbj_identifier = 'msdtb'
        else:
            sbj_identifier = 'sub-{sb:02d}'.format(sb=sb)
        if save == 'gifti':
            # Save Gifti files
            nib.save(
                GIFTIL,
                os.path.join(
                    output_dir,
                    itag
                    + '_' 
                    + sbj_identifier 
                    + '_'
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
                    + '_' 
                    + sbj_identifier 
                    + '_'
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

# Paths of directories
main_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    'roi_analyses_rwls_hrf128_wb_puncorr_unsmoothed', 
    'bothmod_allmain_tasks', 
    'main_tasks'
)

task = 'All Tasks' # 'Production', 'Perception', 'NTFD', 'NTFD Random', 'All Tasks'

surface_space = 'fslr32k'

# ########################## PARAMETERS ###############################

# Parent directories
if os.path.isdir('/home/analu/diedrichsen_data/data'):
    base_dir = '/home/analu/diedrichsen_data/data'
else:
    base_dir = '/cifs/diedrichsen/data'

music = os.path.join(base_dir, 'Cerebellum', 'music-sdtb')
derivatives_folder = os.path.join(music, 'derivatives')

# Tasks definitions
tasks = {'prod': 'Production', 
         'percep': 'Perception', 
         'ntfd': 'NTFD',
         'rand_ntfd': 'NTFD Random',
         'allmain_tasks': 'All Tasks'
}
task_id = {v: k for k, v in tasks.items()}.get(task)

# Contrast dictionary (id -> name)
if task_id != 'rand_ntfd':
    all_contrasts = {
        1: 'Encoding',
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
        18: 'Decision'
    }
    selected_contrasts = {
        10: 'Auditory Beat',
        11: 'Auditory Interval',
        14: 'Visual Beat',
        15: 'Visual Interval'
    }
else:
    assert task_id == 'rand_ntfd'   
    all_contrasts = {
        1: 'Encoding',
        2: 'Auditory Encoding',
        3: 'Visual Encoding',
        4: 'Auditory vs Visual Encoding',
        5: 'Visual vs Auditory Encoding',
        6: 'Beat',
        7: 'Interval',
        8: 'Non-Random',
        9: 'Random',
        10: 'Beat vs Interval',
        11: 'Interval vs Beat',
        12: 'Beat vs Random',
        13: 'Random vs Beat',
        14: 'Interval vs Random',
        15: 'Random vs Interval',
        16: 'Non-Random vs Random',
        17: 'Random vs Non-Random',
        18: 'Auditory Beat',
        19: 'Auditory Interval',
        20: 'Auditory Non-Random',                   
        21: 'Auditory Random',
        22: 'Auditory Beat vs Auditory Interval',
        23: 'Auditory Interval vs Auditory Beat',
        24: 'Auditory Beat vs Auditory Random',
        25: 'Auditory Random vs Auditory Beat',
        26: 'Auditory Interval vs Auditory Random',
        27: 'Auditory Random vs Auditory Interval',
        28: 'Auditory Non-Random vs Auditory Random',
        29: 'Auditory Random vs Auditory Non-Random',
        30: 'Visual Beat',
        31: 'Visual Interval',
        32: 'Visual Non-Random',                   
        33: 'Visual Random',
        34: 'Visual Beat vs Visual Interval',
        35: 'Visual Interval vs Visual Beat',
        36: 'Visual Beat vs Visual Random',
        37: 'Visual Random vs Visual Beat',                    
        38: 'Visual Interval vs Visual Random',
        39: 'Visual Random vs Visual Interval',
        40: 'Visual Non-Random vs Visual Random',
        41: 'Visual Random vs Visual Non-Random',
        42: 'Decision'
    }
    selected_contrasts = {
        18: 'Auditory Beat',
        19: 'Auditory Interval',
        21: 'Auditory Random',
        30: 'Visual Beat',
        31: 'Visual Interval',
        33: 'Visual Random'
    }

surfmaps_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    'results', 
    'parametric_tests', 
    'surface',
    task_id,
    'surface_files'
)

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
                roi_name, surfspace=surface_space, save='gifti'
            )   
            mask2surf(
                roi_folder, derivatives_folder, tag, atlas_name, SUBJECTS,
                roi_name, surfspace=surface_space, save='cifti'
            )

            # # Open the group ROI mask in surface space
            # roi_surfmasks_dir = os.path.join(
            #     roi_folder, 'individual_roi_' + surface_space + '_masks')
            # roi_gifti_left_paths = [
            #     os.path.join(
            #         output_dir,
            #         tag
            #         + '_sub-{sub:02d}_'.format(sb=sb)
            #         + roi_name
            #         + '_'
            #         + surface_space
            #         + '.hem-L.func.gii'
            #     ) for sub in SUBJECTS
            # ]

            # # Do the extraction of the individualized ROIs on surface space
            # for key, value in selected_contrasts.items():
            #     for sub in SUBJECTS:
            #         for hem, roi_gifti in zip(['L', 'R'],
            #                                 [roi_gifti_left,
            #                                 roi_gifti_right]):

            #             # Load individual ROI mask
            #             # masker = SurfaceLabelsMasker(
            #             #     labels_img=roi_gifti,
            #             #     standardize=False,
            #             #     smoothing_fwhm=None,
            #             #     resampling_target='labels',
            #             #     label_indices=None,
            #             #     ensure_finite=True
            #             # )
            #             # masker.fit()

            #             # Path of surface files
            #             fname = 'sub-%02d' % sub + '_' + 'allmain_tasks' + '_' + \
            #                 str(key) + '_' + value.lower().replace(' ', '-') + \
            #                 '_fslr32k.hem-' + hem + '.func.gii'
            #             surf_files_path = os.path.join(surfmaps_dir, fname)
