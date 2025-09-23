"""
This script projects individualized cortical ROIs defined in MNI space
onto fs_LR32k surface space. 

Author: Ana Luisa Pinho

Created: 17th of September 2025
Last update: September 2025

Compatibility: Python 3.10.16

"""

import os
import numpy as np
import nibabel as nib
import nitools as nt

from nilearn.image import load_img
from nilearn.surface import (load_surf_data, load_surf_mesh, SurfaceImage, 
                             PolyMesh, vol_to_surf)
from nilearn.maskers import SurfaceLabelsMasker 


# ########################### FUNCTIONS ###############################

def mask2surf(roi_dir, itag, atlas, subjects, roi, 
              tpl_pial_left, tpl_pial_right, tpl_white_left, tpl_white_right,
              surfspace='fslr32k', save='gifti'):

    # Paths of the NON-NORMALIZED individual contrast map for all subjects
    if itag == 'g':
        masks_dir = os.path.join(roi_dir, 'group_roi_masks')
        masks_lh = [os.path.join(
            masks_dir,
            itag + '_msdtb_' + atlas + '_' + roi + '_lh_mask.nii.gz')
        ]
        masks_rh = [os.path.join(
            masks_dir,
            itag + '_msdtb_' + atlas + '_' + roi + '_rh_mask.nii.gz')
        ]
        output_dir = os.path.join(
            roi_dir, 'group_roi_' + surfspace + '_masks')
    else:
        masks_dir = os.path.join(roi_dir, 'individual_roi_masks')
        masks_lh = [os.path.join(
            masks_dir,
            itag + '_sub-%02d' % sub + '_' + roi + '_lh_mask.nii.gz')
            for sub in subjects
        ]
        masks_rh = [os.path.join(
            masks_dir,
            itag + '_sub-%02d' % sub + '_' + roi + '_rh_mask.nii.gz')
            for sub in subjects
        ]
        output_dir = os.path.join(
            roi_dir, 'individual_roi_' + surfspace + '_masks')

    # For each subject...
    for mask_lh, mask_rh, sb in zip(masks_lh, masks_rh, subjects):

        # Map individual functional data from  Nifti to the surface of...
        # ... left and right hemispheres
        mask_img_lh = load_img(mask_lh)
        mask_img_rh = load_img(mask_rh)
        DL = vol_to_surf(mask_img_lh, 
                         surf_mesh=tpl_pial_left, inner_mesh=tpl_white_left,
                         interpolation="nearest")
        DR = vol_to_surf(mask_img_rh, 
                         surf_mesh=tpl_pial_right, inner_mesh=tpl_white_right,
                         interpolation="nearest")
        print(sb)
        print(mask_lh)
        print(mask_rh)
        print(DL.shape)
        print(DR.shape)
        print("L> nnz:", int(np.count_nonzero(DL)), 
              "R> nnz:", int(np.count_nonzero(DR)))

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


def extract_roi_means_surface(
        roi_gifti_path: str,
        surfmap_path: str,
        mesh_path: str,
        medial_wall_path: str = None,   # 1=cortex, 0=MW
        background_label: int = 0,
        mask_threshold: float = 0.5,
    ):
    """
    Extract mean beta per ROI from a single-hemisphere surface.

    Parameters
    ----------
    roi_gifti_path : str
        .label.gii or thresholdable .func.gii (ROI on the same mesh).
    surfmap_path : str
        Single beta map (.func.gii) on the same mesh.
    mesh_path : str
        Subject's hemi mesh (same used in vol_to_surf).
    medial_wall_path : str | None
        0/1 mask (1=cortex). If given, drop MW from labels and data.
    background_label : int
        Label code for background/medial wall (default 0).
    mask_threshold : float
        Threshold if ROI is a metric mask (>= threshold → inside).

    Returns
    -------
    roi_means : np.ndarray, shape (n_rois,)
    roi_codes : np.ndarray[int], shape (n_rois,)
    """

    # ROI labels (binarize if metric)
    lab = load_surf_data(roi_gifti_path)
    if not np.issubdtype(lab.dtype, np.integer):
        lab = (lab >= mask_threshold).astype(int)

    # Beta map
    data = load_surf_data(surfmap_path).astype(float)

    # Apply medial wall mask (enforce cortex-only)
    if medial_wall_path is not None:
        mw = load_surf_data(medial_wall_path).astype(bool)
        if mw.shape[-1] != data.shape[-1]:
            raise ValueError(
                "Vertex count mismatch: MW "
                f"{mw.shape[-1]} vs data {data.shape[-1]}"
            )
        lab = np.where(mw, lab, background_label).astype(int)
        data = np.where(mw, data, np.nan)

    # Extra safety: drop any remaining non-finite betas from ROI
    if lab.shape[-1] != data.shape[-1]:
        raise ValueError(
            "Vertex count mismatch: labels "
            f"{lab.shape[-1]} vs data {data.shape[-1]}"
        )
    finite = np.isfinite(data)
    lab = lab.astype(int)
    lab[~finite] = background_label

    # Mesh -> PolyMesh for the correct hemisphere
    hemi_left = ("hem-L" in mesh_path or ".L." in mesh_path or
                 "Left" in mesh_path)
    smesh = load_surf_mesh(mesh_path)
    if hemi_left:
        mesh = PolyMesh(left=smesh)
        hemi_key = "left"
    else:
        mesh = PolyMesh(right=smesh)
        hemi_key = "right"

    # Wrap & extract
    labels_img = SurfaceImage(mesh=mesh, data={hemi_key: lab})
    surf_img = SurfaceImage(mesh=mesh, data={hemi_key: data})
    masker = SurfaceLabelsMasker(
        labels_img=labels_img,
        background_label=background_label,
    ).fit()
    out = masker.transform(surf_img)
    roi_means = out if out.ndim == 1 else out.squeeze(0)

    # ROI codes (ascending, excluding background)
    roi_codes = np.array(
        sorted(c for c in np.unique(lab) if c != background_label),
        dtype=int,
    )
    return roi_means, roi_codes


# ############################ INPUTS #################################

# Subjects without pilot
SUBJECTS = [3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26, 28,
            29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

# SUBJECTS = [47]

# Paths of directories
main_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    'roi_analyses_rwls_hrf128_wb_puncorr_unsmoothed', 
    'bothmod_allmain_tasks'
)

# Tasks to extract ROI data
# Note: do not select 'NTFD Random' together w/ any other task
# tasks_roiextract_vals = ['Production', 'Perception', 'NTFD', 'All Tasks']
# tasks_roiextract_vals = ['NTFD Random']
tasks_roiextract_vals = ['Production', 'Perception', 'NTFD', 'All Tasks']
surface_space = 'fslr32k'
contype = 'psc'

# ############ Medial Wall Masks ##################
fslr32k_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'fslr32k_meshes')

fslr32_tpl_pial_left = os.path.join(
    fslr32k_folder, 'templates', 'tpl-fs32k_hemi-L_pial.surf.gii')
fslr32_tpl_pial_right = os.path.join(
    fslr32k_folder, 'templates', 'tpl-fs32k_hemi-R_pial.surf.gii')
fslr32_tpl_white_left = os.path.join(
    fslr32k_folder, 'templates', 'tpl-fs32k_hemi-L_white.surf.gii')
fslr32_tpl_white_right = os.path.join(
    fslr32k_folder, 'templates', 'tpl-fs32k_hemi-R_white.surf.gii')

mask_suffix = '1'
lh_medial_wall_mask_path = os.path.join(
    fslr32k_folder, 'medialwall_masks',
    'fs_LR.32k.L.medialwall.mask' + mask_suffix + '.gii')
rh_medial_wall_mask_path = os.path.join(
    fslr32k_folder, 'medialwall_masks',
    'fs_LR.32k.R.medialwall.mask' + mask_suffix + '.gii')

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
tasks_roiextract = \
    {k: v for k, v in tasks.items() if v in tasks_roiextract_vals}

# Contrast dictionary (id -> name)
if 'rand_ntfd' not in tasks_roiextract.keys():
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
    folder_name = 'main_tasks'
else:
    assert 'rand_ntfd' in tasks_roiextract.keys()
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
    folder_name = 'main_tasks'

# All ROIs: 10 ROIs
# region_names = [
#     'motor_area', 'motor_area', 'motor_area', 'motor_area', 
#     'heschl_gyrus', 
#     'occipital_lobe'
#     ]
# atlas_names = [
#     'hmat', 'hmat', 'hmat', 'hmat',
#     'hos', 
#     'hos'
#     ]
# roi_names = [
#     'pmd', 'pmv', 'sma', 'presma',
#     'heschl',
#     'occipital'
#     ]

region_names = ['occipital_lobe']
atlas_names = ['hos']
roi_names = ['occipital']

# tags = ['i', 'i9a', 'i8a', 'i7a', 'i6a', 'a', 'a4g', 'a3g', 'a2g', 'a1g', 'g']
tags = ['i6a']

# ############################# RUN ###################################

if __name__ == '__main__':

    for region_name, atlas_name, roi_name in zip(region_names,
                                                 atlas_names,
                                                 roi_names):
        # Define ROI-specific folders
        roi_folder = os.path.join(main_dir, folder_name, region_name, 
                                  atlas_name, roi_name)
          
        for tag in tags:

            # Project masks onto surface
            mask2surf(
                roi_folder, tag, atlas_name, SUBJECTS, roi_name,
                fslr32_tpl_pial_left, fslr32_tpl_pial_right,
                fslr32_tpl_white_left, fslr32_tpl_white_right, 
                surfspace=surface_space, save='gifti'
            )   
            mask2surf(
                roi_folder, tag, atlas_name, SUBJECTS, roi_name,
                fslr32_tpl_pial_left, fslr32_tpl_pial_right,
                fslr32_tpl_white_left, fslr32_tpl_white_right, 
                surfspace=surface_space, save='cifti'
            )

            # Open the group ROI mask in surface space
            if tag == 'g':
                prefix = 'group'
                roi_surfmasks_dir = os.path.join(
                    roi_folder, prefix + '_roi_' + surface_space + '_masks')
                roi_gifti_left_paths = [
                    os.path.join(
                        roi_surfmasks_dir,
                        tag
                        + '_msdtb_'
                        + roi_name
                        + '_'
                        + surface_space
                        + '.hem-L.func.gii'
                    ) for sub in SUBJECTS
                ]
                roi_gifti_right_paths = [
                    os.path.join(
                        roi_surfmasks_dir,
                        tag
                        + '_msdtb_'
                        + roi_name
                        + '_'
                        + surface_space
                        + '.hem-R.func.gii'
                    ) for sub in SUBJECTS
                ]
            else:
                prefix = 'individual'
                roi_surfmasks_dir = os.path.join(
                    roi_folder, prefix + '_roi_' + surface_space + '_masks')
                roi_gifti_left_paths = [
                    os.path.join(
                        roi_surfmasks_dir,
                        tag
                        + '_sub-{sub:02d}_'.format(sub=sub)
                        + roi_name
                        + '_'
                        + surface_space
                        + '.hem-L.func.gii'
                    ) for sub in SUBJECTS
                ]
                roi_gifti_right_paths = [
                    os.path.join(
                        roi_surfmasks_dir,
                        tag
                        + '_sub-{sub:02d}_'.format(sub=sub)
                        + roi_name
                        + '_'
                        + surface_space
                        + '.hem-R.func.gii'
                    ) for sub in SUBJECTS
                ]         
                
            hems_rois =  []
            # For each hemisphere
            for hem, roi_gifti_paths, pial, mw in \
                zip(['L', 'R'], 
                    [roi_gifti_left_paths, roi_gifti_right_paths], 
                    [fslr32_tpl_pial_left, fslr32_tpl_pial_right],
                    [lh_medial_wall_mask_path, rh_medial_wall_mask_path]):

                tasks_rois = []
                for task_key in tasks_roiextract.keys():
                    surfmaps_pardir = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)), 
                        'results', 
                        'parametric_tests', 
                        'surface',
                        task_key,
                        'surface_files'
                    )

                    contrasts_rois = []
                    # For each selected contrast
                    for key, value in selected_contrasts.items():

                        subjects_rois = []
                        # For each subject
                        for roi_gifti_path, subject in \
                                zip(roi_gifti_paths, SUBJECTS):

                            # Path of surface files
                            cname = value.lower().replace(
                                ' vs ', '_vs_').replace(' ', '-')
                            surfmaps_dir = os.path.join(
                                surfmaps_pardir, str(key) + '_' + cname)
                            if tag == 'g':
                                fname = 'group' + '_' + \
                                    task_key.replace('_', '-') + '_' + \
                                    cname + '_' + surface_space + '.' + \
                                    hem + '.func.gii'
                            else:
                                fname = 'sub-%02d' % subject + '_' + \
                                    task_key.replace('_', '-') + '_' + \
                                    cname + '_' + surface_space + '.hem-' + \
                                    hem + '.func.gii'                        
                            surfmap_path = os.path.join(surfmaps_dir, fname)

                            roi_means, _ = extract_roi_means_surface(
                                roi_gifti_path=roi_gifti_path,   # mask/labels for this hemi
                                surfmap_path=surfmap_path,       # beta map for this hemi
                                mesh_path=pial,                  # pial template (same used in vol_to_surf)
                                medial_wall_path=mw,             # medial wall mask for this hemi
                                background_label=0               # or -1 if that's your medial wall
                            )

                            # Append: shape (hemisphere, tasks, contrasts, subjects)
                            # hemisphere: lh, rh
                            # tasks: prod, percep, ntfd, allmain_tasks
                            # contrasts: Auditory Beat, Auditory Interval, Visual Beat,
                            #            Visual Interval
                            # subjects: list of subjects' ids
                            subjects_rois.append(roi_means[0])                            
                        contrasts_rois.append(subjects_rois)
                    tasks_rois.append(contrasts_rois)
                hems_rois.append(tasks_rois)

            # Save
            roi_surf_dir = os.path.join(roi_folder, 'rois_surf_extraction')
            os.makedirs(roi_surf_dir, exist_ok=True)
            outpath = os.path.join(
                roi_surf_dir, 
                tag + '_' + roi_name + '_' + contype + '.npy'
            )
            if os.path.exists(outpath):
                os.remove(outpath)
            np.save(outpath, hems_rois, allow_pickle=False)