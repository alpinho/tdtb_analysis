"""
Compute mean activity in ROIs for selected contrasts in Music-SDTB.

Author: Ana Luisa Pinho
email: agrilopi@uwo.ca

Created: October 2023
Last update: May 2026

Compatibility: Python 3.10.14

How to run:
1) Main tasks:
   python roi_extraction_msdtb.py <encoding_type>
   <encoding_type> ∈ {bothmod|auditory|visual}

2) rand_ntfd only:
   python roi_extraction_msdtb.py <encoding_type> <rand_mode>
   <rand_mode> ∈ {pairs|nonrandom}

Examples:
   python roi_extraction_msdtb.py bothmod
   python roi_extraction_msdtb.py auditory pairs
   python roi_extraction_msdtb.py visual nonrandom
"""

import glob
import os
import re
import sys
import numpy as np

from scipy.ndimage import binary_dilation, binary_erosion
from nilearn.image import load_img, new_img_like, resample_to_img
from nilearn.input_data import NiftiLabelsMasker


# ############################ FUNCTIONS ###############################

def nonan_map(con_path):
    """Load image and replace NaNs with zero."""
    con = load_img(con_path)
    con_val = con.get_fdata()
    con_val[np.isnan(con_val)] = 0
    con_map = new_img_like(con, con_val)
    return con_val, con_map


def threshold_map(con_val, thresh_min, thresh_max=None):
    """Apply lower/upper threshold to a 3D array."""
    thr = con_val.copy()
    thr[thr < thresh_min] = 0
    if thresh_max is not None:
        thr[thr > thresh_max] = 0
    return thr


def binary_dilation_with_limit(image, target_count, gmask, atlas_con):
    """Dilate/erode to reach a voxel count limit inside gmask.

    If over target after loop, drop lowest-value voxels by atlas_con.
    """
    target0 = target_count
    s1, s2, s3 = np.random.choice(
        np.random.permutation(np.random.permutation(np.arange(1, 30))), 3
    )
    dil = binary_dilation(
        image, mask=gmask, structure=np.ones((s1, s2, s3))
    )
    cur = np.count_nonzero(dil)

    n_iter, flag = 0, 0
    while cur != target_count:
        n_iter += 1
        s1, s2, s3 = np.random.choice(
            np.random.permutation(
                np.random.permutation(np.arange(1, 30))
            ), 3
        )
        if cur < target_count:
            dil = binary_dilation(
                dil, mask=gmask, structure=np.ones((s1, s2, s3))
            )
        else:
            dil = binary_erosion(
                dil, mask=gmask, structure=np.ones((s1, s2, s3))
            )
        cur = np.count_nonzero(dil)

        if n_iter == 100:
            flag += 1
            target_count += 1
            n_iter = 0

        print('Number of iterations:', flag * 100 + n_iter)

    if cur > target0:
        idx = np.argwhere(dil)
        vals = atlas_con[tuple(idx.T)]
        order = idx[np.argsort(vals)]
        excess = cur - target0
        for i in range(excess):
            x, y, z = order[i]
            dil[x, y, z] = 0

    return dil


def combine_masks(maskpath1, maskpath2, combined_maskpath):
    """Union two binary masks and save the result."""
    m1 = load_img(maskpath1)
    m2 = load_img(maskpath2)
    v1 = m1.get_fdata().astype(np.uint8)
    v2 = m2.get_fdata().astype(np.uint8)
    v = v1 + v2
    v[v == 2] = 1
    out = new_img_like(m1, v)
    out.to_filename(combined_maskpath)


def create_group_roimask(con_path, atlas_maskpath, msdtb_maskpath,
                         con_thresh_min=3.385, con_thresh_max=None):
    """Intersect thresholded encoding map with an atlas mask."""
    con_val, con_map = nonan_map(con_path)

    if con_thresh_max is None:
        thr_val = threshold_map(con_val, con_thresh_min)
    else:
        thr_val = threshold_map(
            con_val, con_thresh_min, thresh_max=con_thresh_max
        )

    bin_con = (thr_val != 0)

    atlas_mask = load_img(atlas_maskpath)
    atlas_r = resample_to_img(atlas_mask, con_map, interpolation='nearest')
    atlas_v = atlas_r.get_fdata()

    msdtb_v = np.logical_and(
        bin_con.astype(bool), atlas_v.astype(bool)
    ).astype(int)

    n_vox = np.count_nonzero(msdtb_v)
    if not n_vox:
        raise ValueError(
            'N_voxels = 0 ! No intersection between thresholded encoding '
            'and atlas mask.'
        )

    msdtb_mask = new_img_like(atlas_r, msdtb_v)
    msdtb_mask.to_filename(msdtb_maskpath)
    return msdtb_mask, n_vox


def create_iroimask(icon_path, atlas_maskpath, gmask, n_voxels,
                    iroi_maskpath, gcon_path=None, weights=None):
    """Create individual ROI with group-size match and optional dilation."""
    icon_v, _ = nonan_map(icon_path)
    gcon_v, gcon_map = nonan_map(gcon_path)

    atlas_mask = load_img(atlas_maskpath)
    atlas_r = resample_to_img(atlas_mask, gcon_map, interpolation='nearest')
    atlas_v = atlas_r.get_fdata()

    con_v = np.average(np.array([icon_v, gcon_v]), axis=0, weights=weights)
    bin_msdtb_v = np.where(atlas_v, con_v, 0)

    iroi_v = np.where(bin_msdtb_v, con_v, 0)
    flat = iroi_v.ravel()

    if np.count_nonzero(flat) >= n_voxels:
        nz_idx = np.flatnonzero(flat)
        nz_val = flat[nz_idx]
        order = np.argpartition(nz_val, -n_voxels)[-n_voxels:]
        sel_idx = nz_idx[order]
        mask = np.zeros_like(flat, dtype=bool)
        mask[sel_idx] = True
        bin_i_v = mask.reshape(iroi_v.shape)
    else:
        bin_i_v = (iroi_v != 0)

    n_sel = np.count_nonzero(bin_i_v)
    if n_sel == 0:
        print('Binarized ROI is empty. Taking the group mask.')
        iroi_mask = gmask
    elif n_sel < n_voxels:
        gmask_v = gmask.get_fdata()
        dil_v = binary_dilation_with_limit(
            bin_i_v, n_voxels, gmask_v, bin_msdtb_v
        )
        print('Dilation performed! ', np.count_nonzero(dil_v))
        iroi_mask = new_img_like(atlas_r, dil_v)
    else:
        print(np.count_nonzero(bin_i_v))
        iroi_mask = new_img_like(atlas_r, bin_i_v)

    iroi_mask.to_filename(iroi_maskpath)
    return iroi_mask


def overlay_masks(pdir, mask_type, roi):
    """Overlay individual masks to show ROI stability."""
    masks_dir = os.path.join(pdir, 'individual_roi_masks')
    out_dir = os.path.join(pdir, 'overlaid_masks')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for hem in ['lh', 'rh']:
        pat = f'{mask_type}_sub-*_{roi}_{hem}_mask.nii.gz'
        paths = glob.glob(os.path.join(masks_dir, pat))
        images = [load_img(p) for p in paths]
        vals = [im.get_fdata().tolist() for im in images]
        sm = np.sum(vals, axis=0)
        nm = sm / len(vals)
        out = new_img_like(images[0], nm)
        name = f'{mask_type}_{roi}_{hem}_mask.nii.gz'
        out.to_filename(os.path.join(out_dir, name))


def extract_roi(rmask, task, contrasts, subject_estimates_dir,
                derivatives_folder, filetype, derivative_type='sm8wbmasked'):
    """Extract mean PSC within ROI for each contrast."""
    out = []
    for key in list(contrasts.keys()):
        cfname = f'{filetype}_{key:04d}_desc-{derivative_type}.nii'
        masker = NiftiLabelsMasker(labels_img=rmask)
        masker.fit()
        mpath = os.path.join(
            subject_estimates_dir, task, derivatives_folder, cfname
        )
        print('Extracting ROI data from:', mpath)
        val = masker.transform(mpath)[0][0]
        out.append(val)
    return out


def iroicon_estimation(main_dir, atlas_dir, atlas, region, roi,
                       group_tmap_path, tasks_list, contrasts_dic, contype,
                       prefix, task_roi_definition, derivatives_folder, mask,
                       con_thresh_min=3.385, weights=None,
                       subregion=False, hems=None,
                       derivative_type='sm8wbmasked'):
    """Extract ROIs and compute overlays.

    Output array shape:
    (hemisphere, tasks, contrasts, subjects)
    """
    if hems is None:
        hems = ['lh', 'rh', 'bh']

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

    roi_hems = []
    for hem in hems:
        atlas_mask = os.path.join(
            atlas_dir, f'{atlas}_{roi}_{hem}_mask.nii.gz'
        )
        gmask_path = os.path.join(
            groi_dir, f'g_msdtb_{atlas}_{roi}_{hem}_mask.nii.gz'
        )

        if hem in ['lh', 'rh']:
            print('Group Encoding t-map for ROI mask:', group_tmap_path)
            gmask, clsize = create_group_roimask(
                group_tmap_path, atlas_mask, gmask_path,
                con_thresh_min=con_thresh_min
            )
        else:
            gmask_lh = os.path.join(
                groi_dir, f'g_msdtb_{atlas}_{roi}_lh_mask.nii.gz'
            )
            gmask_rh = os.path.join(
                groi_dir, f'g_msdtb_{atlas}_{roi}_rh_mask.nii.gz'
            )
            combine_masks(gmask_lh, gmask_rh, gmask_path)
            gmask = load_img(gmask_path)

        subjects_all = []
        for subject in SUBJECTS:
            sdir = os.path.join(data_dir, 'sub-%02d') % subject
            est_dir = os.path.join(sdir, 'estimates')
            iroi_path = os.path.join(
                iroi_dir,
                f'{prefix}_sub-%02d_{roi}_{hem}_mask.nii.gz' % subject
            )

            if weights == (0., 1.):
                irmask = gmask
            else:
                idx = [m.end() for m in re.finditer(
                    'con_', group_tmap_path)][0]
                con_id = int(group_tmap_path[idx: idx + 2])
                subj_tmap = os.path.join(
                    est_dir, task_roi_definition, derivatives_folder,
                    f'wspmT_{con_id:04d}_desc-sm8{mask}masked.nii'
                )
                print('Subject Encoding t-map for ROI mask:', subj_tmap)
                if hem in ['lh', 'rh']:
                    irmask = create_iroimask(
                        subj_tmap, atlas_mask, gmask, clsize, iroi_path,
                        gcon_path=group_tmap_path, weights=weights
                    )
                else:
                    im_lh = os.path.join(
                        iroi_dir,
                        f'{prefix}_sub-%02d_{roi}_lh_mask.nii.gz' % subject
                    )
                    im_rh = os.path.join(
                        iroi_dir,
                        f'{prefix}_sub-%02d_{roi}_rh_mask.nii.gz' % subject
                    )
                    combine_masks(im_lh, im_rh, iroi_path)
                    irmask = load_img(iroi_path)

            itasks = []
            for task in tasks_list.keys():
                vals = extract_roi(
                    irmask, task, contrasts_dic, est_dir, derivatives_folder,
                    contype, derivative_type=derivative_type
                )
                itasks.append(vals)
            subjects_all.append(itasks)

        tasks_all = np.moveaxis(subjects_all, 0, -1).tolist()
        roi_hems.append(tasks_all)

    outpath = os.path.join(roiextr_dir, f'{prefix}_{roi}_{contype[1:]}.npy')
    if os.path.exists(outpath):
        os.remove(outpath)
    np.save(outpath, roi_hems, allow_pickle=False)


# ############################## INPUTS #################################

# Subjects without pilot
SUBJECTS = [
    3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26, 28, 29, 32,
    34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47
]

# Task that defines the ROI mask
# 'Production' | 'Perception' | 'NTFD' | 'NTFD Random' | 'All Tasks'
task_roidef = 'All Tasks'

# Tasks to extract from
tasks_roiextract_vals = ['NTFD Random']

# ============================ PARAMETERS ==============================

if os.path.isdir('/home/analu/diedrichsen_data/data'):
    base_dir = '/home/analu/diedrichsen_data/data'
else:
    base_dir = '/cifs/diedrichsen/data'

data_dir = os.path.join(base_dir, 'Cerebellum/music-sdtb/derivatives')

# Tasks dict
tasks = {
    'prod': 'Production',
    'percep': 'Perception',
    'ntfd': 'NTFD',
    'rand_ntfd': 'NTFD Random',
    'allmain_tasks': 'All Tasks'
}
task_roidef_id = {v: k for k, v in tasks.items()}.get(task_roidef)
tasks_roiextract = {
    k: v for k, v in tasks.items() if v in tasks_roiextract_vals
}

# -------- All contrast dictionaries --------

ALL_CONTRASTS_MAIN = {
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

ALL_CONTRASTS_RAND = {
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

model = 'rwls'
masking = 'wb'
design = 'dbb'
hrf_cutoff = 'hrf128'

individual_derivatives_folder = (
    'masked_derivatives_' + model + '_' + design + '_' + hrf_cutoff
)
group_derivatives_folder = (
    'rfx_onesample_t_' + model + '_' + design + '_' + hrf_cutoff + '_' +
    masking
)

group_relative_path = os.path.join(
    'group', task_roidef_id, group_derivatives_folder
)
gtmap_encoding = os.path.join(
    data_dir, group_relative_path, 'con_01_Encoding', 'spmT_0001.nii'
)
gtmap_audioencoding = os.path.join(
    data_dir, group_relative_path, 'con_02_Auditory_Encoding',
    'spmT_0001.nii'
)
gtmap_visualencoding = os.path.join(
    data_dir, group_relative_path, 'con_03_Visual_Encoding',
    'spmT_0001.nii'
)

# Group-level cluster size threshold (t=3.385 ≈ z=3.1)
t_threshold = 3.385

working_dir = os.path.dirname(os.path.abspath(__file__))
atlases_dir = os.path.join(working_dir, 'atlases')
fsl_dir = os.path.join(atlases_dir, 'fsl_atlases')
atag_dir = os.path.join(atlases_dir, 'atag_atlas')
ntk_dir = os.path.join(atlases_dir, 'nettekoven_atlas')
hmat_dir = os.path.join(atlases_dir, 'hmat_atlas')
roi_dir = os.path.join(
    working_dir,
    f'roi_analyses_{model}_{hrf_cutoff}_{masking}_puncorr_unsmoothed'
)
contrast_type = 'wbmasked'  # or 'sm8wbmasked'

# All ROIs: 10 ROIs
atlas_dirnames = [fsl_dir,
                  ntk_dir, ntk_dir, ntk_dir,
                  hmat_dir, hmat_dir, hmat_dir, hmat_dir,
                  fsl_dir,
                  fsl_dir]

atlas_names = ['hos',
               'ntk_symmni128', 'ntk_symmni128', 'ntk_symmni128',
               'hmat', 'hmat', 'hmat', 'hmat',
               'hos', 
               'hos']

region_names = ['dorsal_striatum',
                'cerebellum', 'cerebellum', 'cerebellum',
                'motor_area', 'motor_area', 'motor_area', 'motor_area', 
                'heschl_gyrus', 
                'occipital_lobe']

roi_names = ['dstr',
             'cereb-s', 'cereb-i', 'cereb',
             'pmd', 'pmv', 'sma', 'presma',
             'heschl',
             'occipital']

# Example: 4 ROIs
# atlas_dirnames = [hmat_dir, hmat_dir, hmat_dir, hmat_dir]
# atlas_names = ['hmat', 'hmat', 'hmat', 'hmat']
# region_names = ['motor_area', 'motor_area', 'motor_area', 'motor_area']
# roi_names = ['pmd', 'pmv', 'sma', 'presma']

# 2 ROIs
# atlas_dirnames = [ntk_dir, ntk_dir]
# atlas_names = ['ntk_symmni128', 'ntk_symmni128']
# region_names = ['cerebellum', 'cerebellum']
# roi_names = ['cereb-s', 'cereb-i']

tags = ['i', 'i9a', 'i8a', 'i7a', 'i6a', 'a', 'a4g', 'a3g', 'a2g', 'a1g', 'g']

# Weights (individual, group) via two-vector average
weights_list = [
    (1., 0.), (.9, .1), (.8, .2), (.7, .3), (.6, .4), (.5, .5),
    (.4, .6), (.3, .7), (.2, .8), (.1, .9), (0., 1.)
]

# ############################### RUN ##################################

if __name__ == '__main__':
    assert len(sys.argv) > 1, (
        'Pass at least one arg: encoding_type '
        '(bothmod|auditory|visual).'
    )

    encoding_type = sys.argv[1]

    # Decide contrast pool and selection
    if 'rand_ntfd' not in tasks_roiextract.keys():
        all_contrasts = ALL_CONTRASTS_MAIN
        selected_contrasts = {
            10: 'Auditory Beat',
            11: 'Auditory Interval',
            14: 'Visual Beat',
            15: 'Visual Interval'
        }
        folder_name = 'main_tasks'
    else:
        all_contrasts = ALL_CONTRASTS_RAND
        assert len(sys.argv) > 2, (
            'For rand_ntfd, pass second arg <rand_mode>: '
            'pairs|nonrandom.'
        )
        rand_mode = sys.argv[2].strip().lower()
        if rand_mode not in ('pairs', 'nonrandom'):
            raise ValueError("rand_mode must be 'pairs' or 'nonrandom'.")

        if rand_mode == 'pairs':
            selected_contrasts = {
                18: 'Auditory Beat',
                19: 'Auditory Interval',
                21: 'Auditory Random',
                30: 'Visual Beat',
                31: 'Visual Interval',
                33: 'Visual Random'
            }
        else:
            assert rand_mode == 'nonrandom', \
                f"Unexpected rand_mode: {rand_mode}"
            selected_contrasts = {
                20: 'Auditory Non-Random',
                21: 'Auditory Random',
                32: 'Visual Non-Random',
                33: 'Visual Random'
            }
        folder_name = 'rand_ntfd_' + rand_mode

    keys = list(selected_contrasts.keys())

    msdtb_dir = os.path.join(
        roi_dir, f'{encoding_type}_{task_roidef_id}', folder_name
    )

    # Pick group t-map and optionally filter contrasts by encoding type
    if encoding_type == 'bothmod':
        gtmap = gtmap_encoding
        filtered_contrasts = selected_contrasts
    elif encoding_type == 'auditory':
        gtmap = gtmap_audioencoding
        aud_keys = keys[:len(keys) // 2]
        filtered_contrasts = {
            k: selected_contrasts[k] for k in aud_keys
            if k in selected_contrasts
        }
    elif encoding_type == 'visual':
        gtmap = gtmap_visualencoding
        vis_keys = keys[len(keys) // 2:]
        filtered_contrasts = {
            k: selected_contrasts[k] for k in vis_keys
            if k in selected_contrasts
        }
    else:
        raise ValueError(
            "encoding_type must be 'bothmod', 'auditory' or 'visual'."
        )

    if not os.path.exists(msdtb_dir):
        os.makedirs(msdtb_dir)

    for tag, wpair in zip(tags, weights_list):
        for adir, aname, rname, rlabel in zip(
            atlas_dirnames, atlas_names, region_names, roi_names
        ):
            if rname == 'dorsal_striatum':
                iroicon_estimation(
                    msdtb_dir, adir, aname, rname, rlabel, gtmap,
                    tasks_roiextract, filtered_contrasts, 'wpsc', tag,
                    task_roidef_id, individual_derivatives_folder, masking,
                    con_thresh_min=t_threshold, weights=wpair,
                    derivative_type=contrast_type
                )
            else:
                iroicon_estimation(
                    msdtb_dir, adir, aname, rname, rlabel, gtmap,
                    tasks_roiextract, filtered_contrasts, 'wpsc', tag,
                    task_roidef_id, individual_derivatives_folder, masking,
                    con_thresh_min=t_threshold, weights=wpair,
                    subregion=True, derivative_type=contrast_type
                )

            if rname == 'dorsal_striatum':
                outdir = os.path.join(msdtb_dir, rname, aname)
            else:
                outdir = os.path.join(msdtb_dir, rname, aname, rlabel)

            if tag != 'g':
                overlay_masks(outdir, tag, rlabel)