"""
This script projects individualized cortical ROIs defined in MNI space
onto fs_LR32k surface space. 

Author: Ana Luisa Pinho

Created: 17th of September 2025
Last update: September 2025

Compatibility: Python 3.10.16

"""

import os
import csv
import numpy as np
import nibabel as nib
import nitools as nt

from datetime import datetime
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
                         surf_mesh=tpl_pial_left, # inner_mesh=tpl_white_left,
                         interpolation="nearest", radius=2.)
        DR = vol_to_surf(mask_img_rh, 
                         surf_mesh=tpl_pial_right, # inner_mesh=tpl_white_right,
                         interpolation="nearest", radius=2.)
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


# ---------- Surface neighbors (cached) ----------
_ADJ_CACHE = {}

def _neighbors_cached(mesh_path_or_obj):
    """
    Build (and cache) per-vertex neighbors from the surface faces.
    Works with nilearn Surface/tuple/dict mesh objects.
    """
    if mesh_path_or_obj in _ADJ_CACHE:
        return _ADJ_CACHE[mesh_path_or_obj]

    smesh = load_surf_mesh(mesh_path_or_obj)
    # Be tolerant to nilearn/nibabel mesh formats
    try:
        coords = np.asarray(smesh.coordinates)
        faces  = np.asarray(smesh.faces, dtype=np.int64)
    except AttributeError:
        try:
            coords = np.asarray(smesh["coordinates"])
            faces  = np.asarray(smesh["faces"], dtype=np.int64)
        except Exception:
            coords = np.asarray(smesh[0])
            faces  = np.asarray(smesh[1], dtype=np.int64)

    n_vertices = int(coords.shape[0])
    neigh = [set() for _ in range(n_vertices)]
    for a, b, c in faces:
        neigh[a].add(b); neigh[a].add(c)
        neigh[b].add(a); neigh[b].add(c)
        neigh[c].add(a); neigh[c].add(b)
    neigh = [np.fromiter(s, dtype=np.int64) if s else np.empty(0, np.int64) for s in neigh]
    _ADJ_CACHE[mesh_path_or_obj] = neigh
    return neigh


def _expand_seed_to_valid(seed_mask, valid_mask, neighbors, max_rings, allowed_mask):
    """
    Expand a boolean seed on the surface up to `max_rings` rings,
    restricted to `allowed_mask`, until any FINITE vertices are reached.

    Returns
    -------
    final_mask : np.ndarray[bool]
    ring_used  : int  (0..max_rings if success, -1 if none found)
    """
    visited  = (seed_mask & allowed_mask).astype(bool, copy=True)
    frontier = visited.copy()

    for ring in range(max_rings + 1):  # includes ring 0
        candidate = visited & valid_mask
        if candidate.any():
            return candidate, ring

        if ring == max_rings or not frontier.any():
            break

        nxt = np.zeros_like(visited, dtype=bool)
        for v in np.where(frontier)[0]:
            nb = neighbors[v]
            if nb.size:
                nxt[nb] = True
        nxt &= allowed_mask & ~visited
        visited |= nxt
        frontier = nxt

    return np.zeros_like(seed_mask, dtype=bool), -1


# ---------- CSV logger ----------
LOG_FIELDS = [
    "timestamp", "region", "atlas", "roi", "tag", "subject", "hemisphere",
    "task", "contrast_id", "contrast_name", "mesh",
    "mask_file", "map_file",
    "n_vertices", "seed_nonzero", "seed_allowed",
    "finite_total", "valid_total", "initial_roi_finite",
    "expanded", "rings_used", "final_mask_size",
    "returned_mean", "used_nan", "cortex_frac"
]

def _write_log_row(csv_path, row):
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if not exists:
            w.writeheader()
        # ensure all fields exist
        for k in LOG_FIELDS:
            row.setdefault(k, "")
        w.writerow(row)


def extract_roi_means_surface(
        roi_gifti_path: str,
        surfmap_path: str,
        mesh_path: str,
        medial_wall_path: str = None,   # 1=cortex, 0=MW
        background_label: int = 0,
        mask_threshold: float = 0.5,
        expand_rings: int = 2,          # try 2–3 rings if needed
        log_csv_path: str | None = None,
        log_context: dict | None = None,
    ):
    """
    Robust single-hemisphere ROI summary using SurfaceLabelsMasker.

    Steps:
      1) Read ROI labels (binarize if metric).
      2) Read beta map; define FINITE data on cortex (exclude medial wall).
      3) Preferred mask = ROI ∩ FINITE ∩ cortex.
      4) If empty, expand ROI on-surface up to `expand_rings` rings
         (restricted to cortex) until FINITE data is found.
      5) Hand the final mask to SurfaceLabelsMasker and compute mean.

    Returns
    -------
    roi_means : np.ndarray, shape (n_rois,)  # here n_rois=1
    roi_codes : np.ndarray[int], shape (n_rois,)  # [1] if non-empty, else []
    """

    # 1) ROI labels (binarize if metric)
    lab = load_surf_data(roi_gifti_path)
    if not np.issubdtype(lab.dtype, np.integer):
        lab = (lab >= mask_threshold).astype(int)
    seed = (lab != background_label)

    # 2) Beta map & masks
    data = load_surf_data(surfmap_path).astype(float)
    if seed.shape[-1] != data.shape[-1]:
        raise ValueError(
            f"Vertex count mismatch: labels {seed.shape[-1]} vs data {data.shape[-1]}"
        )

    if medial_wall_path is not None:
        mw = load_surf_data(medial_wall_path).astype(bool)  # 1=cortex
        if mw.shape[-1] != data.shape[-1]:
            raise ValueError(
                f"Vertex count mismatch: MW {mw.shape[-1]} vs data {data.shape[-1]}"
            )
        allowed = mw
    else:
        allowed = np.ones_like(seed, dtype=bool)

    finite = np.isfinite(data)
    valid  = finite & allowed

    # 3) Preferred mask: inside ROI AND finite AND cortex
    seed_allowed = seed & allowed
    final_mask = seed_allowed & valid

    rings_used = 0
    expanded = False

    # 4) If empty, expand on-surface to nearest FINITE cortical vertices
    if not final_mask.any() and seed_allowed.any():
        neighbors = _neighbors_cached(mesh_path)
        final_mask, rings_used = _expand_seed_to_valid(
            seed_mask=seed_allowed,
            valid_mask=valid,
            neighbors=neighbors,
            max_rings=expand_rings,
            allowed_mask=allowed
        )
        expanded = rings_used >= 1

    used_nan = False
    if not final_mask.any():
        used_nan = True
        roi_means = np.array([np.nan], dtype=float)
        roi_codes = np.array([], dtype=int)
    else:
        # 5) Build labels for the masker: 1 for ROI, 0 background
        lab_for_masker = np.zeros_like(seed, dtype=int)
        lab_for_masker[final_mask] = 1

        # Mesh → PolyMesh (unchanged from your code)
        hemi_left = ("hem-L" in mesh_path or ".L." in mesh_path or "Left" in mesh_path)
        smesh = load_surf_mesh(mesh_path)
        if hemi_left:
            mesh = PolyMesh(left=smesh); hemi_key = "left"
        else:
            mesh = PolyMesh(right=smesh); hemi_key = "right"

        labels_img = SurfaceImage(mesh=mesh, data={hemi_key: lab_for_masker})
        surf_img   = SurfaceImage(mesh=mesh, data={hemi_key: data})

        masker = SurfaceLabelsMasker(labels_img=labels_img, background_label=0).fit()
        out = masker.transform(surf_img)
        roi_means = out if out.ndim == 1 else out.squeeze(0)
        roi_codes = np.array([1], dtype=int)

    # ---------- LOG ----------
    if log_csv_path is not None:
        ctx = log_context.copy() if log_context else {}
        row = dict(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            mesh=os.path.basename(mesh_path),
            mask_file=os.path.basename(roi_gifti_path),
            map_file=os.path.basename(surfmap_path),
            n_vertices=int(seed.size),
            seed_nonzero=int(np.count_nonzero(seed)),
            seed_allowed=int(np.count_nonzero(seed_allowed)),
            finite_total=int(np.count_nonzero(finite)),
            valid_total=int(np.count_nonzero(valid)),
            initial_roi_finite=int(np.count_nonzero(seed_allowed & finite)),
            expanded=int(expanded),
            rings_used=(int(rings_used) if not used_nan else -1),
            final_mask_size=int(np.count_nonzero(final_mask)),
            returned_mean=(float(roi_means[0]) if not used_nan else float("nan")),
            used_nan=int(used_nan),
            cortex_frac=float(allowed.mean())
        )
        row.update(ctx)
        _write_log_row(log_csv_path, row)

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

tags = ['i', 'i9a', 'i8a', 'i7a', 'i6a', 'a', 'a4g', 'a3g', 'a2g', 'a1g', 'g']
# tags = ['i6a']

# ############################# RUN ###################################

if __name__ == '__main__':

    for region_name, atlas_name, roi_name in zip(region_names,
                                                 atlas_names,
                                                 roi_names):
        # Define ROI-specific folders
        roi_folder = os.path.join(main_dir, folder_name, region_name, 
                                  atlas_name, roi_name)
          
        for tag in tags:

            # Ensure output dir exists early (we also save the log here)
            roi_surf_dir = os.path.join(roi_folder, 'rois_surf_extraction')
            os.makedirs(roi_surf_dir, exist_ok=True)

            # One CSV log per tag/ROI/contrast type
            log_csv_path = os.path.join(
                roi_surf_dir,
                f'{tag}_{roi_name}_{contype}_extraction_log.csv'
            )
            # If you prefer a fresh log each run, uncomment:
            if os.path.exists(log_csv_path):
                os.remove(log_csv_path)

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
                                background_label=0,              # or -1 if that's your medial wall
                                expand_rings=3,                  # adjust to 3 if you want to be more aggressive
                                log_csv_path=log_csv_path,
                                log_context={
                                    "region": region_name,
                                    "atlas": atlas_name,
                                    "roi": roi_name,
                                    "tag": tag,
                                    "subject": f"{subject:02d}",
                                    "hemisphere": hem,
                                    "task": task_key,
                                    "contrast_id": key,
                                    "contrast_name": value
                                }
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