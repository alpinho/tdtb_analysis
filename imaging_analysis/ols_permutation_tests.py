"""
Script: One-sample permuted OLS on volume, surface & SUIT contrast maps
        with BH-FDR on z. Plots:
          - Volume: glass-brain (viridis, colorbar at z*).
          - Surface (fs_LR 32k): flatmap (one or two contrasts).
          - SUIT (cerebellum): flatmap (one or two contrasts).
        SUIT projection uses the group volume t-map (as in
        volume_to_suit.py): project volume t to SUIT, convert to z,
        and apply BH-FDR on z for plotting.

Author: Ana Luisa Pinho
Email: agrilopi@uwo.ca

Creation: 15th of August 2025
Last Update: August 2025

Compatibility: Python 3.10.14, nilearn 0.11.1, SUITPy 1.x
"""

import os
import numpy as np
import nibabel as nib
import matplotlib as mpl

from scipy import stats
from nilearn import plotting
from nilearn.maskers import NiftiMasker
from nilearn.mass_univariate import permuted_ols
from nilearn.glm.thresholding import fdr_threshold

# Surface helpers from sibling script (same directory)
from volume_to_surface import (
    individual_surf,
    get_isurf_cifti,
    mask_cortical_activation,
    plot_flatmap,
)

# SUIT helpers/libs
from SUITPy import flatmap as suit_flatmap
import nitools as nt  # optional: write GIFTI for SUIT vectors

# ============================ TOGGLES ==================================

RUN_VOLUME = True
RUN_SURFACE = False
RUN_SUIT = False

# Run all single-contrast maps
RUN_ALL_CONTRASTS = True

# Threshold sources for plotting:
#   - 'volume'  : use volume FDR z-threshold(s)
#   - 'surface' : compute from surface z directly
#   - 'suit'    : compute from SUIT z directly
SURFACE_THR_SOURCE = 'volume'
SUIT_THR_SOURCE = 'suit'   # 'volume' or 'suit'

# ============================ SETTINGS =================================

# Surface settings (use one constant everywhere)
SURFSPACE = 'fslr32k'
make_cifti_if_missing = True

# Two-contrast default colors (requested)
TWO_COLORS = ('#009E73', '#F0E442')

# ============================ INPUTS ===================================

SUBJECTS = [
    3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26,
    28, 29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
]

task_tag = 'Perception'  # 'Production', 'Perception', 'NTFD', 'Randomized NTFD', 'All Tasks'
contrast_name = 'Encoding'       # first contrast (required)
contrast_name2 = None  # None or optional second contrast

n_permutations = 10000
two_sided_test = False
fdr_alpha = 0.05

# ========================= PATHS / LABELS ==============================

# Tasks definitions
tasks = {'prod': 'Production', 
         'percep': 'Perception', 
         'ntfd': 'NTFD',
         'rand_ntfd': 'Randomized NTFD',
         'allmain_tasks': 'All Tasks'
}
task_id = {v: k for k, v in tasks.items()}.get(task_tag)

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
        8: 'Random',
        9: 'Beat vs Interval',
        10: 'Interval vs Beat',
        11: 'Beat vs Random',
        12: 'Random vs Beat',
        13: 'Interval vs Random',
        14: 'Random vs Interval',
        15: 'Non-Random vs Random',
        16: 'Random vs Non-Random',
        17: 'Auditory Beat',
        18: 'Auditory Interval',
        19: 'Auditory Random',
        20: 'Auditory Beat vs Auditory Interval',
        21: 'Auditory Interval vs Auditory Beat',
        22: 'Auditory Beat vs Auditory Random',
        23: 'Auditory Random vs Auditory Beat',
        24: 'Auditory Interval vs Auditory Random',
        25: 'Auditory Random vs Auditory Interval',
        26: 'Auditory Non-Random vs Auditory Random',
        27: 'Auditory Random vs Auditory Non-Random',
        28: 'Visual Beat',
        29: 'Visual Interval',
        30: 'Visual Random',
        31: 'Visual Beat vs Visual Interval',
        32: 'Visual Interval vs Visual Beat',
        33: 'Visual Beat vs Visual Random',
        34: 'Visual Random vs Visual Beat',                    
        35: 'Visual Interval vs Visual Random',
        36: 'Visual Random vs Visual Interval',
        37: 'Visual Non-Random vs Visual Random',
        38: 'Visual Random vs Visual Non-Random',
        39: 'Decision'
    }

contrast_id = {v: k for k, v in all_contrasts.items()}.get(contrast_name)
contrast_id2 = (
    {v: k for k, v in all_contrasts.items()}.get(contrast_name2)
    if contrast_name2 else None
)

if os.path.isdir('/home/analu/diedrichsen_data/data'):
    base_dir = '/home/analu/diedrichsen_data/data'
else:
    base_dir = '/cifs/diedrichsen/data'

music = os.path.join(base_dir, 'Cerebellum', 'music-sdtb')
derivatives_folder = os.path.join(music, 'derivatives')
GM_MASK_PATH = os.path.join(derivatives_folder, 'group', 'anat',
                            'group_mask_gray.nii')

# Volume outputs
out_root_vol = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'results', 'ols_permutation_tests', 'volume', task_id
)

# Surface files (CIFTI) and plots
surf_files_root = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'results', 'surface_files',
)
surf_plots_root = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'results', 'ols_permutation_tests', 'surface',
)

# SUIT outputs (volume t → SUIT)
suit_files_root = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'results', 'ols_permutation_tests', 'suit', 'files',
)
suit_plots_root = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'results', 'ols_permutation_tests', 'suit', 'plots',
)

# Medial wall masks (fs_LR 32k)
fslr32k_folder = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'fslr32k_meshes'
)
mask_suffix = '1'
lh_medial_wall_mask_path = os.path.join(
    fslr32k_folder, 'medialwall_masks',
    f'fs_LR.32k.L.medialwall.mask{mask_suffix}.gii'
)
rh_medial_wall_mask_path = os.path.join(
    fslr32k_folder, 'medialwall_masks',
    f'fs_LR.32k.R.medialwall.mask{mask_suffix}.gii'
)

# ========================== HELPERS ====================================

def sanitize_label_kebab(label: str) -> str:
    """Folder-safe: lowercase, spaces→'-', and '-vs-'→'_vs_'."""
    if label is None:
        return None
    s = label.lower().strip()
    s = s.replace(' ', '-')
    s = s.replace('-vs-', '_vs_')
    return s


def sanitize_label_snake(label: str) -> str:
    """File-stem-safe: lowercase, spaces→'_', and '-vs-'→'_vs_'."""
    if label is None:
        return None
    s = label.lower().strip()
    s = s.replace(' ', '_')
    s = s.replace('-vs-', '_vs_')
    return s


def cap_label(label: str) -> str:
    """Capitalize label for titles (keep full-uppercase acronyms)."""
    if label is None:
        return None
    if label.isupper():
        return label
    return " ".join(w.capitalize() for w in label.split())


def id_label_folder(cid: int, cname: str) -> str:
    """Return '1_encoding' (id + '_' + snake_case label)."""
    return f"{cid}_{sanitize_label_snake(cname)}"


def overlay_folder_kebab(c1: str, c2: str) -> str:
    """Return 'auditory-beat_and_auditory-interval' folder name."""
    return f"{sanitize_label_kebab(c1)}_and_{sanitize_label_kebab(c2)}"


label1_kebab = sanitize_label_kebab(contrast_name)
label1_snake = sanitize_label_snake(contrast_name)
label2_kebab = (
    sanitize_label_kebab(contrast_name2) if contrast_name2 else None
)
label2_snake = (
    sanitize_label_snake(contrast_name2) if contrast_name2 else None
)

cap1 = cap_label(contrast_name)
cap2 = cap_label(contrast_name2) if contrast_name2 else None

# Canonical single-contrast folder names
idlabel1 = id_label_folder(contrast_id, contrast_name)
idlabel2 = (
    id_label_folder(contrast_id2, contrast_name2) if contrast_name2 else None
)

def group_df() -> int:
    """Degrees of freedom for one-sample group t."""
    return len(SUBJECTS) - 1


def load_lr_from_cifti(cifti_path: str):
    """Load CIFTI dense scalar and split LH and RH arrays."""
    img = nib.load(cifti_path)
    data = np.asanyarray(img.get_fdata())
    if data.ndim == 2 and data.shape[0] == 1:
        data = data[0]
    bm_list = img.header.get_index_map(1).brain_models
    lh = rh = None
    for bm in bm_list:
        bs = bm.brain_structure.upper()
        start = bm.index_offset
        stop = start + bm.index_count
        if bs == 'CIFTI_STRUCTURE_CORTEX_LEFT':
            lh = data[start:stop]
        elif bs == 'CIFTI_STRUCTURE_CORTEX_RIGHT':
            rh = data[start:stop]
    if lh is None or rh is None:
        raise ValueError("Both hemispheres not found in CIFTI.")
    return lh, rh


def run_permuted_ols_one_sample(
    Y, n_perm=10000, two_sided=False, n_jobs=1, random_state=42
):
    """One-sample test across subjects: mean effect > 0 or != 0."""
    n_subj = Y.shape[0]
    tested = np.ones((n_subj, 1))
    outputs = permuted_ols(
        tested_vars=tested,
        target_vars=Y,
        confounding_vars=None,
        model_intercept=False,
        n_perm=n_perm,
        two_sided_test=two_sided,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=1,
        output_type='dict',
    )
    return outputs


def save_stat_maps_zfdr(
    masker, t_vec, df, out_dir, prefix, alpha_fdr=0.05, two_sided=False
):
    """
    Save t-map and z-map. Compute FDR(z) threshold for plotting.

    Returns
    -------
    t_path, z_path, z_thr
    """
    os.makedirs(out_dir, exist_ok=True)

    if two_sided:
        p_unc = 2.0 * stats.t.sf(np.abs(t_vec), df)
        p_unc = np.clip(p_unc, 1e-300, 1.0)
        z_abs = stats.norm.isf(p_unc / 2.0)
        z_vec = np.sign(t_vec) * z_abs
        valid = np.isfinite(z_abs) & (z_abs > 0)
        if not np.any(valid):
            raise RuntimeError("No finite z for two-sided FDR.")
        z_thr = fdr_threshold(z_abs[valid], alpha=alpha_fdr)
    else:
        p_unc = stats.t.sf(t_vec, df)
        p_unc = np.clip(p_unc, 1e-300, 1.0)
        z_vec = stats.norm.isf(p_unc)
        valid = np.isfinite(z_vec) & (z_vec > 0)
        if not np.any(valid):
            raise RuntimeError("No positive z for one-sided FDR.")
        z_thr = fdr_threshold(z_vec[valid], alpha=alpha_fdr)

    z_vec = np.asarray(z_vec, dtype=float)
    z_vec[~np.isfinite(z_vec)] = 0.0

    t_img = masker.inverse_transform(t_vec)
    z_img = masker.inverse_transform(z_vec)

    t_path = os.path.join(out_dir, f"{prefix}_tmap.nii.gz")
    z_path = os.path.join(out_dir, f"{prefix}_zmap.nii.gz")

    nib.save(t_img, t_path)
    nib.save(z_img, z_path)

    return t_path, z_path, float(z_thr)


def fdr_z_threshold_from_arrays(arrays, alpha=0.05, two_sided=False):
    """Compute BH-FDR z-threshold from one or more arrays."""
    zcat = np.concatenate([np.ravel(a) for a in arrays])
    zuse = np.abs(zcat) if two_sided else zcat
    valid = np.isfinite(zuse) & (zuse > 0)
    if not np.any(valid):
        print("[FDR] No valid z > 0; using z_thr = inf.")
        return float('inf')
    return float(fdr_threshold(zuse[valid], alpha=alpha))


def plot_glass_brain_z(
    z_map_path, z_threshold, two_sided, title, out_png,
    cbar_contrast_label=None
):
    """
    Plot z-map with viridis. Colorbar starts at z-threshold.
    """
    z_img = nib.load(z_map_path)
    z = np.asanyarray(z_img.dataobj)

    z_for_vmax = np.abs(z) if two_sided else z
    finite = np.isfinite(z_for_vmax)
    above = finite & (z_for_vmax > z_threshold)

    if np.any(above):
        vmax = float(np.nanmax(z_for_vmax[above]))
        colorbar_flag = True
    else:
        vmax = float(z_threshold) + 1e-6
        colorbar_flag = False
        print("[plot] No voxels exceed FDR z-threshold; omitting colorbar.")

    disp = plotting.plot_glass_brain(
        z_map_path,
        threshold=z_threshold,
        colorbar=colorbar_flag,
        display_mode='lyrz',
        cmap='viridis',
        plot_abs=True if two_sided else False,
        vmin=z_threshold,
        vmax=vmax,
        symmetric_cbar=False,
    )
    disp.title(title, size=10)

    if colorbar_flag and cbar_contrast_label:
        cbar_obj = getattr(disp, "_colorbar", None) or getattr(
            disp, "cbar", None
        )
        if isinstance(cbar_obj, mpl.colorbar.Colorbar):
            cbar_obj.set_label(
                f"Z-values ({cbar_contrast_label})",
                fontsize=9,
                labelpad=6,
            )

    disp.savefig(out_png, dpi=300)
    disp.close()

# ---- NEW: helper to mask the z-map for visualization only --------------

def apply_visual_mask(src_img_path, mask_path, out_path):
    """
    Save a copy of src_img masked by mask_path (0 outside mask).
    Does not change the original stats; for visualization only.
    """
    if not mask_path:
        # no-op: return original
        return src_img_path
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"GM mask not found: {mask_path}")

    z_img = nib.load(src_img_path)
    z = np.asanyarray(z_img.dataobj).astype(float)

    m_img = nib.load(mask_path)
    m = np.asanyarray(m_img.dataobj).astype(bool)

    if z.shape != m.shape:
        raise ValueError(
            f"Mask shape {m.shape} != image shape {z.shape}. "
            "Resample your GM mask to match the z-map grid."
        )

    z_masked = np.where(m, z, 0.0)
    nib.save(nib.Nifti1Image(z_masked, z_img.affine, z_img.header), out_path)
    return out_path

# ============================ VOLUME ===================================

def get_single_contrast_paths(
    derivatives_dir, subjects, task_key, contrast_key,
    derivative_type='sm8wbmasked'
):
    """Build paths to a single individual contrast map per subject."""
    fname = f"wcon_{contrast_key:04d}_desc-{derivative_type}.nii"
    conpaths = [
        os.path.join(
            derivatives_dir,
            f"sub-{sub:02d}",
            "estimates",
            task_key,
            "masked_derivatives_rwls_dbb_hrf128",
            fname,
        )
        for sub in subjects
    ]
    return conpaths


def fit_masker_and_stack(conpaths, mask_img=None, smoothing_fwhm=None):
    """Fit a masker and transform all subject images into a 2D array."""
    masker = NiftiMasker(
        mask_img=mask_img,
        smoothing_fwhm=smoothing_fwhm,
        standardize=False,
        detrend=False,
        memory=None,
        verbose=0,
    )
    Y = masker.fit_transform(conpaths)
    ref_img = nib.load(conpaths[0])
    return masker, Y, ref_img.affine, ref_img.header


def run_volume_pipeline_one(contrast_nm, contrast_k, label_out, cap_out):
    """
    Run permuted OLS in volume for a single contrast.

    Returns
    -------
    z_thr, z_max, z_path
    """
    con_folder = os.path.join(out_root_vol, label_out)
    os.makedirs(con_folder, exist_ok=True)

    conpaths = get_single_contrast_paths(
        derivatives_folder, SUBJECTS, task_id, contrast_k,
        derivative_type='sm8wbmasked',
    )

    # NOTE: keep mask_img=None to avoid changing stats
    masker, Y, _, _ = fit_masker_and_stack(
        conpaths, mask_img=None, smoothing_fwhm=None,
    )

    n_subj = Y.shape[0]
    df = n_subj - 1

    cache_npz = os.path.join(
        con_folder,
        f"{label_out}_permuted_ols_{n_permutations}perm_"
        f"{'2s' if two_sided_test else '1s'}.npz",
    )

    if os.path.exists(cache_npz):
        print(f"[volume] Loading cached: {cache_npz}")
        data = np.load(cache_npz)
        outputs = {'t': data['t']}
    else:
        print(f"[volume] Running permuted_ols: {contrast_nm}")
        outputs = run_permuted_ols_one_sample(
            Y, n_perm=n_permutations, two_sided=two_sided_test,
            n_jobs=-1, random_state=42,
        )
        np.savez(cache_npz, t=outputs['t'])
        print(f"[volume] Saved: {cache_npz}")

    t_vec = outputs['t'][0]
    t_path, z_path, z_thr = save_stat_maps_zfdr(
        masker=masker, t_vec=t_vec, df=df, out_dir=con_folder,
        prefix=label_out, alpha_fdr=fdr_alpha, two_sided=two_sided_test,
    )

    # ---- NEW: produce a GM-masked copy of z-map for visualization only
    z_path_for_plot = z_path
    if GM_MASK_PATH:
        z_path_for_plot = os.path.join(
            con_folder, f"{label_out}_zmap_gmmasked.nii.gz"
        )
        apply_visual_mask(z_path, GM_MASK_PATH, z_path_for_plot)

    png_path = os.path.join(
        con_folder,
        f"{label_out}_glassbrain_zmap_FDRz{int(fdr_alpha*100)}.png",
    )
    title = (
        f"{contrast_nm}: one-sample permuted OLS "
        f"(FDR on z < {fdr_alpha}, "
        f"{'two-sided |z|' if two_sided_test else 'one-sided z+'}; "
        f"z* = {z_thr:.3f})"
    )
    plot_glass_brain_z(
        z_map_path=z_path_for_plot,  # <- masked copy (if provided)
        z_threshold=z_thr,
        two_sided=two_sided_test,
        title=title,
        out_png=png_path,
        cbar_contrast_label=cap_out,
    )

    z_img = nib.load(z_path)
    z = np.asanyarray(z_img.dataobj)
    zmax = float(np.nanmax(np.abs(z)))

    print("[volume] Outputs:")
    print(f"  t-map: {t_path}")
    print(f"  z-map: {z_path}")
    if GM_MASK_PATH:
        print(f"  z-map (GM-masked for plot): {z_path_for_plot}")
    print(f"  figure: {png_path}")

    return z_thr, zmax, z_path

# ============================ SURFACE ==================================

def build_surface_matrices(contrast_k, contrast_lbl_kebab):
    """
    Ensure subject CIFTIs exist; return LH/RH matrices of shape
    (subjects, vertices) in fs_LR 32k space.

    Note: CIFTI folder uses '{id}_{kebab-label}' for compatibility
    with your existing surface export convention.
    """
    cifti_dir = os.path.join(
        surf_files_root, f"{contrast_k}_{contrast_lbl_kebab}"
    )
    cifti0 = os.path.join(
        cifti_dir,
        f"sub-{SUBJECTS[0]:02d}_{task_id.replace('_','-')}_"
        f"{contrast_lbl_kebab}_{SURFSPACE}.dscalar.nii",
    )
    if make_cifti_if_missing and not os.path.exists(cifti0):
        print("[surface] Creating subject CIFTI (vol->surf)...")
        individual_surf(
            derivatives_dir=derivatives_folder,
            subjects=SUBJECTS,
            task_key=task_id,
            contrast_key=contrast_k,
            surf_dir=surf_files_root,
            surfspace=SURFSPACE,
            save='cifti',
        )

    cifti_files = get_isurf_cifti(
        surf_files_root, SUBJECTS, task_id, contrast_k, contrast_lbl_kebab,
        surfspace=SURFSPACE,
    )

    lh_list, rh_list = [], []
    for path in cifti_files:
        lh, rh = load_lr_from_cifti(path)
        lh_list.append(lh)
        rh_list.append(rh)

    Y_lh = np.vstack(lh_list).astype(float)
    Y_rh = np.vstack(rh_list).astype(float)
    Y_lh[~np.isfinite(Y_lh)] = 0.0
    Y_rh[~np.isfinite(Y_rh)] = 0.0
    return Y_lh, Y_rh


def surface_z_for_one_contrast(contrast_k, contrast_nm, contrast_lbl_kebab):
    """Per-vertex permuted OLS on surface; return masked LH/RH z."""
    Y_lh, Y_rh = build_surface_matrices(contrast_k, contrast_lbl_kebab)
    n_subj = Y_lh.shape[0]
    df = n_subj - 1

    print(f"[surface] permuted_ols LH: {contrast_nm}")
    out_lh = run_permuted_ols_one_sample(
        Y_lh, n_perm=n_permutations, two_sided=two_sided_test,
        n_jobs=-1, random_state=42,
    )
    print(f"[surface] permuted_ols RH: {contrast_nm}")
    out_rh = run_permuted_ols_one_sample(
        Y_rh, n_perm=n_permutations, two_sided=two_sided_test,
        n_jobs=-1, random_state=42,
    )

    t_lh = out_lh['t'][0]
    t_rh = out_rh['t'][0]
    if two_sided_test:
        p_lh = 2.0 * stats.t.sf(np.abs(t_lh), df)
        p_rh = 2.0 * stats.t.sf(np.abs(t_rh), df)
        z_lh = np.sign(t_lh) * stats.norm.isf(p_lh / 2.0)
        z_rh = np.sign(t_rh) * stats.norm.isf(p_rh / 2.0)
    else:
        p_lh = stats.t.sf(t_lh, df)
        p_rh = stats.t.sf(t_rh, df)
        z_lh = stats.norm.isf(p_lh)
        z_rh = stats.norm.isf(p_rh)

    z_lh = mask_cortical_activation(z_lh, lh_medial_wall_mask_path)
    z_rh = mask_cortical_activation(z_rh, rh_medial_wall_mask_path)
    return z_lh, z_rh


def run_surface_plot_single(thr_mode='volume', zthr_from_volume=None):
    """Single-contrast surface flatmap with volume/surface FDR(z)."""
    z_lh1, z_rh1 = surface_z_for_one_contrast(
        contrast_id, contrast_name, label1_kebab
    )

    if thr_mode == 'surface' or not np.isfinite(zthr_from_volume or np.nan):
        zthr1 = fdr_z_threshold_from_arrays(
            [z_lh1, z_rh1], alpha=fdr_alpha, two_sided=two_sided_test
        )
    else:
        zthr1 = float(zthr_from_volume)

    vmax1 = float(
        max(np.nanmax(np.abs(z_lh1)), np.nanmax(np.abs(z_rh1)))
    )
    vmax1 = max(vmax1, float(zthr1) + 1e-6)

    # Single-contrast surface plots folder: '<id>_<snake>'
    outdir = os.path.join(surf_plots_root, idlabel1)
    os.makedirs(outdir, exist_ok=True)

    # Use kebab-case tag (no spaces) for filenames inside the folder
    plot_flatmap(
        stats=[z_lh1, z_rh1],
        threshold=zthr1,
        task_key=task_id,
        contrast_tag=label1_kebab,
        output_dir=outdir,
        hemi=['L', 'R'],
        colormap='viridis',
        vmax=vmax1,
    )
    print("[surface] Flatmap (single) saved to:", outdir)


def run_surface_plot_two(
    thr_mode="volume", zthr1_vol=None, zthr2_vol=None, colors=TWO_COLORS
):
    """
    Two-contrast surface overlay (fs_LR 32k), with BH-FDR on z.

    Colorbar labels must be capitalized (only labels). We pass a
    capitalized-kebab tag with `_vs_` to plot_flatmap so it derives
    capitalized labels, but we still save the final file using `_and_`.
    """
    if contrast_name2 is None or contrast_id2 is None:
        raise ValueError("contrast_name2/contrast_id2 must be set.")

    # Compute surface z for both contrasts via per-vertex permuted OLS
    z_lh1, z_rh1 = surface_z_for_one_contrast(
        contrast_id, contrast_name, label1_kebab
    )
    z_lh2, z_rh2 = surface_z_for_one_contrast(
        contrast_id2, contrast_name2, label2_kebab
    )

    # Thresholds: from surface z or from volume (if provided)
    if thr_mode == "surface":
        zthr1 = fdr_z_threshold_from_arrays(
            [z_lh1, z_rh1], alpha=fdr_alpha, two_sided=two_sided_test
        )
        zthr2 = fdr_z_threshold_from_arrays(
            [z_lh2, z_rh2], alpha=fdr_alpha, two_sided=two_sided_test
        )
    else:
        if zthr1_vol is None or zthr2_vol is None:
            raise ValueError("Volume z-thresholds must be provided.")
        zthr1 = float(zthr1_vol)
        zthr2 = float(zthr2_vol)

    # Color scaling per contrast
    v1 = float(max(np.nanmax(np.abs(z_lh1)), np.nanmax(np.abs(z_rh1))))
    v2 = float(max(np.nanmax(np.abs(z_lh2)), np.nanmax(np.abs(z_rh2))))
    v1 = max(v1, float(zthr1) + 1e-6)
    v2 = max(v2, float(zthr2) + 1e-6)

    # Output folder: rgba/<kebab1>_and_<kebab2>
    two_folder = overlay_folder_kebab(contrast_name, contrast_name2)
    outdir = os.path.join(surf_plots_root, "rgba", two_folder)
    os.makedirs(outdir, exist_ok=True)

    # Capitalized-kebab tag with `_vs_` for colorbar labels (no spaces)
    cap1_kebab = cap1.replace(" ", "-")
    cap2_kebab = cap2.replace(" ", "-")
    contrast_tag_cap = f"{cap1_kebab}_vs_{cap2_kebab}"

    # Plot (volume_to_surface.plot_flatmap remains unchanged)
    plot_flatmap(
        stats=[[z_lh1, z_rh1], [z_lh2, z_rh2]],
        threshold=[zthr1, zthr2],
        task_key=task_id,
        contrast_tag=contrast_tag_cap,
        output_dir=outdir,
        hemi=["L", "R"],
        colors=list(colors),
        vmax=[v1, v2],
    )

    # Ensure final filename uses "_and_" instead of "_vs_"
    task_dash = task_id.replace("_", "-")
    produced_tag = contrast_tag_cap.lower()
    src_png = os.path.join(
        outdir,
        f"group_{task_dash}_{produced_tag}_flat_overlay_{SURFSPACE}.png",
    )
    alt_src_png = os.path.join(
        outdir,
        f"group_{task_dash}_{contrast_tag_cap}_flat_overlay_{SURFSPACE}.png",
    )
    dst_png = os.path.join(
        outdir,
        f"group_{task_dash}_{two_folder}_flat_overlay_{SURFSPACE}.png",
    )

    try:
        if os.path.exists(src_png) and src_png != dst_png:
            os.replace(src_png, dst_png)
            print("[surface] Renamed output to:", dst_png)
        elif os.path.exists(alt_src_png) and alt_src_png != dst_png:
            os.replace(alt_src_png, dst_png)
            print("[surface] Renamed output to:", dst_png)
        else:
            if os.path.exists(dst_png):
                print("[surface] Flatmap (two-contrast) saved to:", dst_png)
            else:
                print("[surface] Flatmap (two-contrast) saved to:", outdir)
    except Exception as e:
        print("[surface] Rename skipped due to error:", e)
        print("[surface] Check outputs in:", outdir)

# ============================== SUIT ===================================

def volume_tmap_path(label_out: str) -> str:
    """Path to saved group t-map in volume pipeline."""
    return os.path.join(out_root_vol, label_out, f"{label_out}_tmap.nii.gz")


def project_volume_t_to_suit_z(label_out: str, contrast_nm: str) -> tuple:
    """
    Project the group volume t-map to SUIT and convert to z.

    Returns
    -------
    z_vec : np.ndarray (n_vertices,)
    zmax  : float
    gii_path : str (optional saved GIFTI path)
    """
    tmap_path = volume_tmap_path(label_out)
    if not os.path.exists(tmap_path):
        raise FileNotFoundError(
            f"Group t-map not found: {tmap_path}. "
            "Run the volume pipeline first or provide the file."
        )

    arr = suit_flatmap.vol_to_surf(tmap_path, space='SUIT')
    t_vec = np.squeeze(np.asarray(arr), axis=1).astype(float)
    t_vec[~np.isfinite(t_vec)] = 0.0

    df = group_df()
    if two_sided_test:
        p_unc = 2.0 * stats.t.sf(np.abs(t_vec), df)
        p_unc = np.clip(p_unc, 1e-300, 1.0)
        z_abs = stats.norm.isf(p_unc / 2.0)
        z_vec = np.sign(t_vec) * z_abs
    else:
        p_unc = stats.t.sf(t_vec, df)
        p_unc = np.clip(p_unc, 1e-300, 1.0)
        z_vec = stats.norm.isf(p_unc)

    zmax = float(np.nanmax(np.abs(z_vec)))

    # Save SUIT z as GIFTI under single-contrast folder '<id>_<snake>'
    os.makedirs(os.path.join(suit_files_root, label_out), exist_ok=True)
    z_gii = nt.gifti.make_func_gifti(
        z_vec,
        anatomical_struct='Cerebellum',
        column_names=[contrast_nm.replace(' ', '_')],
    )
    gii_path = os.path.join(
        suit_files_root, label_out, f"{label_out}_suit_z.func.gii"
    )
    nib.save(z_gii, gii_path)
    print(f"[SUIT] Saved z GIFTI from volume t-map: {gii_path}")

    return z_vec, zmax, gii_path


def run_suit_plot_single(thr_mode='suit', zthr_from_volume=None):
    """
    Single-contrast SUIT flatmap from volume t-map. If thr_mode='suit',
    compute BH-FDR on SUIT z; else use provided volume z-threshold.
    """
    out_label = idlabel1
    z_vec, zmax, _ = project_volume_t_to_suit_z(out_label, contrast_name)

    if thr_mode == 'suit' or not np.isfinite(zthr_from_volume or np.nan):
        zthr = fdr_z_threshold_from_arrays(
            [z_vec], alpha=fdr_alpha, two_sided=two_sided_test
        )
    else:
        zthr = float(zthr_from_volume)

    vmax = max(zmax, float(zthr) + 1e-6 if np.isfinite(zthr) else zmax)

    # Single-contrast SUIT plots folder: '<id>_<snake>'
    os.makedirs(os.path.join(suit_plots_root, out_label), exist_ok=True)
    out_png = os.path.join(
        suit_plots_root, out_label,
        f"group_{task_id.replace('_', '-')}_{out_label}_suit.png",
    )

    from volume_to_suit import plot_suitflat
    plot_suitflat(
        stats=z_vec,
        threshold=zthr,
        outpath=out_png,
        colormap='viridis',
        vmax=vmax,
    )
    print("[SUIT] Flatmap (single) saved to:", out_png)


def run_suit_plot_two(
    thr_mode='suit', zthr1_vol=None, zthr2_vol=None, colors=TWO_COLORS
):
    """
    Two-contrast SUIT overlay from volume t-maps. If thr_mode='suit',
    compute BH-FDR on each SUIT z; else use volume z-thresholds.
    """
    if contrast_name2 is None or contrast_id2 is None:
        raise ValueError("contrast_name2/contrast_id2 must be set.")

    z1, z1max, _ = project_volume_t_to_suit_z(idlabel1, contrast_name)
    z2, z2max, _ = project_volume_t_to_suit_z(idlabel2, contrast_name2)

    if thr_mode == 'suit':
        zthr1 = fdr_z_threshold_from_arrays(
            [z1], alpha=fdr_alpha, two_sided=two_sided_test
        )
        zthr2 = fdr_z_threshold_from_arrays(
            [z2], alpha=fdr_alpha, two_sided=two_sided_test
        )
    else:
        zthr1 = float(zthr1_vol)
        zthr2 = float(zthr2_vol)

    v1 = max(z1max, float(zthr1) + 1e-6 if np.isfinite(zthr1) else z1max)
    v2 = max(z2max, float(zthr2) + 1e-6 if np.isfinite(zthr2) else z2max)

    # Two-contrast SUIT plots under:
    #   '<suit_plots_root>/rgba/<kebab1>_and_<kebab2>'
    two_folder = overlay_folder_kebab(contrast_name, contrast_name2)
    out_dir = os.path.join(suit_plots_root, 'rgba', two_folder)
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(
        out_dir,
        f"group_{task_id.replace('_', '-')}_{two_folder}_suit.png",
    )

    from volume_to_suit import plot_suitflat
    plot_suitflat(
        stats=[z1, z2],
        threshold=[zthr1, zthr2],
        colors=list(colors),
        vmax=[v1, v2],
        outpath=out_png,
    )
    print("[SUIT] Flatmap (two-contrast) saved to:", out_png)

# ============================ MAIN =====================================

if __name__ == '__main__':
    if RUN_ALL_CONTRASTS:
        # --- run every single contrast in all_contrasts (no overlays) ---
        for cid, cname in all_contrasts.items():  # dict {id: name}
            print(f"\n[MAIN] Running single-contrast: {cid} – {cname}")

            # Canonical labels for this contrast
            idlabel = id_label_folder(cid, cname)            # e.g. "14_visual_beat"
            cap = cap_label(cname)                           # title-cased label

            zthr_vol = None

            # ---------- VOLUME ----------
            if RUN_VOLUME:
                zthr_vol, _, _ = run_volume_pipeline_one(     # uses your existing code
                    contrast_nm=cname,
                    contrast_k=cid,
                    label_out=idlabel,
                    cap_out=cap,
                )

            # ---------- SURFACE ----------
            if RUN_SURFACE:
                thr_mode = (
                    'volume'
                    if (RUN_VOLUME and SURFACE_THR_SOURCE == 'volume')
                    else 'surface'
                )
                # single-contrast surface plot
                # temporarily bind globals used by helpers
                contrast_id = cid
                contrast_name = cname
                label1_kebab = sanitize_label_kebab(cname)
                run_surface_plot_single(
                    thr_mode=thr_mode,
                    zthr_from_volume=zthr_vol,
                )

            # ---------- SUIT ----------
            if RUN_SUIT:
                need = not os.path.exists(volume_tmap_path(idlabel))
                if need:
                    raise FileNotFoundError(
                        f"Group t-map not found for {cname}. "
                        "Enable RUN_VOLUME or precompute volume t-maps."
                    )
                thr_mode = (
                    'volume'
                    if (RUN_VOLUME and SUIT_THR_SOURCE == 'volume')
                    else 'suit'
                )
                # temporarily bind globals used by helpers
                contrast_name = cname
                idlabel1 = idlabel
                run_suit_plot_single(
                    thr_mode=thr_mode,
                    zthr_from_volume=zthr_vol,
                )

    else:
        # --- original behavior: one contrast (and optional overlay) ---
        zthr1_vol = None
        zthr2_vol = None

        if RUN_VOLUME:
            zthr1_vol, _, _ = run_volume_pipeline_one(
                contrast_nm=contrast_name,
                contrast_k=contrast_id,
                label_out=idlabel1,
                cap_out=cap1,
            )
            if RUN_SURFACE and contrast_name2 and contrast_id2:
                zthr2_vol, _, _ = run_volume_pipeline_one(
                    contrast_nm=contrast_name2,
                    contrast_k=contrast_id2,
                    label_out=idlabel2,
                    cap_out=cap2,
                )

        if RUN_SURFACE:
            thr_mode = (
                'volume'
                if (RUN_VOLUME and SURFACE_THR_SOURCE == 'volume')
                else 'surface'
            )
            if contrast_name2 and contrast_id2:
                run_surface_plot_two(
                    thr_mode=thr_mode,
                    zthr1_vol=zthr1_vol,
                    zthr2_vol=zthr2_vol,
                    colors=TWO_COLORS,
                )
            else:
                run_surface_plot_single(
                    thr_mode=thr_mode,
                    zthr_from_volume=zthr1_vol,
                )

        if RUN_SUIT:
            need1 = not os.path.exists(volume_tmap_path(idlabel1))
            need2 = contrast_name2 and contrast_id2 and \
                (not os.path.exists(volume_tmap_path(idlabel2)))
            if need1 or need2:
                raise FileNotFoundError(
                    "Required group t-map(s) not found. "
                    "Enable RUN_VOLUME or precompute the volume t-maps."
                )
            thr_mode = (
                'volume'
                if (RUN_VOLUME and SUIT_THR_SOURCE == 'volume')
                else 'suit'
            )
            if contrast_name2 and contrast_id2:
                run_suit_plot_two(
                    thr_mode=thr_mode,
                    zthr1_vol=zthr1_vol,
                    zthr2_vol=zthr2_vol,
                    colors=TWO_COLORS,
                )
            else:
                run_suit_plot_single(
                    thr_mode=thr_mode,
                    zthr_from_volume=zthr1_vol,
                )