"""
Script: One-sample permuted OLS on volume & surface contrast maps with
        BH-FDR on z. Plots:
          - Volume: glass-brain (viridis, colorbar at z*).
          - Surface: static flatmap (fs_LR 32k), one or two contrasts.
          - Surface can run without running volume; thresholds can be
            computed from surface z directly.

Author: Ana Luisa Pinho
Email: agrilopi@uwo.ca

Creation: 15 Aug 2025
Last Update: Aug 2025

Compatibility: Python 3.10.14, nilearn 0.11.1

Notes
-----
- Volume: nilearn.mass_univariate.permuted_ols → voxelwise t.
- Convert t → p (parametric) → z; apply BH-FDR on z.
- Save unthresholded t/z NIfTI and a glass-brain PNG.
- Surface: ensure fs_LR 32k CIFTI, per-vertex permuted OLS, z, medial
  wall mask, flatmap(s).
- Two-contrast flatmap uses named colors (e.g., 'red', 'blue').
- Colorbar labels include the contrast name in parentheses (capitalized)
  for the volume glass-brain; surface labels/titles follow
  volume_to_surface.py behavior but receive capitalized contrast tags.
"""

import os
import numpy as np
import nibabel as nib

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

# ============================ TOGGLES ==================================

RUN_VOLUME = False
RUN_SURFACE = True

# Surface threshold source:
#   - 'volume'  : use volume FDR z-threshold(s) for surface plotting
#   - 'surface' : compute FDR z-threshold(s) from surface z directly
SURFACE_THR_SOURCE = 'volume'    # change to 'surface' if RUN_VOLUME = False

# ========================== HELPERS ====================================

def sanitize_label(label: str) -> str:
    """Lowercase, spaces → '-', and '-vs-' → '_vs_' for file names."""
    s = label.lower().strip().replace(' ', '-')
    s = s.replace('-vs-', '_vs_')
    return s


def cap_label(label: str) -> str:
    """
    Capitalize a contrast label for titles/labels without mangling
    acronyms that are already uppercase.
    """
    if label is None:
        return None
    if label.isupper():
        return label
    return " ".join(w.capitalize() for w in label.split())


def load_lr_from_cifti(cifti_path: str):
    """Load fs_LR 32k CIFTI dense scalar and split LH and RH arrays."""
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


def run_permuted_ols_one_sample(Y, n_perm=10000, two_sided=False, n_jobs=1,
                                random_state=42):
    """One-sample test across subjects: is mean effect > 0 or != 0."""
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


def save_stat_maps_zfdr(masker, t_vec, df, out_dir, prefix,
                        alpha_fdr=0.05, two_sided=False):
    """
    Save t-map and z-map. Compute FDR(z) threshold for plotting.

    Returns
    -------
    t_path, z_path, z_thr
    """
    os.makedirs(out_dir, exist_ok=True)

    if two_sided:
        p_unc = 2.0 * stats.t.sf(np.abs(t_vec), df)
        z_abs = stats.norm.isf(p_unc / 2.0)
        z_vec = np.sign(t_vec) * z_abs
        valid = np.isfinite(z_abs) & (z_abs > 0)
        if not np.any(valid):
            raise RuntimeError("No finite z for two-sided FDR.")
        z_thr = fdr_threshold(z_abs[valid], alpha=alpha_fdr)
    else:
        p_unc = stats.t.sf(t_vec, df)
        z_vec = stats.norm.isf(p_unc)
        valid = np.isfinite(z_vec) & (z_vec > 0)
        if not np.any(valid):
            raise RuntimeError("No positive z for one-sided FDR.")
        z_thr = fdr_threshold(z_vec[valid], alpha=alpha_fdr)

    t_img = masker.inverse_transform(t_vec)
    z_img = masker.inverse_transform(z_vec)

    t_path = os.path.join(out_dir, f"{prefix}_tmap.nii.gz")
    z_path = os.path.join(out_dir, f"{prefix}_zmap.nii.gz")

    nib.save(t_img, t_path)
    nib.save(z_img, z_path)

    return t_path, z_path, float(z_thr)


def fdr_z_threshold_from_surface(z_lh, z_rh, alpha=0.05, two_sided=False):
    """
    Compute BH-FDR threshold on surface z.

    One-sided: FDR on positive z only. Two-sided: FDR on |z|.
    Returns np.inf if no valid positive values are present.
    """
    if two_sided:
        z_abs = np.concatenate([np.abs(z_lh), np.abs(z_rh)])
        valid = np.isfinite(z_abs) & (z_abs > 0)
        if not np.any(valid):
            print("[surface] No valid |z| for FDR; using z_thr = inf.")
            return float('inf')
        return float(fdr_threshold(z_abs[valid], alpha=alpha))
    else:
        z_pos = np.concatenate([z_lh, z_rh])
        valid = np.isfinite(z_pos) & (z_pos > 0)
        if not np.any(valid):
            print("[surface] No positive z for FDR; using z_thr = inf.")
            return float('inf')
        return float(fdr_threshold(z_pos[valid], alpha=alpha))


def plot_glass_brain_z(z_map_path, z_threshold, two_sided, title, out_png,
                       cbar_contrast_label=None):
    """
    Plot z-map with viridis. Colorbar starts at z-threshold.

    If no voxel exceeds threshold, hide colorbar and use a tiny epsilon
    above vmin for vmax to avoid Normalize errors. If a contrast label
    is given, append it to the colorbar label in parentheses.
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
        try:
            cbar = getattr(disp, '_colorbar', None) or getattr(disp, 'cbar',
                                                               None)
            if cbar is not None:
                cbar.set_label(
                    f"Z-values ({cbar_contrast_label})",
                    fontsize=9,
                    labelpad=6,
                )
        except Exception as e:
            print(f"[plot] Could not set colorbar label: {e}")

    disp.savefig(out_png, dpi=300)
    disp.close()


# ============================ INPUTS ===================================

SUBJECTS = [
    3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26,
    28, 29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
]

task_tag = 'All Tasks'
contrast_name = 'Beat'        # first contrast (required)
contrast_name2 = 'Interval'             # e.g., 'Interval' for overlay

n_permutations = 10000
two_sided_test = False
fdr_alpha = 0.05

# Surface settings
surfspace = 'fslr32k'            # fs_LR 32k
make_cifti_if_missing = True     # create per-subject CIFTI if absent

# ========================= PATHS / LABELS ==============================

if os.path.isdir('/home/analu/diedrichsen_data/data'):
    base_dir = '/home/analu/diedrichsen_data/data'
else:
    base_dir = '/cifs/diedrichsen/data'

music = os.path.join(base_dir, 'Cerebellum', 'music-sdtb')
derivatives_folder = os.path.join(music, 'derivatives')

# Volume outputs
out_root_vol = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'results', 'ols_permutation_tests', 'volume',
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

tasks = {
    'prod': 'Production',
    'percep': 'Perception',
    'ntfd': 'NTFD',
    'allmain_tasks': 'All Tasks',
}
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
    18: 'Decision',
}

task_id = {v: k for k, v in tasks.items()}.get(task_tag)
contrast_id = {v: k for k, v in all_contrasts.items()}.get(contrast_name)
contrast_id2 = (
    {v: k for k, v in all_contrasts.items()}.get(contrast_name2)
    if contrast_name2 else None
)

label1 = sanitize_label(contrast_name)
label2 = sanitize_label(contrast_name2) if contrast_name2 else None

cap1 = cap_label(contrast_name)
cap2 = cap_label(contrast_name2) if contrast_name2 else None

# ============================ VOLUME ===================================

def get_single_contrast_paths(derivatives_dir, subjects, task_key,
                              contrast_key, derivative_type='sm8wbmasked'):
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
        z_map_path=z_path,
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
    print(f"  figure: {png_path}")

    return z_thr, zmax, z_path


# ============================ SURFACE ==================================

def build_surface_matrices(contrast_k, contrast_lbl):
    """
    Ensure subject CIFTIs exist; return LH/RH matrices shape
    (subjects, vertices) in fs_LR 32k space.
    """
    need_cifti = False
    if make_cifti_if_missing:
        cifti_dir = os.path.join(surf_files_root, f"{contrast_k}_{contrast_lbl}")
        cifti0 = os.path.join(
            cifti_dir,
            f"sub-{SUBJECTS[0]:02d}_{task_id.replace('_','-')}_"
            f"{contrast_lbl}_{surfspace}.dscalar.nii",
        )
        need_cifti = not os.path.exists(cifti0)

    if need_cifti:
        print("[surface] Creating subject CIFTI (vol->surf)...")
        individual_surf(
            derivatives_dir=derivatives_folder,
            subjects=SUBJECTS,
            task_key=task_id,
            contrast_key=contrast_k,
            surf_dir=surf_files_root,
            surfspace=surfspace,
            save='cifti',
        )

    cifti_files = get_isurf_cifti(
        surf_files_root, SUBJECTS, task_id, contrast_k, contrast_lbl,
        surfspace=surfspace,
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


def surface_z_for_one_contrast(contrast_k, contrast_nm, contrast_lbl):
    """
    Run per-vertex permuted OLS for one contrast; return masked LH/RH z.
    """
    Y_lh, Y_rh = build_surface_matrices(contrast_k, contrast_lbl)
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
    """
    Single-contrast flatmap. If thr_mode='surface', compute BH-FDR on
    surface z; otherwise use zthr_from_volume.
    """
    z_lh1, z_rh1 = surface_z_for_one_contrast(
        contrast_id, contrast_name, label1
    )

    if thr_mode == 'surface' or not np.isfinite(zthr_from_volume or np.nan):
        zthr1 = fdr_z_threshold_from_surface(
            z_lh1, z_rh1, alpha=fdr_alpha, two_sided=two_sided_test
        )
    else:
        zthr1 = float(zthr_from_volume)

    vmax1 = float(max(np.nanmax(np.abs(z_lh1)), np.nanmax(np.abs(z_rh1))))
    vmax1 = max(vmax1, float(zthr1) + 1e-6)

    outdir = os.path.join(surf_plots_root, label1)
    os.makedirs(outdir, exist_ok=True)

    plot_flatmap(
        stats=[z_lh1, z_rh1],
        threshold=zthr1,
        task_key=task_id,
        contrast_tag=cap1,
        output_dir=outdir,
        hemi=['L', 'R'],
        colormap='viridis',
        vmax=vmax1,
    )
    print("[surface] Flatmap (single) saved to:", outdir)


def run_surface_plot_two(thr_mode='volume', zthr1_vol=None, zthr2_vol=None,
                         colors=('red', 'blue')):
    """
    Two-contrast flatmap. If thr_mode='surface', compute BH-FDR on each
    contrast's surface z independently; else use volume thresholds.
    """
    if contrast_name2 is None or contrast_id2 is None:
        raise ValueError("contrast_name2/contrast_id2 must be set.")

    z_lh1, z_rh1 = surface_z_for_one_contrast(
        contrast_id, contrast_name, label1
    )
    z_lh2, z_rh2 = surface_z_for_one_contrast(
        contrast_id2, contrast_name2, label2
    )

    if thr_mode == 'surface':
        zthr1 = fdr_z_threshold_from_surface(
            z_lh1, z_rh1, alpha=fdr_alpha, two_sided=two_sided_test
        )
        zthr2 = fdr_z_threshold_from_surface(
            z_lh2, z_rh2, alpha=fdr_alpha, two_sided=two_sided_test
        )
    else:
        zthr1 = float(zthr1_vol)
        zthr2 = float(zthr2_vol)

    v1 = float(max(np.nanmax(np.abs(z_lh1)), np.nanmax(np.abs(z_rh1))))
    v2 = float(max(np.nanmax(np.abs(z_lh2)), np.nanmax(np.abs(z_rh2))))
    v1 = max(v1, float(zthr1) + 1e-6)
    v2 = max(v2, float(zthr2) + 1e-6)

    out_label_cap = f"{cap1}_vs_{cap2}"
    outdir = os.path.join(surf_plots_root, sanitize_label(out_label_cap))
    os.makedirs(outdir, exist_ok=True)

    plot_flatmap(
        stats=[[z_lh1, z_rh1], [z_lh2, z_rh2]],
        threshold=[zthr1, zthr2],
        task_key=task_id,
        contrast_tag=out_label_cap,
        output_dir=outdir,
        hemi=['L', 'R'],
        colors=list(colors),
        vmax=[v1, v2],
    )
    print("[surface] Flatmap (two-contrast) saved to:", outdir)


# ============================ MAIN =====================================

if __name__ == '__main__':
    zthr1_vol = None
    zthr2_vol = None

    if RUN_VOLUME:
        # Volume (contrast 1)
        zthr1_vol, _, _ = run_volume_pipeline_one(
            contrast_nm=contrast_name,
            contrast_k=contrast_id,
            label_out=label1,
            cap_out=cap1,
        )
        # Volume (contrast 2) if set
        if RUN_SURFACE and contrast_name2 and contrast_id2:
            zthr2_vol, _, _ = run_volume_pipeline_one(
                contrast_nm=contrast_name2,
                contrast_k=contrast_id2,
                label_out=label2,
                cap_out=cap2,
            )

    if RUN_SURFACE:
        if contrast_name2 and contrast_id2:
            # Two-contrast surface
            thr_mode = ('volume' if (RUN_VOLUME and
                                     SURFACE_THR_SOURCE == 'volume')
                        else 'surface')
            run_surface_plot_two(
                thr_mode=thr_mode,
                zthr1_vol=zthr1_vol,
                zthr2_vol=zthr2_vol,
                colors=('#009E73','#F0E442'),
            )
        else:
            # Single-contrast surface
            thr_mode = ('volume' if (RUN_VOLUME and
                                     SURFACE_THR_SOURCE == 'volume')
                        else 'surface')
            run_surface_plot_single(
                thr_mode=thr_mode, zthr_from_volume=zthr1_vol
            )