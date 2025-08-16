"""
Script: One-sample permuted OLS on volume & surface contrast maps with
        BH-FDR on z. Plots:
          - Volume: glass-brain (viridis, colorbar starts at z*).
          - Surface: static flatmap (fs_LR 32k), single contrast.

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
- Surface: project/ensure per-subject fs_LR 32k CIFTI, run per-vertex
  permuted OLS, convert to z, mask medial wall, plot a flatmap.
- For a non-empty colorbar in flatmap, ensure your
  volume_to_surface.plot_flatmap calls:
      ScalarMappable(...).set_array([])
"""

import os
import numpy as np
import nibabel as nib

from scipy import stats
from nilearn import plotting
from nilearn.maskers import NiftiMasker
from nilearn.mass_univariate import permuted_ols
from nilearn.glm.thresholding import fdr_threshold

from volume_to_surface import (  # same dir imports
    individual_surf,
    get_isurf_cifti,
    mask_cortical_activation,
    plot_flatmap,
)

# ========================== HELPERS ====================================

def sanitize_label(label: str) -> str:
    """
    Prepare labels for file and folder names.

    - lowercase
    - spaces -> '-'
    - replace '-vs-' with '_vs_'
    """
    s = label.lower().strip().replace(' ', '-')
    s = s.replace('-vs-', '_vs_')
    return s


def load_lr_from_cifti(cifti_path: str):
    """
    Load an fs_LR 32k CIFTI dense scalar and split into LH and RH arrays.
    """
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
    """
    One-sample test across subjects: is mean effect > 0 or != 0.
    """
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


def plot_glass_brain_z(z_map_path, z_threshold, two_sided, title, out_png):
    """
    Plot z-map with viridis. Colorbar starts at z-threshold.

    Robustness: if no voxels exceed threshold, hide colorbar and use a
    tiny epsilon above vmin for vmax to avoid Normalize errors.
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
        print(
            "[plot] No voxels exceed FDR z-threshold; "
            "omitting colorbar."
        )

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
    disp.savefig(out_png, dpi=300)
    disp.close()


# ============================ INPUTS ===================================

SUBJECTS = [
    3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26,
    28, 29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
]

task_tag = 'All Tasks'
contrast_name = 'Beat vs Interval'

n_permutations = 10000
two_sided_test = False
fdr_alpha = 0.05

# Surface settings
surfspace = 'fslr32k'           # fs_LR 32k
make_cifti_if_missing = True    # create per-subject CIFTI if absent

# ========================= PATHS / LABELS ==============================

if os.path.isdir('/home/analu/diedrichsen_data/data'):
    base_dir = '/home/analu/diedrichsen_data/data'
else:
    base_dir = '/cifs/diedrichsen/data'

music = os.path.join(base_dir, 'Cerebellum', 'music-sdtb')
derivatives_folder = os.path.join(music, 'derivatives')

out_root_vol = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'results', 'ols_permutation_tests', 'volume',
)

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

label = sanitize_label(contrast_name)

# ============================ VOLUME RUN ===============================

def get_single_contrast_paths(derivatives_dir, subjects, task_key,
                              contrast_key, derivative_type='sm8wbmasked'):
    """
    Build paths to a single individual contrast map for each subject.
    """
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
    """
    Fit a masker and transform all subject images into a 2D array.
    """
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


def run_volume_pipeline():
    """
    Run permuted OLS in volume, save t/z, and plot glass-brain.
    """
    con_folder = os.path.join(out_root_vol, label)
    os.makedirs(con_folder, exist_ok=True)

    conpaths = get_single_contrast_paths(
        derivatives_folder,
        SUBJECTS,
        task_id,
        contrast_id,
        derivative_type='sm8wbmasked',
    )

    masker, Y, affine, header = fit_masker_and_stack(
        conpaths,
        mask_img=None,
        smoothing_fwhm=None,
    )

    n_subj = Y.shape[0]
    df = n_subj - 1

    cache_npz = os.path.join(
        con_folder,
        (
            f"{label}_permuted_ols_{n_permutations}perm_"
            f"{'2s' if two_sided_test else '1s'}.npz"
        ),
    )

    if os.path.exists(cache_npz):
        print(f"Loading cached results: {cache_npz}")
        data = np.load(cache_npz)
        outputs = {'t': data['t']}
    else:
        print("Running permuted_ols (volume)...")
        outputs = run_permuted_ols_one_sample(
            Y,
            n_perm=n_permutations,
            two_sided=two_sided_test,
            n_jobs=-1,
            random_state=42,
        )
        np.savez(cache_npz, t=outputs['t'])
        print(f"Saved permutation outputs: {cache_npz}")

    t_vec = outputs['t'][0]
    t_path, z_path, z_thr = save_stat_maps_zfdr(
        masker=masker,
        t_vec=t_vec,
        df=df,
        out_dir=con_folder,
        prefix=label,
        alpha_fdr=fdr_alpha,
        two_sided=two_sided_test,
    )

    png_path = os.path.join(
        con_folder,
        f"{label}_glassbrain_zmap_FDRz{int(fdr_alpha * 100)}.png",
    )
    title = (
        f"{contrast_name}: one-sample permuted OLS "
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
    )

    print("Volume outputs:")
    print(f"  t-map: {t_path}")
    print(f"  z-map: {z_path}")
    print(f"  Figure: {png_path}")

    z_img = nib.load(z_path)
    z = np.asanyarray(z_img.dataobj)
    zmax = float(np.nanmax(np.abs(z)))
    return z_thr, zmax


# ============================ SURFACE RUN ==============================

def build_surface_matrices():
    """
    Ensure subject CIFTIs exist; return LH/RH matrices with shape
    (subjects, vertices) in fs_LR 32k space.
    """
    need_cifti = False
    if make_cifti_if_missing:
        cifti_dir = os.path.join(surf_files_root, f"{contrast_id}_{label}")
        cifti0 = os.path.join(
            cifti_dir,
            (
                f"sub-{SUBJECTS[0]:02d}_{task_id.replace('_', '-')}_"
                f"{label}_{surfspace}.dscalar.nii"
            ),
        )
        need_cifti = not os.path.exists(cifti0)

    if need_cifti:
        print("Creating subject CIFTI (vol->surf)...")
        individual_surf(
            derivatives_dir=derivatives_folder,
            subjects=SUBJECTS,
            task_key=task_id,
            contrast_key=contrast_id,
            surf_dir=surf_files_root,
            surfspace=surfspace,
            save='cifti',
        )

    cifti_files = get_isurf_cifti(
        surf_files_root,
        SUBJECTS,
        task_id,
        contrast_id,
        label,
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


def run_surface_pipeline(zthr_vol):
    """
    Run per-vertex permutation on LH and RH, convert to z, mask medial
    wall, and plot a static flatmap (single contrast).
    """
    Y_lh, Y_rh = build_surface_matrices()
    n_subj = Y_lh.shape[0]
    df = n_subj - 1

    print("Running permuted_ols (surface LH)...")
    out_lh = run_permuted_ols_one_sample(
        Y_lh,
        n_perm=n_permutations,
        two_sided=two_sided_test,
        n_jobs=-1,
        random_state=42,
    )
    print("Running permuted_ols (surface RH)...")
    out_rh = run_permuted_ols_one_sample(
        Y_rh,
        n_perm=n_permutations,
        two_sided=two_sided_test,
        n_jobs=-1,
        random_state=42,
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

    vmax_surf = float(
        max(np.nanmax(np.abs(z_lh)), np.nanmax(np.abs(z_rh)))
    )
    vmax_surf = max(vmax_surf, float(zthr_vol) + 1e-6)

    surf_outdir = os.path.join(surf_plots_root, label)
    os.makedirs(surf_outdir, exist_ok=True)

    plot_flatmap(
        stats=[z_lh, z_rh],
        threshold=zthr_vol,
        task_key=task_id,
        contrast_tag=label,
        output_dir=surf_outdir,
        hemi=['L', 'R'],
        colormap='viridis',
        vmax=vmax_surf,
    )

    print("Surface output:")
    print(f"  Flatmap PNG in: {surf_outdir}")


# ============================ MAIN =====================================

if __name__ == '__main__':
    zthr_vol, _ = run_volume_pipeline()
    run_surface_pipeline(zthr_vol=zthr_vol)