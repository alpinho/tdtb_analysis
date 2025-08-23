"""
Script: Group-level (second-level) volume maps with BH-FDR on z and
        glass-brain visualization. Supports:
          - One contrast, or iterate all contrasts (toggle).
          - Gray-matter mask for visualization only (stats unchanged).
          - Storage under results/volume/<id>_<name>/

Author: Ana Luisa Pinho
Email: agrilopi@uwo.ca

Creation: 22nd of August 2025
Last Update: August 2025

Compatibility: Python 3.10.14, nilearn 0.11.1
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd

from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm.thresholding import fdr_threshold

# Re-use your glass-brain plotter from ols_permutation_tests.py
from ols_permutation_tests import plot_glass_brain_z


# ============================ TOGGLES ==================================

# Run a single contrast (set below) or all contrasts in all_contrasts
RUN_ALL_CONTRASTS = True

# Two-sided vs one-sided BH-FDR on z for reporting/plotting
TWO_SIDED_TEST = False

# Smoothing at second level (nilearn SecondLevelModel)
SMOOTHING_FWHM = 8.0

# FDR alpha
FDR_ALPHA = 0.05

# ============================ INPUTS ===================================

SUBJECTS = [
    3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26,
    28, 29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
]

# Task selection
tasks = {
    'prod': 'Production',
    'percep': 'Perception',
    'ntfd': 'NTFD',
    'rand_ntfd': 'Randomized NTFD',
    'allmain_tasks': 'All Tasks'
}
task_tag = 'Randomized NTFD'  # display label
task_id = {v: k for k, v in tasks.items()}[task_tag]  # e.g. 'allmain_tasks'

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
        18: 'Decision',
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
        15: 'Auditory Beat',
        16: 'Auditory Interval',
        17: 'Auditory Random',
        18: 'Auditory Beat vs Auditory Interval',
        19: 'Auditory Interval vs Auditory Beat',
        20: 'Auditory Beat vs Auditory Random',
        21: 'Auditory Random vs Auditory Beat',
        22: 'Auditory Interval vs Auditory Random',
        23: 'Auditory Random vs Auditory Interval',
        24: 'Visual Beat',
        25: 'Visual Interval',
        26: 'Visual Random',
        27: 'Visual Beat vs Visual Interval',
        28: 'Visual Interval vs Visual Beat',
        29: 'Visual Beat vs Visual Random',
        30: 'Visual Random vs Visual Beat',                 
        31: 'Visual Interval vs Visual Random',
        32: 'Visual Random vs Visual Interval',
        33: 'Decision'
    }

# Single-contrast selection used when RUN_ALL_CONTRASTS = False
contrast_name = 'Encoding'  # change when running a single contrast
contrast_id = {v: k for k, v in all_contrasts.items()}[contrast_name]

# ========================= PATHS / LABELS ==============================

if os.path.isdir('/home/analu/diedrichsen_data/data'):
    base_dir = '/home/analu/diedrichsen_data/data'
else:
    base_dir = '/cifs/diedrichsen/data'

music = os.path.join(base_dir, 'Cerebellum', 'music-sdtb')
derivatives_folder = os.path.join(music, 'derivatives')
GM_MASK_PATH = os.path.join(derivatives_folder, 'group', 'anat',
                            'group_mask_gray.nii')

# Where to save results (as requested)
out_root_vol = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'results', 'parametric_tests', 'volume', task_id
)

# Subject contrast location/pattern (second-level inputs)
# Adjust here if your single-subject outputs differ.
DERIVATIVE_SUBFOLDER = 'ffx_rwls_dbb_hrf128'
FILENAME_TEMPLATE = 'wcon_{cid:04d}.nii'  # e.g., wcon_0014.nii


# ========================== HELPERS ====================================

def sanitize_label_snake(label: str) -> str:
    if label is None:
        return None
    s = label.lower().strip().replace(' ', '_')
    s = s.replace('-vs-', '_vs_')
    return s


def cap_label(label: str) -> str:
    if label is None:
        return None
    if label.isupper():
        return label
    return ' '.join(w.capitalize() for w in label.split())


def id_label_folder(cid: int, cname: str) -> str:
    return f"{cid}_{sanitize_label_snake(cname)}"


def build_single_contrast_paths(
    derivatives_dir: str,
    subjects: list,
    task_key: str,
    contrast_key: int,
) -> list:
    """
    Build paths to the per-subject contrast images used for second level.
    """
    fname = FILENAME_TEMPLATE.format(cid=contrast_key)
    paths = [
        os.path.join(
            derivatives_dir,
            f"sub-{sub:02d}",
            "estimates",
            task_key,
            DERIVATIVE_SUBFOLDER,
            fname,
        )
        for sub in subjects
    ]
    return paths


def fdr_z_threshold_from_img(z_img, alpha=0.05, two_sided=False) -> float:
    """
    Compute BH-FDR threshold on z from an image, following the convention:
      - one-sided: use z>0
      - two-sided: use |z|
    """
    z = np.asanyarray(z_img.dataobj)
    zuse = np.abs(z) if two_sided else z
    valid = np.isfinite(zuse) & (zuse > 0)
    if not np.any(valid):
        print("[FDR] No valid z > 0; using z_thr = inf.")
        return float('inf')
    return float(fdr_threshold(zuse[valid], alpha=alpha))


def apply_visual_mask(src_img_path, mask_path, out_path):
    """
    Save a copy of src_img masked by mask_path (0 outside mask).
    For visualization only; stats remain unchanged.
    """
    if not mask_path:
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


def second_level_one(
    cid: int,
    cname: str,
    task_key: str,
    subjects: list,
    out_root: str,
    smoothing_fwhm: float = 8.0,
    alpha_fdr: float = 0.05,
    two_sided: bool = False,
) -> tuple:
    """
    Fit a one-sample second-level model on subject maps for a single
    contrast, save z-map and glass brain, and return (z_thr, z_max, z_path).
    """
    idlabel = id_label_folder(cid, cname)
    out_dir = os.path.join(out_root, idlabel)
    os.makedirs(out_dir, exist_ok=True)

    conpaths = build_single_contrast_paths(
        derivatives_folder, subjects, task_key, cid
    )
    print(f"[volume] Found {len(conpaths)} subject contrast maps.")
    print(f"conpaths: {np.array(conpaths)}")

    # Design: one-sample t-test (intercept-only)
    design = pd.DataFrame({'intercept': [1] * len(conpaths)})

    # Fit the second-level model; rely on implicit mask
    slm = SecondLevelModel(mask_img=None, smoothing_fwhm=smoothing_fwhm)
    slm = slm.fit(conpaths, design_matrix=design)

    # z-map
    z_img = slm.compute_contrast(output_type='z_score')

    # Save z-map
    z_path = os.path.join(out_dir, f"{idlabel}_zmap.nii.gz")
    nib.save(z_img, z_path)

    # FDR threshold (BH on z)
    z_thr = fdr_z_threshold_from_img(
        z_img, alpha=alpha_fdr, two_sided=two_sided
    )

    # z-maximum (absolute if two-sided)
    z = np.asanyarray(z_img.dataobj)
    z_abs = np.abs(z) if two_sided else z
    if np.any(np.isfinite(z_abs)):
        z_max = float(np.nanmax(z_abs))
    else:
        z_max = float('nan')

    # Optional: GM mask for visualization only
    z_path_for_plot = z_path
    if GM_MASK_PATH:
        z_path_for_plot = os.path.join(out_dir, f"{idlabel}_zmap_gmmasked.nii.gz")
        apply_visual_mask(z_path, GM_MASK_PATH, z_path_for_plot)

    # Glass-brain figure
    png_path = os.path.join(
        out_dir, f"{idlabel}_glassbrain_zmap_FDRz{int(alpha_fdr*100)}.png"
    )
    title = (
        f"{cname}: second-level one-sample "
        f"(FDR on z < {alpha_fdr}, "
        f"{'two-sided |z|' if two_sided else 'one-sided z+'}; "
        f"z* = {z_thr:.3f})"
    )
    plot_glass_brain_z(
        z_map_path=z_path_for_plot,
        z_threshold=z_thr,
        two_sided=two_sided,
        title=title,
        out_png=png_path,
        cbar_contrast_label=cap_label(cname),
    )

    print("[volume] Outputs:")
    print(f"  z-map: {z_path}")
    if GM_MASK_PATH:
        print(f"  z-map (GM-masked for plot): {z_path_for_plot}")
    print(f"  figure: {png_path}")
    print(f"  FDR z-threshold: {z_thr:.4f}")
    print(f"  z-max: {z_max:.4f}")

    return z_thr, z_max, z_path


# ============================ MAIN =====================================

if __name__ == '__main__':
    if RUN_ALL_CONTRASTS:
        # Iterate over all contrasts (single-contrast runs only)
        for cid, cname in all_contrasts.items():
            print(f"\n[MAIN] Running contrast {cid}: {cname}")
            second_level_one(
                cid=cid,
                cname=cname,
                task_key=task_id,
                subjects=SUBJECTS,
                out_root=out_root_vol,
                smoothing_fwhm=SMOOTHING_FWHM,
                alpha_fdr=FDR_ALPHA,
                two_sided=TWO_SIDED_TEST,
            )
    else:
        # Single contrast only (as set at the top)
        print(f"[MAIN] Running single contrast {contrast_id}: {contrast_name}")
        second_level_one(
            cid=contrast_id,
            cname=contrast_name,
            task_key=task_id,
            subjects=SUBJECTS,
            out_root=out_root_vol,
            smoothing_fwhm=SMOOTHING_FWHM,
            alpha_fdr=FDR_ALPHA,
            two_sided=TWO_SIDED_TEST,
        )