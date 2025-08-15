"""
Script: One-sample permuted OLS on single-condition contrast maps
        with BH-FDR on z and viridis colorbar starting at z-threshold.

Author: Ana Luisa Pinho
Email: agrilopi@uwo.ca

Creation: 15 Aug 2025
Last Update: Aug 2025

Compatibility: Python 3.10.14, nilearn 0.11.1

Notes
-----
- One-sample test across subjects for a single contrast 
  (e.g., "Encoding").
- Uses nilearn.mass_univariate.permuted_ols to get voxelwise t.
- Converts t -> p (parametric) -> z; applies BH-FDR on z:
    * one-sided: FDR on positive z only (H1: mean > 0)
    * two-sided: FDR on |z|; saved z keeps the sign of t
- Saves: tmap, zmap, and a glass-brain PNG. No thresholded NIfTI.
"""

import os
import numpy as np
import nibabel as nib

from scipy import stats
from nilearn import plotting
from nilearn.maskers import NiftiMasker
from nilearn.mass_univariate import permuted_ols
from nilearn.glm.thresholding import fdr_threshold


# ========================== HELPERS ====================================

def sanitize_label(label: str) -> str:
    """
    Normalize labels for file and folder names.

    - lowercase
    - spaces -> '-'
    - replace '-vs-' with '_vs_'
    """
    s = label.lower().strip().replace(' ', '-')
    s = s.replace('-vs-', '_vs_')
    return s


# ========================== FUNCTIONS ==================================

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

    Returns
    -------
    masker : NiftiMasker
    Y : array, shape (n_subjects, n_voxels)
    affine, header : from a representative image
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


def run_permuted_ols_one_sample(Y, n_perm=10000, two_sided=False, n_jobs=1,
                                random_state=42):
    """
    One-sample test across subjects: is mean effect > 0 or != 0.

    Returns a dict with key 't' (voxelwise t).
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
                        alpha_fdr=.05, two_sided=False):
    """
    Save t-map and z-map. Compute FDR(z) threshold for plotting.

    Parameters
    ----------
    masker : NiftiMasker
        Fitted masker used for inverse transforms.
    t_vec : array-like
        Voxelwise t-statistics (flattened in mask space).
    df : int
        Degrees of freedom (n_subjects - 1).
    out_dir : str
        Output directory.
    prefix : str
        Filename prefix (sanitized label).
    alpha_fdr : float
        FDR alpha for BH procedure on z.
    two_sided : bool
        If True, run FDR on |z|; else on positive z only.

    Returns
    -------
    t_path : str
        Path to saved t-map NIfTI.
    z_path : str
        Path to saved z-map NIfTI.
    z_thr : float
        FDR z-threshold used for plotting.
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

    Robustness: compute vmax from supra-threshold voxels. If none, hide
    the colorbar and set a tiny epsilon above vmin to avoid errors.
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


# ============================ INPUTS =================================

SUBJECTS = [
    3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26,
    28, 29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
]

task_tag = 'All Tasks'
contrast_name = 'Encoding'

n_permutations = 10000
two_sided_test = False
fdr_alpha = .05

# ========================= PARAMETERS ===============================

if os.path.isdir('/home/analu/diedrichsen_data/data'):
    base_dir = '/home/analu/diedrichsen_data/data'
else:
    base_dir = '/cifs/diedrichsen/data'

music = os.path.join(base_dir, 'Cerebellum', 'music-sdtb')
derivatives_folder = os.path.join(music, 'derivatives')

out_root = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'results',
    'ols_permutation_tests',
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


# ============================ RUN ====================================

if __name__ == '__main__':

    label = sanitize_label(contrast_name)

    con_folder = os.path.join(out_root, label)
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
        outputs = {
            't': data['t'],
            'logp_max_t': data.get('logp_max_t', None),
            'h0_max_t': data.get('h0_max_t', None),
        }
    else:
        print("Running permuted_ols (this can take a while)...")
        outputs = run_permuted_ols_one_sample(
            Y,
            n_perm=n_permutations,
            two_sided=two_sided_test,
            n_jobs=-1,
            random_state=42,
        )
        np.savez(
            cache_npz,
            t=outputs['t'],
            logp_max_t=outputs.get('logp_max_t', None),
            h0_max_t=outputs.get('h0_max_t', None),
        )
        print(f"Saved permutation outputs: {cache_npz}")

    prefix = label
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

    print("Outputs:")
    print(f"  t-map: {t_path}")
    print(f"  z-map: {z_path}")
    print(f"  Figure: {png_path}")