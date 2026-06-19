"""
Script to do the volume to surface projection of data from the
 Music-SDTB project. It plots flatmaps and dynamic inflated maps.

Plotting modes
Use flatmaps (default) or dynamic HTML maps (Plotly) by passing a flag:

    - Single contrast (flatmap): python volume_to_surface.py

    - Single contrast (dynamic): python volume_to_surface.py --dyn

    - Batch all contrasts (flatmaps): set contrast_name = 'ALL' and run

    - Batch all contrasts (dynamic): 
        set contrast_name = 'ALL' and run with --dyn
         (Dynamic maps are not produced for the two-contrast overlay.)

    - Single or Multiple IROIs (flatmaps): 
        python volume_to_surface.py --irois

    - Two-contrast overlay with the SECOND contrast drawn as a contour
      (first filled, second as a thresholded outline):
        set contrast_name and contrast_name2 and run with --contour

    - Combined regime map of a signed difference contrast (e.g.
      'Random vs Non-Random'), re-colored by the rest-referenced regime of its
      two conditions so activation, deactivation and crossover are visually
      separable on one flatmap:
        set contrast_name to the difference and run with --regime
      The --contour outline is independent and can be added on top by also
      passing --contour (with contrast_name2); run either or both at once.

Sidedness of the FDR threshold / display (set in the INPUTS section):
    contrast_sides : sidedness of the FILLED map (contrast_name).
    contour_sides  : sidedness of the OUTLINE (contrast_name2, --contour only).
    Each may be 'one-sided' (positive tail only, activations), 'two-sided'
    (|z| FDR, both activations and deactivations, diverging colormap for the
    fill), or None (auto: two-sided for signed difference contrasts such as
    'Random vs Non-Random', one-sided otherwise). These options apply in every
    branch that thresholds a contrast.

Author: Ana Luisa Pinho
Email: agrilopi@uwo.ca

Creation: 24th of February 2025
Last Update: June 2026

Compatibility: Python 3.10.14, nilearn 0.11.1

Note: The all pipeline of this script only works for surf_files saved 
      as cifti in fs_LR32k.
"""

import sys
import os

import numpy as np
import pandas as pd
import nibabel as nib
import nitools as nt

import plotly.graph_objects as go
import plotly.io as pio

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib import colors as mcolors
import matplotlib.tri as mtri

from scipy import stats
from nilearn.image import load_img
from nilearn.surface import vol_to_surf
from nilearn.maskers import NiftiMasker
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm.thresholding import fdr_threshold
from Functional_Fusion.util import smooth_fs32k_data
from SUITPy import flatmap

# setting path
sys.path.append('../')
# importing
from utils import zval_conversion

# %%
# ========================== FUNCTIONS ================================


def build_contrasts(task_id):
    """Return the {id: name} contrast dictionary for a given task_id."""
    if task_id != 'rand_ntfd':
        return {
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
    return {
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


def get_imeshes(derivatives_dir, subjects, surfspace='fslr32k'):

    if surfspace == 'fsaverage':
        surfspace_dir = os.path.join(os.path.dirname(derivatives_dir),
                                     'surfaceFreeSurfer')

        pial_left = [os.path.join(
            surfspace_dir, 'xsub-%02d' % sub, 'surf', 'lh.pial')
                     for sub in subjects]
        pial_right = [os.path.join(
            surfspace_dir, 'xsub-%02d' % sub, 'surf', 'rh.pial')
                      for sub in subjects]
        
        white_left = [os.path.join(
            surfspace_dir, 'xsub-%02d' % sub, 'surf', 'lh.white')
                      for sub in subjects]
        white_right = [os.path.join(
            surfspace_dir, 'xsub-%02d' % sub, 'surf', 'rh.white')
                       for sub in subjects]
    else:
        assert surfspace == 'fslr32k'
        
        surfspace_dir = os.path.join(os.path.dirname(derivatives_dir),
                                     'surfaceWB', 'data')
        subjects_dir = [os.path.join(surfspace_dir, 'sub-%02d' % sub)
                        for sub in subjects]

        pial_left = [os.path.join(subjects_dir[s],
                                  'sub-%02d' % sub + '.L.pial.32k.surf.gii')
                     for s, sub in enumerate(subjects)]
        pial_right = [os.path.join(subjects_dir[s],
                                   'sub-%02d' % sub + '.R.pial.32k.surf.gii')
                      for s, sub in enumerate(subjects)]
        
        white_left = [os.path.join(subjects_dir[s],
                                   'sub-%02d' % sub + '.L.white.32k.surf.gii')
                      for s, sub in enumerate(subjects)]
        white_right = [os.path.join(subjects_dir[s],
                                    'sub-%02d' % sub + '.R.white.32k.surf.gii')
                       for s, sub in enumerate(subjects)]

    return pial_left, pial_right, white_left, white_right


def individual_surf(
    derivatives_dir,
    subjects,
    task_key,
    contrasts_dic,
    contrast_key,
    surf_dir,
    surfspace='fslr32k',
    save='gifti',
):

    # Paths of the NON-NORMALIZED individual contrast map for all subjects
    encoding_maps = [os.path.join(derivatives_dir, 'sub-%02d' % sub,
                                  'estimates', task_key, 'ffx_rwls_dbb_hrf128',
                                  'con_%04d' % contrast_key + '.nii')
                     for sub in subjects]

    # Paths of individual meshes per hemisphere
    pial_left, pial_right, white_left, white_right = get_imeshes(
        derivatives_dir, subjects, surfspace=surfspace)

    # For each subject...
    for emap, pl, pr, wl, wr, sb in zip(encoding_maps, pial_left, pial_right,
                                        white_left, white_right, subjects):

        # Map individual functional data from  Nifti to the surface of...
        # ... left and right hemispheres
        emap_img = load_img(emap)
        DL = vol_to_surf(emap_img, surf_mesh=pl, inner_mesh=wl)
        DR = vol_to_surf(emap_img, surf_mesh=pr, inner_mesh=wr)
        print(sb)
        print(DL.shape)
        print(DR.shape)

        # Transform numpy arrays in gifti files
        contrast = (
            contrasts_dic[contrast_key]
            .replace(' vs ', '_vs_')
            .replace(' ', '-')
        )
        GIFTIL = nt.gifti.make_func_gifti(DL, anatomical_struct='CortexLeft',
                                          column_names=[contrast])
        GIFTIR = nt.gifti.make_func_gifti(DR, anatomical_struct='CortexRight',
                                          column_names=[contrast])

        # Create directory to save outputs if does not exist
        contrast_dir = os.path.join(
            surf_dir, str(contrast_key) + '_' + contrast.lower())
        if not os.path.exists(contrast_dir):
            os.makedirs(contrast_dir)

        # Save output
        if save == 'gifti':
            # Save Gifti files
            nib.save(
                GIFTIL,
                os.path.join(
                    contrast_dir,
                    'sub-{sb:02d}_'.format(sb=sb)
                    + task_key.replace('_', '-')
                    + '_'
                    + contrast.lower()
                    + '_'
                    + surfspace
                    + '.hem-L.func.gii'
                )
            )
            nib.save(
                GIFTIR,
                os.path.join(
                    contrast_dir,
                    'sub-{sb:02d}_'.format(sb=sb)
                    + task_key.replace('_', '-')
                    + '_'
                    + contrast.lower()
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
                    contrast_dir,
                    f'sub-{sb:02d}_{task_key.replace("_", "-")}_'
                    f'{contrast.lower()}_{surfspace}.dscalar.nii'
                )
            )


def get_isurf_gifti(surf_dir, subjects, task_key, contrast,
                    surfspace='fslr32k'):

    # Paths of individual files per hemisphere
    gifti_left = [
        os.path.join(
            surf_dir,
            f"sub-{sub:02d}_"
            f"{task_key.replace('_', '-')}_"
            f"{contrast}_{surfspace}.hem-L.func.gii",
        )
        for sub in subjects
    ]

    gifti_right = [
        os.path.join(
            surf_dir,
            f"sub-{sub:02d}_"
            f"{task_key.replace('_', '-')}_"
            f"{contrast}_{surfspace}.hem-R.func.gii",
        )
        for sub in subjects
    ]

    return gifti_left, gifti_right


def get_isurf_cifti(surf_dir, subjects, task_key, contrast_key, contrast,
                    surfspace='fslr32k'):

    # Paths of individual files
    cifti_file = [
        os.path.join(
            surf_dir, str(contrast_key) + '_' + contrast,
            f"sub-{sub:02d}_"
            f"{task_key.replace('_', '-')}_"
            f"{contrast}_{surfspace}.dscalar.nii",
        )
        for sub in subjects
    ]

    return cifti_file


def group_surf(surf_dir, subjects, task_key, contrast_key, contrast_tag,
               surfspace='fslr32k'):

    contrast = contrast_tag.lower()

    # Get paths of files with individual functional data projected on...
    # ... the surface
    cifti_paths = get_isurf_cifti(surf_dir, subjects, task_key, contrast_key,
                                  contrast, surfspace=surfspace)

    # Substitute nan's by zeros and smooth the data
    smoothed_data = np.array([smooth_fs32k_data(cifti_path, smooth=8,
                                                kernel='fwhm',
                                                return_data_only=True)
                              for cifti_path in cifti_paths])

    # Remove the middle dimension
    smoothed_data = np.squeeze(smoothed_data, axis=1)

    # Substitute nan's by 0's
    smoothed_data[np.isnan(smoothed_data)] = 0

    # Calculate the one sample t-test
    tvals, _ = stats.ttest_1samp(smoothed_data, 0, axis=0,
                                 alternative='greater')

    # Compute z-values from t-values
    zvals = zval_conversion(tvals, len(subjects)-1)

    # Substitute nan's by 0's
    zvals[np.isnan(zvals)] = 0

    return zvals


def mask_cortical_activation(activation_data, medial_wall_mask_path):
    """
    Masks activation data using a medial wall mask so that only cortical
    vertices retain their activation. Vertices where the mask is 0
    (non-cortical) will be set to 0.
    
    Parameters:
      activation_data : np.array
          1D array of activation values for each vertex.
      medial_wall_mask_path : str
          Path to the medial wall mask GIFTI file
    
    Returns:
      masked_activation : np.array
          Activation values with non-cortical vertices set to 0.
    """
    
    # Load the medial wall mask
    mask_img = nib.load(medial_wall_mask_path)
    
    # Depending on the file, the data may be stored...
    # ... in the first data array.
    mask_data = mask_img.darrays[0].data
    
    # Ensure mask is boolean
    # (True = cortical, False = non-cortical)
    cortex_mask = mask_data.astype(bool)
    
    # Apply the mask: set activation to 0 for non-cortical vertices.
    masked_activation = activation_data.copy()
    masked_activation[~cortex_mask] = 0
    
    return masked_activation


def roi_to_surf(lh_roi_path, rh_roi_path, pl, pr, wl, wr, surf_dir, roi_name, 
                surfspace='fslr32k', save='gifti', individualization='i'):

    # Map volumetric roi data in MNI from Nifti to the surface of...
    # ... left and right hemispheres
    lh_roi_img = load_img(lh_roi_path)
    rh_roi_img = load_img(rh_roi_path)
    DL = vol_to_surf(lh_roi_img, surf_mesh=pl, inner_mesh=wl, 
                     interpolation='nearest')
    DR = vol_to_surf(rh_roi_img, surf_mesh=pr, inner_mesh=wr, 
                     interpolation='nearest')
    
    # Binarize the data if group-level individualization
    if individualization == 'g':
        DL[:] = (DL >= 0.5)
        DR[:] = (DR >= 0.5)

    print(DL.shape)
    print(DR.shape)

    # Transform numpy arrays in gifti files
    GIFTIL = nt.gifti.make_func_gifti(DL, anatomical_struct='CortexLeft',
                                      column_names=['PMD'])
    GIFTIR = nt.gifti.make_func_gifti(DR, anatomical_struct='CortexRight',
                                      column_names=['PMD'])
    
    # Create directory to save outputs if does not exist
    irois_files_folder = os.path.join(surf_dir, 'surface_irois_files')
    os.makedirs(irois_files_folder, exist_ok=True)

    # Save output
    if save == 'gifti':
        # Save Gifti files
        nib.save(
            GIFTIL,
            os.path.join(
                irois_files_folder,
                individualization + '_' + roi_name + '_mask'
                + '_'
                + surfspace
                + '.hem-L.func.gii'
            )
        )
        nib.save(
            GIFTIR,
            os.path.join(
                irois_files_folder,
                individualization + '_' + roi_name + '_mask'
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
                + irois_files_folder,
                + individualization + '_' + roi_name + '_mask'
                + '_'
                + surfspace
                + '.dscalar.nii'
            )
        )

    return DL, DR


def whole_brain_thresholds(derivatives_dir, subjects, task_key, contrast_key,
                           gmask):

    # Paths of the NORMALIZED individual contrast map for all subjects
    encoding_maps = [os.path.join(derivatives_dir, 'sub-%02d' % sub,
                                  'estimates', task_key, 'ffx_rwls_dbb_hrf128',
                                  'wcon_%04d' % contrast_key + '.nii')
                     for sub in subjects]

    # Create design matrix (one-sample t-test)
    design_matrix = pd.DataFrame(
        [1] * len(encoding_maps),
        columns=['intercept'],
    )

    # Initialize a NiftiMasker (creates an implicit mask from the Z-map).
    masker = NiftiMasker(mask_img=gmask)

    # Initialize and fit the SecondLevelModel
    second_level_model = SecondLevelModel(mask_img=masker, smoothing_fwhm=8)
    second_level_model = second_level_model.fit(encoding_maps,
                                                design_matrix=design_matrix)

    # Compute the Z-Map
    z_map = second_level_model.compute_contrast(output_type='z_score')

    # Extract voxel values using fit_transform()
    z_values = masker.fit_transform(z_map)  # Output shape: (1, p)

    # Get FDR threshold at alpha = 0.05 (5% false discovery rate)
    # One side: greater than (so, no need to divide by 2)
    fdr_thresh = fdr_threshold(z_values.ravel(), alpha=0.05)

    # Print the estimated FDR threshold
    print(f'Estimated FDR threshold: {fdr_thresh}')

    # Get maximum peak value
    z_max = np.amax(z_values[~np.isnan(z_values)])

    # Print z_max
    print(f'Maximum Z-value is: {z_max}')

    return fdr_thresh, z_max


def whole_brain_thresholds_signed(derivatives_dir, subjects, task_key,
                                  contrast_key, gmask):
    """Proper two-sided BH-FDR threshold (on |z|) and symmetric vmax for a
    signed diverging map. Reuses the same second-level fit as
    whole_brain_thresholds, but corrects both tails by doubling the p-values
    (NOT fdr_threshold(|z|), which under-corrects); e.g. z* = 3.34 rather than
    the folded 2.92 for Random vs Non-Random."""

    # Paths of the NORMALIZED individual contrast map for all subjects
    encoding_maps = [os.path.join(derivatives_dir, 'sub-%02d' % sub,
                                  'estimates', task_key, 'ffx_rwls_dbb_hrf128',
                                  'wcon_%04d' % contrast_key + '.nii')
                     for sub in subjects]

    # Create design matrix (one-sample t-test)
    design_matrix = pd.DataFrame(
        [1] * len(encoding_maps),
        columns=['intercept'],
    )

    # Initialize a NiftiMasker (creates an implicit mask from the Z-map).
    masker = NiftiMasker(mask_img=gmask)

    # Initialize and fit the SecondLevelModel
    second_level_model = SecondLevelModel(mask_img=masker, smoothing_fwhm=8)
    second_level_model = second_level_model.fit(encoding_maps,
                                                design_matrix=design_matrix)

    # Compute the Z-Map and extract voxel values
    z_map = second_level_model.compute_contrast(output_type='z_score')
    z_values = masker.fit_transform(z_map).ravel()
    z_values = z_values[~np.isnan(z_values)]

    # Two-sided BH-FDR. Note fdr_threshold(|z|) does NOT give this: it treats
    # |z| as a one-sided statistic (p = 1 - Phi(|z|)) and so under-corrects,
    # returning the folded one-sided z* (e.g. 2.92 instead of 3.34) and roughly
    # twice the suprathreshold voxels. For a proper two-sided control we form
    # two-sided p = 2*(1 - Phi(|z|)), run Benjamini-Hochberg on those, and
    # convert the cutoff back to a |z| threshold.
    absz = np.abs(z_values)
    p_two = np.clip(2.0 * stats.norm.sf(absz), 0.0, 1.0)
    p_sorted = np.sort(p_two)
    n = p_sorted.size
    bh_line = (np.arange(1, n + 1) / n) * 0.05
    below = p_sorted <= bh_line
    if np.any(below):
        p_cut = p_sorted[np.max(np.nonzero(below))]
        thr_signed = float(stats.norm.isf(p_cut / 2.0))
    else:
        thr_signed = float('inf')

    # Symmetric color limit
    vmax = np.amax(absz)

    print(f'Signed FDR threshold (|z|): {thr_signed}; symmetric vmax: {vmax}')

    return thr_signed, vmax


def overlay_region_contour(ax, surf_path, values, threshold,
                           color='k', linewidth=1.0, positive_only=True,
                           tail=None):
    """
    Draw the outline of the supra-threshold region of `values` on an
    existing flatmap axis `ax`, using the flat-surface mesh in `surf_path`.

    Parameters
    ----------
    ax : matplotlib axis already holding a flatmap (from flatmap.plot).
    surf_path : str, path to the hemisphere flat .surf.gii (same one passed
        to flatmap.plot, so coordinates align).
    values : 1D np.array, per-vertex map of the second contrast.
    threshold : float, defines the region to outline.
    positive_only : if True (default), one-sided inference: only vertices
        with value >= threshold (activations) are outlined. If False,
        two-sided: |value| >= threshold (activations + deactivations).
    color, linewidth : contour styling.
    """
    gii = nib.load(surf_path)
    coords = gii.darrays[0].data
    faces = np.asarray(gii.darrays[1].data, dtype=int)
    x, y = coords[:, 0].astype(float), coords[:, 1].astype(float)

    v = np.nan_to_num(values, nan=0.0)
    # `tail` (when given) overrides positive_only and selects which side of the
    # threshold is outlined: 'positive' (right, v>=thr), 'negative' (left,
    # v<=-thr) or 'both' (|v|>=thr). positive_only is kept for back-compat
    # (positive_only=True == tail='positive'; False == tail='both').
    t = (str(tail).strip().lower() if tail is not None
         else ('positive' if positive_only else 'both'))
    if t in ('positive', 'pos', 'right', 'r', 'greater', '+'):
        mask = (v >= threshold).astype(float)
    elif t in ('negative', 'neg', 'left', 'l', 'less', '-'):
        mask = (v <= -threshold).astype(float)
    else:  # 'both'
        mask = (np.abs(v) >= threshold).astype(float)
    if mask.sum() == 0:
        return  # nothing supra-threshold -> no contour

    # Guard against non-finite flat coordinates (e.g. medial wall cut)
    good = np.isfinite(x) & np.isfinite(y)
    if not good.all():
        x = np.where(good, x, 0.0)
        y = np.where(good, y, 0.0)
    triang = mtri.Triangulation(x, y, faces)
    if not good.all():
        triang.set_mask(~good[faces].all(axis=1))

    # Freeze the framing that flatmap.plot established: tricontour otherwise
    # re-autoscales the axes, which (with bbox_inches='tight') clips the
    # flatmap tips and shifts the colorbar.
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.tricontour(triang, mask, levels=[0.5],
                  colors=[color], linewidths=linewidth)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_autoscale_on(False)


def resolve_signed(sides, contrast_name, signed_contrasts):
    """Resolve the requested sidedness into a two-sided (signed) boolean.

    Parameters
    ----------
    sides : {'one-sided', 'two-sided', None}
        Explicit choice; None falls back to auto-detection, i.e. two-sided for
        the signed difference contrasts listed in `signed_contrasts` and
        one-sided otherwise (legacy behaviour).
    contrast_name : str
        Name of the contrast whose sidedness is being resolved.
    signed_contrasts : sequence of str
        Contrasts treated as two-sided when `sides` is None.

    Returns
    -------
    bool
        True  -> two-sided: |z| FDR, both tails (diverging fill / outline of
                 activations + deactivations).
        False -> one-sided: positive tail only (activations).
    """
    if sides is None:
        return contrast_name in signed_contrasts
    s = str(sides).strip().lower()
    if s in ('two-sided', 'two_sided', 'twosided', 'two', '2'):
        return True
    if s in ('one-sided', 'one_sided', 'onesided', 'one', '1'):
        return False
    raise ValueError(
        f"Invalid sidedness {sides!r}; use 'one-sided', 'two-sided', or None.")


def resolve_contour_display(contour_display, contour_twosided):
    """Resolve which tail(s) of the contour OUTLINE to draw, independent of the
    threshold sidedness.

    This decouples *what is drawn* from *how the threshold is computed*, so the
    outline can be thresholded two-sided (|z| FDR) yet trace only the positive
    (activation) tail.

    Parameters
    ----------
    contour_display : {'positive', 'both', None}
        'positive' (or 'activations') -> draw the positive tail only;
        'both' (or 'two-sided')       -> draw activations and deactivations;
        None                          -> mirror the threshold sidedness, i.e.
                                          positive-only when the threshold is
                                          one-sided and both tails when it is
                                          two-sided (legacy behaviour).
    contour_twosided : bool
        Whether the contour THRESHOLD was computed two-sided (from
        resolve_signed); used only for the None fallback.

    Returns
    -------
    bool
        positive_only flag for overlay_region_contour.
    """
    if contour_display is None:
        return not contour_twosided
    s = str(contour_display).strip().lower()
    if s in ('positive', 'pos', 'activations', 'one', 'one-sided', '1'):
        return True
    if s in ('both', 'signed', 'two', 'two-sided', '2'):
        return False
    raise ValueError(
        f"Invalid contour_display {contour_display!r}; "
        "use 'positive', 'both', or None.")


def parse_sides(spec):
    """Unify a contrast's sidedness into (test, tail) from one string.

    The FDR *test* and the displayed *tail* are dissociated, so a two-sided
    test can still classify or outline a single tail:

        'greater'  -> ('greater',   'positive')   one-sided +FDR, right tail
        'less'     -> ('less',      'negative')   one-sided -FDR, left tail
        'two-sided'        -> ('two-sided', 'positive')   |z| FDR, right tail
        'two-sided:right'  -> ('two-sided', 'positive')
        'two-sided:left'   -> ('two-sided', 'negative')
        'two-sided:both'   -> ('two-sided', 'both')

    The test fixes the threshold value (whole_brain_thresholds for one-sided,
    whole_brain_thresholds_signed for two-sided); the tail fixes which side of
    that threshold is drawn/classified.
    """
    s = str(spec).strip().lower()
    one = {
        'greater': ('greater', 'positive'),
        'one-sided': ('greater', 'positive'),
        'one-sided-greater': ('greater', 'positive'),
        'positive': ('greater', 'positive'), 'pos': ('greater', 'positive'),
        '+': ('greater', 'positive'),
        'less': ('less', 'negative'), 'one-sided-less': ('less', 'negative'),
        'negative': ('less', 'negative'), 'neg': ('less', 'negative'),
        '-': ('less', 'negative'),
    }
    if s in one:
        return one[s]
    head, _, tail = s.partition(':')
    if head in ('two-sided', 'two_sided', 'twosided', 'two', '2', 'signed'):
        tmap = {'': 'positive', 'right': 'positive', 'r': 'positive',
                'positive': 'positive', 'pos': 'positive',
                'left': 'negative', 'l': 'negative', 'negative': 'negative',
                'neg': 'negative', 'both': 'both'}
        if tail not in tmap:
            raise ValueError(
                f"Invalid two-sided tail {tail!r} in {spec!r}; "
                "use right/left/both.")
        return 'two-sided', tmap[tail]
    raise ValueError(
        f"Invalid sides spec {spec!r}; use 'greater', 'less', or "
        "'two-sided[:right|:left|:both]'.")


def make_signed_gray_threshold_cmap(
    thr, vlim,
    base_colors=('#1A85FF', '#FFFFFF', '#D41159'),
    gray='#999999', n=256,
):
    """Diverging colormap for signed (two-sided) thresholded maps in which the
    sub-threshold band [-thr, +thr] is rendered as a flat neutral gray and only
    the two supra-threshold tails keep color.

    Outside [-thr, +thr] the colormap is identical to the plain ``base_colors``
    diverging map; inside that band it is overwritten with ``gray``. Data are
    spread linearly over [-vlim, +vlim] both by SUITPy's ``flatmap.plot``
    (cscale=[-vlim, vlim]) and by the colorbar's Normalize, so the gray band
    occupies exactly the |z| < thr portion of the scale. On the surface the
    sub-threshold vertices are set to NaN (transparent) before plotting, so the
    gray entries appear only in the colorbar, while the colored tails match the
    on-surface colors exactly.

    Parameters
    ----------
    thr : float
        FDR critical value Z(q), i.e. the |z| threshold. Expected 0 < thr < vlim.
    vlim : float
        Symmetric color limit; the scale spans [-vlim, +vlim].
    base_colors : sequence of 3 colors
        (negative, center, positive) anchors of the underlying diverging map.
    gray : color
        Color of the [-thr, +thr] band.
    n : int
        Number of lookup-table samples.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
    """
    base = LinearSegmentedColormap.from_list(
        'signed_base', list(base_colors), N=n)
    if not (np.isfinite(thr) and np.isfinite(vlim)) or vlim <= 0 or thr <= 0:
        return base
    thr = min(float(thr), float(vlim))            # clamp degenerate thr >= vlim
    pos = np.linspace(0.0, 1.0, n)                # normalized scale positions
    rgba = base(pos)
    # normalized positions of -thr and +thr over [-vlim, +vlim]
    p_lo = 0.5 - thr / (2.0 * vlim)
    p_hi = 0.5 + thr / (2.0 * vlim)
    band = (pos >= p_lo) & (pos <= p_hi)
    rgba[band] = mcolors.to_rgba(gray)
    return LinearSegmentedColormap.from_list('signed_gray_thr', rgba, N=n)


def plot_flatmap(
    stats,
    threshold,
    task_key,
    contrast_tag,
    output_dir,
    hemi=['L', 'R'],
    colormap='viridis',
    colors=['Reds', 'Blues'],
    vmin=None,
    vmax=None,
    cbar_title='Z-values',
    n_ticks=4,
    tick_decimals=1,
    show_colorbar=True,
    signed=False,
    contour_stat=None,
    contour_threshold=None,
    contour_color='k',
    contour_linewidth=1.0,
    contour_positive_only=True,
    show_borders=True,
):
    """
    Plot one or two contrasts on a flat cortical map.

    Single-contrast:
      stats: [lh_array, rh_array]
      threshold: float
      colormap: str
      vmax: float

    Two-contrast RGB overlay:
      stats: [[lh1, rh1], [lh2, rh2]]
      threshold: [thr1, thr2]
      colors: [color1, color2]  # any matplotlib color
      vmax: [v1, v2]

    Two-contrast contour overlay:
      stats: [lh1, rh1]               # contrast 1, drawn filled
      threshold: thr1                 # fill threshold (or signed |z| thr)
      vmax: v1                        # fill vmax (symmetric if signed=True)
      contour_stat: [lh2, rh2]        # contrast 2, drawn as an outline
      contour_threshold: thr2         # |z| threshold defining the outline
      (set signed=True to fill contrast 1 with the diverging colormap)

    Note on magnitude of co-activation for the RGB overlay:
    -------------------------------------------------------
    The “Overlap” colorbar is a third gradient that tells you how
    strongly both contrasts co-activate at each vertex. Concretely:

    1. Range
    --------
    Minimum tick (vmin) = thr1/v1 + thr2/v2
    (i.e. both maps just at their respective statistical thresholds)

    Maximum tick (vmax) = 1.0
    (i.e. at least one map at its peak, and the other possibly
    contributing up to its peak)

    2. Color gradient
    -----------------
    The start color is the sum of the two threshold-level colors

    The end color is the sum of the two full-bright colors
    (clipped to [0,1])

    Intermediate hues reflect intermediate sums of the two normalized
    intensities

    3. Tick labels
    --------------
    Show four values:

    Min sum-of-fractions (thr1/v1 + thr2/v2)

    Two intermediate sums (one- and two-thirds along the range)

    Max = 1.0
    """

    contrast = contrast_tag.lower()
    task_name = task_key.replace('_', '-')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    surf_dir = os.path.join(script_dir, 'fslr32k_meshes')

    borders = {
        h: os.path.join(
            surf_dir, 'borders', f'fs_LR.32k.{h}.border')
        for h in hemi
    }
    underlays = {
        h: os.path.join(
            surf_dir, 'flat', f'fs_LR.32k.{h}.shape.gii')
        for h in hemi
    }
    surfaces = {
        h: os.path.join(
            surf_dir, 'flat', f'fs_LR.32k.{h}.flat.surf.gii')
        for h in hemi
    }

    two_rgb = (
        isinstance(stats, (list, tuple))
        and len(stats) == 2
        and isinstance(stats[0], (list, tuple))
        and colors is not None
        and isinstance(vmax, (list, tuple))
    )

    fig, axs = plt.subplots(
        1, len(hemi), figsize=(8, 4),
        gridspec_kw={'wspace': 0.05}
    )

    # contour-overlay branch: contrast 1 filled, contrast 2 as an outline
    if contour_stat is not None:
        lh, rh = stats
        lh2, rh2 = contour_stat
        cthr = float(contour_threshold)

        if signed:
            vlim = (vmax if (vmax is not None and np.isfinite(vmax))
                    else np.nanmax([np.nanmax(np.abs(lh)),
                                    np.nanmax(np.abs(rh))]))
            if not np.isfinite(vlim) or vlim <= 0:
                vlim = float(threshold) + 1e-6
            # Gray [-Z(q), +Z(q)] band; color only on the significant tails.
            fill_cmap = make_signed_gray_threshold_cmap(threshold, vlim)
            cscale = [-vlim, vlim]
        else:
            cscale_lo = threshold
            data_max = np.nanmax([np.nanmax(lh), np.nanmax(rh)])
            cscale_hi = data_max if vmax is None else vmax
            if not np.isfinite(cscale_hi) or cscale_hi <= cscale_lo:
                cscale_hi = cscale_lo + 1e-6
            fill_cmap = colormap
            cscale = [cscale_lo, cscale_hi]

        for ax, stat, c2, h in zip(axs, (lh, rh), (lh2, rh2), hemi):
            s_arr = np.asarray(stat, float).copy()
            plt.sca(ax)
            plot_kwargs = dict(
                surf=surfaces[h], underlay=underlays[h], undermap='gray',
                underscale=[-1.5, 1], cscale=cscale, cmap=fill_cmap,
                borders=(borders[h] if show_borders else None), new_figure=False, frame=None,
            )
            if signed:
                s_arr[np.abs(s_arr) < threshold] = np.nan
            else:
                plot_kwargs['threshold'] = threshold
            flatmap.plot(s_arr, **plot_kwargs)
            overlay_region_contour(
                ax, surfaces[h], c2, cthr,
                color=contour_color, linewidth=contour_linewidth,
                positive_only=contour_positive_only)

        if show_colorbar:
            lo, hi = (-vlim, vlim) if signed else (cscale[0], cscale[1])
            sm = ScalarMappable(norm=Normalize(vmin=lo, vmax=hi),
                                cmap=fill_cmap)
            cbar = fig.colorbar(
                sm, ax=list(axs), orientation='horizontal',
                fraction=0.05, pad=0.02)
            cbar.set_label(cbar_title, fontsize=12, labelpad=8)
            if signed:
                # Tick the edges of the gray (non-significant) band.
                ticks = np.array([-vlim, -float(threshold),
                                  float(threshold), vlim])
            else:
                ticks = np.linspace(lo, hi, n_ticks)
            cbar.set_ticks(ticks)
            dec = int(tick_decimals) if tick_decimals is not None else 2
            cbar.ax.set_xticklabels(
                [f'{t:.{dec}f}' for t in ticks], fontsize=12)

        plt.subplots_adjust(left=0, right=1, top=0.97, bottom=0.05)
        fig.set_size_inches(6, 2.75)
        fname = (
            f'group_{task_name}_{contrast}_flat_contour_fslr32k.png'
            if len(hemi) == 2 else
            f'group_{task_name}_{contrast}_flat_contour_fslr32k_{hemi[0]}.png'
        )
        fig.savefig(
            os.path.join(output_dir, fname),
            dpi=300, bbox_inches='tight', pad_inches=0
        )
        return

    # signed (diverging) single-contrast branch
    if not two_rgb and signed:
        lh, rh = stats
        vlim = vmax if vmax is not None else np.nanmax(
            [np.nanmax(np.abs(lh)), np.nanmax(np.abs(rh))])
        if not np.isfinite(vlim) or vlim <= 0:
            vlim = float(threshold) + 1e-6
        # blue (Non-Random > Random) <-> gray (n.s.) <-> red (Random > Non-Random)
        diverging = make_signed_gray_threshold_cmap(threshold, vlim)
        for ax, stat, h in zip(axs, (lh, rh), hemi):
            s_arr = np.asarray(stat, float).copy()
            # blank sub-threshold vertices on BOTH tails (NaN -> transparent)
            s_arr[np.abs(s_arr) < threshold] = np.nan
            plt.sca(ax)
            flatmap.plot(
                s_arr,
                surf=surfaces[h],
                underlay=underlays[h],
                undermap='gray',
                underscale=[-1.5, 1],
                cscale=[-vlim, vlim],
                cmap=diverging,
                borders=(borders[h] if show_borders else None),
                new_figure=False,
                frame=None
            )
        if show_colorbar:
            sm = ScalarMappable(
                norm=Normalize(vmin=-vlim, vmax=vlim), cmap=diverging)
            cbar = fig.colorbar(
                sm, ax=list(axs), orientation='horizontal',
                fraction=0.05, pad=0.02
            )
            cbar.set_label(cbar_title, fontsize=12, labelpad=8)
            dec = int(tick_decimals) if tick_decimals is not None else 2
            # If every supra-threshold vertex is one-signed (e.g. a pure
            # deactivation), show a single-direction colorbar instead of the
            # symmetric diverging one -- the symmetric bar would otherwise imply
            # a non-existent opposite tail and print zero twice. Colors stay
            # matched to the surface (same cmap and norm); we just crop the view.
            _sv = np.concatenate([np.asarray(lh, float).ravel(),
                                  np.asarray(rh, float).ravel()])
            _sv = _sv[np.isfinite(_sv) & (np.abs(_sv) >= float(threshold))]
            one_sided = ('neg' if (_sv.size and np.all(_sv < 0)) else
                         'pos' if (_sv.size and np.all(_sv > 0)) else None)
            if one_sided == 'neg':
                cbar.ax.set_xlim(-vlim, 0.0)
                ticks = np.array([-vlim, -float(threshold), 0.0])
            elif one_sided == 'pos':
                cbar.ax.set_xlim(0.0, vlim)
                ticks = np.array([0.0, float(threshold), vlim])
            else:
                # genuinely diverging: keep the symmetric bar (edges of the
                # gray non-significant band plus the two extremes)
                ticks = np.array([-vlim, -float(threshold),
                                  float(threshold), vlim])
            cbar.set_ticks(ticks)
            cbar.ax.set_xticklabels(
                [f'{t:.{dec}f}' for t in ticks], fontsize=12)
        plt.subplots_adjust(left=0, right=1, top=0.97, bottom=0.05)
        fig.set_size_inches(6, 2.75)
        contrast = contrast_tag.lower()
        task_name = task_key.replace('_', '-')
        fname = (
            f'group_{task_name}_{contrast}_flat_signed_fslr32k.png'
            if len(hemi) == 2 else
            f'group_{task_name}_{contrast}_flat_signed_fslr32k_{hemi[0]}.png'
        )
        fig.savefig(
            os.path.join(output_dir, fname),
            dpi=300, bbox_inches='tight', pad_inches=0
        )
        return

    # single-contrast branch
    if not two_rgb:
        lh, rh = stats

        # Ensure SUITPy uses a deterministic color scale:
        # - lower bound defaults to `threshold` (unless `vmin` provided)
        # - upper bound defaults to data max (unless `vmax` provided)
        cscale_lo = threshold if vmin is None else vmin
        data_max = np.nanmax([np.nanmax(lh), np.nanmax(rh)])
        cscale_hi = data_max if vmax is None else vmax
        if not np.isfinite(cscale_hi):
            cscale_hi = cscale_lo
        if cscale_hi <= cscale_lo:
            cscale_hi = cscale_lo + 1e-6

        for ax, stat, h in zip(axs, (lh, rh), hemi):
            plt.sca(ax)
            flatmap.plot(
                stat,
                surf=surfaces[h],
                underlay=underlays[h],
                undermap='gray',
                underscale=[-1.5, 1],
                threshold=threshold,
                cscale=[cscale_lo, cscale_hi],
                cmap=colormap,
                borders=(borders[h] if show_borders else None),
                new_figure=False,
                frame=None
            )

        # only show colorbar if threshold is finite and...
        # ... at least one value ≥ thr
        show_cbar1 = (
            bool(show_colorbar)
            and bool(np.isfinite(threshold) and np.nanmax(stats) >= threshold)
        )

        if show_cbar1:
            norm = plt.Normalize(vmin=threshold, vmax=cscale_hi)
            sm = ScalarMappable(norm=norm, cmap=colormap)
            cbar = fig.colorbar(
                sm, ax=list(axs), orientation='horizontal',
                fraction=0.05, pad=0.02
            )
            cbar.set_label(cbar_title, fontsize=12, labelpad=8)
            ticks = np.linspace(threshold, cscale_hi, n_ticks)
            cbar.set_ticks(ticks)
            dec = int(tick_decimals) if tick_decimals is not None else 2
            cbar.ax.set_xticklabels(
                [f'{t:.{dec}f}' for t in ticks], fontsize=12
            )

    # two-contrast RGB overlay
    else:
        (lh1, rh1), (lh2, rh2) = stats
        thr1, thr2 = threshold
        v1, v2 = vmax

        # only show bars if at least one of the maps has something...
        # ... above its thr
        show_cbar2 = bool((np.isfinite(thr1) and np.isfinite(thr2) 
                           and (np.nanmax((lh1, rh1)) >= thr1 
                                or np.nanmax((lh2, rh2)) >= thr2)))

        # parse colors
        color1 = colors[0]
        color2 = colors[1]
        rgb1 = np.array(mcolors.to_rgb(color1))
        rgb2 = np.array(mcolors.to_rgb(color2))

        for ax, h in zip(axs, hemi):
            plt.sca(ax)
            arr1 = lh1 if h == 'L' else rh1
            arr2 = lh2 if h == 'L' else rh2
            nvert = arr1.shape[0]

            # normalize each map to [0,1]
            norm1 = np.clip(arr1 / v1, 0, 1)
            norm2 = np.clip(arr2 / v2, 0, 1)

            # zero out below threshold
            thr_frac1 = thr1 / v1
            thr_frac2 = thr2 / v2
            norm1[norm1 < thr_frac1] = 0
            norm2[norm2 < thr_frac2] = 0

            # build RGB by weighted sum of color vectors
            rgb_data = np.outer(norm1, rgb1) + np.outer(norm2, rgb2)
            rgb_data = np.clip(rgb_data, 0, 1)

            # assemble RGBA
            data = np.zeros((nvert, 4), float)
            data[:, :3] = rgb_data
            alpha = ((norm1 > 0) | (norm2 > 0)).astype(float)
            data[:, 3] = alpha

            # transparent where no activity
            zero_mask = (alpha == 0)
            data[zero_mask, :] = np.nan

            flatmap.plot(
                data,
                overlay_type='rgb',
                surf=surfaces[h],
                underlay=underlays[h],
                undermap='gray',
                underscale=[-1.5, 1],
                borders=(borders[h] if show_borders else None),
                bordersize=1.5,
                bordercolor='k',
                new_figure=False,
                frame=None
            )

        if show_cbar2:
            # ################### LEGEND LABELS ########################

            # Split off the two contrast parts at the last 
            # '_vs_' or '_and_'
            if '_vs_' in contrast_tag:
                left, right = contrast_tag.rsplit('_vs_', 1)
            elif '_and_' in contrast_tag:
                left, right = contrast_tag.rsplit('_and_', 1)
            else:
                raise ValueError(f"Can't find '_vs_' or '_and_' separator in "
                                 f"'{fname}'")

            # Pull only the contrast key from the left side 
            # (drop any 'group_…_' prefix)
            c1_key = left.split('_')[-1]
            c2_key = right  # this is already just the second contrast

            # Format into nice labels
            label1 = c1_key.replace('-', ' ').replace('_', ' ')
            label2 = c2_key.replace('-', ' ').replace('_', ' ')

            # #################### COLORBARS ##########################

            # Compute fractions & RGB vectors
            thr_frac1 = thr1 / v1
            thr_frac2 = thr2 / v2
            rgb1 = np.array(to_rgb(colors[0]))
            rgb2 = np.array(to_rgb(colors[1]))

            # Define start/end colors for each bar
            # contrast1: from thr_color1 -> rgb1
            thr_color1 = tuple(rgb1 * thr_frac1)
            # contrast2: from thr_color2 -> rgb2
            thr_color2 = tuple(rgb2 * thr_frac2)
            # overlap: from thr_color1 + thr_color2 -> overlap_max
            thr_overlap = np.clip(rgb1 * thr_frac1 + rgb2 * thr_frac2, 0, 1)
            max_overlap = np.clip(rgb1 + rgb2, 0, 1)

            # Build colormaps & mappables
            cmap1 = LinearSegmentedColormap.from_list(
                "c1", [thr_color1, rgb1])
            sm1 = ScalarMappable(norm=Normalize(vmin=thr1, vmax=v1), 
                                 cmap=cmap1)
            sm1.set_array([])

            cmap2 = LinearSegmentedColormap.from_list(
                "c2", [thr_color2, rgb2])
            sm2 = ScalarMappable(norm=Normalize(vmin=thr2, vmax=v2), 
                                 cmap=cmap2)
            sm2.set_array([])

            cmap3 = LinearSegmentedColormap.from_list(
                "c3", [thr_overlap, max_overlap])
            # We normalize overlap on a 0–1 scale of (norm1+norm2)... 
            # ... clipped -> [thr_frac1 + thr_frac2, 1]
            min_ol = thr_frac1 + thr_frac2
            sm3 = ScalarMappable(norm=Normalize(vmin=min_ol, vmax=1.0), 
                                 cmap=cmap3)
            sm3.set_array([])

            # Compute mid‑ticks
            m1_1 = thr1 + (v1 - thr1) / 3
            m1_2 = thr1 + 2*(v1 - thr1) / 3
            m2_1 = thr2 + (v2 - thr2) / 3
            m2_2 = thr2 + 2*(v2 - thr2) / 3
            m3_1 = min_ol + (1.0 - min_ol) / 3
            m3_2 = min_ol + 2*(1.0 - min_ol) / 3

            # Place three horizontal bars
            bars = [
                # colorbar positions: [left, bottom, width, height]
                (sm1, [.04, .08, .25, .04], thr1, m1_1, m1_2, v1,
                 f"Z-values ({label1})"),
                (sm2, [.3825, .08, .25, .04], thr2, m2_1, m2_2, v2,
                 f"Z-values ({label2})"),
                (sm3, [.715, .08, .25, .04], min_ol, m3_1, m3_2, 1.0,
                 "Co-activation")
            ]

            # Do colorbars
            fig = plt.gcf()
            dec = int(tick_decimals) if tick_decimals is not None else 2
            for sm, rect, lo, m1, m2, hi, lbl in bars:
                cax = fig.add_axes(rect)
                cb = fig.colorbar(sm, cax=cax, orientation='horizontal',
                                ticks=[lo, m1, m2, hi])
                cb.set_label(lbl, fontsize=9, labelpad=5)
                cb.ax.set_xticklabels(
                    [f"{lo:.{dec}f}", f"{m1:.{dec}f}", f"{m2:.{dec}f}",
                     f"{hi:.{dec}f}"]
                )
                cb.ax.tick_params(labelsize=8)

    plt.subplots_adjust(left=0, right=1, top=0.97, bottom=0.05)
    fig.set_size_inches(6, 2.75)
    suffix = 'flat' if not two_rgb else 'flat_overlay'
    fname = (
        f'group_{task_name}_{contrast}_{suffix}_'
        f'fslr32k.png'
        if len(hemi) == 2 else
        f'group_{task_name}_{contrast}_{suffix}_'
        f'fslr32k_{hemi[0]}.png'
    )
    fig.savefig(
        os.path.join(output_dir, fname),
        dpi=300,
        bbox_inches='tight',
        pad_inches=0
    )


def make_regime_cmaps(
    warm=('#E1261C', '#FF8C00', '#FFE100'),
    cool=('#1A6FE0', '#22C2E8', '#9BE8FF'),
    neutral=('#DB2777', '#F9A8D4'),
    undetermined=('#7B7239', '#C6BC82'),
    n=256,
):
    """Four intensity colormaps (low |z| -> high |z|) for the regimes of the
    combined Random / Non-Random map: activation (warm), deactivation (cool),
    crossover (neutral), and undetermined (muted olive). The neutral ramp is an
    arbitrary flag colour for a regime expected to be empty; change it freely.
    The undetermined ramp is a low-saturation, non-grey hue so those vertices
    read as a distinct 'no rest anchor' class without dissolving into the grey
    flat underlay."""
    return (
        LinearSegmentedColormap.from_list('regime_act', list(warm), N=n),
        LinearSegmentedColormap.from_list('regime_deact', list(cool), N=n),
        LinearSegmentedColormap.from_list('regime_cross', list(neutral), N=n),
        LinearSegmentedColormap.from_list('regime_undet',
                                          list(undetermined), N=n),
    )


def regime_rgba(z_diff, z_active, z_passive, thr_diff, vmax,
                thr_active=0.0, thr_passive=0.0, gate=True,
                show_undetermined=True, min_vertices=0,
                display_tail='positive', cmaps=None):
    """Per-vertex RGBA for one hemisphere of a paired-condition difference map.

    The difference is ``diff = active - passive`` (e.g. Random - Non-Random).
    Classification proceeds in two explicit steps so the regime always refers
    to the contrast direction of interest and never to its opposite:

    1. Keep only vertices in the requested *displayed tail* of the difference:
           display_tail='positive' -> diff >=  thr_diff   (active > passive)
           display_tail='negative' -> diff <= -thr_diff   (passive > active)
           display_tail='both'     -> |diff| >= thr_diff   (each vertex by tail)
       This replaces the old symmetric base, which mixed the active > passive
       and passive > active tails and could mislabel the opposite direction.

    2. Within that tail, read the regime off the two conditions vs baseline,
       driven by the ACTIVE condition (the larger one in that tail):
           Activation   : active sig > baseline AND passive NOT sig < baseline
                          -> a genuine rise of the active condition.
           Deactivation : passive sig < baseline AND active NOT sig > baseline
                          -> carried by suppression of the passive condition,
                             not a rise above baseline.
           Crossover    : active sig > baseline AND passive sig < baseline
                          -> opposite anchors.
           Undetermined : neither anchor -> difference significant but neither
                          condition resolved vs baseline (the paired contrast
                          is more sensitive than either condition alone).

    The four classes are the 2x2 of (active-above) x (passive-below), so they
    partition the displayed tail exactly. On the positive tail the active
    condition is ``z_active`` and the passive ``z_passive``; on the negative
    tail the roles swap, so "activation"/"deactivation" keep their meaning in
    either direction.

    Parameters
    ----------
    z_diff : (nvert,) signed Z of (active - passive).
    z_active, z_passive : (nvert,) signed Z of each condition vs baseline, in
        contrast order (active = minuend, passive = subtrahend).
    thr_diff : |Z| threshold of the difference map (its FDR z*; the two-sided
        value when a two-sided test was used -- the caller decides, see
        parse_sides / the --regime block).
    vmax : upper |Z| of the shared intensity ramp.
    thr_active, thr_passive : |Z| FDR thresholds of each condition vs baseline.
        With gate=True they set the per-condition significance; with gate=False
        the raw sign of each group mean is used instead.
    gate : significance-gated classification (default) vs raw-sign.
    show_undetermined : if False, the undetermined class is left transparent.
    min_vertices : if >0, drop any class with fewer than this many vertices in
        the hemisphere (so sub-visible specks neither paint nor get a colorbar).
    display_tail : 'positive' | 'negative' | 'both'.
    cmaps : (warm, cool, neutral, undetermined); defaults via make_regime_cmaps.

    Returns
    -------
    rgba : (nvert, 4); unclassified vertices NaN (transparent under 'rgb').
    counts : (n_activation, n_deactivation, n_crossover, n_undetermined).
    """
    if cmaps is None:
        cmaps = make_regime_cmaps()
    cmap_act, cmap_deact, cmap_cross, cmap_undet = cmaps

    z_diff = np.asarray(z_diff, float)
    z_active = np.asarray(z_active, float)
    z_passive = np.asarray(z_passive, float)
    nvert = z_diff.size

    thr_diff = float(thr_diff)
    vmax = float(vmax)
    if not np.isfinite(vmax) or vmax <= thr_diff:
        vmax = thr_diff + 1e-6

    finite = (np.isfinite(z_diff) & np.isfinite(z_active)
              & np.isfinite(z_passive))

    # Step 1: restrict to the displayed tail of the difference.
    dt = str(display_tail).strip().lower()
    if dt in ('positive', 'pos', 'right', 'r', 'greater', '+'):
        in_pos = finite & (z_diff >= thr_diff)
        in_neg = np.zeros(nvert, bool)
    elif dt in ('negative', 'neg', 'left', 'l', 'less', '-'):
        in_pos = np.zeros(nvert, bool)
        in_neg = finite & (z_diff <= -thr_diff)
    elif dt in ('both', 'two-sided', 'two_sided', 'two', '2'):
        in_pos = finite & (z_diff >= thr_diff)
        in_neg = finite & (z_diff <= -thr_diff)
    else:
        raise ValueError(
            f"Invalid display_tail {display_tail!r}; "
            "use 'positive', 'negative', or 'both'.")
    base = in_pos | in_neg

    # Step 2: anchor each tail by its active (larger) and passive conditions.
    # Positive tail: active=z_active, passive=z_passive. Negative tail: swapped,
    # so the larger condition is always the 'active' driver of activation.
    if gate:
        a_above = z_active >= float(thr_active)     # active sig > baseline
        a_below = z_active <= -float(thr_active)    # active sig < baseline
        p_above = z_passive >= float(thr_passive)   # passive sig > baseline
        p_below = z_passive <= -float(thr_passive)  # passive sig < baseline
    else:
        a_above, a_below = z_active > 0, z_active < 0
        p_above, p_below = z_passive > 0, z_passive < 0

    active_above = (in_pos & a_above) | (in_neg & p_above)
    passive_below = (in_pos & p_below) | (in_neg & a_below)

    activ = base & active_above & ~passive_below
    cross = base & active_above & passive_below
    deact = base & ~active_above & passive_below
    undet = base & ~active_above & ~passive_below

    # Honour the display switches at mask level so the returned counts match
    # what is painted (and therefore what the colorbar legend shows).
    if not show_undetermined:
        undet = np.zeros(nvert, bool)
    if min_vertices and int(min_vertices) > 0:
        for m in (activ, deact, cross, undet):
            if 0 < int(m.sum()) < int(min_vertices):
                m[:] = False

    mag = np.clip((np.abs(z_diff) - thr_diff) / (vmax - thr_diff), 0.0, 1.0)
    rgba = np.full((nvert, 4), np.nan, float)
    if activ.any():
        rgba[activ] = cmap_act(mag[activ])
    if deact.any():
        rgba[deact] = cmap_deact(mag[deact])
    if cross.any():
        rgba[cross] = cmap_cross(mag[cross])
    if undet.any():
        rgba[undet] = cmap_undet(mag[undet])

    return rgba, (int(activ.sum()), int(deact.sum()),
                  int(cross.sum()), int(undet.sum()))


def regime_exclusion_stats(z_diff, z_rand, thr_diff, thr_rand,
                           display_tail='positive', cortex_mask=None):
    """Quantify how much of a Random - Non-Random significant map is NOT backed
    by a significant Random vs Rest activation, i.e. the vertices that would be
    excluded if Random significance vs rest were required. At those vertices the
    Random - Non-Random difference is carried by suppression of the Non-Random
    condition rather than by Random rising above rest.

    Parameters
    ----------
    z_diff : (nvert,) signed Z of Random - Non-Random (concatenated L+R).
    z_rand : (nvert,) signed Z of Random vs Rest (concatenated L+R).
    thr_diff : |Z| FDR threshold of the difference map.
    thr_rand : |Z| FDR threshold of Random vs Rest. A vertex counts as
        "Random not significant" when |z_rand| < thr_rand (two-sided, i.e.
        Random is indistinguishable from rest in either direction). Pass the
        one-sided positive threshold instead to mean "not significantly
        activated".
    cortex_mask : optional (nvert,) bool, True for cortical (non-medial-wall)
        vertices; used as the 'total' denominator. If None, all finite vertices
        are used.

    Returns
    -------
    dict with vertex counts and the two requested percentages:
        pct_within_contrast : excluded / (Random - Non-Random significant)
        pct_of_total_cortex : excluded / (all cortical vertices)
    """
    z_diff = np.asarray(z_diff, float)
    z_rand = np.asarray(z_rand, float)
    finite = np.isfinite(z_diff) & np.isfinite(z_rand)
    cortex = finite if cortex_mask is None else (
        np.asarray(cortex_mask, bool) & finite)

    n_total = int(cortex.sum())
    dt = str(display_tail).strip().lower()
    if dt.startswith('pos') or dt in ('right', 'r', 'greater', '+'):
        sig = z_diff >= float(thr_diff)
    elif dt.startswith('neg') or dt in ('left', 'l', 'less', '-'):
        sig = z_diff <= -float(thr_diff)
    else:  # 'both'
        sig = np.abs(z_diff) >= float(thr_diff)
    in_contrast = cortex & sig
    n_contrast = int(in_contrast.sum())
    excluded = in_contrast & (np.abs(z_rand) < float(thr_rand))
    n_excluded = int(excluded.sum())

    pct_contrast = (100.0 * n_contrast / n_total) if n_total else float('nan')
    pct_within = (100.0 * n_excluded / n_contrast) if n_contrast \
        else float('nan')
    pct_total = (100.0 * n_excluded / n_total) if n_total else float('nan')
    return {
        'n_total_cortex': n_total,
        'n_contrast': n_contrast,
        'pct_contrast_of_total': pct_contrast,
        'n_excluded_random_ns': n_excluded,
        'pct_within_contrast': pct_within,
        'pct_of_total_cortex': pct_total,
    }


def plot_regime_flatmap(
    diff_stats, cond_rand_stats, cond_nonrand_stats,
    thr_diff, vmax, task_key, contrast_tag, output_dir,
    thr_rand=0.0, thr_nonrand=0.0, gate=True,
    show_undetermined=True, min_vertices=0,
    display_tail='positive',
    hemi=['L', 'R'],
    warm=('#E1261C', '#FF8C00', '#FFE100'),
    cool=('#1A6FE0', '#22C2E8', '#9BE8FF'),
    neutral=('#DB2777', '#F9A8D4'),
    undetermined=('#7B7239', '#C6BC82'),
    show_colorbar=True, n_ticks=4, tick_decimals=1,
    contour_stat=None, contour_threshold=None, contour_color='k',
    contour_linewidth=1.0, contour_tail='positive',
    show_borders=True,
):
    """Combined Random/Non-Random flatmap coloured by rest-referenced regime.

    The difference contrast (Random - Non-Random) supplies the displayed
    magnitude; the two condition-vs-Rest maps supply the regime (activation /
    deactivation / crossover). See regime_rgba for the classification.

    diff_stats         : [lh, rh] signed Z of Random - Non-Random.
    cond_rand_stats    : [lh, rh] signed Z of Random vs Rest.
    cond_nonrand_stats : [lh, rh] signed Z of Non-Random vs Rest.

    The network outline is optional and independent: pass contour_stat
    ([lh, rh]) and contour_threshold to draw it, or leave them None.
    """
    contrast = contrast_tag.lower()
    task_name = task_key.replace('_', '-')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    surf_dir = os.path.join(script_dir, 'fslr32k_meshes')
    borders = {h: os.path.join(surf_dir, 'borders', f'fs_LR.32k.{h}.border')
               for h in hemi}
    underlays = {h: os.path.join(surf_dir, 'flat', f'fs_LR.32k.{h}.shape.gii')
                 for h in hemi}
    surfaces = {h: os.path.join(surf_dir, 'flat', f'fs_LR.32k.{h}.flat.surf.gii')
                for h in hemi}

    cmaps = make_regime_cmaps(warm=warm, cool=cool, neutral=neutral,
                              undetermined=undetermined)

    fig, axs = plt.subplots(1, len(hemi), figsize=(8, 4),
                            gridspec_kw={'wspace': 0.05})
    if len(hemi) == 1:
        axs = [axs]

    per_h = {
        'L': (diff_stats[0], cond_rand_stats[0], cond_nonrand_stats[0]),
        'R': (diff_stats[1], cond_rand_stats[1], cond_nonrand_stats[1]),
    }
    total = np.zeros(4, int)

    for ax, h in zip(axs, hemi):
        zd, zr, znr = per_h[h]
        rgba, counts = regime_rgba(
            zd, zr, znr, thr_diff, vmax,
            thr_active=thr_rand, thr_passive=thr_nonrand, gate=gate,
            show_undetermined=show_undetermined,
            min_vertices=min_vertices, display_tail=display_tail, cmaps=cmaps)
        total += np.asarray(counts, int)
        plt.sca(ax)
        flatmap.plot(
            rgba, overlay_type='rgb',
            surf=surfaces[h], underlay=underlays[h], undermap='gray',
            underscale=[-1.5, 1],
            borders=(borders[h] if show_borders else None),
            bordersize=1.5, bordercolor='k',
            new_figure=False, frame=None,
        )
        if contour_stat is not None and contour_threshold is not None:
            c2 = contour_stat[0] if h == 'L' else contour_stat[1]
            overlay_region_contour(
                ax, surfaces[h], c2, float(contour_threshold),
                color=contour_color, linewidth=contour_linewidth,
                tail=contour_tail)

    print(f"[regime] displayed vertices -> activation={total[0]} "
          f"deactivation={total[1]} crossover={total[2]} "
          f"undetermined={total[3]}"
          f"{' (undetermined hidden)' if not show_undetermined else ''}"
          f"{f' (regimes <{int(min_vertices)} vtx/hemi dropped)' if min_vertices else ''}")

    if show_colorbar:
        dec = int(tick_decimals) if tick_decimals is not None else 2
        vlim = float(vmax) if (np.isfinite(vmax)
                               and vmax > thr_diff) else float(thr_diff) + 1e-6
        ticks = np.linspace(float(thr_diff), vlim, n_ticks)
        # Left -> right: deactivation, crossover, activation, then undetermined
        # (a separate 'no rest anchor' class). The regime name sits over each
        # bar; the Z(...) label below gives the rest-referenced ordering of the
        # conditions, with "R>NR" in bold for the single (Random > Non-Random)
        # direction (valid for one-sided runs). Only regimes with >=1 vertex get
        # a bar (a hidden or empty regime drops out), and the survivors are
        # re-spaced and centred.
        bar_w, bar_gap, bar_y, bar_h = 0.24, 0.09, 0.085, 0.035
        undet_n = int(total[3]) if show_undetermined else 0
        candidates = [
            (cmaps[1], 'Deactivation',
             r'$\mathrm{Z\ (Baseline}\gg\mathbf{R{>}NR}\mathrm{)}$', int(total[1])),
            (cmaps[2], 'Crossover',
             r'$\mathrm{Z\ (}\mathbf{R\gg}\mathrm{Baseline}\mathbf{\gg NR}\mathrm{)}$',
             int(total[2])),
            (cmaps[3], 'Undetermined',
             r'$\mathrm{Z\ (}\mathbf{R{>}NR}\mathrm{,\ n.s.\ vs\ Baseline)}$',
             undet_n),
            (cmaps[0], 'Activation',
             r'$\mathrm{Z\ (}\mathbf{R{>}NR}\mathrm{\gg Baseline)}$', int(total[0])),
        ]
        present = [(c, ref, lbl) for (c, ref, lbl, n) in candidates if n > 0]
        nb = len(present)
        if nb:
            span = nb * bar_w + (nb - 1) * bar_gap
            x0 = 0.5 - span / 2.0
            for i, (cmap_i, ref, lbl) in enumerate(present):
                rect = [x0 + i * (bar_w + bar_gap), bar_y, bar_w, bar_h]
                sm = ScalarMappable(
                    norm=Normalize(vmin=float(thr_diff), vmax=vlim), cmap=cmap_i)
                sm.set_array([])
                cax = fig.add_axes(rect)
                cb = fig.colorbar(sm, cax=cax, orientation='horizontal',
                                  ticks=ticks)
                cb.ax.set_title(ref, fontsize=9, pad=3)    # reference over bar
                cb.set_label(lbl, fontsize=8, labelpad=4)  # Z(...) below bar
                cb.ax.set_xticklabels([f'{t:.{dec}f}' for t in ticks])
                cb.ax.tick_params(labelsize=7)

    plt.subplots_adjust(left=0, right=1, top=0.98, bottom=0.06)
    fig.set_size_inches(6, 3.0)
    suffix = 'flat_regime_contour' if contour_stat is not None else 'flat_regime'
    fname = (
        f'group_{task_name}_{contrast}_{suffix}_fslr32k.png'
        if len(hemi) == 2 else
        f'group_{task_name}_{contrast}_{suffix}_fslr32k_{hemi[0]}.png'
    )
    fig.savefig(os.path.join(output_dir, fname),
                dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    return tuple(int(x) for x in total)


def plot_multirois_flatmap(
    stats,
    threshold,
    task_key,
    contrast_tag,
    output_dir,
    hemi=['L', 'R'],
    colormaps=None,
    labels=None,
    vmax=None,
    cbar_title='Fraction of Participants',
    cbar_ticks=None,
    tick_decimals=2,
    show_borders=True,
):
    """
    Plot one or two contrasts on a flat cortical map.

    Single-contrast:
      stats: [lh_array, rh_array]
      threshold: float
      colormap: str
      vmax: float

    Two-contrast RGB overlay:
      stats: [[lh1, rh1], [lh2, rh2]]
      threshold: [thr1, thr2]
      colors: [color1, color2]  # any matplotlib color
      vmax: [v1, v2]

    Note on magnitude of co-activation for the RGB overlay:
    -------------------------------------------------------
    The “Overlap” colorbar is a third gradient that tells you how
    strongly both contrasts co-activate at each vertex. Concretely:

    1. Range
    --------
    Minimum tick (vmin) = thr1/v1 + thr2/v2
    (i.e. both maps just at their respective statistical thresholds)

    Maximum tick (vmax) = 1.0
    (i.e. at least one map at its peak, and the other possibly
    contributing up to its peak)

    2. Color gradient
    -----------------
    The start color is the sum of the two threshold-level colors

    The end color is the sum of the two full-bright colors
    (clipped to [0,1])

    Intermediate hues reflect intermediate sums of the two normalized
    intensities

    3. Tick labels
    --------------
    Show four values:

    Min sum-of-fractions (thr1/v1 + thr2/v2)

    Two intermediate sums (one- and two-thirds along the range)

    Max = 1.0
    """

    contrast = contrast_tag.lower()
    task_name = task_key.replace('_', '-')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    surf_dir = os.path.join(script_dir, 'fslr32k_meshes')

    borders = {
        h: os.path.join(
            surf_dir, 'borders', f'fs_LR.32k.{h}.border')
        for h in hemi
    }
    underlays = {
        h: os.path.join(
            surf_dir, 'flat', f'fs_LR.32k.{h}.shape.gii')
        for h in hemi
    }
    surfaces = {
        h: os.path.join(
            surf_dir, 'flat', f'fs_LR.32k.{h}.flat.surf.gii')
        for h in hemi
    }

    fig, axs = plt.subplots(
        1, len(hemi), figsize=(8, 4),
        gridspec_kw={'wspace': 0.05}
    )
    
    # multi-iroi overlay (N > 2): alpha-composited RGBA layers
    # stats is a list like: [[lh1, rh1], [lh2, rh2], ...]
    # We alpha-composite per-ROI RGBA layers, each using its own colormap.
    thr = float(threshold)
    vhi = None
    if vmax is not None:
        vhi = (
            float(vmax)
            if not isinstance(vmax, (list, tuple))
            else float(vmax[0])
        )

    if labels is None:
        labels = [f"map-{i+1}" for i in range(len(stats))]

    if len(colormaps) != len(stats):
        raise ValueError(
            "When using multi_overlay, 'colormaps' must have the same "
            "length as 'stats'."
        )

    # Plot hemispheres
    for ax, h in zip(axs, hemi):
        plt.sca(ax)

        # Initialize composite RGBA
        arr0 = stats[0][0] if h == 'L' else stats[0][1]
        nvert = int(arr0.shape[0])

        # Build a categorical RGB overlay: each vertex is assigned to
        # the iROI with the maximum value (after thresholding). This
        # avoids any alpha-compositing darkening when iROIs do not
        # overlap (the intended use-case here).
        vals = []
        for (lh_i, rh_i) in stats:
            arr = lh_i if h == 'L' else rh_i
            arr = np.asarray(arr, float)
            arr[arr < thr] = np.nan
            vals.append(arr)

        vals = np.vstack(vals)  # shape: (n_maps, nvert)
        n_maps = int(vals.shape[0])
        valid = np.any(np.isfinite(vals), axis=0)

        # Winner-take-all index and winning value per vertex.
        vals_fill = np.where(np.isfinite(vals), vals, -np.inf)
        winner = np.argmax(vals_fill, axis=0)
        win_val = np.max(vals_fill, axis=0)

        data = np.full((nvert, 4), np.nan, float)

        # Set the global upper bound for scaling. If no explicit vmax was
        # provided, fall back to the maximum winning value for this hemi.
        if vhi is None:
            if np.any(valid):
                vhi = float(np.nanmax(win_val[valid]))
            else:
                vhi = float(thr)

        denom = vhi - thr
        if denom <= 0:
            denom = 1.0

        for i, cmap_i in enumerate(colormaps):
            idx = valid & (winner == i)
            if not np.any(idx):
                continue

            norm = np.clip((win_val[idx] - thr) / denom, 0.0, 1.0)
            cmap_obj = plt.get_cmap(cmap_i)
            rgba = cmap_obj(norm)

            data[idx, :3] = rgba[:, :3]
            data[idx, 3] = 1.0

        flatmap.plot(
            data,
            overlay_type='rgb',
            surf=surfaces[h],
            underlay=underlays[h],
            undermap='gray',
            underscale=[-1.5, 1],
            borders=(borders[h] if show_borders else None),
            bordersize=1.5,
            bordercolor='k',
            new_figure=False,
            frame=None
        )

    # 6 small colorbars (2x3) using the same normalization
    cticks = (
        cbar_ticks
        if cbar_ticks is not None
        else np.linspace(thr, vhi, 5)
    )
    dec = int(tick_decimals) if tick_decimals is not None else 2

    fig = plt.gcf()

    # Leave room on the right for a vertical stack of horizontal colorbars
    plt.subplots_adjust(left=0.0, right=0.77, top=0.97, bottom=0.10)

    n_maps = len(labels)
    x0 = 0.78
    w = 0.20

    # Stack horizontal colorbars vertically without overlap.
    # Define geometry in figure-relative coordinates.
    top = 0.75
    bottom = 0.10
    gap = 0.025
    avail = top - bottom

    # Add a title above the colorbars
    fig.text(
        x0 + w / 2,
        top + 0.035,
        cbar_title,
        ha="center",
        va="bottom",
        fontsize=9,
    )

    # Compute a bar height that fits all maps. If needed, shrink the gap.
    BAR_THICKNESS_SCALE = 0.40  # <--- new (0.45–0.65 are reasonable)
    bar_h = (avail / n_maps) * BAR_THICKNESS_SCALE
    if bar_h < 0.01:
        gap = 0.005
        bar_h = (avail - (n_maps - 1) * gap) / max(n_maps, 1)
    bar_h = max(bar_h, 0.01)

    # Desired top-to-bottom order of colorbars
    CBAR_ORDER = [
        "PreSMA",
        "SMA",
        "PMD",
        "PMV",
        "Auditory Cortex",
        "Visual Cortex",
    ]

    label_to_cmap = dict(zip(labels, colormaps))

    labels = [lab for lab in CBAR_ORDER if lab in label_to_cmap]
    colormaps = [label_to_cmap[lab] for lab in labels]

    for i, (cmap_i, lab) in enumerate(zip(colormaps, labels)):
        y = top - (i + 1) * bar_h - i * gap

        rect = [x0, y, w, bar_h]
        cax = fig.add_axes(rect)

        sm = ScalarMappable(
            norm=Normalize(vmin=thr, vmax=vhi),
            cmap=cmap_i,
        )
        sm.set_array([])

        cb = fig.colorbar(
            sm,
            cax=cax,
            orientation='horizontal',
            ticks=cticks,
        )
        if i < n_maps - 1:
            cb.ax.set_xticklabels([])
            cb.ax.tick_params(labelbottom=False, labelsize=7, length=2)
        else:
            cb.ax.set_xticklabels([f"{t:.{dec}f}" for t in cticks])
            cb.ax.tick_params(labelsize=7, length=2)

        # Put the ROI label on the right side of the bar
        cax.text(
            1.02,
            0.5,
            lab,
            va='center',
            ha='left',
            transform=cax.transAxes,
            fontsize=8,
        )

    fig.set_size_inches(7.2, 2.75)

    suffix = 'flat'
    fname = (
        f'group_{task_name}_{contrast}_{suffix}_fslr32k.png'
        if len(hemi) == 2 else
        f'group_{task_name}_{contrast}_{suffix}_fslr32k_{hemi[0]}.png'
    )
    fig.savefig(
        os.path.join(output_dir, fname),
        dpi=300,
        bbox_inches='tight',
        pad_inches=0
    )    


def split_and_save_sulc_cifti(cifti_path, output_dir):
    """
    Load a CIFTI-2 dscalar file containing sulcal depth, split the data
    into left and right hemispheres, and save them as GIFTI metric files
    with the correct metadata for Workbench.

    Parameters
    ----------
    cifti_path : str
        Path to the CIFTI file (.dscalar.nii).
    output_dir : str
        Directory where the GIFTI files will be saved.

    The files will be saved as:
      sulc.L.32k_fs_LR.gii   (for CortexLeft)
      sulc.R.32k_fs_LR.gii   (for CortexRight)
    """
    # Load the CIFTI file
    cifti = nib.load(cifti_path)
    data = cifti.get_fdata() # Expected shape: (num_maps, num_vertices)

    # Get the brain models (a list of brain model objects)
    brain_models = cifti.header.get_index_map(1).brain_models

    # Initialize variables for left and right hemisphere data
    lh_data = None
    rh_data = None

    # Loop through the brain models to extract data by hemisphere
    for bm in brain_models:
        if bm.brain_structure.upper() == 'CIFTI_STRUCTURE_CORTEX_LEFT':
            lh_data = data[0, bm.index_offset:bm.index_offset + bm.index_count]
        elif bm.brain_structure.upper() == 'CIFTI_STRUCTURE_CORTEX_RIGHT':
            rh_data = data[0, bm.index_offset:bm.index_offset + bm.index_count]

    if lh_data is None or rh_data is None:
        error_message = (
            "Could not find both left and right cortical data in "
            "the CIFTI file."
        )
        raise ValueError(error_message)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define output file paths
    gifti_l_path = os.path.join(output_dir, 'fs_LR.32k.L.sulc.dscalar.gii')
    gifti_r_path = os.path.join(output_dir, 'fs_LR.32k.R.sulc.dscalar.gii')

    # Define metadata for left and right hemispheres
    meta_l_dict = {
        "Name": "Sulcal Depth",
        "GeometricType": "Metric",
        "Caret-Version": "5.64",
        "encoding": "GZipBase64Binary",
        "structure": "CortexLeft",
    }
    meta_r_dict = {
        "Name": "Sulcal Depth",
        "GeometricType": "Metric",
        "Caret-Version": "5.64",
        "encoding": "GZipBase64Binary",
        "structure": "CortexRight",
    }

    # Convert dictionaries to GiftiMetaData objects
    meta_l = nib.gifti.GiftiMetaData(meta_l_dict)
    meta_r = nib.gifti.GiftiMetaData(meta_r_dict)

    # Create and save left hemisphere GIFTI file
    gifti_img_l = nib.gifti.GiftiImage()
    darray_l = nib.gifti.GiftiDataArray(lh_data.astype(np.float32))
    darray_l.intent = 2005  # NIFTI_INTENT_SHAPE for surface-based data
    darray_l.meta = meta_l
    gifti_img_l.add_gifti_data_array(darray_l)
    nib.save(gifti_img_l, gifti_l_path)

    # Create and save right hemisphere GIFTI file
    gifti_img_r = nib.gifti.GiftiImage()
    darray_r = nib.gifti.GiftiDataArray(rh_data.astype(np.float32))
    darray_r.intent = 2005
    darray_r.meta = meta_r
    gifti_img_r.add_gifti_data_array(darray_r)
    nib.save(gifti_img_r, gifti_r_path)

    print("Saved Left Hemisphere sulc:", gifti_l_path)
    print("Saved Right Hemisphere sulc:", gifti_r_path)


def grid_sample_border_vertices_snap(coords, cell_size=5.0):
    """
    Divide the space into a 3D grid with cells of size `cell_size`
    and for each cell that contains border vertices, compute the centroid
    and then snap it to the nearest original vertex in that cell.
    
    Parameters:
      coords : (N, 3) array of border vertex coordinates.
      cell_size : float, the size of each grid cell
                  (in the same units as your surface).
    
    Returns:
      representative : (M, 3) array of representative vertex coordinates
      (one per occupied cell).
    """
    coords = np.array(coords)
    # Determine grid indices for each vertex.
    min_coords = np.min(coords, axis=0)
    grid_indices = np.floor((coords - min_coords) / cell_size).astype(int)
 
    # Group indices by grid cell.
    cell_dict = {}
    for i, idx in enumerate(grid_indices):
        key = tuple(idx)
        if key not in cell_dict:
            cell_dict[key] = []
        cell_dict[key].append(i)
    
    # For each cell, compute the centroid and then pick the vertex...
    # ... closest to the centroid.
    representative = []
    for key, indices in cell_dict.items():
        cell_points = coords[indices]
        centroid = np.mean(cell_points, axis=0)
        distances = np.linalg.norm(cell_points - centroid, axis=1)
        min_index_in_cell = np.argmin(distances)
        # Snap: choose the actual vertex from the cell.
        representative.append(coords[indices[min_index_in_cell]])
    
    return np.array(representative)


def generate_sphere(res=3, radius=1.0):
    """
    Generate a UV sphere mesh with subdivisions defined by `res`
    and given radius.
    
    Returns:
      verts : numpy array of shape (N,3) for sphere vertices.
      faces : numpy array of shape (M,3) for triangular faces.
    """
    verts = []
    faces = []
    for i in range(res + 1):
        theta = np.pi * i / res  # 0 to pi
        for j in range(2 * res + 1):
            phi = 2 * np.pi * j / (2 * res)  # 0 to 2pi
            x = radius * np.sin(theta) * np.cos(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = radius * np.cos(theta)
            verts.append([x, y, z])
    verts = np.array(verts)
    n_verts_per_row = 2 * res + 1
    for i in range(res):
        for j in range(2 * res):
            idx = i * n_verts_per_row + j
            idx_next = idx + n_verts_per_row
            faces.append([idx, idx + 1, idx_next])
            faces.append([idx + 1, idx_next + 1, idx_next])
    return np.array(verts), np.array(faces, dtype=int)


def replicate_spheres_for_vertices(coords, sphere_verts, sphere_faces):
    """
    Replicate a sphere mesh (sphere_verts, sphere_faces)
    at each coordinate in `coords`.
    
    Returns:
      all_verts : combined vertices from all replicated spheres.
      all_faces : combined faces (with proper index offsets).
    """
    all_verts = []
    all_faces = []
    vert_count = 0
    for c in coords:
        shifted = sphere_verts + c
        all_verts.append(shifted)
        all_faces.append(sphere_faces + vert_count)
        vert_count += len(sphere_verts)
    return np.vstack(all_verts), np.vstack(all_faces)


def plotly_surfmap(
        sulc_path, borders_path, surf_path, data, threshold, outfname,
        gray_scale=[[0, 'rgb(105,105,105)'], [1, 'rgb(211,211,211)']],
        resolution=3, radius=0.5, plot_title=None, cmap='viridis',
        cbar_title='Z-values', cell_size=5.0, marker_size=5, borders=False):
    """
    Generates an interactive Plotly HTML render of an inflated brain
    surface with:
      - Sulcal depth background.
      - Thresholded activation overlay.
      - Border markers computed via grid-based sampling with snapping to
        the nearest border vertex.
    
    Parameters:
      sulc_path: str, path to the sulcal depth GIFTI file (.gii).
      borders_path: str, path to the borders GIFTI file (.label.gii).
      surf_path: str, path to the inflated surface mesh (.surf.gii).
      data: np.array, activation values per vertex.
      threshold: float, activation threshold.
      outfname: str, output filename (without extension) for the HTML
                file.
      gray_scale: list, colorscale for sulcal depth.
      resolution: int, resolution for sphere used as border markers.
      radius: float, radius of the sphere used for border markers.
      plot_title: str, title of the figure.
      cmap: str, colormap for activation overlay.
      cbar_title: str, label for the activation colorbar.
      cell_size: float, grid cell size for sampling border vertices.
      marker_size: int, size (scale factor) for the sphere markers.
    """
    # Load sulcal depth.
    sulc_img = nib.load(sulc_path)
    sulc_data = sulc_img.darrays[0].data

    # Load surface mesh.
    surf_img = nib.load(surf_path)
    surf_coords = surf_img.darrays[0].data  # shape: (N,3)
    faces = surf_img.darrays[1].data        # shape: (M,3)

    # Load border data.
    borders_img = nib.load(borders_path)
    borders_data = borders_img.darrays[0].data
    border_indices = np.where(borders_data > 0)[0]
    # Extract border vertex coordinates.
    border_coords = surf_coords[border_indices]

    # Determine active faces
    # (where all vertices have activation >= threshold).
    active_face_mask = np.all(data[faces] >= threshold, axis=1)
    active_faces = faces[active_face_mask]

    # Create sulcal background surface.
    background_surface = go.Mesh3d(
        x=surf_coords[:, 0],
        y=surf_coords[:, 1],
        z=surf_coords[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        intensity=sulc_data,
        colorscale=gray_scale,
        cmin=np.min(sulc_data),
        cmax=np.max(sulc_data),
        showscale=False,
        opacity=1,
        name='Sulc Background'
    )

    # Create activation overlay.
    activation_surface = go.Mesh3d(
        x=surf_coords[:, 0],
        y=surf_coords[:, 1],
        z=surf_coords[:, 2],
        i=active_faces[:, 0],
        j=active_faces[:, 1],
        k=active_faces[:, 2],
        intensity=data,
        colorscale=cmap,
        cmin=threshold,
        cmax=np.max(data),
        colorbar=dict(title=cbar_title, x=0.8, y=0.5, len=0.5),
        showscale=True,
        opacity=1,
        name='Activation Map'
    )

    # Grid-sample the border vertices:
    # compute one representative per cell,
    # then snap to the nearest vertex in that cell.
    sampled_border_coords = grid_sample_border_vertices_snap(
        border_coords, cell_size=cell_size)

    # Now, generate a sphere mesh (for a single dot).
    sphere_verts, sphere_faces = generate_sphere(res=resolution, radius=radius)
    # Replicate the sphere at each sampled border vertex.
    all_verts, all_faces = replicate_spheres_for_vertices(
        sampled_border_coords, sphere_verts, sphere_faces)
    border_spheres = go.Mesh3d(
        x=all_verts[:, 0],
        y=all_verts[:, 1],
        z=all_verts[:, 2],
        i=all_faces[:, 0],
        j=all_faces[:, 1],
        k=all_faces[:, 2],
        color='black',
        opacity=1.0,
        flatshading=True,
        name='Border Markers',
        showscale=False
    )

    # Combine all layers into a figure.
    if borders:
        fig = go.Figure(data=[background_surface, activation_surface,
                              border_spheres])
    else:
        fig = go.Figure(data=[background_surface, activation_surface])

    fig.update_layout(
        title=plot_title,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )

    # Save as interactive HTML.
    pio.write_html(fig, outfname, auto_open=True)
    print(f"Saved HTML figure: {outfname}.html")


# %%
# ============================ INPUTS =================================

# Subjects without pilot
SUBJECTS = [3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26, 28,
            29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

# Parent dir for output folders
surfparametric_folder = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'results', 'parametric_tests', 
    'surface')

# Production/Perception/NTFD/NTFD Random/All Main Tasks
task_tag = 'NTFD Random'
 
# To run every contrast:
# contrast_name = 'ALL' and contrast_name2 = None.
# To run a subset sequentially:
# contrast_name = ['Beat', 'Interval', 'Decision']
# contrast_name2 = None.
# To run overlays of pairs:
# contrast_name = 'Beat'
# contrast_name2 = 'Interval' (must be contrast name or list of names)
# For single or overlay, keep contrast_name/contrast_name2 as strings
contrast_name = 'Random vs Non-Random' # E.g. 'Beat', 'Interval', 'ALL', etc.
contrast_name2 = 'Encoding' # E.g. 'Interval'

# Contour overlay (used only with --contour). The second contrast
# (contrast_name2) is drawn as an outline and MAY come from a different task
# than the filled map. Set contour_task_tag to that task (e.g.
# 'All Main Tasks') or leave None to use the same task as contrast_name.
# contour_threshold_override sets the |z| threshold of the outline directly;
# leave None to use that contrast's whole-brain FDR threshold. For a dense
# contrast such as Encoding vs. Rest, the auto-FDR boundary is very extensive,
# so a fixed value (e.g. 3.09 for p <= 0.001) usually gives a cleaner outline.
contour_task_tag = 'All Main Tasks'  # e.g. 'All Main Tasks'
contour_threshold_override = None  # e.g. 3.09

# Sidedness of the FDR threshold / display. Each is 'one-sided' (positive tail
# only, activations), 'two-sided' (|z| FDR, both activations and deactivations;
# diverging colormap for the fill), or None (auto: two-sided for signed
# difference contrasts, one-sided otherwise).
#   contrast_sides : applies to the FILLED map (contrast_name), every branch.
#   contour_sides  : applies to the OUTLINE (contrast_name2), --contour only.
# Note: with contour_threshold_override set, that fixed |z| value is used for
# the outline regardless of contour_sides; contour_sides then only controls
# whether the outline traces activations only or both tails.
contrast_sides = None        # e.g. 'one-sided', 'two-sided', or None (auto)
# In the --regime path, contour_sides is parsed by parse_sides into (FDR test,
# drawn tail), so encode the tail here:
#   'two-sided:both'  -> two-sided z*, outline activations AND deactivations
#                        (Fig 6a-style full predictive-timing outline)
#   'two-sided:right' -> two-sided z*, outline the activation tail only
#                        (Fig 7-style, same z*, same-sign with the conjunctions)
#   'two-sided:left'  -> two-sided z*, deactivation tail only
#   'greater'/'less'  -> one-sided +/- FDR, right/left tail
# (In the legacy batch/single contour paths below, the tail still comes from
# contour_display; only the --regime path reads the tail from contour_sides.)
contour_sides = 'two-sided:both'

# contour_display : tail of the OUTLINE for the LEGACY batch/single contour
# paths only (the --regime path encodes the tail in contour_sides via
# parse_sides, and ignores contour_display). 'positive' draws the
# Encoding > Rest (activation) side only; 'both' draws activations and
# deactivations; None mirrors contour_sides.
contour_display = 'positive'

# Display the sulcal/gyral border outlines (the dotted lines from the fs_LR
# border files) on the cortical flatmaps. Set to False for a clean underlay
# with no sulcal borders. Applies to every flatmap (single, overlay, regime,
# contour and multi-ROI). It does NOT affect the region contour drawn with
# --contour (e.g. the Encoding-vs-Rest outline), which is controlled separately.
SHOW_SULCI_BORDERS = True

# ------------- Combined regime visualization (--contour) -------------
# Used only with: python volume_to_surface.py --regime
# Re-colours the signed difference map `contrast_name` (e.g.
# 'Random vs Non-Random') by the rest-referenced regime of its two underlying
# conditions, so that on a single flatmap one can tell apart:
#   activation   (both conditions above rest; warm),
#   deactivation (both conditions below rest; cool), and
#   crossover    (the two conditions disagree in sign; neutral, expected empty).
# The intensity within each regime is |Z(Random - Non-Random)|.
#
# REGIME_COMPONENTS maps each supported difference contrast to its two
# component conditions (each vs Rest). This is where you choose which contrast
# the regime visualization is applied to (set contrast_name accordingly).
REGIME_COMPONENTS = {
    'Random vs Non-Random': ('Random', 'Non-Random'),
    'Auditory Random vs Auditory Non-Random':
        ('Auditory Random', 'Auditory Non-Random'),
    'Visual Random vs Visual Non-Random':
        ('Visual Random', 'Visual Non-Random'),
}
# If True, a vertex receives a regime label only when its dominant condition
# (the one with the larger |z|) is significant vs rest at its own two-sided
# whole-brain FDR, so vertices hovering at rest are not labelled. If False,
# classify by the raw sign of each condition's group mean.
REGIME_GATE_SIGNIF = True
# Crossover is now defined directionally and unambiguously inside regime_rgba:
# a vertex is 'crossover' iff the ACTIVE condition is significantly above
# baseline AND the PASSIVE condition significantly below it (opposite anchors).
# A vertex significant in only one condition is folded into activation or
# deactivation by that condition. (The old REGIME_CROSS_REQUIRE_BOTH_SIGNIF
# toggle is gone: with the active/passive framing the strict definition is the
# only sensible one.) The strict crossover is empty for this dataset.
# Sidedness of the difference (Random - Non-Random) on the regime map, parsed
# by parse_sides into an FDR test (sets the threshold) and a displayed tail:
#   'greater' / 'less'              one-sided +/- FDR, right/left tail
#   'two-sided:right' (default)     two-sided z* (e.g. 3.343), Random>Non-Random
#   'two-sided:left'                two-sided z*, Non-Random>Random
#   'two-sided:both'                two-sided z*, both tails
# 'two-sided:right' is what matches Table E.2 / the supplement (two-sided z*,
# Random>Non-Random tail), so the colorbar magnitude is unambiguously Z (R>NR)
# and the hue is purely the rest-referenced regime.
REGIME_DIFF_SIDES = 'two-sided:right'
# Per-regime intensity colormaps (low |z| -> high |z|).
REGIME_WARM = ('#E1261C', '#FF8C00', '#FFE100')      # activation
REGIME_COOL = ('#1A6FE0', '#22C2E8', '#9BE8FF')      # deactivation
REGIME_CROSS = ('#DB2777', '#F9A8D4')                # crossover (pink, dark->light)
# Undetermined regime: R>NR significant but NEITHER condition significant vs
# Rest, so no activation/deactivation/crossover label is defensible. Muted
# olive (low saturation, but not grey, so it does not dissolve into the grey
# flat underlay), dark->light.
REGIME_UNDETERMINED = ('#7B7239', '#C6BC82')
# If True, drop the undetermined regime from the flatmap (those vertices stay
# transparent). If False, show it as its own colour/colorbar. Set True here:
# the undetermined band is a transition gradient between activation and
# deactivation, not a distinct regime, so it is filtered from the figure; the
# quantitative fraction is kept for the text via regime_exclusion_stats / the
# printed counts (set this False to inspect it on the map if a reviewer asks).
REGIME_FILTER_UNDETERMINED = False
# Minimum vertices per hemisphere for a regime to be drawn and to receive a
# colorbar. Regimes below this are sub-visible specks, so showing a full
# colorbar for them is misleading; dropping them keeps the legend matched to
# what is actually on the map. The strict crossover is a handful of vertices on
# the surface (zero in the volume), so this removes its empty-looking bar.
# Check the printed crossover count and raise this just above it if needed; set
# 0 to disable.
REGIME_MIN_VERTICES = 10

# ========================= PARAMETERS ================================

# Parent directories
if os.path.isdir('/home/analu/diedrichsen_data/data'):
    base_dir = '/home/analu/diedrichsen_data/data'
else:
    base_dir = '/cifs/diedrichsen/data'

music = os.path.join(base_dir, 'Cerebellum', 'music-sdtb')
derivatives_folder = os.path.join(music, 'derivatives')
wb_gmask = os.path.join(derivatives_folder, 'group', 'anat',
                        'group_mask_noskull.nii')

rois_pardir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'roi_analyses_rwls_hrf128_wb_puncorr_unsmoothed',
    'bothmod_allmain_tasks',
    'main_tasks'
)

# ---------------- ROI overlap (cortex) inputs ------------------------
# Used only when running:  python volume_to_surface.py --iroi
#
# This simplified implementation assumes the ROI overlap masks are 
# already in fs_LR32k surface space (one 1D value per vertex), 
# stored as NIfTI files per hemisphere (lh/rh).
#
# All possible levels: 
# ['i', 'i9a', 'i8a', 'i7a', 'i6a', 
#  'a', 
#  'a4g', 'a3g', 'a2g', 'a1g', 'g']
IROI_LEVELS = ['i']

# All ROIs: 8 ROIs
region_names = ['motor_area', 'motor_area', 'motor_area', 'motor_area',
                'auditory_cortex',
                'visual_cortex']

atlas_names = ['hmat', 'hmat', 'hmat', 'hmat',
               'hos',
               'hos']

roi_names = ['pmd', 'pmv', 'sma', 'presma',
             'auditory_cortex',
             'visual_cortex']

ROI_LABELS = {
    "dstr": "Dorsal Striatum",
    "cereb": "Cerebellum",
    "pmv": "PMV",
    "pmd": "PMD",
    "presma": "PreSMA",
    "sma": "SMA",
    "auditory_cortex": "Auditory Cortex",
    "visual_cortex": "Visual Cortex",
}

# ###################### fs_LR32k Meshes ##############################
fslr32k_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'fslr32k_meshes')
lh_veryinflated = os.path.join(fslr32k_folder, 'templates',
                               'tpl-fs32k_hemi-L_veryinflated.surf.gii')
rh_veryinflated = os.path.join(fslr32k_folder, 'templates',
                               'tpl-fs32k_hemi-R_veryinflated.surf.gii')
lh_tpl_pial = os.path.join(fslr32k_folder, 'templates', 
                           'tpl-fs32k_hemi-L_pial.surf.gii')
rh_tpl_pial = os.path.join(fslr32k_folder, 'templates', 
                           'tpl-fs32k_hemi-R_pial.surf.gii')
lh_tpl_white = os.path.join(fslr32k_folder, 'templates', 
                            'tpl-fs32k_hemi-L_white.surf.gii')
rh_tpl_white = os.path.join(fslr32k_folder, 'templates', 
                            'tpl-fs32k_hemi-R_white.surf.gii')
sulc_folder = os.path.join(fslr32k_folder, 'sulc')
lr_sulc_path = os.path.join(sulc_folder,
                            'fs_LR.32k.LR.sulc.dscalar.nii')
lh_sulc_path = os.path.join(sulc_folder,
                            'fs_LR.32k.L.sulc.dscalar.gii')
rh_sulc_path = os.path.join(sulc_folder,
                            'fs_LR.32k.R.sulc.dscalar.gii')
lh_borders_path = os.path.join(fslr32k_folder, 'borders',
                               'fs_LR.32k.L.border.label.gii')
rh_borders_path = os.path.join(fslr32k_folder, 'borders',
                               'fs_LR.32k.R.border.label.gii')

mask_suffix = '1'
lh_medial_wall_mask_path = os.path.join(
    fslr32k_folder, 'medialwall_masks',
    'fs_LR.32k.L.medialwall.mask' + mask_suffix + '.gii')
rh_medial_wall_mask_path = os.path.join(
    fslr32k_folder, 'medialwall_masks',
    'fs_LR.32k.R.medialwall.mask' + mask_suffix + '.gii')

# #####################################################################

# Tasks definitions
tasks = {'prod': 'Production', 
         'percep': 'Perception', 
         'ntfd': 'NTFD',
         'rand_ntfd': 'NTFD Random',
         'allmain_tasks': 'All Main Tasks'
}
task_id = {v: k for k, v in tasks.items()}.get(task_tag)

# Contrast dictionary (id -> name)
all_contrasts = build_contrasts(task_id)

# Output folders
surf_folder = os.path.join(surfparametric_folder, task_id, 
                           'surface_files')
contrasts_folder = os.path.join(surfparametric_folder, task_id, 
                                'surface_images')

# Output folder for ROI overlap (cortex) flatmaps
irois_parfolder = os.path.join(surfparametric_folder, task_id)

# Contrasts definitions
contrast_id = {v: k for k, v in all_contrasts.items()}.get(contrast_name)
cname = contrast_name.replace(' vs ', '_vs_').replace(' ', '-')

# Check if a second contrast is provided
if contrast_name2:
    contrast_id2 = \
        {v: k for k, v in all_contrasts.items()}.get(contrast_name2)
    cname2 = contrast_name2.replace(' vs ', '_vs_').replace(' ', '-')

# %%
# ============================ RUN ====================================

if __name__ == '__main__':

    # ------------------ ROI overlap (cortex) ------------------
    # Run independently of contrasts (flatmaps only).
    if '--irois' in sys.argv:

        vmin = 1 / len(SUBJECTS)
        vmax = 1.0

        iroi_cmaps = ['Blues_r', 'Oranges_r', 'Purples_r', 'Reds_r',
                      'PuBuGn_r', 'PuRd_r']

        IROI_SELECTED = ['pmv', 'pmd', 'sma', 'presma', 'auditory_cortex', 
                         'visual_cortex']

        # Create directory to save outputs if does not exist
        irois_imgs_folder = os.path.join(irois_parfolder, 
                                         'surface_irois_images')
        os.makedirs(irois_imgs_folder, exist_ok=True)

        for lvl in IROI_LEVELS:

            print(f"[iROI] Individualization level '{lvl}'...")

            all_lh = []
            all_rh = []
            all_labels = []
            all_rois = []
            all_cmaps = []

            for idx, (region, atlas, roi) in enumerate(
                zip(region_names, atlas_names, roi_names)
            ):
                if roi not in IROI_SELECTED:
                    continue

                if lvl == 'g':
                    pattern = os.path.join(
                        rois_pardir,
                        region,
                        atlas,
                        roi,
                        'group_roi_masks',
                        f'g_msdtb_{atlas}_{roi}_' + '{hemi}_mask.nii.gz'
                    )
                    lh_path = pattern.format(hemi='lh')
                    rh_path = pattern.format(hemi='rh')
                else:
                    pattern = os.path.join(
                        rois_pardir,
                        region,
                        atlas,
                        roi,
                        'overlaid_masks',
                        f'{lvl}_{roi}_' + '{hemi}_mask.nii.gz'
                    )
                    lh_path = pattern.format(hemi='lh')
                    rh_path = pattern.format(hemi='rh')

                print(f"[iROI]   ROI '{roi}'...")

                lh_arr, rh_arr = roi_to_surf(
                    lh_path, rh_path,
                    lh_tpl_pial, rh_tpl_pial, lh_tpl_white, rh_tpl_white,
                    irois_parfolder, roi, individualization=lvl
                )

                # Discard values below the minimum observable fraction.
                lh_arr = np.asarray(lh_arr, float)
                rh_arr = np.asarray(rh_arr, float)
                lh_arr[lh_arr < vmin] = 0
                rh_arr[rh_arr < vmin] = 0

                # Collect for combined plot
                all_lh.append(lh_arr)
                all_rh.append(rh_arr)
                all_labels.append(ROI_LABELS.get(roi, roi))

                # Individual ROI plot
                cticks = np.linspace(vmin, vmax, 5)
                cmap_used = iroi_cmaps[idx % len(iroi_cmaps)]
                plot_flatmap(
                    stats=[lh_arr, rh_arr],
                    threshold=vmin,
                    task_key=task_id,
                    contrast_tag=f"{lvl}_{roi}",
                    output_dir=irois_imgs_folder,
                    hemi=['L', 'R'],
                    colormap=cmap_used,
                    vmax=vmax,
                    cbar_title='Fraction of Participants',
                    n_ticks=5,
                    tick_decimals=2,
                    show_borders=SHOW_SULCI_BORDERS,
                )
                all_rois.append(roi)
                all_cmaps.append(cmap_used)
            # Desired top-to-bottom order of iROI colorbars
            IROI_ORDER = list(IROI_SELECTED)

            roi_to_lh = {r: a for r, a in zip(all_rois, all_lh)}
            roi_to_rh = {r: a for r, a in zip(all_rois, all_rh)}
            roi_to_label = {r: a for r, a in zip(all_rois, all_labels)}
            roi_to_cmap = {r: a for r, a in zip(all_rois, all_cmaps)}

            missing = [r for r in IROI_ORDER if r not in roi_to_lh]
            if missing:
                raise ValueError(
                    "Requested iROIs missing from loaded maps: "
                    f"{missing}. Available: {sorted(list(roi_to_lh.keys()))}"
                )

            all_lh = [roi_to_lh[r] for r in IROI_ORDER]
            all_rh = [roi_to_rh[r] for r in IROI_ORDER]
            all_labels = [roi_to_label[r] for r in IROI_ORDER]
            all_cmaps = [roi_to_cmap[r] for r in IROI_ORDER]

            # Combined plot: all iROIs in the same flatmap (L/R)
            combined_stats = [[lh, rh] for lh, rh in zip(all_lh, all_rh)]
            plot_multirois_flatmap(
                stats=combined_stats,
                threshold=vmin,
                task_key=task_id,
                contrast_tag=f"{lvl}_all-irois",
                output_dir=irois_imgs_folder,
                hemi=['L', 'R'],
                colormaps=all_cmaps[:len(combined_stats)],
                labels=all_labels,
                vmax=vmax,
                cbar_ticks=np.linspace(vmin, vmax, 5),
                tick_decimals=2,
                show_borders=SHOW_SULCI_BORDERS,
            )

        sys.exit(0)

    # ----------------- Combined regime map (--regime) -----------------
    # Re-colour a signed difference contrast (contrast_name) by the
    # rest-referenced regime of its two conditions. Independent of --contour:
    # pass --contour as well (with contrast_name2) to add the network outline.
    if '--regime' in sys.argv:
        if contrast_name not in REGIME_COMPONENTS:
            raise ValueError(
                "--regime needs a difference contrast with defined "
                f"components; got {contrast_name!r}. "
                f"Known: {list(REGIME_COMPONENTS)}")
        rand_name, nonrand_name = REGIME_COMPONENTS[contrast_name]
        _name2id = {v: k for k, v in all_contrasts.items()}
        diff_id = _name2id.get(contrast_name)
        rand_id = _name2id.get(rand_name)
        nonrand_id = _name2id.get(nonrand_name)
        if None in (diff_id, rand_id, nonrand_id):
            raise ValueError(
                "Could not resolve regime contrast ids for "
                f"{contrast_name!r} / {rand_name!r} / {nonrand_name!r}.")
        cname = contrast_name.replace(' vs ', '_vs_').replace(' ', '-')

        def _group_lr(cid, cnm):
            """Group surface z-map (signed) -> medial-wall-masked L, R."""
            zv = group_surf(
                surf_folder, SUBJECTS, task_id, cid,
                cnm.replace(' vs ', '_vs_').replace(' ', '-'),
                surfspace='fslr32k')
            zl = mask_cortical_activation(
                np.split(zv, 2, axis=0)[0], lh_medial_wall_mask_path)
            zr = mask_cortical_activation(
                np.split(zv, 2, axis=0)[1], rh_medial_wall_mask_path)
            return zl, zr

        dL, dR = _group_lr(diff_id, contrast_name)
        rL, rR = _group_lr(rand_id, rand_name)
        nL, nR = _group_lr(nonrand_id, nonrand_name)

        # Difference threshold + vmax. parse_sides splits REGIME_DIFF_SIDES into
        # the FDR *test* (which sets the threshold) and the displayed *tail*
        # (which side is classified). The default 'two-sided:right' uses the
        # two-sided z* (e.g. 3.343, matching Table E.2 / the supplement) and
        # shows only the Random > Non-Random tail, reproducing the reported
        # regime counts and overlaps. A one-sided test would instead lower the
        # threshold (~2.956) and desync the figure from the volume tables.
        regime_test, regime_tail = parse_sides(REGIME_DIFF_SIDES)
        if regime_test == 'two-sided':
            thr_diff, vmax_diff = whole_brain_thresholds_signed(
                derivatives_folder, SUBJECTS, task_id, diff_id, wb_gmask)
        else:
            thr_diff, vmax_diff = whole_brain_thresholds(
                derivatives_folder, SUBJECTS, task_id, diff_id, wb_gmask)

        # Condition |z| FDR thresholds vs rest. Always computed: the gate uses
        # them when REGIME_GATE_SIGNIF, and thr_rand is also used by the
        # Random-vs-Rest exclusion diagnostic below.
        thr_rand, _ = whole_brain_thresholds_signed(
            derivatives_folder, SUBJECTS, task_id, rand_id, wb_gmask)
        thr_nonrand, _ = whole_brain_thresholds_signed(
            derivatives_folder, SUBJECTS, task_id, nonrand_id, wb_gmask)

        # Optional network outline (independent toggle: --contour).
        contour_kw = {}
        contour_tag = ''
        if '--contour' in sys.argv and contrast_name2:
            contour_task_id = (
                {v: k for k, v in tasks.items()}.get(contour_task_tag)
                if contour_task_tag else task_id)
            if contour_task_id is None:
                raise ValueError(
                    f"Unknown contour_task_tag: {contour_task_tag}")
            contour_contrasts = build_contrasts(contour_task_id)
            contrast_id2 = \
                {v: k for k, v in contour_contrasts.items()}.get(contrast_name2)
            if contrast_id2 is None:
                raise ValueError(
                    f"Unknown contour contrast '{contrast_name2}' "
                    f"for task '{contour_task_tag or task_tag}'")
            cname2 = contrast_name2.replace(' vs ', '_vs_').replace(' ', '-')
            contour_surf_folder = os.path.join(
                surfparametric_folder, contour_task_id, 'surface_files')
            z_values2 = group_surf(
                contour_surf_folder, SUBJECTS, contour_task_id, contrast_id2,
                cname2, surfspace='fslr32k')
            cL2 = mask_cortical_activation(
                np.split(z_values2, 2, axis=0)[0], lh_medial_wall_mask_path)
            cR2 = mask_cortical_activation(
                np.split(z_values2, 2, axis=0)[1], rh_medial_wall_mask_path)
            signed_contrasts = (
                'Random vs Non-Random',
                'Auditory Random vs Auditory Non-Random',
                'Visual Random vs Visual Non-Random',
            )
            # parse_sides splits contour_sides into the FDR test (threshold) and
            # the drawn tail: e.g. 'two-sided:both' for Fig 6a (outline of both
            # activations and deactivations) or 'two-sided:right' for Fig 7
            # (same z*, only the activation tail outlined).
            contour_test, contour_tail = parse_sides(contour_sides)
            if contour_threshold_override is not None:
                thr2 = float(contour_threshold_override)
            elif contour_test == 'two-sided':
                thr2, _ = whole_brain_thresholds_signed(
                    derivatives_folder, SUBJECTS, contour_task_id,
                    contrast_id2, wb_gmask)
            else:
                thr2, _ = whole_brain_thresholds(
                    derivatives_folder, SUBJECTS, contour_task_id,
                    contrast_id2, wb_gmask)
            contour_tag = '_with_' + cname2
            contour_kw = dict(
                contour_stat=[cL2, cR2], contour_threshold=thr2,
                contour_tail=contour_tail)

        # Diagnostic: fraction of the Random - Non-Random significant map that
        # is NOT backed by a significant Random vs Rest activation (would be
        # excluded if Random significance were required). Total denominator =
        # cortical (non-medial-wall) vertices from the medial-wall masks.
        _cortexL = nib.load(
            lh_medial_wall_mask_path).darrays[0].data.astype(bool)
        _cortexR = nib.load(
            rh_medial_wall_mask_path).darrays[0].data.astype(bool)
        _excl = regime_exclusion_stats(
            np.concatenate([dL, dR]),
            np.concatenate([rL, rR]),
            thr_diff, thr_rand,
            display_tail=regime_tail,
            cortex_mask=np.concatenate([_cortexL, _cortexR]))
        print(f"[regime] Random-vs-Rest exclusion within {contrast_name!r}:")
        print(f"  total cortical vertices          : "
              f"{_excl['n_total_cortex']}")
        print(f"  Random-vs-Non-Random significant : {_excl['n_contrast']} "
              f"({_excl['pct_contrast_of_total']:.2f}% of cortex)")
        print(f"  of those, Random vs Rest n.s.    : "
              f"{_excl['n_excluded_random_ns']}")
        print(f"    -> {_excl['pct_within_contrast']:.2f}% within contrast, "
              f"{_excl['pct_of_total_cortex']:.2f}% of cortex")

        regime_dir = os.path.join(
            contrasts_folder, 'regime', (cname + contour_tag).lower())
        os.makedirs(regime_dir, exist_ok=True)
        plot_regime_flatmap(
            diff_stats=[dL, dR],
            cond_rand_stats=[rL, rR],
            cond_nonrand_stats=[nL, nR],
            thr_diff=thr_diff, vmax=vmax_diff,
            task_key=task_id, contrast_tag=cname + contour_tag,
            output_dir=regime_dir,
            thr_rand=thr_rand, thr_nonrand=thr_nonrand,
            gate=REGIME_GATE_SIGNIF,
            show_undetermined=not REGIME_FILTER_UNDETERMINED,
            min_vertices=REGIME_MIN_VERTICES,
            display_tail=regime_tail,
            hemi=['L', 'R'],
            warm=REGIME_WARM, cool=REGIME_COOL, neutral=REGIME_CROSS,
            undetermined=REGIME_UNDETERMINED,
            show_borders=SHOW_SULCI_BORDERS,
            **contour_kw,
        )
        sys.exit(0)

    # -------- detect batch mode without reordering inputs ---------------
    _batch = None
    if isinstance(contrast_name, (list, tuple, np.ndarray)):
        _batch = list(contrast_name)
    elif isinstance(contrast_name, str) and \
            contrast_name.strip().upper() == 'ALL':
        _batch = list(all_contrasts.values())

    # draw the second contrast as a contour instead of an RGB overlay
    _contour = '--contour' in sys.argv

    # =================== BATCH: multiple single runs =================
    if _batch is not None and not contrast_name2:
        for _cname in _batch:
            _cid = {v: k for k, v in all_contrasts.items()}.get(_cname)
            if _cid is None:
                print(f"[skip] Unknown contrast: {_cname}")
                continue

            _tag = _cname.replace(' vs ', '_vs_').replace(' ', '-')
            _cdir = os.path.join(surf_folder, f"{_cid}_{_tag.lower()}")
            os.makedirs(_cdir, exist_ok=True)

            # ---- compute individual surfaces (gifti + cifti) --------
            individual_surf(derivatives_folder, SUBJECTS, task_id, 
                            all_contrasts, _cid, surf_folder, 
                            surfspace='fslr32k', save='gifti')
            individual_surf(derivatives_folder, SUBJECTS, task_id, 
                            all_contrasts, _cid, surf_folder, 
                            surfspace='fslr32k', save='cifti')

            # ---- compute group CIFTI → split → mask -----------------
            z_values = group_surf(surf_folder, SUBJECTS, task_id, _cid, _tag,
                                  surfspace='fslr32k')
            zvals_lh = np.split(z_values, 2, axis=0)[0]
            zvals_rh = np.split(z_values, 2, axis=0)[1]

            zvals_lh_masked = mask_cortical_activation(
                zvals_lh, lh_medial_wall_mask_path)
            zvals_rh_masked = mask_cortical_activation(
                zvals_rh, rh_medial_wall_mask_path)

            # ---- save group z-maps as GIFTI (L/R) -------------------
            for zm, structure, hemi in zip(
                [zvals_lh_masked, zvals_rh_masked],
                ['CortexLeft', 'CortexRight'],
                ['lh', 'rh']
            ):
                gifti_img = nt.gifti.make_func_gifti(
                    zm, anatomical_struct=structure, column_names=[_tag])
                nib.save(
                    gifti_img,
                    os.path.join(
                        _cdir,
                        'group_'
                        + task_id.replace('_', '-')
                        + '_'
                        + _tag.lower()
                        + '_'
                        + 'fslr32k.' + hemi[0].capitalize() + '.func.gii',
                    )
                )

            # ---- thresholds (volume) + plot flatmaps ----------------
            thresh, v_max = whole_brain_thresholds(
                derivatives_folder, SUBJECTS, task_id, _cid, wb_gmask
            )
            out_dir = os.path.join(contrasts_folder, f"{_cid}_{_tag.lower()}")
            os.makedirs(out_dir, exist_ok=True)
            plot_flatmap(
                [zvals_lh_masked, zvals_rh_masked],
                thresh, task_id, _tag, out_dir,
                hemi=['L', 'R'], colormap='viridis', vmax=v_max,
                show_borders=SHOW_SULCI_BORDERS,
            )

        sys.exit(0)  # do not fall through to single/overlay

    # ======================== SINGLE CONTRAST ========================
    if not contrast_name2:
        contrast_id = \
            {v: k for k, v in all_contrasts.items()}.get(contrast_name)
        if contrast_id is None:
            raise ValueError(f"Unknown contrast_name: {contrast_name}")
        cname = contrast_name.replace(' vs ', '_vs_').replace(' ', '-')
        cdir = os.path.join(surf_folder, f"{contrast_id}_{cname.lower()}")
        os.makedirs(cdir, exist_ok=True)

        # ---- compute individual (gifti + cifti) ---------------------
        individual_surf(derivatives_folder, SUBJECTS, task_id, all_contrasts, 
                        contrast_id, surf_folder, 
                        surfspace='fslr32k', save='gifti')
        individual_surf(derivatives_folder, SUBJECTS, task_id, all_contrasts, 
                        contrast_id, surf_folder, 
                        surfspace='fslr32k', save='cifti')

        # ---- compute group → split → mask ---------------------------
        z_values = group_surf(surf_folder, SUBJECTS, task_id, contrast_id, 
                              cname, surfspace='fslr32k')
        zvals_lh = np.split(z_values, 2, axis=0)[0]
        zvals_rh = np.split(z_values, 2, axis=0)[1]
        zvals_lh_masked = mask_cortical_activation(
            zvals_lh, lh_medial_wall_mask_path)
        zvals_rh_masked = mask_cortical_activation(
            zvals_rh, rh_medial_wall_mask_path)

        # ---- save group z-maps (L/R) --------------------------------
        for zm, structure, hemi in zip(
            [zvals_lh_masked, zvals_rh_masked],
            ['CortexLeft', 'CortexRight'],
            ['lh', 'rh']
        ):
            gifti_img = nt.gifti.make_func_gifti(
                zm, anatomical_struct=structure, column_names=[cname])
            nib.save(
                gifti_img,
                os.path.join(
                    cdir,
                    'group_'
                    + task_id.replace('_', '-')
                    + '_'
                    + cname.lower()
                    + '_'
                    + 'fslr32k.' + hemi[0].capitalize() + '.func.gii',
                )
            )

        # ---- thresholds (volume) + plot flatmaps --------------------
        thresh, v_max = whole_brain_thresholds(
            derivatives_folder, SUBJECTS, task_id, contrast_id, wb_gmask
        )
        surfplots_folder = os.path.join(
            contrasts_folder, f"{contrast_id}_{cname.lower()}"
        )
        os.makedirs(surfplots_folder, exist_ok=True)
        signed_contrasts = (
            'Random vs Non-Random',
            'Auditory Random vs Auditory Non-Random',
            'Visual Random vs Visual Non-Random',
        )
        if resolve_signed(contrast_sides, contrast_name, signed_contrasts):
            thr_s, vmax_s = whole_brain_thresholds_signed(
                derivatives_folder, SUBJECTS, task_id, contrast_id, wb_gmask)
            plot_flatmap(
                [zvals_lh_masked, zvals_rh_masked],
                thr_s, task_id, cname, surfplots_folder,
                hemi=['L', 'R'], vmax=vmax_s, signed=True,
                show_borders=SHOW_SULCI_BORDERS,
                cbar_title='Z  (Random \u2212 Non-Random)'
            )
        else:
            plot_flatmap(
                [zvals_lh_masked, zvals_rh_masked],
                thresh, task_id, cname, surfplots_folder,
                hemi=['L', 'R'], colormap='viridis', vmax=v_max,
                show_colorbar=False,
                show_borders=SHOW_SULCI_BORDERS,
            )

    # ====================== TWO-CONTRAST OVERLAY =====================
    else:
        contrast_id = \
            {v: k for k, v in all_contrasts.items()}.get(contrast_name)
        if contrast_id is None:
            raise ValueError(f"Unknown contrast: {contrast_name}")
        cname = contrast_name.replace(' vs ', '_vs_').replace(' ', '-')

        # ---- compute + group + mask for contrast 1 ------------------
        cdir1 = os.path.join(surf_folder, f"{contrast_id}_{cname.lower()}")
        os.makedirs(cdir1, exist_ok=True)
        # individual_surf(derivatives_folder, SUBJECTS, task_id, all_contrasts,
        #                 contrast_id, surf_folder, 
        #                 surfspace='fslr32k', save='gifti')
        # individual_surf(derivatives_folder, SUBJECTS, task_id, all_contrasts,
        #                 contrast_id, surf_folder, 
        #                 surfspace='fslr32k', save='cifti')
        z_values1 = group_surf(surf_folder, SUBJECTS, task_id, contrast_id,
                               cname, surfspace='fslr32k')
        zL1 = mask_cortical_activation(
            np.split(z_values1, 2, axis=0)[0], lh_medial_wall_mask_path)
        zR1 = mask_cortical_activation(
            np.split(z_values1, 2, axis=0)[1], rh_medial_wall_mask_path)
        for zm, structure, hemi in zip(
            [zL1, zR1],
            ['CortexLeft', 'CortexRight'],
            ['lh', 'rh']):
            gifti_img = nt.gifti.make_func_gifti(
                zm, anatomical_struct=structure, column_names=[cname])
            nib.save(
                gifti_img,
                os.path.join(
                    cdir1,
                    'group_'
                    + task_id.replace('_', '-')
                    + '_'
                    + cname.lower()
                    + '_'
                    + 'fslr32k.' + hemi[0].capitalize() + '.func.gii',
                )
            )

        # ---- thresholds + overlay plot ------------------------------
        if _contour:
            # Contrast 2 (the outline) may come from a DIFFERENT task than
            # the filled contrast 1. Resolve its task, dictionary, surface
            # folder and threshold independently.
            contour_task_id = (
                {v: k for k, v in tasks.items()}.get(contour_task_tag)
                if contour_task_tag else task_id
            )
            if contour_task_id is None:
                raise ValueError(f"Unknown contour_task_tag: {contour_task_tag}")
            contour_contrasts = build_contrasts(contour_task_id)
            contrast_id2 = \
                {v: k for k, v in contour_contrasts.items()}.get(contrast_name2)
            if contrast_id2 is None:
                raise ValueError(
                    f"Unknown contour contrast '{contrast_name2}' "
                    f"for task '{contour_task_tag or task_tag}'")
            cname2 = contrast_name2.replace(' vs ', '_vs_').replace(' ', '-')
            contour_surf_folder = os.path.join(
                surfparametric_folder, contour_task_id, 'surface_files')

            # group surface z-map of the contour contrast (read from its
            # own task's surface files; no re-save into another task tree)
            z_values2 = group_surf(
                contour_surf_folder, SUBJECTS, contour_task_id, contrast_id2,
                cname2, surfspace='fslr32k')
            zL2 = mask_cortical_activation(
                np.split(z_values2, 2, axis=0)[0], lh_medial_wall_mask_path)
            zR2 = mask_cortical_activation(
                np.split(z_values2, 2, axis=0)[1], rh_medial_wall_mask_path)

            # contrast 1 fill threshold (signed for the Random-Non-Random map)
            signed_contrasts = (
                'Random vs Non-Random',
                'Auditory Random vs Auditory Non-Random',
                'Visual Random vs Visual Non-Random',
            )
            is_signed = resolve_signed(
                contrast_sides, contrast_name, signed_contrasts)
            if is_signed:
                thr1, v1 = whole_brain_thresholds_signed(
                    derivatives_folder, SUBJECTS, task_id, contrast_id,
                    wb_gmask)
            else:
                thr1, v1 = whole_brain_thresholds(
                    derivatives_folder, SUBJECTS, task_id, contrast_id,
                    wb_gmask)

            # contour sidedness: two-sided traces activations AND
            # deactivations (|z| FDR); one-sided traces activations only.
            contour_twosided = resolve_signed(
                contour_sides, contrast_name2, signed_contrasts)

            # contour threshold: explicit override, else whole-brain FDR of
            # the contour contrast computed in ITS OWN task (two-sided FDR on
            # |z| when the outline is two-sided, one-sided FDR otherwise).
            if contour_threshold_override is not None:
                thr2 = float(contour_threshold_override)
            elif contour_twosided:
                thr2, _ = whole_brain_thresholds_signed(
                    derivatives_folder, SUBJECTS, contour_task_id,
                    contrast_id2, wb_gmask)
            else:
                thr2, _ = whole_brain_thresholds(
                    derivatives_folder, SUBJECTS, contour_task_id,
                    contrast_id2, wb_gmask)

            contourplots_folder = os.path.join(
                contrasts_folder, 'contour',
                cname.lower() + '_with_' + cname2.lower()
            )
            os.makedirs(contourplots_folder, exist_ok=True)
            plot_flatmap(
                stats=[zL1, zR1],
                threshold=thr1,
                task_key=task_id,
                contrast_tag=cname + '_with_' + cname2,
                output_dir=contourplots_folder,
                hemi=['L', 'R'],
                colormap='viridis',
                vmax=v1,
                signed=is_signed,
                cbar_title=('Z-values (Random \u2212 Non-Random)' if is_signed
                            else 'Z-values'),
                contour_stat=[zL2, zR2],
                contour_threshold=thr2,
                contour_color='k',
                contour_linewidth=1.0,
                contour_positive_only=resolve_contour_display(
                    contour_display, contour_twosided),
                show_borders=SHOW_SULCI_BORDERS,
            )
        else:
            contrast_id2 = \
                {v: k for k, v in all_contrasts.items()}.get(contrast_name2)
            if contrast_id2 is None:
                raise ValueError(f"Unknown contrast: {contrast_name2}")
            cname2 = contrast_name2.replace(' vs ', '_vs_').replace(' ', '-')

            # ---- compute + group + mask for contrast 2 --------------
            cdir2 = os.path.join(
                surf_folder, f"{contrast_id2}_{cname2.lower()}")
            os.makedirs(cdir2, exist_ok=True)
            z_values2 = group_surf(
                surf_folder, SUBJECTS, task_id, contrast_id2, cname2,
                surfspace='fslr32k'
            )
            zL2 = mask_cortical_activation(
                np.split(z_values2, 2, axis=0)[0], lh_medial_wall_mask_path)
            zR2 = mask_cortical_activation(
                np.split(z_values2, 2, axis=0)[1], rh_medial_wall_mask_path)
            for zm, structure, hemi in zip([zL2, zR2],
                                           ['CortexLeft', 'CortexRight'],
                                           ['lh', 'rh']):
                gifti_img = nt.gifti.make_func_gifti(
                    zm, anatomical_struct=structure, column_names=[cname2])
                nib.save(
                    gifti_img,
                    os.path.join(
                        cdir2,
                        'group_'
                        + task_id.replace('_', '-')
                        + '_'
                        + cname2.lower()
                        + '_'
                        + 'fslr32k.' + hemi[0].capitalize() + '.func.gii',
                    )
                )

            thr1, v1 = whole_brain_thresholds(
                derivatives_folder, SUBJECTS, task_id, contrast_id, wb_gmask
            )
            thr2, v2 = whole_brain_thresholds(
                derivatives_folder, SUBJECTS, task_id, contrast_id2, wb_gmask
            )
            rgbaplots_folder = os.path.join(
                contrasts_folder, 'rgba',
                cname.lower() + '_and_' + cname2.lower()
            )
            os.makedirs(rgbaplots_folder, exist_ok=True)
            thr = max(thr1, thr2)
            vmax = max(v1, v2)
            plot_flatmap(
                stats=[[zL1, zR1], [zL2, zR2]],
                threshold=[thr, thr],
                task_key=task_id,
                contrast_tag=cname + '_and_' + cname2,
                output_dir=rgbaplots_folder,
                hemi=['L', 'R'],
                colors=["#FFF200", "#F42DFF"], # colors=['#D41159', '#1A85FF']
                vmax=[vmax, vmax], # [vmax, vmax]
                show_borders=SHOW_SULCI_BORDERS,
            )