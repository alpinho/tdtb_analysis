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

Author: Ana Luisa Pinho
Email: agrilopi@uwo.ca

Creation: 24th of February 2025
Last Update: February 2026

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
                borders=borders[h],
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
                borders=borders[h],
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
                f"Z-Values ({label1})"),
                (sm2, [.3825, .08, .25, .04], thr2, m2_1, m2_2, v2,
                f"Z-Values ({label2})"),
                (sm3, [.715, .08, .25, .04], min_ol, m3_1, m3_2, 1.0, 
                "Co-activation")
            ]

            # Do colorbars
            fig = plt.gcf()
            for sm, rect, lo, m1, m2, hi, lbl in bars:
                cax = fig.add_axes(rect)
                cb = fig.colorbar(
                    sm, cax=cax, orientation='horizontal',
                    ticks=[lo, m1, m2, hi]
                )
                cb.set_label(lbl, fontsize=9, labelpad=5)
                cb.ax.set_xticklabels([f"{lo:.2f}", f"{m1:.2f}", f"{m2:.2f}", 
                                       f"{hi:.2f}"])
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


def plot_multirois_flatmap(
        stats,
        threshold,
        task_key,
        contrast_tag,
        output_dir,
        hemi=['L', 'R'],
        colormaps=None,
        labels=None,
        vmax=10,
        cbar_title='Fraction of Participants',
        cbar_ticks=None,
        tick_decimals=2,
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

        roi_vmax = []
        for i in range(n_maps):
            has_vals = np.any(np.isfinite(vals[i]))
            vmax_i = np.nanmax(vals[i]) if has_vals else thr
            if not np.isfinite(vmax_i) or vmax_i <= thr:
                vmax_i = thr
            roi_vmax.append(float(vmax_i))

        for i, cmap_i in enumerate(colormaps):
            idx = valid & (winner == i)
            if not np.any(idx):
                continue

            denom_i = roi_vmax[i] - thr
            if denom_i <= 0:
                denom_i = 1.0

            norm = np.clip((win_val[idx] - thr) / denom_i, 0.0, 1.0)
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
            borders=borders[h],
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
        "Heschl's Gyrus",
        "Occipital Lobe",
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

    suffix = 'flat_all_irois'
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

    # Leave room on the right for a vertical stack of horizontal colorbars
    plt.subplots_adjust(left=0.0, right=0.77, top=0.97, bottom=0.10)

    n_maps = len(labels)
    x0 = 0.79
    w = 0.20

    # Stack horizontal colorbars vertically without overlap.
    # Define geometry in figure-relative coordinates.
    top = 0.95
    bottom = 0.10
    gap = 0.012
    avail = top - bottom

    # Compute a bar height that fits all maps. If needed, shrink the gap.
    bar_h = (avail - (n_maps - 1) * gap) / max(n_maps, 1)
    if bar_h < 0.01:
        gap = 0.005
        bar_h = (avail - (n_maps - 1) * gap) / max(n_maps, 1)
    bar_h = max(bar_h, 0.01)

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

# Production/Perception/NTFD/NTFD Random/All Tasks
task_tag = 'All Tasks'
 
# To run every contrast:
# contrast_name = 'ALL' and contrast_name2 = None.
# To run a subset sequentially:
# contrast_name = ['Beat', 'Interval', 'Decision']
# contrast_name2 = None.
# To run overlays of pairs:
# contrast_name = 'Beat'
# contrast_name2 = 'Interval' (must be contrast name or list of names)
# For single or overlay, keep contrast_name/contrast_name2 as strings
contrast_name = 'ALL' # ''E.g. 'Beat', 'Interval', 'ALL', etc.
contrast_name2 = None # E.g. 'Interval'

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
                'heschl_gyrus',
                'occipital_lobe']

atlas_names = ['hmat', 'hmat', 'hmat', 'hmat',
               'hos',
               'hos']

roi_names = ['pmd', 'pmv', 'sma', 'presma',
             'heschl',
             'occipital']

ROI_LABELS = {
    "dstr": "Dorsal Striatum",
    "cereb": "Cerebellum",
    "pmv": "PMV",
    "pmd": "PMD",
    "presma": "PreSMA",
    "sma": "SMA",
    "heschl": "Heschl's Gyrus",
    "occipital": "Occipital Lobe",
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
    if '--iroi' in sys.argv:

        vmin = 1 / len(SUBJECTS)
        vmax = 1.0

        iroi_cmaps = ['Blues_r', 'Oranges_r', 'Purples_r', 'Reds_r',
                      'PuBuGn_r', 'PuRd_r']

        IROI_SELECTED = ['pmv', 'pmd', 'sma', 'presma', 'heschl', 'occipital']

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
                    vmin=vmin,
                    task_key=task_id,
                    contrast_tag=f"{lvl}_{roi}",
                    output_dir=irois_imgs_folder,
                    hemi=['L', 'R'],
                    colormap=cmap_used,
                    vmax=vmax,
                    cbar_title='Fraction of Participants',
                    n_ticks=5,
                    tick_decimals=2,
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
                tick_decimals=2
            )

        sys.exit(0)

    # -------- detect batch mode without reordering inputs ---------------
    _batch = None
    if isinstance(contrast_name, (list, tuple, np.ndarray)):
        _batch = list(contrast_name)
    elif isinstance(contrast_name, str) and \
            contrast_name.strip().upper() == 'ALL':
        _batch = list(all_contrasts.values())

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
            # individual_surf(derivatives_folder, SUBJECTS, task_id, 
            #                 all_contrasts, _cid, surf_folder, 
            #                 surfspace='fslr32k', save='gifti')
            # individual_surf(derivatives_folder, SUBJECTS, task_id, 
            #                 all_contrasts, _cid, surf_folder, 
            #                 surfspace='fslr32k', save='cifti')

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
                hemi=['L', 'R'], colormap='viridis', vmax=v_max
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
        # individual_surf(derivatives_folder, SUBJECTS, task_id, all_contrasts, 
        #                 contrast_id, surf_folder, 
        #                 surfspace='fslr32k', save='gifti')
        # individual_surf(derivatives_folder, SUBJECTS, task_id, all_contrasts, 
        #                 contrast_id, surf_folder, 
        #                 surfspace='fslr32k', save='cifti')

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
        plot_flatmap(
            [zvals_lh_masked, zvals_rh_masked],
            thresh, task_id, cname, surfplots_folder,
            hemi=['L', 'R'], colormap='viridis', vmax=v_max,
            show_colorbar=True
        )

    # ====================== TWO-CONTRAST OVERLAY =====================
    else:
        contrast_id = \
            {v: k for k, v in all_contrasts.items()}.get(contrast_name)
        contrast_id2 = \
            {v: k for k, v in all_contrasts.items()}.get(contrast_name2)
        if contrast_id is None or contrast_id2 is None:
            raise ValueError(
                f"Unknown contrasts: {contrast_name}, {contrast_name2}")
        cname = contrast_name.replace(' vs ', '_vs_').replace(' ', '-')
        cname2 = contrast_name2.replace(' vs ', '_vs_').replace(' ', '-')

        # ---- compute + group + mask for contrast 1 ------------------
        cdir1 = os.path.join(surf_folder, f"{contrast_id}_{cname.lower()}")
        os.makedirs(cdir1, exist_ok=True)
        individual_surf(derivatives_folder, SUBJECTS, task_id, all_contrasts,
                        contrast_id, surf_folder, 
                        surfspace='fslr32k', save='gifti')
        individual_surf(derivatives_folder, SUBJECTS, task_id, all_contrasts,
                        contrast_id, surf_folder, 
                        surfspace='fslr32k', save='cifti')
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

        # ---- compute + group + mask for contrast 2 ------------------
        cdir2 = os.path.join(surf_folder, f"{contrast_id2}_{cname2.lower()}")
        os.makedirs(cdir2, exist_ok=True)
        individual_surf(derivatives_folder, SUBJECTS, task_id, all_contrasts,
                        contrast_id2, surf_folder, 
                        surfspace='fslr32k', save='gifti')
        individual_surf(derivatives_folder, SUBJECTS, task_id, all_contrasts, 
                        contrast_id2, surf_folder, 
                        surfspace='fslr32k', save='cifti')
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

        # ---- thresholds + overlay plot ------------------------------
        thr1, v1 = whole_brain_thresholds(
            derivatives_folder, SUBJECTS, task_id, contrast_id, wb_gmask
        )
        thr2, v2 = whole_brain_thresholds(
            derivatives_folder, SUBJECTS, task_id, contrast_id2, wb_gmask
        )
        rgbaplots_folder = os.path.join(
            contrasts_folder, 'rgba', cname.lower() + '_and_' + cname2.lower()
        )
        os.makedirs(rgbaplots_folder, exist_ok=True)
        plot_flatmap(
            stats=[[zL1, zR1], [zL2, zR2]],
            threshold=[thr1, thr2],
            task_key=task_id,
            contrast_tag=cname + '_and_' + cname2,
            output_dir=rgbaplots_folder,
            hemi=['L', 'R'],
            colors=['#D41159', '#1A85FF'],
            vmax=[v1, v2]
        )