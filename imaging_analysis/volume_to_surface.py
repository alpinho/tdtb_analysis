"""
Script to do the volume to surface projection of data from the
 Music-SDTB project

Author: Ana Luisa Pinho
Email: agrilopi@uwo.ca

Creation: 24th of February 2025
Last Update: March 2025

Compatibility: Python 3.10.14, nilearn 0.11.1

Note: The all pipeline of this script only works for surf_files saved as
      cifti in fs_LR32k.
"""

import os

import numpy as np
import pandas as pd
import nibabel as nib
import nitools as nt
import surfAnalysisPy as surf

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

from scipy import stats
from nilearn.image import load_img
from nilearn.surface import load_surf_data, vol_to_surf
from nilearn.maskers import NiftiMasker
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm.thresholding import fdr_threshold
from Functional_Fusion.util import smooth_fs32k_data


# %%
# ========================== FUNCTIONS =================================

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


def individual_surf(derivatives_dir, subjects, task_key, contrast_key,
                    surf_dir, surfspace='fslr32k', save='gifti'):

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
                                        white_left, white_right, SUBJECTS):

        # Map individual functional data from  Nifti to the surface of...
        # ... left and right hemispheres
        emap_img = load_img(emap)
        DL = vol_to_surf(emap_img, surf_mesh=pl, inner_mesh=wl)
        DR = vol_to_surf(emap_img, surf_mesh=pr, inner_mesh=wr)
        print(sb)
        print(DL.shape)
        print(DR.shape)

        # Transform numpy arrays in gifti files
        contrast = all_contrasts[contrast_key].replace(' ', '-')
        GIFTIL = nt.gifti.make_func_gifti(DL, anatomical_struct='CortexLeft',
                                          column_names=[contrast])
        GIFTIR = nt.gifti.make_func_gifti(DR, anatomical_struct='CortexRight',
                                          column_names=[contrast])

        # Create directory to save outputs if does not exist
        if not os.path.exists(surf_dir):
            os.makedirs(surf_dir)

        # Save output
        if save == 'gifti':
            # Save Gifti files
            nib.save(
                GIFTIL,
                os.path.join(
                    surf_dir,
                    'sub-{sb:02d}_'.format(sb=sb)
                    + contrast.lower()
                    + '_'
                    + surfspace
                    + '.hem-L.func.gii',
                ),
            )
            nib.save(
                GIFTIR,
                os.path.join(
                    surf_dir,
                    'sub-{sb:02d}_'.format(sb=sb)
                    + contrast.lower()
                    + '_'
                    + surfspace
                    + '.hem-R.func.gii',
                ),
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
                    surf_dir,
                    f"sub-{sb:02d}_{contrast.lower()}_{surfspace}.dscalar.nii"
                ),
            )


def get_isurf_gifti(surf_dir, subjects, contrast, surfspace='fslr32k'):

    # Paths of individual files per hemisphere
    gifti_left = [
        os.path.join(
            surf_dir,
            f'sub-{sub:02d}_{contrast}_{surfspace}.hem-L.func.gii'
        )
        for sub in subjects
    ]

    gifti_right = [
        os.path.join(
            surf_dir,
            f'sub-{sub:02d}_{contrast}_{surfspace}.hem-R.func.gii'
        )
        for sub in subjects
    ]

    return gifti_left, gifti_right


def get_isurf_cifti(surf_dir, subjects, contrast, surfspace='fslr32k'):

    # Paths of individual files
    cifti_file = [
        os.path.join(
            surf_dir,
            f'sub-{sub:02d}_{contrast}_{surfspace}.dscalar.nii'
        )
        for sub in subjects
    ]

    return cifti_file


def zval_conversion(tval, dof):
    pval = stats.t.sf(tval, dof)
    one_minus_pval = stats.t.cdf(tval, dof)
    zval_sf = stats.norm.isf(pval)
    zval_cdf = stats.norm.ppf(one_minus_pval)
    zval = np.empty(pval.shape)
    use_cdf = zval_sf < 0
    use_sf = np.logical_not(use_cdf)
    zval[np.atleast_1d(use_cdf)] = zval_cdf[use_cdf]
    zval[np.atleast_1d(use_sf)] = zval_sf[use_sf]

    return zval


def group_surf(surf_dir, subjects, contrast_tag, surfspace='fslr32k'):

    contrast = contrast_tag.lower().replace(' ', '-')

    # Get paths of files with individual functional data projected on...
    # ... the surface
    cifti_paths = get_isurf_cifti(surf_dir, subjects, contrast,
                                  surfspace=surfspace)

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

    return zvals


def whole_brain_fdr(derivatives_dir, subjects, task_key, contrast_key, gmask):

    # Paths of the NORMALIZED individual contrast map for all subjects
    encoding_maps = [os.path.join(derivatives_dir, 'sub-%02d' % sub,
                                  'estimates', task_key, 'ffx_rwls_dbb_hrf128',
                                  'wcon_%04d' % contrast_key + '.nii')
                     for sub in subjects]

    # Create design matrix (one-sample t-test)
    design_matrix = pd.DataFrame([1] * len(encoding_maps), columns=['intercept'])

    # Initialize a NiftiMasker (it will create an implicit mask from the Z-map)
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

    return fdr_thresh


def plot_flatmap(stats, threshold, contrast_tag, hemi=['L', 'R'],
                 colormap='copper', vmax=10):

    contrast = contrast_tag.lower().replace(' ', '-')

    # Get border files
    meshes_dir = os.path.join(home, 'mygit', 'surfAnalysisPy', 'standard_mesh')
    borders = {'L': os.path.join(meshes_dir, 'fs_L', 'fs_LR.32k.L.border'),
               'R': os.path.join(meshes_dir, 'fs_R', 'fs_LR.32k.R.border')
               }

    # Define figure with two subplots
    fig, axs = plt.subplots(1, len(hemi), figsize=(8, 4),
                            gridspec_kw={'wspace': 0.05})
    for ax, stat, h in zip(axs, stats, hemi):
        plt.sca(ax)
        ax = surf.plot.plotmap(stat,
                               surf=f'fs32k_{h}',
                               threshold=threshold,
                               cmap=colormap,
                               borders=borders[h],
                               )

    # Define lower bound of color limits
    vmin = threshold

    # Make colorbar
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=colormap)
    cbar = fig.colorbar(sm, ax=[axs[0], axs[1]], orientation='horizontal',
                        fraction=0.05, pad=0.02)

    # Add label below colorbar
    cbar.set_label('z-values', fontsize=12, labelpad=8)

    # Set 4 evenly spaced tick positions
    tick_positions = np.linspace(vmin, vmax, 4)
    cbar.set_ticks(tick_positions)

    # Format tick labels to 1 decimal place
    cbar.ax.set_xticklabels([f'{tick:.1f}' for tick in tick_positions],
                            fontsize=12)

    # Reduce extra whitespace
    plt.subplots_adjust(left=0, right=1, top=0.97, bottom=0.05, wspace=0.02)

    # Force a small, tight figure
    fig.set_size_inches(6, 2.5)
    
    # Save figure with tight cropping
    output_name = f'group_{contrast}_fslr32k.png' if len(hemi) == 2 else \
        f'group_{contrast}_fslr32k_{hemi[0]}.png'
    fig.savefig(output_name, dpi=300, bbox_inches='tight', pad_inches=0)


# %%
# =========================== INPUTS ===================================

# Subjects without pilot
SUBJECTS = [3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26, 28,
            29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

# Relative path for output folder
surf_folder = 'surface_files'

task_tag = 'All Tasks'
contrast_name = 'Auditory Encoding'

# %%
# ========================= PARAMETERS =================================

# Parent directories
home = os.path.expanduser('~')
music = os.path.join(home, 'diedrichsen_data/data/Cerebellum/music-sdtb')
derivatives_folder = os.path.join(music, 'derivatives')
wb_gmask = os.path.join(derivatives_folder, 'group', 'anat',
                        'group_mask_noskull.nii')

tasks = {'prod': 'Production', 'percep': 'Perception', 'ntfd': 'NTFD',
         'allmain_tasks': 'All Tasks'}

all_contrasts = {1: 'Encoding',
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
                 18: 'Decision'}

task_id = {v: k for k, v in tasks.items()}.get(task_tag)
contrast_id = {v: k for k, v in all_contrasts.items()}.get(contrast_name)

# %%
# ============================ RUN =====================================

if __name__ == '__main__':

    # Compute individual gifti/cifti files with the volume to surface...
    # ... projection of the contrast map
    individual_surf(derivatives_folder, SUBJECTS, task_id, contrast_id,
                    surf_folder, surfspace='fslr32k', save='cifti')

    # Compute group func cifti
    z_values = group_surf(surf_folder, SUBJECTS, contrast_name,
                          surfspace='fslr32k')

    # Compute whole-brain fdr threshold of volumetric data
    fdr_thresh = whole_brain_fdr(derivatives_folder, SUBJECTS, task_id,
                                 contrast_id, wb_gmask)

    # ################## Plot ##################
    # Note: This plotting only works for surfspace='fslr32k'
    
    # Split results into the two hemispheres
    zvals_lh = np.split(z_values, 2, axis=0)[0]
    zvals_rh = np.split(z_values, 2, axis=0)[1]
    split_maps = [zvals_lh, zvals_rh]

    # Plot static flatmap
    v_max = np.max(z_values[~np.isnan(z_values)])
    print(f'Maximum Z value is: {v_max}')
    plot_flatmap(split_maps, fdr_thresh, contrast_name, hemi=['L', 'R'],
                 vmax=v_max)
