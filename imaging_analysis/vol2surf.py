"""
Script to do the volume to surface projection of data from the
 Music-SDTB project + smoothing

Author: Ana Luisa Pinho
Email: agrilopi@uwo.ca

Creation: 24th of February 2025
Last Update: February 2025

Compatibility: Python 3.10.14, nilearn 0.11.1
"""

import os
import numpy as np

import nibabel as nib
import nitools as nt
import surfAnalysisPy as surf

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

from scipy import stats
from nilearn.image import load_img
from nilearn.surface import load_surf_data, vol_to_surf


# %%
# ========================== FUNCTIONS =================================

def get_imeshes(derivatives_dir, subjects, surfspace='fsaverage'):

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
                    surf_dir, surfspace='fsaverage', save='gifti'):

    # Paths of non-normalized individual contrast map for all subjects
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
        contrast = all_contrasts[contrast_key].replace(" ", "-")
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
                    "sub-{sb:02d}_".format(sb=sb)
                    + contrast.lower()
                    + "_"
                    + surfspace
                    + ".hem-L.func.gii",
                ),
            )
            nib.save(
                GIFTIR,
                os.path.join(
                    surf_dir,
                    "sub-{sb:02d}_".format(sb=sb)
                    + contrast.lower()
                    + "_"
                    + surfspace
                    + ".hem-R.func.gii",
                ),
            )
        else:
            assert save == 'cifti'
            # Create CIFTI
            CIFTI = nt.cifti.join_giftis_to_cifti([GIFTIL, GIFTIR],
                                                  mask=[None, None])
            # Save CIFT file
            nib.save(CIFTI, os.path.join(
                surf_dir,
                'sub-%02d_' % sb + contrast.lower() + '.dscalar.nii'))


def get_isurf(surf_dir, subjects, contrast, surfspace='fsaverage'):
    
    # Paths of individual gifti files per hemisphere
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


def threshold(z_vals, p_vals, alpha, height_control='fdr'):
    """
    Return the Benjamini-Hochberg FDR or Bonferroni threshold for
    the input correlations + corresponding p-values.
    """
    if alpha < 0 or alpha > 1:
        raise ValueError(
            'alpha should be between 0 and 1. {} was provided'.format(alpha))

    p_vals_ = np.sort(p_vals)
    idx = np.argsort(p_vals)

    z_vals_abs = np.abs(z_vals)
    z_vals_ = z_vals_abs[idx]

    n_samples = len(p_vals_)

    if height_control == 'fdr':
        pos = p_vals_ < alpha * np.linspace(1 / n_samples, 1, n_samples)
    elif height_control == 'bonferroni':
        pos = p_vals_ < alpha / n_samples
    else:
        raise ValueError('Height-control method not valid.')

    return (z_vals_[pos][-1] - 1.e-12) if pos.any() else np.infty


def group_surf(surf_dir, subjects, contrast_tag, surfspace='fsaverage'):

    contrast = contrast_tag.lower().replace(" ", "-")

    # Get individual functional data projected on the surface
    gifti_left, gifti_right = get_isurf(surf_dir, subjects, contrast,
                                        surfspace=surfspace)

    # Load individual surface data
    data_left = np.array([load_surf_data(gl) for gl in gifti_left])
    data_right = np.array([load_surf_data(gr) for gr in gifti_right])

    # Substitute nan's by 0's
    data_left[np.isnan(data_left)] = 0
    data_right[np.isnan(data_right)] = 0

    # Stack data
    data = np.hstack((data_left, data_right))

    # Calculate the one sample t-test
    tvals, pvals = stats.ttest_1samp(data, 0, axis=0, alternative='two-sided')

    # Compute z-values from t-values
    zvals = zval_conversion(tvals, len(subjects)-1)

    # Threshold z-values, ...
    # ... because pvalues are two-sided, ...
    # ... alpha is corrected by a factor of 2
    fdr_thresh = threshold(zvals, pvals, 0.05/2, height_control='fdr')
    print(fdr_thresh)

    return zvals, fdr_thresh


def plot_flatmap(stats, threshold, contrast_tag, hemi=['L', 'R'],
                 colormap='copper'):

    contrast = contrast_tag.lower().replace(" ", "-")

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

    # Define color limits
    vmin, vmax = threshold, 10

    # Make colorbar
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=colormap)
    cbar = fig.colorbar(sm, ax=[axs[0], axs[1]], orientation='horizontal',
                        fraction=0.05, pad=0.02)

    # Add label below colorbar
    cbar.set_label("z-values", fontsize=12, labelpad=8)

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
home = os.path.expanduser("~")
music = os.path.join(home, 'diedrichsen_data/data/Cerebellum/music-sdtb')
derivatives_folder = os.path.join(music, 'derivatives')

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

    # Get individual cifti files with the volume to surface projection of...
    # ... the contrast map per participant
    individual_surf(derivatives_folder, SUBJECTS, task_id, contrast_id,
                    surf_folder, surfspace='fslr32k')

    # Compute group func gifti
    z_values, thresh = group_surf(surf_folder, SUBJECTS, contrast_name,
                                  surfspace='fslr32k')

    # Split results into the two hemispheres
    zvals_lh = np.split(z_values, 2, axis=0)[0]
    zvals_rh = np.split(z_values, 2, axis=0)[1]
    split_maps = [zvals_lh, zvals_rh]

    # Plot static
    plot_flatmap(split_maps, thresh, contrast_name, hemi=['L', 'R'])
    
    # from nilearn.plotting import plot_surf_stat_map
    # from nilearn.datasets import load_fsaverage
    # from nilearn.datasets import load_fsaverage_data

    # fsaverage_meshes = load_fsaverage(mesh='fsaverage')

    # curv_sign = load_fsaverage_data(mesh='fsaverage', data_type="curvature")
    # for hemi, data in curv_sign.data.parts.items():
    #     curv_sign.data.parts[hemi] = np.sign(data)

    # # In this example we will plot both hemispheres, but you can choose one of
    # # "left", "right" or "both".
    # hemi = "left"

    # fig = plot_surf_stat_map(
    #     stat_map=zvals_lh,
    #     surf_mesh=fsaverage_meshes["inflated"],
    #     hemi=hemi,
    #     title="Surface with matplotlib",
    #     threshold=thresh,
    #     bg_map=curv_sign,
    #     output_file='left_hemi.png'
    # )




