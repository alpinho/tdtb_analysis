"""
Script to do the volume to suit projection of data from the
 Music-SDTB project

Author: Ana Luisa Pinho
Email: agrilopi@uwo.ca

Creation: 27th of February 2025
Last Update: March 2025

Compatibility: Python 3.10.14, SUITPy 1.3.2
"""

import os

import numpy as np
import nibabel as nib
import nitools as nt
import matplotlib.pyplot as plt

from scipy import stats
from SUITPy import flatmap

from volume_to_surface import whole_brain_thresholds


# %%
# ========================== FUNCTIONS =================================

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


def group_suit(group_dir, task_key, contrast_key, subjects, suit_dir):

    contrast = all_contrasts[contrast_key].replace(' ', '_')

    # Paths of the group contrast t-map
    encoding_map = os.path.join(
        group_dir, task_key, 'rfx_onesample_t_rwls_dbb_hrf128_wb',
        'con_%02d' % contrast_key + '_%s' % contrast, 'spmT_0001.nii')

    # Maps volume-based data onto the suit surface as numpy arrays
    suit_tvals = flatmap.vol_to_surf(encoding_map, space='SUIT')

    # Remove the second dimension
    suit_tvals = np.squeeze(suit_tvals, axis=1)

    # Compute z-values from t-values
    suit_zvals = zval_conversion(suit_tvals, len(subjects)-1)

    # Transform numpy arrays in gifti files
    contrast = all_contrasts[contrast_key].replace('_', '-')
    gifti = nt.gifti.make_func_gifti(suit_zvals,
                                     anatomical_struct='Cerebellum',
                                     column_names=[contrast])

    # Create directory to save outputs if does not exist
    if not os.path.exists(suit_dir):
        os.makedirs(suit_dir)

    # Save the data
    nib.save(
        gifti,
        os.path.join(
            suit_dir,
            'group_'
            + task_key.replace('_', '-')
            + '_'
            + contrast.lower()
            + '_'
            + 'suit.func.gii',
        ),
    )

    return suit_zvals


def plot_suitflat(stats, threshold, task_key, contrast_tag, output_dir,
                  colormap='viridis', vmax=10):

    contrast = contrast_tag.lower().replace(' ', '-')
    task_name = task_key.replace('_', '-')

    # Define lower bound of color limits
    vmin = threshold

    # Do the flatmap
    flatmap.plot(stats,
                 cmap=colormap,
                 cscale=[vmin, vmax],
                 underscale=[-5., 5.], # (default: [-1, 0.5])
                 threshold=vmin,
                 colorbar=True,
                 render='matplotlib')

    # Get the current figure created by flatmap.plot()
    fig = plt.gcf()  # Get the figure from the active Matplotlib state

    # Change fontsize of numbers in colorbar
    cbar = fig.axes[-1]
    cbar.tick_params(labelsize=14)

    # Title of colormap
    plt.text(-1., 7.9, "Z-values", fontsize=15, color="black")

    # Save figure
    output_name = f'group_{task_name}_{contrast}_suit.png'
    output_path = os.path.join(output_dir, output_name)
    fig.savefig(output_path, dpi=300)


# %%
# =========================== INPUTS ===================================

# Subjects without pilot
SUBJECTS = [3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26, 28,
            29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

# Relative path for output folder
suit_folder = 'results/suit_files'
contrasts_folder = 'results/control_contrasts'

task_tag = 'All Tasks'
contrast_name = 'Encoding'

# %%
# ========================= PARAMETERS =================================

# Parent directories
home = os.path.expanduser('~')
music = os.path.join(home, 'diedrichsen_data/data/Cerebellum/music-sdtb')
derivatives_folder = os.path.join(music, 'derivatives')
group_folder = os.path.join(derivatives_folder, 'group')
wb_gmask = os.path.join(group_folder, 'anat', 'group_mask_noskull.nii')

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

# Already pre-computed thresholds for the three first contrasts...
# ... considering all tasks together (allmain_tasks)
fdr_thresh_encoding = 2.7166013496886174
zmax_encoding = 6.796930745609075

fdr_thresh_audio_encoding = 2.7051156945711403
zmax_audio_encoding = 7.366581723533498

fdr_thresh_visual_encoding = 2.6649611311019035
zmax_visual_encoding = 6.896651056145507

# %%
# ============================ RUN =====================================

if __name__ == '__main__':

    # Create contrasts folder if it does not exist
    os.makedirs(contrasts_folder, exist_ok=True)

    # Get z-values of group contrast in SUIT space
    z_values = group_suit(group_folder, task_id, contrast_id, SUBJECTS,
                          suit_folder)

    # Compute whole-brain fdr threshold of volumetric data
    # fdr_thresh, zmax = whole_brain_thresholds(
    #     derivatives_folder, SUBJECTS, task_id, contrast_id, wb_gmask)

    # Plot cerebellum flatmap
    v_max = np.amax(z_values[~np.isnan(z_values)])
    print(f'Maximum Z value is: {v_max}')
    plot_suitflat(z_values, fdr_thresh_encoding, task_id, contrast_name,
                  contrasts_folder, vmax=zmax_encoding)
