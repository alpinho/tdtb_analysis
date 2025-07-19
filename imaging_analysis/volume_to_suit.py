"""
Script to do the volume to suit projection of data from the
 Music-SDTB project

Author: Ana Luisa Pinho
Email: agrilopi@uwo.ca

Creation: 27th of February 2025
Last Update: July 2025

Compatibility: Python 3.10.14, SUITPy 1.3.2

Note about overlap colorbar:
The "Overlap" colorbar is simply a third gradient that tells you how 
strongly both contrasts co-activate at each vertex. Concretely:

    Range

        Minimum tick (vmin) = thr1/v1 + thr2/v2
        (i.e. both maps just at their respective statistical 
         thresholds)

        Maximum tick (vmax) = 1.0
        (i.e. at least one map at its peak, and the other possibly 
         contributing up to its peak)

    Color gradient

        The start color is the sum of the two threshold-level colors

        The end color is the sum of the two full-bright colors 
        (clipped to [0, 1])

        Intermediate hues reflect intermediate sums of the two 
        normalized intensities

    Tick labels

        Show four values:

            Min sum-of-fractions (thr1/v1 + thr2/v2)

            Two intermediate sums (one- and two-thirds along the range)

            Max = 1.0
"""

import sys
import os

import numpy as np
import nibabel as nib
import nitools as nt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import to_rgb, Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable

from SUITPy import flatmap
from volume_to_surface import whole_brain_thresholds

# setting path
sys.path.append('../')
# importing
from utils import zval_conversion


# %%
# ========================== FUNCTIONS =================================

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


def scientific_notation(x, pos):
    return f"{x:.1e}"  # 1 decimal in scientific notation (e.g., 1.2e3)


def plot_suitflat(stats,
                  threshold,
                  outpath,
                  colormap='viridis',
                  vmax=10,
                  colors=('r', 'b'),
                  sci_notation=False,
                  cmap_title_loc=(.775, .69),
                  cmap_title='Z-values'):
    """
    Plot one or two contrasts on a SUIT flatmap.

    Single‐contrast:
      stats : 1D array (n_verts,)
      threshold : float
      vmax : float
      colormap : str

    Two‐contrast RGB overlay:
      stats : [stats1, stats2]  # each a 1D array
      threshold : [thr1, thr2]
      vmax : [vmax1, vmax2]
      colors : (c1, c2)  # valid matplotlib colors

    In the overlay case we compute normalized fractions,
    zero‐out below threshold, build RGB channels, set alpha=1
    wherever either map survives, and plot with overlay_type='rgb'.
    """

    # Detect overlay mode
    is_overlay = isinstance(stats, (list, tuple)) and len(stats) == 2

    if not is_overlay:
        # ——— single contrast ———
        vmin = threshold
        flatmap.plot(
            stats,
            cmap=colormap,
            cscale=[vmin, vmax],
            underscale=[-5., 5.],
            threshold=vmin,
            colorbar=True,
            render='matplotlib'
        )
        fig = plt.gcf()
        cbar = fig.axes[-1]
        cbar.tick_params(labelsize=14)
        cbar.set_position([.825, .2, .03, .6])
        if sci_notation:
            formatter = ticker.FuncFormatter(lambda x, pos: f"{x:.1e}")
            cbar.yaxis.set_major_formatter(formatter)
        fig.text(*cmap_title_loc, cmap_title, fontsize=15, color='black')
        fig.savefig(outpath, dpi=300)
        return

    # ——— two-contrast overlay ———
    stat1, stat2 = stats

    # unpack thresholds & vmax
    try:
        thr1, thr2 = threshold
    except Exception:
        thr1 = thr2 = threshold
    try:
        v1, v2 = vmax
    except Exception:
        v1 = v2 = vmax

    # normalize and threshold
    norm1 = np.clip(stat1 / v1, 0, 1)
    norm2 = np.clip(stat2 / v2, 0, 1)
    frac1 = thr1 / v1
    frac2 = thr2 / v2
    norm1[norm1 < frac1] = 0
    norm2[norm2 < frac2] = 0

    # build RGB data
    rgb1 = np.array(to_rgb(colors[0]))
    rgb2 = np.array(to_rgb(colors[1]))
    nverts = stat1.shape[0]
    rgb_data = np.outer(norm1, rgb1) + np.outer(norm2, rgb2)
    rgb_data = np.clip(rgb_data, 0, 1)

    # assemble RGBA
    data = np.zeros((nverts, 4), float)
    data[:, :3] = rgb_data
    alpha = ((norm1 > 0) | (norm2 > 0)).astype(float)
    data[:, 3] = alpha
    data[alpha == 0, :] = np.nan

    # plot RGB overlay
    flatmap.plot(
        data,
        overlay_type='rgb',
        new_figure=True,
        colorbar=False,
        render='matplotlib',
        underscale=[-5., 5.]
    )

    # ##################### LEGEND LABELS #############################

    # Get base filename (no path, no extension)
    fname = os.path.splitext(os.path.basename(outpath))[0]

    # Strip trailing '_suit' if present
    base = fname[:-5] if fname.endswith('_suit') else fname

    # Split off the two contrast parts at the last '_vs_' or '_and_'
    if '_vs_' in base:
        left, right = base.rsplit('_vs_', 1)
    elif '_and_' in base:
        left, right = base.rsplit('_and_', 1)
    else:
        raise ValueError(f"Can't find '_vs_' or '_and_' separator in "
                         f"'{fname}'")

    # Pull only the contrast key from the left side 
    # (drop any 'group_…_' prefix)
    c1_key = left.split('_')[-1]
    c2_key = right  # this is already just the second contrast

    # Format into nice labels
    s1 = c1_key.replace('-', ' ').replace('_', ' ')
    label1 = ' '.join(word.capitalize() for word in s1.split())

    s2 = c2_key.replace('-', ' ').replace('_', ' ')
    label2 = ' '.join(word.capitalize() for word in s2.split())

    # ###################### COLORBARS ################################

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
    cmap1 = LinearSegmentedColormap.from_list("c1", [thr_color1, rgb1])
    sm1 = ScalarMappable(norm=Normalize(vmin=thr1, vmax=v1), cmap=cmap1)
    sm1.set_array([])

    cmap2 = LinearSegmentedColormap.from_list("c2", [thr_color2, rgb2])
    sm2 = ScalarMappable(norm=Normalize(vmin=thr2, vmax=v2), cmap=cmap2)
    sm2.set_array([])

    cmap3 = LinearSegmentedColormap.from_list("c3", [thr_overlap, max_overlap])
    # We normalize overlap on a 0–1 scale of (norm1+norm2)... 
    # ... clipped -> [thr_frac1 + thr_frac2, 1]
    min_ol = thr_frac1 + thr_frac2
    sm3 = ScalarMappable(norm=Normalize(vmin=min_ol, vmax=1.0), cmap=cmap3)
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
        (sm1, [.04, .125, .25, .03], thr1, m1_1, m1_2, v1,
         f"Z-Values ({label1})"),
        (sm2, [.3825, .125, .25, .03], thr2, m2_1, m2_2, v2,
         f"Z-Values ({label2})"),
        (sm3, [.715, .125, .25, .03], min_ol, m3_1, m3_2, 1.0, 
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
        cb.set_label(lbl, fontsize=11, labelpad=10)
        cb.ax.set_xticklabels([f"{lo:.2f}", f"{m1:.2f}", f"{m2:.2f}", 
                               f"{hi:.2f}"])
        cb.ax.tick_params(labelsize=10)

    # ########################### SAVE ################################
    fig = plt.gcf()
    fig.savefig(outpath, dpi=300)


# %%
# =========================== INPUTS ===================================

# Subjects without pilot
SUBJECTS = [3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26, 28,
            29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

# Relative path for output folder
suit_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'results', 'suit_files')
contrasts_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'results', 'suit_images')
irois_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'results', 'irois')

task_tag = 'All Tasks'
contrast_name = 'Auditory Encoding'
contrast_name2 = 'Visual Encoding' # Set to None if not used

# %%
# ========================= PARAMETERS =================================

# Parent directories
if os.path.isdir('/home/analu/diedrichsen_data/data'):
    base_dir = '/home/analu/diedrichsen_data/data'
else:
    base_dir = '/cifs/diedrichsen/data'

music = os.path.join(base_dir, 'Cerebellum/music-sdtb')
derivatives_folder = os.path.join(music, 'derivatives')
group_folder = os.path.join(derivatives_folder, 'group')
wb_gmask_path = os.path.join(group_folder, 'anat', 'group_mask_noskull.nii')

iroi_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'roi_analyses_rwls_hrf128_wb_puncorr', 'all',
                         'cerebellum', 'ntk_symmni128', 'cereb',
                         'overlaid_masks', 'i8a_cereb_bh_mask.nii.gz')

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
cname = contrast_name.replace(' vs ', '_vs_').replace(' ', '-')

# Check if a second contrast is provided
if contrast_name2:
    contrast_id2 = \
        {v: k for k, v in all_contrasts.items()}.get(contrast_name2)
    cname2 = contrast_name2.replace(' vs ', '_vs_').replace(' ', '-')

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

    # Create output folders if they do not exist
    suitplots_folder = os.path.join(
        contrasts_folder, str(contrast_id) + '_' + cname.lower())
    os.makedirs(suitplots_folder, exist_ok=True)
    os.makedirs(irois_folder, exist_ok=True)

    # #################################################################

    # Compute z-values of group contrast in SUIT space, save and return
    z_values = group_suit(group_folder, task_id, contrast_id, SUBJECTS,
                          suit_folder)

    # Compute whole-brain fdr threshold of volumetric data
    fdr_thresh, zmax = whole_brain_thresholds(
        derivatives_folder, SUBJECTS, task_id, contrast_id, wb_gmask_path)

    # #### Plot cerebellum stat flatmap ####
    # One contrast
    contrast_fname = (
        f"group_{task_id.replace('_', '-')}_{cname.lower()}_suit.png"
    )
    contrast_fpath = os.path.join(suitplots_folder, contrast_fname)
    # plot_suitflat(z_values, fdr_thresh, contrast_fpath, vmax=zmax)
    
    # Two contrasts
    if contrast_name2:
        z_values2 = group_suit(group_folder, task_id, contrast_id2,
                               SUBJECTS, suit_folder)
        fdr_thresh2, zmax2 = whole_brain_thresholds(
            derivatives_folder, SUBJECTS, task_id, contrast_id2,
            wb_gmask_path
            )
        rgbaplots_folder = os.path.join(
            contrasts_folder, 'rgba',
            cname.lower() + '_and_' + cname2.lower()
            )
        os.makedirs(rgbaplots_folder, exist_ok=True)
        contrasts_fname = (
            f"group_{task_id.replace('_', '-')}_"
            f"{cname.lower()}_and_{cname2.lower()}_suit.png"
        )
        contrast_fpath2 = os.path.join(rgbaplots_folder, contrasts_fname)
        plot_suitflat(
            stats=[z_values, z_values2],
            threshold=[fdr_thresh, fdr_thresh2],
            outpath=contrast_fpath2,
            vmax=[zmax, zmax2],
            colors=('#009E73','#F0E442')
        )

    # #################################################################

    # # Plot cerebellum overlaid iroi
    # iroi = nib.load(iroi_path)
    # iroi_suitdata = flatmap.vol_to_surf(iroi, space='SUIT')
    # thresh = np.unique(iroi_suitdata)[1]
    # # iroi_suitdata[iroi_suitdata == 0.] = np.nan
    # iroi_fname = 'iroi_cerebellum_suit.png'
    # iroi_fpath = os.path.join(irois_folder, iroi_fname)
    # plot_suitflat(iroi_suitdata, 1/len(SUBJECTS), iroi_fpath,
    #               colormap='cividis', vmax=1, sci_notation=True,
    #               cmap_title_loc=(.7, .69),
    #               cmap_title='Fraction of \n Participants')
