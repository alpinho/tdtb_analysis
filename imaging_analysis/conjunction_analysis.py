"""
Script: Cross-modal vs. modality-specific unpredictability maps via
        CONJUNCTION analysis, in both VOLUME and SURFACE (fs_LR32k), with the
        predictive-timing network used as a mask.

----------------------------------------------------------------------------
WHY A CONJUNCTION (and not the pooled Random - Non-Random contrast)
----------------------------------------------------------------------------
The pooled Random > Non-Random contrast collapses two distinctions that are
actually orthogonal, and cannot recover either:

  (a) Is the unpredictability EFFECT (Random > Non-Random) present in BOTH
      modalities (cross-modal) or in only ONE (modality-specific)?  A pooled
      effect can be driven entirely by one modality, so pooled significance
      does not imply cross-modality; and a purely modality-specific effect can
      fail to survive pooling altogether.

  (b) Relative to REST, is the response an activation or a deactivation?  The
      Random - Non-Random difference is orthogonal to the baseline level, so
      its sign says nothing about activation vs. deactivation; that requires
      the condition-vs-Rest maps.

A conjunction under the conjunction-null / minimum-statistic logic (Nichols et
al., 2005) requires each conjoined contrast to be INDIVIDUALLY significant,
i.e. a logical AND. That is exactly the cross-modality guarantee, and because
each arm is one-sided it also fixes the activation/deactivation sign. We run
the conjunctions on the per-modality condition-vs-Rest contrasts, never on the
difference.

----------------------------------------------------------------------------
THE THREE CONJUNCTIONS (intersections of FDR-thresholded maps)
----------------------------------------------------------------------------
Per-modality FDR-thresholded maps (NTFD Random task):
  A_eff / V_eff  = Auditory / Visual (Random > Non-Random)        [side 'pos']
  A_act / V_act  = Auditory / Visual (Random vs Rest)             [side 'pos']
  A_deact/V_deact= Auditory / Visual (Non-Random vs Rest)         [side 'neg']

  1. Cross-modal activation     = A_eff & V_eff & A_act & V_act
        (unpredictability effect in both modalities AND Random above rest in
         both -> activated cross-modally by unpredictability)

  2. Cross-modal deactivation   = A_eff & V_eff & A_deact & V_deact
        (effect in both AND Non-Random below rest in both -> the predictable
         condition suppresses the region cross-modally)

  3. Modality-specific activation
        =  (A_eff & A_act & ~V_eff)            # auditory only
        OR (V_eff & V_act & ~A_eff)            # visual only
        (effect + activation in one modality, NO effect in the other; the
         '~' is exclusive masking that isolates 'one modality only')

The '&' across modalities is the conjunction (minimum statistic); the '~' is
exclusive masking. Modality-specific deactivation and sensory dissociation are
intentionally NOT computed here.

----------------------------------------------------------------------------
MASKING BY THE PREDICTIVE-TIMING NETWORK
----------------------------------------------------------------------------
The predictive-timing network is Encoding vs. Rest = Beat + Interval, pooled
over Production / Perception / NTFD (the 'All Main Tasks' Encoding contrast);
it deliberately EXCLUDES the Random condition. We build it two-sided:

  NET = ( |z(Encoding, All Main Tasks)| >= FDR threshold )

Then, for every conjunction result C:

  beyond timing  = C & ~NET     (Encoding sub-threshold: outside the network)
  within timing  = C &  NET     (kept for comparison)

The three conjunctions are computed once; each is then intersected with ~NET
to keep only what lies outside the predictive-timing network.

----------------------------------------------------------------------------
OUTPUTS
----------------------------------------------------------------------------
Volume  : results/parametric_tests/volume/<task>/conjunctions/<category>/
            <category>_zmap.nii.gz            (signed minimum-statistic map)
            <category>_mask.nii.gz            (binary, beyond timing)
            <category>_glassbrain_*.png
Surface : results/parametric_tests/surface/<task>/surface_images/
          conjunction/<category>/
            group_<task>_<category>_flat_contour_fslr32k*.png
          (flatmap of the conjunction statistic, filled, with the Encoding /
           predictive-timing network drawn as a two-sided contour)

The displayed statistic is the conjunction-null minimum statistic: the least
significant of the conjoined condition-vs-Rest arms (min z for activations,
max z i.e. nearest-zero for deactivations), so a voxel/vertex is shown only
where every arm individually survives FDR.

Author: (generated to match volume_maps.py / volume_to_surface.py)
Compatibility: Python 3.10, nilearn 0.11.1
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib

from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm.thresholding import fdr_threshold
from nilearn.image import resample_to_img, load_img

# Re-use the project's machinery (resource paths inside these functions
# resolve relative to volume_to_surface.py / ols_permutation_tests.py).
from ols_permutation_tests import plot_glass_brain_z
from volume_to_surface import (
    build_contrasts,
    individual_surf,
    group_surf,
    mask_cortical_activation,
    plot_flatmap,
    lh_medial_wall_mask_path,
    rh_medial_wall_mask_path,
)


# %%
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


def subject_contrast_paths(derivatives_dir, subjects, task_key, cid):
    """Per-subject normalized contrast images (wcon) for a second-level fit."""
    fname = FILENAME_TEMPLATE.format(cid=cid)
    return [
        os.path.join(derivatives_dir, f"sub-{sub:02d}", "estimates",
                     task_key, DERIVATIVE_SUBFOLDER, fname)
        for sub in subjects
    ]


def second_level_zmap(cid, task_key, subjects, fitting_mask, smoothing_fwhm):
    """Fit a one-sample second-level model (intercept-only) on the per-subject
    contrast maps and return the group z-map image. Same model as
    volume_maps.py; the fitting mask constrains smoothing to the brain."""
    conpaths = subject_contrast_paths(derivatives_folder, subjects,
                                      task_key, cid)
    design = pd.DataFrame({'intercept': [1] * len(conpaths)})
    slm = SecondLevelModel(mask_img=fitting_mask,
                           smoothing_fwhm=smoothing_fwhm)
    slm = slm.fit(conpaths, design_matrix=design)
    return slm.compute_contrast(output_type='z_score')


def load_or_fit_volume_z(cid, cname, task_key, subjects, ref_img=None):
    """Return the group volume z-map for a contrast. Loads the z-map written by
    volume_maps.py if present; otherwise re-fits it. Resampled onto ref_img
    so all contrasts share one grid (the network task may differ)."""
    zpath = os.path.join(VOL_RESULTS_ROOT, task_key,
                         id_label_folder(cid, cname),
                         f"{id_label_folder(cid, cname)}_zmap.nii.gz")
    if LOAD_PRECOMPUTED_VOLUME_ZMAPS and os.path.exists(zpath):
        z_img = nib.load(zpath)
    else:
        z_img = second_level_zmap(cid, task_key, subjects,
                                  FITTING_MASK_PATH, SMOOTHING_FWHM)
    if ref_img is not None and z_img.shape != ref_img.shape:
        z_img = resample_to_img(z_img, ref_img, interpolation='continuous',
                                force_resample=True, copy_header=True)
    return z_img


def mask_on_grid(mask_path, ref_img):
    """Load a NIfTI mask, resample to ref_img grid (nearest), return bool."""
    m = resample_to_img(nib.load(mask_path), ref_img, interpolation='nearest',
                        force_resample=True, copy_header=True)
    return np.asanyarray(m.dataobj).astype(bool)


def fdr_bool(z, in_mask, side, alpha=0.05, thr_override=None):
    """Boolean supra-threshold map for one tail of a z-map.

    side='pos' : z >= +thr           (one-sided BH-FDR on the positive tail)
    side='neg' : z <= -thr           (one-sided BH-FDR on the negative tail)
    side='two' : |z| >= thr          (two-sided BH-FDR on |z|)

    thr_override forces the |z| threshold directly (used for the network
    contour). Returns (boolean array shaped like z, signed threshold).
    """
    vals = z[in_mask]
    vals = vals[np.isfinite(vals)]
    if side == 'pos':
        thr = float(thr_override) if thr_override else \
            float(fdr_threshold(vals, alpha))
        return (z >= thr) & in_mask, thr
    if side == 'neg':
        thr = float(thr_override) if thr_override else \
            float(fdr_threshold(-vals, alpha))
        return (z <= -thr) & in_mask, -thr
    # two-sided
    thr = float(thr_override) if thr_override else \
        float(fdr_threshold(np.abs(vals), alpha))
    return (np.abs(z) >= thr) & in_mask, thr


def conjoin(bools, include, exclude):
    """Logical AND over `include` term-masks, AND-NOT over `exclude` ones."""
    ref = bools[include[0]]
    out = np.ones(ref.shape, dtype=bool)
    for k in include:
        out &= bools[k]
    for k in exclude:
        out &= ~bools[k]
    return out


def min_statistic(zmaps, display_terms, keep):
    """Conjunction-null (minimum) statistic for display, restricted to `keep`.

    All display arms share one sign. For 'pos' arms the statistic is the
    voxel/vertex-wise minimum z (the weakest arm); for 'neg' arms it is the
    maximum z (the nearest-zero, i.e. weakest deactivation). Voxels outside
    `keep` are set to 0."""
    sides = {TERMS[t]['side'] for t in display_terms}
    stack = np.stack([zmaps[t] for t in display_terms], axis=0)
    stat = stack.max(axis=0) if sides == {'neg'} else stack.min(axis=0)
    out = np.zeros(stat.shape, dtype=float)
    out[keep] = stat[keep]
    return out


def load_cortex_mask(lh_path, rh_path):
    """Concatenated (L,R) boolean cortex mask from the medial-wall GIFTIs."""
    lh = nib.load(lh_path).darrays[0].data.astype(bool)
    rh = nib.load(rh_path).darrays[0].data.astype(bool)
    return np.concatenate([lh, rh], axis=0)


def surface_group_z(task_key, cid, cname, surf_dir, subjects,
                    compute_individual=True):
    """Group surface z-map (concatenated L,R vertices) for one contrast.

    Mirrors volume_to_surface.py: optionally (re)project each subject to
    fs_LR32k, then group via group_surf, and zero the medial wall. Returns the
    concatenated z (length = 2 * n_vertices), already medial-wall-masked."""
    tag = cname.replace(' vs ', '_vs_').replace(' ', '-')
    if compute_individual:
        cdir = os.path.join(surf_dir, f"{cid}_{tag.lower()}")
        os.makedirs(cdir, exist_ok=True)
        for save in ('gifti', 'cifti'):
            individual_surf(derivatives_folder, subjects, task_key,
                            build_contrasts(task_key), cid, surf_dir,
                            surfspace='fslr32k', save=save)
    z = group_surf(surf_dir, subjects, task_key, cid, tag, surfspace='fslr32k')
    zL = mask_cortical_activation(np.split(z, 2, axis=0)[0],
                                  lh_medial_wall_mask_path)
    zR = mask_cortical_activation(np.split(z, 2, axis=0)[1],
                                  rh_medial_wall_mask_path)
    return np.concatenate([zL, zR], axis=0)


# %%
# ========================== ANALYSIS STEPS =============================

def compute_volume_category(category, spec, zcache, brain, net_bool):
    """Build one conjunction in the volume, mask it to outside the predictive-
    timing network, and write the signed minimum-statistic z-map, the binary
    mask, and the glass brain.

    Steps: threshold each conjoined contrast (fdr_bool, per side) -> AND/AND-NOT
    them (conjoin) -> intersect with ~NET (beyond timing) -> save."""
    bools = {t: fdr_bool(zcache[t], brain, TERMS[t]['side'], FDR_ALPHA)[0]
             for t in set(spec['include'] + spec['exclude'] + spec['display'])}
    cat = conjoin(bools, spec['include'], spec['exclude'])
    beyond = cat & ~net_bool

    stat = min_statistic(zcache, spec['display'], beyond)

    out_dir = os.path.join(VOL_CONJ_ROOT, category)
    os.makedirs(out_dir, exist_ok=True)
    if GM_MASK_PATH:
        gm = mask_on_grid(GM_MASK_PATH, REF_IMG)
        stat = np.where(gm, stat, 0.0)

    z_path = os.path.join(out_dir, f"{category}_zmap.nii.gz")
    nib.save(nib.Nifti1Image(stat, REF_IMG.affine, REF_IMG.header), z_path)
    nib.save(nib.Nifti1Image(beyond.astype(np.int16), REF_IMG.affine,
                             REF_IMG.header),
             os.path.join(out_dir, f"{category}_mask.nii.gz"))

    is_deact = {TERMS[t]['side'] for t in spec['display']} == {'neg'}
    png = os.path.join(out_dir, f"{category}_glassbrain.png")
    plot_glass_brain_z(
        z_map_path=z_path,
        z_threshold=1e-6,                 # the FDR cut is already in the mask
        two_sided=is_deact,
        title=f"{category.replace('_', ' ')} (beyond predictive timing)",
        out_png=png,
        cbar_contrast_label=cap_label(category.replace('_', ' ')),
    )
    print(f"[volume] {category}: {int(beyond.sum())} voxels -> {z_path}")


def compute_surface_category(category, spec, zcache, cortex, net_bool,
                             enc_zL, enc_zR, plot_contour, net_thr):
    """Same conjunction on the fs_LR32k surface; plot the conjunction statistic
    as a filled flatmap. When `plot_contour` is True the predictive-timing
    network (Encoding) is drawn as a two-sided contour at `net_thr` -- the same
    threshold that defined `net_bool`, so the outline matches the mask. Filled
    regions of `beyond` lie outside that contour by construction."""
    bools = {t: fdr_bool(zcache[t], cortex, TERMS[t]['side'], FDR_ALPHA)[0]
             for t in set(spec['include'] + spec['exclude'] + spec['display'])}
    cat = conjoin(bools, spec['include'], spec['exclude'])
    beyond = cat & ~net_bool

    stat = min_statistic(zcache, spec['display'], beyond)
    statL, statR = np.split(stat, 2, axis=0)

    is_deact = {TERMS[t]['side'] for t in spec['display']} == {'neg'}
    vmax = float(np.nanmax(np.abs(stat))) if np.any(stat) else 1.0

    # Draw the network outline only if requested (display only).
    contour_kwargs = (
        dict(contour_stat=[enc_zL, enc_zR], contour_threshold=net_thr,
             contour_color='k', contour_linewidth=1.0,
             contour_positive_only=False)        # two-sided Encoding network
        if plot_contour else {})

    out_dir = os.path.join(SURF_CONJ_IMGS, category)
    os.makedirs(out_dir, exist_ok=True)
    plot_flatmap(
        stats=[statL, statR],
        threshold=1e-6,
        task_key=analysis_task_id,
        contrast_tag=category,
        output_dir=out_dir,
        hemi=['L', 'R'],
        colormap=('Blues_r' if is_deact else 'autumn'),
        vmax=vmax,
        signed=is_deact,
        cbar_title=('Conjunction z (deactivation)' if is_deact
                    else 'Conjunction z (activation)'),
        **contour_kwargs,
    )
    print(f"[surface] {category}: {int(beyond.sum())} vertices -> {out_dir}")


# %%
# ============================ TOGGLES ==================================

RUN_VOLUME = True
RUN_SURFACE = True

# (Re)project each subject to the surface before grouping. Set False if the
# per-subject fs_LR32k ciftis already exist for every contrast below.
COMPUTE_INDIVIDUAL_SURF = False

# Load volume z-maps already written by volume_maps.py when available
# (else re-fit the second-level model here).
LOAD_PRECOMPUTED_VOLUME_ZMAPS = True

FDR_ALPHA = 0.05
SMOOTHING_FWHM = 8.0

# Predictive-timing-network (Encoding) threshold. This SINGLE value is used in
# BOTH the computation and the display, deliberately kept consistent:
#   (1) COMPUTATION / masking: it defines the network, hence the ~NET in
#       `beyond = cat & ~net_bool` -- changing it changes which voxels/vertices
#       are kept as "beyond the predictive-timing network"; and
#   (2) surface DISPLAY: the same |z| at which the network is drawn as a
#       contour on the conjunction flatmaps (when PLOT_NETWORK_CONTOUR=True).
# None -> two-sided BH-FDR of the Encoding map (principled definition); a float
# forces |z| >= value (e.g. 2.7, as contour_threshold_override in
# volume_to_surface.py), which both restricts the mask and gives a cleaner
# outline for the dense Encoding contrast.
NET_THRESHOLD_OVERRIDE = None

# Surface DISPLAY only: draw the predictive-timing network as a contour on the
# conjunction flatmaps (True) or omit it and show just the filled conjunction
# statistic (False). This affects the figures only; the masking uses
# NET_THRESHOLD_OVERRIDE either way.
PLOT_NETWORK_CONTOUR = False


# %%
# ============================ INPUTS ===================================

SUBJECTS = [
    3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26,
    28, 29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
]

tasks = {
    'prod': 'Production',
    'percep': 'Perception',
    'ntfd': 'NTFD',
    'rand_ntfd': 'NTFD Random',
    'allmain_tasks': 'All Main Tasks',
}

# Task supplying the Random / Non-Random contrasts (the conjunction arms)
analysis_task_tag = 'NTFD Random'
# Task supplying the predictive-timing network (Encoding = Beat + Interval,
# Random excluded)
network_task_tag = 'All Main Tasks'

analysis_task_id = {v: k for k, v in tasks.items()}[analysis_task_tag]
network_task_id = {v: k for k, v in tasks.items()}[network_task_tag]

# ---- Conjunction arms (term -> contrast id / name / task / tail) ------
# ids follow build_contrasts('rand_ntfd'); see volume_to_surface.py
TERMS = {
    'aud_eff':   dict(cid=29, name='Auditory Random vs Auditory Non-Random',
                      task=analysis_task_tag, side='pos'),
    'vis_eff':   dict(cid=41, name='Visual Random vs Visual Non-Random',
                      task=analysis_task_tag, side='pos'),
    'aud_act':   dict(cid=21, name='Auditory Random',
                      task=analysis_task_tag, side='pos'),
    'vis_act':   dict(cid=33, name='Visual Random',
                      task=analysis_task_tag, side='pos'),
    'aud_deact': dict(cid=20, name='Auditory Non-Random',
                      task=analysis_task_tag, side='neg'),
    'vis_deact': dict(cid=32, name='Visual Non-Random',
                      task=analysis_task_tag, side='neg'),
}

# Predictive-timing network (masking term)
NET_TERM = dict(cid=1, name='Encoding', task=network_task_tag, side='two')

# ---- The three conjunctions (include = AND, exclude = AND-NOT) ---------
# 'display' = the arms whose minimum statistic is shown (one sign each)
CONJUNCTIONS = {
    'cross_modal_activation': dict(
        include=['aud_eff', 'vis_eff', 'aud_act', 'vis_act'],
        exclude=[],
        display=['aud_act', 'vis_act']),
    'cross_modal_deactivation': dict(
        include=['aud_eff', 'vis_eff', 'aud_deact', 'vis_deact'],
        exclude=[],
        display=['aud_deact', 'vis_deact']),
    # Modality-specific activation has two disjoint branches; computed and
    # written separately (auditory-only and visual-only).
    'modality_specific_activation_auditory': dict(
        include=['aud_eff', 'aud_act'],
        exclude=['vis_eff'],
        display=['aud_eff', 'aud_act']),
    'modality_specific_activation_visual': dict(
        include=['vis_eff', 'vis_act'],
        exclude=['aud_eff'],
        display=['vis_eff', 'vis_act']),
}


# %%
# ========================= PATHS / LABELS ==============================

if os.path.isdir('/home/analu/diedrichsen_data/data'):
    base_dir = '/home/analu/diedrichsen_data/data'
else:
    base_dir = '/cifs/diedrichsen/data'

music = os.path.join(base_dir, 'Cerebellum', 'music-sdtb')
derivatives_folder = os.path.join(music, 'derivatives')

# Whole-brain mask for second-level fitting / smoothing containment
FITTING_MASK_PATH = os.path.join(derivatives_folder, 'group', 'anat',
                                 'group_mask_noskull.nii')
# Gray-matter mask for visualization only (stats unchanged)
GM_MASK_PATH = os.path.join(derivatives_folder, 'group', 'anat',
                            'group_mask_gray.nii')

DERIVATIVE_SUBFOLDER = 'ffx_rwls_dbb_hrf128'
FILENAME_TEMPLATE = 'wcon_{cid:04d}.nii'

# ---- Volume results tree (same root as volume_maps.py) ----------------
VOL_RESULTS_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'results', 'parametric_tests', 'volume')
VOL_CONJ_ROOT = os.path.join(VOL_RESULTS_ROOT, analysis_task_id,
                             'conjunctions')

# ---- Surface results tree (same root as volume_to_surface.py) ---------
SURFPARAMETRIC_FOLDER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'results', 'parametric_tests', 'surface')
# group surface files per task (re-used / created by group_surf)
SURF_FOLDER = os.path.join(SURFPARAMETRIC_FOLDER, analysis_task_id,
                           'surface_files')
NET_SURF_FOLDER = os.path.join(SURFPARAMETRIC_FOLDER, network_task_id,
                               'surface_files')
# conjunction flatmaps, sibling of the existing 'contour' folder
SURF_CONJ_IMGS = os.path.join(SURFPARAMETRIC_FOLDER, analysis_task_id,
                              'surface_images', 'conjunction')


# %%
# ============================ RUN ======================================

def main():
    # ---------------------------- VOLUME -------------------------------
    if RUN_VOLUME:
        print('\n[VOLUME] computing conjunction maps')

        # group z-maps for every conjunction arm (shared analysis grid)
        global REF_IMG
        first = next(iter(TERMS.values()))
        REF_IMG = load_or_fit_volume_z(first['cid'], first['name'],
                                       analysis_task_id, SUBJECTS)
        zcache = {}
        for t, d in TERMS.items():
            z_img = load_or_fit_volume_z(d['cid'], d['name'],
                                         {v: k for k, v in tasks.items()}
                                         [d['task']], SUBJECTS, ref_img=REF_IMG)
            zcache[t] = np.asanyarray(z_img.get_fdata(), dtype=float)

        brain = mask_on_grid(FITTING_MASK_PATH, REF_IMG)

        # predictive-timing network (Encoding, two-sided), resampled to grid
        net_img = load_or_fit_volume_z(NET_TERM['cid'], NET_TERM['name'],
                                       network_task_id, SUBJECTS,
                                       ref_img=REF_IMG)
        net_z = np.asanyarray(net_img.get_fdata(), dtype=float)
        net_bool, net_thr = fdr_bool(net_z, brain, 'two', FDR_ALPHA,
                                     thr_override=NET_THRESHOLD_OVERRIDE)
        print(f"[VOLUME] predictive-timing network |z| >= {net_thr:.3f} "
              f"({int(net_bool.sum())} voxels)")

        for category, spec in CONJUNCTIONS.items():
            compute_volume_category(category, spec, zcache, brain, net_bool)

    # ---------------------------- SURFACE ------------------------------
    if RUN_SURFACE:
        print('\n[SURFACE] computing conjunction maps')

        cortex = load_cortex_mask(lh_medial_wall_mask_path,
                                  rh_medial_wall_mask_path)

        zcache = {}
        for t, d in TERMS.items():
            zcache[t] = surface_group_z(
                {v: k for k, v in tasks.items()}[d['task']], d['cid'],
                d['name'], SURF_FOLDER, SUBJECTS,
                compute_individual=COMPUTE_INDIVIDUAL_SURF)

        # predictive-timing network on the surface (read from its own task)
        net_z = surface_group_z(network_task_id, NET_TERM['cid'],
                                NET_TERM['name'], NET_SURF_FOLDER, SUBJECTS,
                                compute_individual=COMPUTE_INDIVIDUAL_SURF)
        net_bool, net_thr = fdr_bool(net_z, cortex, 'two', FDR_ALPHA,
                                     thr_override=NET_THRESHOLD_OVERRIDE)
        enc_zL, enc_zR = np.split(net_z, 2, axis=0)
        print(f"[SURFACE] predictive-timing network |z| >= {net_thr:.3f} "
              f"({int(net_bool.sum())} vertices); "
              f"contour {'on' if PLOT_NETWORK_CONTOUR else 'off'}")

        for category, spec in CONJUNCTIONS.items():
            compute_surface_category(category, spec, zcache, cortex, net_bool,
                                     enc_zL, enc_zR, PLOT_NETWORK_CONTOUR,
                                     net_thr)


if __name__ == '__main__':
    main()