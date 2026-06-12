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
MASKING / RELATING TO THE PREDICTIVE-TIMING NETWORK (optional)
----------------------------------------------------------------------------
The predictive-timing network is Encoding vs. Rest = Beat + Interval, pooled
over Production / Perception / NTFD (the 'All Main Tasks' Encoding contrast);
it deliberately EXCLUDES the Random condition. It is built two-sided:

  NET = ( |z(Encoding, All Main Tasks)| >= FDR threshold )

Each conjunction is related to its own network, set explicitly per case in
CONJUNCTION_NETWORK: which Encoding variant ('network': pooled / Auditory /
Visual), which tail defines it ('side': 'pos' = activations, 'neg' =
deactivations, 'two' = both), and the cut ('threshold': None = data-driven
qFDR, or a forced |z|). The cross-modal maps default to pooled Encoding,
two-sided, qFDR; the modality-specific maps to their own modality's network
(the pooled Encoding cancels modality-specific cortex). The SAME thresholded
network is used for both the mask and the flatmap contour, so the outline
traces exactly the masking region.

Masking is OPTIONAL and OFF by default (MASK_BY_NETWORK). When on, each result
C is intersected with ~NET (beyond timing = C & ~NET). Note this "beyond"
statement rests on absence of evidence (Encoding sub-threshold is not evidence
the timing response is absent), which is why it is not run by default; the
network can instead be shown only as a display contour (PLOT_NETWORK_CONTOUR).

A minimum cluster-EXTENT filter (MIN_CLUSTER_MM3 / MIN_CLUSTER_VERTICES) is
applied to every result, since voxel/vertex-wise FDR does not control extent.

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
import inspect
import numpy as np
import pandas as pd
import nibabel as nib

from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm.thresholding import fdr_threshold
from nilearn.image import resample_to_img, load_img

# Re-use the project's machinery (resource paths inside these functions
# resolve relative to volume_to_surface.py / ols_permutation_tests.py).
import volume_to_surface as _vts
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

# plot_flatmap gained a `show_borders` parameter; detect whether the installed
# volume_to_surface.py has it, so the script runs either way (the border toggle
# only takes effect once that file is updated).
_PLOT_FLATMAP_SUPPORTS_BORDERS = (
    'show_borders' in inspect.signature(plot_flatmap).parameters)


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


# ---- Cluster-extent filtering -----------------------------------------
# A voxel/vertex-wise FDR conjunction controls the per-element false-discovery
# rate but NOT spatial extent, so isolated elements can survive with no spatial
# support. We drop connected components smaller than a minimum extent. Volume
# extent is specified physically (mm^3) and converted to voxels at the data
# resolution so it is independent of the grid; surface extent is in vertices.

# fs_LR32k flat meshes, resolved exactly as plot_flatmap does
VTS_MESH_DIR = os.path.join(os.path.dirname(os.path.abspath(_vts.__file__)),
                            'fslr32k_meshes')


def cluster_filter_volume(keep, ref_img, min_mm3):
    """Drop 3-D connected components (26-connectivity) below `min_mm3`.
    Returns (filtered_bool, min_voxels)."""
    if not min_mm3 or min_mm3 <= 0:
        return keep, 0
    vox_mm3 = float(np.prod(ref_img.header.get_zooms()[:3]))
    min_vox = int(np.ceil(min_mm3 / vox_mm3))
    structure = ndimage.generate_binary_structure(3, 3)   # faces+edges+corners
    lab, n = ndimage.label(keep, structure=structure)
    if n == 0:
        return keep, min_vox
    sizes = np.bincount(lab.ravel())
    small = np.where(sizes[1:] < min_vox)[0] + 1           # skip background (0)
    out = keep.copy()
    out[np.isin(lab, small)] = False
    return out, min_vox


def _hemi_vertex_adjacency(hemi, n_vert):
    """Sparse vertex adjacency for one fs_LR32k hemisphere, from the flat-mesh
    triangulation (darrays[1] = faces). Returns None if the mesh cannot be read
    or does not match the data's vertex count (surface clustering then skipped).
    """
    surf = os.path.join(VTS_MESH_DIR, 'flat',
                        f'fs_LR.32k.{hemi}.flat.surf.gii')
    try:
        faces = np.asarray(nib.load(surf).darrays[1].data, dtype=np.int64)
    except Exception as exc:                                # noqa: BLE001
        print(f"[surface] adjacency: cannot read {surf} ({exc}); "
              f"skipping cluster filter for {hemi}")
        return None
    if faces.max() >= n_vert:
        print(f"[surface] adjacency: {hemi} mesh has {faces.max() + 1} vertices "
              f"but data has {n_vert}; skipping cluster filter")
        return None
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]])
    e = np.vstack([e, e[:, ::-1]])                          # undirected
    data = np.ones(e.shape[0], dtype=np.int8)
    return coo_matrix((data, (e[:, 0], e[:, 1])),
                      shape=(n_vert, n_vert)).tocsr()


def build_surface_adjacency(n_vert_per_hemi):
    """(adjL, adjR) sparse adjacencies, or (None, None) if unavailable."""
    return (_hemi_vertex_adjacency('L', n_vert_per_hemi),
            _hemi_vertex_adjacency('R', n_vert_per_hemi))


def _filter_hemi(keep_h, adj, min_vertices):
    if adj is None or keep_h.sum() == 0:
        return keep_h
    idx = np.where(keep_h)[0]
    n_comp, labels = connected_components(adj[idx][:, idx], directed=False)
    out = keep_h.copy()
    for c in range(n_comp):
        members = idx[labels == c]
        if members.size < min_vertices:
            out[members] = False
    return out


def cluster_filter_surface(keep, min_vertices, adjL, adjR):
    """Drop surface connected components (mesh adjacency) below `min_vertices`.
    `keep` is the concatenated (L, R) boolean vector."""
    if not min_vertices or min_vertices <= 0:
        return keep
    n = keep.shape[0] // 2
    kL = _filter_hemi(keep[:n], adjL, min_vertices)
    kR = _filter_hemi(keep[n:], adjR, min_vertices)
    return np.concatenate([kL, kR])


# %%
# ========================== ANALYSIS STEPS =============================

def compute_volume_category(category, spec, zcache, brain, net_bool):
    """Build one conjunction in the volume and write the signed minimum-
    statistic z-map, the binary mask, and the glass brain. If `net_bool` is
    given, the conjunction is additionally intersected with ~NET (kept only
    outside the predictive-timing network); if `net_bool` is None it is left
    whole-brain.

    Steps: threshold each conjoined contrast (fdr_bool, per side) -> AND/AND-NOT
    (conjoin) -> optionally intersect with ~NET -> save."""
    bools = {t: fdr_bool(zcache[t], brain, TERMS[t]['side'], FDR_ALPHA)[0]
             for t in set(spec['include'] + spec['exclude'] + spec['display'])}
    cat = conjoin(bools, spec['include'], spec['exclude'])
    keep = cat if net_bool is None else (cat & ~net_bool)
    keep, min_vox = cluster_filter_volume(keep, REF_IMG, MIN_CLUSTER_MM3)

    stat = min_statistic(zcache, spec['display'], keep)

    out_dir = os.path.join(VOL_CONJ_ROOT, category)
    os.makedirs(out_dir, exist_ok=True)
    if GM_MASK_PATH:
        gm = mask_on_grid(GM_MASK_PATH, REF_IMG)
        stat = np.where(gm, stat, 0.0)

    z_path = os.path.join(out_dir, f"{category}_zmap.nii.gz")
    nib.save(nib.Nifti1Image(stat, REF_IMG.affine, REF_IMG.header), z_path)
    nib.save(nib.Nifti1Image(keep.astype(np.int16), REF_IMG.affine,
                             REF_IMG.header),
             os.path.join(out_dir, f"{category}_mask.nii.gz"))

    is_deact = {TERMS[t]['side'] for t in spec['display']} == {'neg'}
    png = os.path.join(out_dir, f"{category}_glassbrain.png")
    plot_glass_brain_z(
        z_map_path=z_path,
        z_threshold=1e-6,                 # the FDR cut is already in the mask
        two_sided=is_deact,
        title=(f"{category.replace('_', ' ')}"
               + (' (beyond predictive timing)'
                  if net_bool is not None else '')),
        out_png=png,
        cbar_contrast_label=cap_label(category.replace('_', ' ')),
    )
    print(f"[volume] {category}: {int(keep.sum())} voxels "
          f"(>= {min_vox}-voxel clusters) -> {z_path}")


def compute_surface_category(category, spec, zcache, cortex, net_bool,
                             net_contour, adjL, adjR):
    """Same conjunction on the fs_LR32k surface; plot the conjunction statistic
    as a filled flatmap. If `net_bool` is None the conjunction is whole-brain;
    otherwise it is intersected with ~`net_bool`. If `net_contour` (a
    concatenated L,R boolean) is given, its boundary is drawn as the network
    outline -- this is the SAME thresholded network used for masking, so the
    contour traces exactly the masking region."""
    bools = {t: fdr_bool(zcache[t], cortex, TERMS[t]['side'], FDR_ALPHA)[0]
             for t in set(spec['include'] + spec['exclude'] + spec['display'])}
    cat = conjoin(bools, spec['include'], spec['exclude'])
    keep = cat if net_bool is None else (cat & ~net_bool)
    keep = cluster_filter_surface(keep, MIN_CLUSTER_VERTICES, adjL, adjR)

    stat = min_statistic(zcache, spec['display'], keep)
    statL, statR = np.split(stat, 2, axis=0)

    is_deact = {TERMS[t]['side'] for t in spec['display']} == {'neg'}
    vmax = float(np.nanmax(np.abs(stat))) if np.any(stat) else 1.0
    # colorbar starts at the display threshold = the lowest surviving |z|
    # (the FDR cut is already applied via `keep`, so nothing below it is shown)
    nz = np.abs(stat[stat != 0])
    thr_disp = float(nz.min()) if nz.size else 1e-6

    # Outline the thresholded network boolean itself (boundary at 0.5), so the
    # contour is exactly the masking region regardless of its side/threshold.
    if net_contour is not None:
        cbL, cbR = np.split(net_contour.astype(float), 2, axis=0)
        contour_kwargs = dict(contour_stat=[cbL, cbR], contour_threshold=0.5,
                              contour_color='k', contour_linewidth=1.0,
                              contour_positive_only=True)
    else:
        contour_kwargs = {}

    out_dir = os.path.join(SURF_CONJ_IMGS, category)
    os.makedirs(out_dir, exist_ok=True)
    border_kw = {}
    if _PLOT_FLATMAP_SUPPORTS_BORDERS:
        border_kw['show_borders'] = SHOW_FLATMAP_BORDERS
    elif not SHOW_FLATMAP_BORDERS:
        print("[surface] note: installed plot_flatmap has no 'show_borders'; "
              "borders cannot be suppressed until volume_to_surface.py is "
              "updated")
    plot_flatmap(
        stats=[statL, statR],
        threshold=thr_disp,
        task_key=analysis_task_id,
        contrast_tag=category,
        output_dir=out_dir,
        hemi=['L', 'R'],
        colormap=('Blues_r' if is_deact else 'autumn'),
        vmax=vmax,
        signed=is_deact,
        cbar_title='Z-values',
        **border_kw,
        **contour_kwargs,
    )
    print(f"[surface] {category}: {int(keep.sum())} vertices "
          f"(>= {MIN_CLUSTER_VERTICES}-vertex clusters) -> {out_dir}")


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

FDR_ALPHA = 0.05      # BH-FDR level: the conjunction arms (always) and any
                      # network whose threshold is None
SMOOTHING_FWHM = 8.0

# Minimum cluster EXTENT, to drop isolated voxels/vertices that pass the
# voxelwise FDR conjunction but have no spatial support (FDR controls the
# per-element rate, not extent). Volume is given in physical volume (mm^3) and
# converted to a voxel count at the data resolution, so it does not depend on
# the grid; surface is given directly in fs_LR32k vertices. Set either to 0 to
# disable that space's filter. The defaults below are conventional minimal
# extents for whole-brain reporting (small enough to retain genuine clusters,
# large enough to remove specks); adjust if your reporting convention differs.
MIN_CLUSTER_MM3 = 100.0       # volume; ~ a small contiguous cluster
MIN_CLUSTER_VERTICES = 20     # surface (fs_LR32k; ~60 mm^2)

# Mask each conjunction by the predictive-timing network (Encoding vs Rest),
# keeping only what lies OUTSIDE it (`beyond = cat & ~NET`). The "beyond the
# network" statement rests on absence of evidence -- a region being non-
# significant for Encoding is not evidence the timing response is absent -- so
# it is dropped by default: with MASK_BY_NETWORK = False the conjunctions are
# reported whole-brain and the Encoding map is never computed (unless a contour
# is explicitly requested below).
MASK_BY_NETWORK = False

# Surface DISPLAY only: draw the predictive-timing network as a contour on the
# conjunction flatmaps. Independent of the masking. With this and
# MASK_BY_NETWORK both False, the network is dropped from the analysis entirely.
PLOT_NETWORK_CONTOUR = True

# Surface DISPLAY only: draw the fs_LR32k anatomical borders (the dotted sulcal
# landmark lines) on the conjunction flatmaps. Purely cosmetic.
SHOW_FLATMAP_BORDERS = False

# Predictive-timing network related to EACH conjunction. One explicit spec per
# case, and the SAME thresholded network is used for both the mask (~NET, when
# MASK_BY_NETWORK is on) and the flatmap contour (when PLOT_NETWORK_CONTOUR is
# on) -- the contour outlines exactly the masking region.
#   'network'   : which NET_TERMS entry (defined in INPUTS):
#                   'encoding'     = pooled Encoding vs Rest   (Beat+Interval)
#                   'aud_encoding' = Auditory Encoding vs Rest
#                   'vis_encoding' = Visual Encoding vs Rest
#   'side'      : which tail defines the network:
#                   'pos' = z >= +thr        (Encoding activations only)
#                   'neg' = z <= -thr        (Encoding deactivations only)
#                   'two' = |z| >= thr       (both, two-sided)
#   'threshold' : None  -> data-driven BH-FDR (qFDR) threshold on that tail;
#                 float -> forced |z| cut (e.g. 2.7), interpreted per `side`.
CONJUNCTION_NETWORK = {
    'cross_modal_activation':
        dict(network='encoding',     side='pos', threshold=None),
    'cross_modal_deactivation':
        dict(network='encoding',     side='neg', threshold=None),
    'modality_specific_activation_auditory':
        dict(network='aud_encoding', side='pos', threshold=None),
    'modality_specific_activation_visual':
        dict(network='vis_encoding', side='pos', threshold=None),
}


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
# 'side' is INTRINSIC to each arm and fixed by the conjunction's meaning
# (effects/activations one-sided 'pos', deactivations one-sided 'neg'); it is
# not a tunable choice. Every arm is BH-FDR thresholded at FDR_ALPHA -- this
# fixed, FDR-corrected, one-sided thresholding is exactly what the Nichols
# conjunction-null requires, so it is coded directly, not exposed as an input.
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

# Predictive-timing networks (masking / contour terms), all Encoding vs Rest
# (Beat + Interval, Random excluded) from the network task. The tail (`side`)
# and threshold are set per conjunction in CONJUNCTION_NETWORK, not here.
NET_TERMS = {
    'encoding':     dict(cid=1, name='Encoding',          task=network_task_tag),
    'aud_encoding': dict(cid=2, name='Auditory Encoding', task=network_task_tag),
    'vis_encoding': dict(cid=3, name='Visual Encoding',   task=network_task_tag),
}

# ---- The conjunctions (include = AND, exclude = AND-NOT) ---------------
# 'display' = the arms whose minimum statistic is shown (one sign each)
# The network related to each conjunction (mask + contour) is set explicitly in
# the TOGGLES section (CONJUNCTION_NETWORK).
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
    # written separately (auditory-only and visual-only), each related to its
    # own modality's predictive-timing network.
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
    # config check: every conjunction has a valid network spec
    for c in CONJUNCTIONS:
        assert c in CONJUNCTION_NETWORK, \
            f"CONJUNCTION_NETWORK has no entry for '{c}'"
        cn = CONJUNCTION_NETWORK[c]
        assert cn['network'] in NET_TERMS, \
            f"CONJUNCTION_NETWORK['{c}']['network']='{cn['network']}' " \
            f"not in NET_TERMS"
        assert cn['side'] in ('pos', 'neg', 'two'), \
            f"CONJUNCTION_NETWORK['{c}']['side'] must be 'pos'/'neg'/'two'"
        assert cn['threshold'] is None or isinstance(cn['threshold'],
                                                     (int, float)), \
            f"CONJUNCTION_NETWORK['{c}']['threshold'] must be None or a number"

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

        # predictive-timing network z-maps, cached per distinct network and
        # thresholded per case with that case's own side + threshold
        net_z_vol = {}
        if MASK_BY_NETWORK:
            for nk in {CONJUNCTION_NETWORK[c]['network'] for c in CONJUNCTIONS}:
                nt = NET_TERMS[nk]
                net_img = load_or_fit_volume_z(nt['cid'], nt['name'],
                                               network_task_id, SUBJECTS,
                                               ref_img=REF_IMG)
                net_z_vol[nk] = np.asanyarray(net_img.get_fdata(), dtype=float)
        else:
            print('[VOLUME] whole-brain conjunctions (no network masking)')

        for category, spec in CONJUNCTIONS.items():
            net_bool = None
            if MASK_BY_NETWORK:
                cn = CONJUNCTION_NETWORK[category]
                net_bool, net_thr = fdr_bool(net_z_vol[cn['network']], brain,
                                             cn['side'], FDR_ALPHA,
                                             thr_override=cn['threshold'])
                print(f"[VOLUME] {category}: "
                      f"{NET_TERMS[cn['network']]['name']} side={cn['side']} "
                      f"thr={net_thr:.3f} ({int(net_bool.sum())} voxels)")
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

        # predictive-timing network z-maps, cached per distinct network used
        # (as mask and/or contour); thresholded per case below
        net_z_surf = {}
        use_net = MASK_BY_NETWORK or PLOT_NETWORK_CONTOUR
        if use_net:
            for nk in {CONJUNCTION_NETWORK[c]['network'] for c in CONJUNCTIONS}:
                nt = NET_TERMS[nk]
                net_z_surf[nk] = surface_group_z(
                    network_task_id, nt['cid'], nt['name'], NET_SURF_FOLDER,
                    SUBJECTS, compute_individual=COMPUTE_INDIVIDUAL_SURF)
        print(f"[SURFACE] masking {'on' if MASK_BY_NETWORK else 'off'}, "
              f"contour {'on' if PLOT_NETWORK_CONTOUR else 'off'}")

        # vertex adjacency for surface cluster filtering (built once)
        n_vert = next(iter(zcache.values())).shape[0] // 2
        if MIN_CLUSTER_VERTICES and MIN_CLUSTER_VERTICES > 0:
            adjL, adjR = build_surface_adjacency(n_vert)
        else:
            adjL, adjR = None, None

        for category, spec in CONJUNCTIONS.items():
            net_bool, net_contour = None, None
            if use_net:
                cn = CONJUNCTION_NETWORK[category]
                nb, net_thr = fdr_bool(net_z_surf[cn['network']], cortex,
                                       cn['side'], FDR_ALPHA,
                                       thr_override=cn['threshold'])
                if MASK_BY_NETWORK:
                    net_bool = nb           # same thresholded network ...
                if PLOT_NETWORK_CONTOUR:
                    net_contour = nb        # ... used for the contour too
                print(f"[SURFACE] {category}: "
                      f"{NET_TERMS[cn['network']]['name']} side={cn['side']} "
                      f"thr={net_thr:.3f} ({int(nb.sum())} vertices)")
            compute_surface_category(category, spec, zcache, cortex, net_bool,
                                     net_contour, adjL, adjR)


if __name__ == '__main__':
    main()