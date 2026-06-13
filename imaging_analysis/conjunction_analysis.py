"""
Script: Cross-modal vs. modality-specific unpredictability maps via CONJUNCTION
        analysis, in both VOLUME and SURFACE (fs_LR32k). Everything is derived
        WITHIN the NTFD-Random GLM; the predictive-timing network appears only
        as an optional display contour, never as a mask.

----------------------------------------------------------------------------
WHY A CONJUNCTION (and not the pooled Random - Non-Random contrast)
----------------------------------------------------------------------------
The pooled Random > Non-Random contrast collapses two distinctions that are
actually orthogonal, and cannot recover either:

  (a) Is the unpredictability EFFECT (Random > Non-Random) present in BOTH
      modalities (cross-modal) or in only ONE (modality-specific)? A pooled
      effect can be driven entirely by one modality, so pooled significance
      does not imply cross-modality, and a purely modality-specific effect can
      fail to survive pooling.

  (b) Relative to REST, is the response an activation or a deactivation? The
      Random - Non-Random difference is orthogonal to the baseline level, so
      its sign says nothing about activation vs. deactivation; that needs the
      condition-vs-Rest maps.

A conjunction under the conjunction-null / minimum-statistic logic (Nichols et
al., 2005) requires each conjoined contrast to be INDIVIDUALLY significant
(a logical AND). That is exactly the cross-modality guarantee, and because each
arm is one-sided it also fixes the activation/deactivation sign. The
conjunctions run on the per-modality condition-vs-Rest contrasts, never on the
pooled difference.

----------------------------------------------------------------------------
ARMS (per-modality, FDR-thresholded; NTFD-Random GLM)
----------------------------------------------------------------------------
  A_eff / V_eff  = Auditory / Visual (Random > Non-Random)          [pos]
  A_act / V_act  = Auditory / Visual (Random  vs Rest)              [pos]
  A_NR  / V_NR   = Auditory / Visual (Non-Random vs Rest)           [neg]
  AxV            = interaction (A_eff - V_eff), built WITHIN subject
                   (con29 - con41) then taken to the group level; used in two
                   tails, 'aud_gt_vis' (>0) and 'vis_gt_aud' (<0).

The interaction is formed at the SUBJECT level and modelled once at the group
level. It is NOT the difference of the two group z-maps (that would discard the
within-subject covariance). con29 - con41 equals wcon_0029 - wcon_0041 because
normalization is linear in the voxel values, so no GLM/normalization re-run is
needed -- only access to each subject's individual contrast images (the two
loaders flagged TODO(paths) below).

----------------------------------------------------------------------------
THE CONJUNCTIONS (intersections of FDR-thresholded maps)
----------------------------------------------------------------------------
Shared cross-modal NETWORK (both modalities activate above rest for the
unpredictable condition; the shared timing network itself, independent of
whether the unpredictability EFFECT is significant). Report it alongside the
modality-specific enhancement to show "shared network + specific bias":
  0. cross_modal_shared_activation = A_act & V_act

Cross-modal (effect in both modalities AND condition-vs-rest in both):
  1. cross_modal_activation    = A_eff & V_eff & A_act & V_act
  2. cross_modal_deactivation  = A_eff & V_eff & A_NR  & V_NR

Cross-modal CROSSOVER (the within-GLM replacement for "beyond predictable
timing"): activated by the unpredictable condition AND suppressed by the
predictable one, in both modalities --
  3. cross_modal_crossover     = A_act & V_act & A_NR & V_NR
A sign dissociation that is opposite the predictive-timing response and is
established entirely within one GLM, with no absence-of-evidence step.

Modality-specific (one modality AND the interaction in that direction -- a
TESTED specificity claim that replaces the old exclusive '~other' mask):
  4. modality_specific_activation_auditory = A_eff & A_act & (AxV>0)
  5. modality_specific_activation_visual   = V_eff & V_act & (AxV<0)
  6. modality_specific_crossover_auditory  = A_act & A_NR & (AxV>0)
  7. modality_specific_crossover_visual    = V_act & V_NR & (AxV<0)

Every '&' is the conjunction (minimum statistic). There is no exclusive masking
and no cross-task masking anywhere in the analysis.

----------------------------------------------------------------------------
PREDICTIVE-TIMING / PREDICTABLE NETWORK -- DISPLAY CONTOUR ONLY
----------------------------------------------------------------------------
No result is masked by any network. For orientation, the surface flatmaps can
outline a reference network (PLOT_NETWORK_CONTOUR), chosen by CONTOUR_SOURCE:

  'encoding'    : the canonical predictive-timing network, Encoding vs Rest
                  (Beat + Interval, Random excluded) pooled over all main tasks;
                  drawn two-sided. Cross-task, so kept purely descriptive.
  'predictable' : the WITHIN-GLM predictable response, Non-Random vs Rest from
                  the NTFD-Random task itself; drawn one-sided positive
                  (Non-Random > Rest). Same GLM as the arms, and the crossover
                  categories fall outside it by construction (they require
                  Non-Random < Rest), so the outline visually confirms them.

CONJUNCTION_CONTOUR maps each category to a contour key ('pooled' for cross-
modal, 'auditory'/'visual' for the modality-specific ones); CONTOUR_TERMS holds
the actual contrast (cid / side / threshold) for each key under both sources.

A minimum cluster-EXTENT filter (MIN_CLUSTER_MM3 / MIN_CLUSTER_VERTICES) is
applied to every result, since voxel/vertex-wise FDR does not control extent.

----------------------------------------------------------------------------
OUTPUTS
----------------------------------------------------------------------------
Volume  : results/parametric_tests/volume/<task>/conjunctions/<category>/
            <category>_zmap.nii.gz   (signed minimum-statistic map)
            <category>_mask.nii.gz   (binary surviving conjunction)
            <category>_glassbrain.png
Surface : results/parametric_tests/surface/<task>/surface_images/
          conjunction/<category>/
            group_<task>_<category>_flat_contour_fslr32k*.png
          (filled flatmap of the conjunction statistic, with the chosen
           reference network drawn as a contour)

The displayed statistic is the conjunction-null minimum statistic: the least
significant of the conjoined arms shown (min z for activations, max z i.e.
nearest-zero for deactivations), so an element is shown only where every
displayed arm individually survives FDR.

Author: Ana Luisa Pinho
Email: agrilopi@uwo.ca

Creation: 10th of June 2026
Last Update: June 2026

Compatibility: Python 3.10.14, nilearn 0.11.1

Note: Generated to match volume_maps.py / volume_to_surface.py
"""

import os
import inspect
import numpy as np
import pandas as pd
import nibabel as nib

from scipy import ndimage, stats
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm.thresholding import fdr_threshold
from nilearn.image import resample_to_img, load_img, math_img

# Re-use the project's machinery (resource paths inside these functions
# resolve relative to volume_to_surface.py / ols_permutation_tests.py).
import volume_to_surface as _vts
from ols_permutation_tests import plot_glass_brain_z
from volume_to_surface import (
    build_contrasts,
    individual_surf,
    group_surf,
    get_isurf_cifti,
    mask_cortical_activation,
    plot_flatmap,
    lh_medial_wall_mask_path,
    rh_medial_wall_mask_path,
)
# Same per-subject surface smoothing used inside group_surf (8 mm fwhm), so the
# interaction arms are built on identical footing with the other surface arms.
from Functional_Fusion.util import smooth_fs32k_data

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


# ---- Interaction A_eff - V_eff (con29 - con41), within subject ---------
# The modality x structure interaction is built WITHIN each subject as
# con(cid_a) - con(cid_b) and then modelled once at the group (second) level.
# It CANNOT be formed by subtracting the two group z-maps: that discards the
# within-subject covariance between the two contrasts. We therefore need each
# subject's individual contrast images.
#
# con(cid_a) - con(cid_b) equals wcon_{a} - wcon_{b}: normalization is linear in
# the voxel values, so subtracting the already-normalized images equals
# normalizing the native difference -- no GLM/normalization re-run is required.
#
# >>> Fill in the two loaders below for your file layout. Once they return the
#     right images/arrays, the interaction arms ('aud_gt_vis' / 'vis_gt_aud')
#     compute automatically; nothing else needs editing. <<<

def subject_contrast_volume_path(subject, cid, task_key):
    """Path to <subject>'s NORMALIZED individual contrast image `cid` (the wcon).

    Reuses subject_contrast_paths(), so the interaction is built from exactly the
    same per-subject images the other volume arms use at the second level:
    per subject, wcon_{a} - wcon_{b} IS the normalized within-subject interaction
    (normalization is linear in the voxel values). If your individual wcons live
    elsewhere, fix subject_contrast_paths() once and both routes follow.
    NOTE: only exercised when RUN_VOLUME=True; verify this path if you have only
    ever loaded precomputed group z-maps (LOAD_PRECOMPUTED_VOLUME_ZMAPS)."""
    return subject_contrast_paths(derivatives_folder, [subject], task_key, cid)[0]


def subject_contrast_surface(subject, cid, task_key):
    """Return <subject>'s individual contrast `cid`, projected to fs_LR32k and
    smoothed exactly as group_surf does (8 mm fwhm), as a concatenated (L, R)
    vertex vector (medial wall still included; masked downstream).

    Reads the SAME per-subject .dscalar.nii that every other surface arm uses
    and applies the SAME smoothing, so the interaction is on identical footing
    with them; the only difference is that here the per-subject maps are
    subtracted (cid_a - cid_b) before the one-sample t. The contrast name is
    taken from build_contrasts() so the filename matches what individual_surf
    wrote. Requires those per-subject ciftis to exist (they do for cid 29 / 41,
    since aud_eff / vis_eff are themselves arms and were just grouped above)."""
    cname = build_contrasts(task_key)[cid]
    contrast = cname.replace(' vs ', '_vs_').replace(' ', '-').lower()
    cifti_path = get_isurf_cifti(SURF_FOLDER, [subject], task_key, cid,
                                 contrast, surfspace='fslr32k')[0]
    data = smooth_fs32k_data(cifti_path, smooth=8, kernel='fwhm',
                             return_data_only=True)
    data = np.squeeze(np.asarray(data, dtype=float))
    return np.nan_to_num(data, nan=0.0)


def onesample_t_to_z(X):
    """Vertexwise one-sample t across subjects (axis 0) -> signed z.

    X has shape (n_subjects, n_vertices). Constant/empty columns yield z = 0.
    Mirrors a second-level intercept-only model, but on surface arrays."""
    n = X.shape[0]
    mean = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0, ddof=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        t = mean / (sd / np.sqrt(n))
    t = np.where(np.isfinite(t), t, 0.0)
    p = stats.t.sf(np.abs(t), df=n - 1) * 2.0                 # two-sided
    p = np.clip(p, np.finfo(float).tiny, 1.0)
    z = np.sign(t) * stats.norm.isf(p / 2.0)                  # signed z
    return np.where(np.isfinite(z), z, 0.0)


def interaction_group_zmap_volume(cid_a, cid_b, task_key, subjects,
                                  ref_img=None):
    """Group volume z-map of the within-subject interaction con_a - con_b.

    Per subject: load the two individual contrast images and subtract; then a
    one-sample (intercept-only) second-level model -> z. Same fitting mask and
    smoothing as the other arms. Resampled to ref_img if given."""
    diff_imgs = []
    for s in subjects:
        a = nib.load(subject_contrast_volume_path(s, cid_a, task_key))
        b = nib.load(subject_contrast_volume_path(s, cid_b, task_key))
        diff_imgs.append(math_img("a - b", a=a, b=b))
    design = pd.DataFrame({'intercept': [1] * len(diff_imgs)})
    slm = SecondLevelModel(mask_img=FITTING_MASK_PATH,
                           smoothing_fwhm=SMOOTHING_FWHM)
    slm = slm.fit(diff_imgs, design_matrix=design)
    z_img = slm.compute_contrast(output_type='z_score')
    if ref_img is not None and z_img.shape != ref_img.shape:
        z_img = resample_to_img(z_img, ref_img, interpolation='continuous',
                                force_resample=True, copy_header=True)
    return z_img


def interaction_group_zmap_surface(cid_a, cid_b, task_key, subjects):
    """Group surface z-map of the within-subject interaction con_a - con_b.

    Per subject: subtract the two individual surface contrasts; then a one-
    sample t -> z across subjects, medial-wall masked. Surface projection is
    linear, so the surface of the native difference equals the difference of the
    projected contrasts."""
    diffs = []
    for s in subjects:
        a = np.asarray(subject_contrast_surface(s, cid_a, task_key), dtype=float)
        b = np.asarray(subject_contrast_surface(s, cid_b, task_key), dtype=float)
        diffs.append(a - b)
    z = onesample_t_to_z(np.stack(diffs, axis=0))
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

def compute_volume_category(category, spec, zcache, brain):
    """Build one conjunction in the volume and write the signed minimum-
    statistic z-map, the binary surviving mask, and the glass brain. Whole-
    brain: no network masking is applied.

    Steps: threshold each arm (fdr_bool, per side) -> AND/AND-NOT (conjoin) ->
    cluster-extent filter -> save."""
    bools = {t: fdr_bool(zcache[t], brain, TERMS[t]['side'], FDR_ALPHA)[0]
             for t in set(spec['include'] + spec['exclude'] + spec['display'])}
    cat = conjoin(bools, spec['include'], spec['exclude'])
    keep, min_vox = cluster_filter_volume(cat, REF_IMG, MIN_CLUSTER_MM3)

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
        title=category.replace('_', ' '),
        out_png=png,
        cbar_contrast_label=cap_label(category.replace('_', ' ')),
    )
    print(f"[volume] {category}: {int(keep.sum())} voxels "
          f"(>= {min_vox}-voxel clusters) -> {z_path}")


def compute_surface_category(category, spec, zcache, cortex, net_contour,
                             adjL, adjR):
    """Same conjunction on the fs_LR32k surface; plot the conjunction statistic
    as a filled flatmap. Whole-brain (no masking). If `net_contour` (a
    concatenated L,R boolean) is given, its boundary is drawn as the reference-
    network outline -- display only, it does not affect the statistic."""
    bools = {t: fdr_bool(zcache[t], cortex, TERMS[t]['side'], FDR_ALPHA)[0]
             for t in set(spec['include'] + spec['exclude'] + spec['display'])}
    cat = conjoin(bools, spec['include'], spec['exclude'])
    keep = cluster_filter_surface(cat, MIN_CLUSTER_VERTICES, adjL, adjR)

    stat = min_statistic(zcache, spec['display'], keep)
    statL, statR = np.split(stat, 2, axis=0)

    is_deact = {TERMS[t]['side'] for t in spec['display']} == {'neg'}
    vmax = float(np.nanmax(np.abs(stat))) if np.any(stat) else 1.0
    # colorbar starts at the display threshold = the lowest surviving |z|
    # (the FDR cut is already applied via `keep`, so nothing below it is shown)
    nz = np.abs(stat[stat != 0])
    thr_disp = float(nz.min()) if nz.size else 1e-6

    # Outline the thresholded reference-network boolean (boundary at 0.5),
    # regardless of its side/threshold. Display only; does not affect the stat.
    if net_contour is not None:
        cbL, cbR = np.split(net_contour.astype(float), 2, axis=0)
        contour_kwargs = dict(contour_stat=[cbL, cbR], contour_threshold=0.5,
                              contour_color='k', contour_linewidth=1.0,
                              contour_positive_only=True)
    else:
        contour_kwargs = {}

    out_dir = os.path.join(SURF_CONJ_IMGS, category)
    os.makedirs(out_dir, exist_ok=True)
    # Record the contour source in the filename (plot_flatmap builds the name
    # from contrast_tag), so 'encoding' and 'predictable' runs don't overwrite.
    surf_tag = category
    if net_contour is not None:
        surf_tag = f"{category}_{CONTOUR_SOURCE_TAG.get(CONTOUR_SOURCE, CONTOUR_SOURCE)}"
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
        contrast_tag=surf_tag,
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

RUN_VOLUME = False
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

# No masking. Conjunctions are reported whole-brain; the predictive-timing /
# predictable network is used ONLY as an optional display contour below.

# Surface DISPLAY only: draw a reference network as a contour on the conjunction
# flatmaps (does not affect any statistic). Choose which network with
# CONTOUR_SOURCE.
PLOT_NETWORK_CONTOUR = True

# Which reference network to outline (see CONTOUR_TERMS in INPUTS):
#   'encoding'    = canonical predictive-timing network, Encoding vs Rest
#                   (Beat+Interval, Random excluded), pooled over all main tasks;
#                   two-sided. Cross-task -> purely descriptive.
#   'predictable' = within-GLM predictable response, Non-Random vs Rest from the
#                   NTFD-Random task itself; one-sided positive. Same GLM as the
#                   arms; the crossover categories sit outside it by construction.
CONTOUR_SOURCE = 'predictable'        # 'encoding' | 'predictable'

# Short tag appended to the conjunction figure filename so different
# CONTOUR_SOURCE runs are written side by side instead of overwriting.
CONTOUR_SOURCE_TAG = {'encoding': 'encoding', 'predictable': 'nonrandom'}

# Surface DISPLAY only: draw the fs_LR32k anatomical borders (the dotted sulcal
# landmark lines) on the conjunction flatmaps. Purely cosmetic.
SHOW_FLATMAP_BORDERS = False

# Contour network per conjunction: each category -> a contour KEY, resolved
# within the chosen CONTOUR_SOURCE catalog (CONTOUR_TERMS). 'pooled' for the
# cross-modal categories, 'auditory' / 'visual' for the modality-specific ones.
# The side and threshold live with each contrast in CONTOUR_TERMS.
CONJUNCTION_CONTOUR = {
    'cross_modal_shared_activation':         'pooled',
    'cross_modal_activation':                'pooled',
    'cross_modal_deactivation':              'pooled',
    'cross_modal_crossover':                 'pooled',
    'modality_specific_activation_auditory': 'auditory',
    'modality_specific_activation_visual':   'visual',
    'modality_specific_crossover_auditory':  'auditory',
    'modality_specific_crossover_visual':    'visual',
}


# %%
# ====================== RUN CONFIGURATION ==============================
# Genuine run-level choices (whom to analyse, which task supplies the arms vs
# the contour network). The fixed analysis design is defined further below.

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


# %%
# ================= FIXED ANALYSIS DEFINITIONS ==========================
# NOT tuning knobs. The arms (TERMS), the conjunction definitions
# (CONJUNCTIONS), and the contour catalogs (CONTOUR_TERMS) below are fixed by
# the MEANING of each conjunction and are documented in full in the top
# docstring. Each arm's contrast id, task, and tail follow directly from the
# Nichols conjunction-null logic; editing them changes what the analysis tests,
# not how it is run. They are kept here (rather than hidden) only so the design
# is auditable in one place -- treat them as read-only.

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
    # Modality x structure interaction AxV = A_eff - V_eff = con29 - con41,
    # built WITHIN subject then taken to the group level (see
    # interaction_group_zmap_*). One map, two tails: 'aud_gt_vis' = AxV > 0
    # (auditory-leaning), 'vis_gt_aud' = AxV < 0 (visual-leaning). Used to TEST
    # modality specificity, replacing the old exclusive '~other_eff' mask.
    # 'interaction' = (cid_a, cid_b); these arms have no single 'cid'.
    'aud_gt_vis': dict(name='Auditory vs Visual (Random vs Non-Random)',
                       task=analysis_task_tag, side='pos', interaction=(29, 41)),
    'vis_gt_aud': dict(name='Visual vs Auditory (Random vs Non-Random)',
                       task=analysis_task_tag, side='neg', interaction=(29, 41)),
}

# Display-only contour networks. Two catalogs, selected by CONTOUR_SOURCE
# (TOGGLES). CONJUNCTION_CONTOUR maps each category to a key below.
#   'encoding'    = canonical predictive-timing network, Encoding vs Rest
#                   (Beat+Interval, Random excluded), pooled over all main tasks;
#                   two-sided. Cross-task -> purely descriptive.
#   'predictable' = within-GLM predictable response, Non-Random vs Rest from the
#                   NTFD-Random task itself; one-sided positive (NonRandom>Rest).
# Each entry: cid / name / task / side / threshold (None = data-driven qFDR).
CONTOUR_TERMS = {
    'encoding': {
        'pooled':   dict(cid=1, name='Encoding',          task=network_task_tag,
                         side='pos', threshold=None),
        'auditory': dict(cid=2, name='Auditory Encoding', task=network_task_tag,
                         side='pos', threshold=None),
        'visual':   dict(cid=3, name='Visual Encoding',   task=network_task_tag,
                         side='pos', threshold=None),
    },
    'predictable': {
        'pooled':   dict(cid=8,  name='Non-Random',       task=analysis_task_tag,
                         side='pos', threshold=None),
        'auditory': dict(cid=20, name='Auditory Non-Random',
                         task=analysis_task_tag, side='pos', threshold=None),
        'visual':   dict(cid=32, name='Visual Non-Random',
                         task=analysis_task_tag, side='pos', threshold=None),
    },
}

# ---- The conjunctions (include = AND, exclude = AND-NOT) ---------------
# 'display' = the arms whose minimum statistic is shown (one sign each).
# No exclusive masking is used: modality specificity is now TESTED via the
# interaction arm (aud_gt_vis / vis_gt_aud) instead of an exclusive '~' mask.
# The display-only contour for each category is set in TOGGLES
# (CONJUNCTION_CONTOUR + CONTOUR_SOURCE).
CONJUNCTIONS = {
    # ---- shared cross-modal NETWORK: both modalities activate above rest for
    #      the unpredictable condition (Random > rest), regardless of whether
    #      the unpredictability effect is significant. This is the shared timing
    #      network; pair it with the modality-specific enhancement to report
    #      "shared network + specific bias" together.
    'cross_modal_shared_activation': dict(
        include=['aud_act', 'vis_act'],
        exclude=[],
        display=['aud_act', 'vis_act']),
    # ---- cross-modal: effect AND condition-vs-rest in BOTH modalities -----
    'cross_modal_activation': dict(
        include=['aud_eff', 'vis_eff', 'aud_act', 'vis_act'],
        exclude=[],
        display=['aud_act', 'vis_act']),
    'cross_modal_deactivation': dict(
        include=['aud_eff', 'vis_eff', 'aud_deact', 'vis_deact'],
        exclude=[],
        display=['aud_deact', 'vis_deact']),
    # ---- cross-modal CROSSOVER: activated by the unpredictable condition AND
    #      suppressed by the predictable one, in BOTH modalities
    #      (Random>Rest & Non-Random<Rest). A within-GLM sign dissociation,
    #      opposite the predictive-timing response; no absence-of-evidence step.
    #      Display the activation arms (the deactivation arms have opposite sign
    #      and cannot share one minimum statistic).
    'cross_modal_crossover': dict(
        include=['aud_act', 'vis_act', 'aud_deact', 'vis_deact'],
        exclude=[],
        display=['aud_act', 'vis_act']),
    # ---- modality-specific activation: effect + activation in ONE modality AND
    #      a significant interaction in that modality's direction (TESTED
    #      specificity, replacing the old exclusive '~other_eff' mask).
    'modality_specific_activation_auditory': dict(
        include=['aud_eff', 'aud_act', 'aud_gt_vis'],
        exclude=[],
        display=['aud_eff', 'aud_act']),
    'modality_specific_activation_visual': dict(
        include=['vis_eff', 'vis_act', 'vis_gt_aud'],
        exclude=[],
        display=['vis_eff', 'vis_act']),
    # ---- modality-specific CROSSOVER: crossover in one modality AND the
    #      interaction in that direction.
    'modality_specific_crossover_auditory': dict(
        include=['aud_act', 'aud_deact', 'aud_gt_vis'],
        exclude=[],
        display=['aud_act']),
    'modality_specific_crossover_visual': dict(
        include=['vis_act', 'vis_deact', 'vis_gt_aud'],
        exclude=[],
        display=['vis_act']),
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
    task_id = {v: k for k, v in tasks.items()}

    # config check: the contour is display-only (masking has been removed).
    # Every conjunction must map to a key present in the chosen source catalog.
    assert CONTOUR_SOURCE in CONTOUR_TERMS, \
        f"CONTOUR_SOURCE must be one of {list(CONTOUR_TERMS)}"
    contour_terms = CONTOUR_TERMS[CONTOUR_SOURCE]
    for c in CONJUNCTIONS:
        assert c in CONJUNCTION_CONTOUR, \
            f"CONJUNCTION_CONTOUR has no entry for '{c}'"
        key = CONJUNCTION_CONTOUR[c]
        assert key in contour_terms, \
            f"CONTOUR_TERMS['{CONTOUR_SOURCE}'] has no key '{key}' (for '{c}')"
        ct = contour_terms[key]
        assert ct['side'] in ('pos', 'neg', 'two'), \
            f"CONTOUR_TERMS['{CONTOUR_SOURCE}']['{key}']['side'] invalid"
        assert ct['threshold'] is None or isinstance(ct['threshold'],
                                                     (int, float)), \
            f"CONTOUR_TERMS['{CONTOUR_SOURCE}']['{key}']['threshold'] invalid"

    # ---------------------------- VOLUME -------------------------------
    if RUN_VOLUME:
        print('\n[VOLUME] computing conjunction maps (whole-brain, no masking)')

        # group z-maps for every conjunction arm (shared analysis grid). The
        # first single-contrast arm sets the reference grid.
        global REF_IMG
        first = next(d for d in TERMS.values() if 'interaction' not in d)
        REF_IMG = load_or_fit_volume_z(first['cid'], first['name'],
                                       task_id[first['task']], SUBJECTS)
        zcache = {}
        int_cache = {}
        for t, d in TERMS.items():
            if 'interaction' in d:
                pair = d['interaction']
                if pair not in int_cache:
                    int_cache[pair] = interaction_group_zmap_volume(
                        pair[0], pair[1], task_id[d['task']], SUBJECTS,
                        ref_img=REF_IMG)
                z_img = int_cache[pair]
            else:
                z_img = load_or_fit_volume_z(d['cid'], d['name'],
                                             task_id[d['task']], SUBJECTS,
                                             ref_img=REF_IMG)
            zcache[t] = np.asanyarray(z_img.get_fdata(), dtype=float)

        brain = mask_on_grid(FITTING_MASK_PATH, REF_IMG)

        for category, spec in CONJUNCTIONS.items():
            compute_volume_category(category, spec, zcache, brain)

    # ---------------------------- SURFACE ------------------------------
    if RUN_SURFACE:
        print('\n[SURFACE] computing conjunction maps (whole-brain, no masking)')

        cortex = load_cortex_mask(lh_medial_wall_mask_path,
                                  rh_medial_wall_mask_path)

        zcache = {}
        int_cache = {}
        for t, d in TERMS.items():
            if 'interaction' in d:
                pair = d['interaction']
                if pair not in int_cache:
                    int_cache[pair] = interaction_group_zmap_surface(
                        pair[0], pair[1], task_id[d['task']], SUBJECTS)
                zcache[t] = int_cache[pair]
            else:
                zcache[t] = surface_group_z(
                    task_id[d['task']], d['cid'], d['name'], SURF_FOLDER,
                    SUBJECTS, compute_individual=COMPUTE_INDIVIDUAL_SURF)

        # reference-network z-maps for the DISPLAY contour (no masking). Cached
        # per distinct contour key; the surface-files folder follows the
        # contour contrast's task.
        net_z_surf = {}
        if PLOT_NETWORK_CONTOUR:
            for key in {CONJUNCTION_CONTOUR[c] for c in CONJUNCTIONS}:
                ct = contour_terms[key]
                surf_dir = (NET_SURF_FOLDER if ct['task'] == network_task_tag
                            else SURF_FOLDER)
                net_z_surf[key] = surface_group_z(
                    task_id[ct['task']], ct['cid'], ct['name'], surf_dir,
                    SUBJECTS, compute_individual=COMPUTE_INDIVIDUAL_SURF)
            print(f"[SURFACE] contour on, source = '{CONTOUR_SOURCE}'")
        else:
            print('[SURFACE] contour off')

        # vertex adjacency for surface cluster filtering (built once)
        n_vert = next(iter(zcache.values())).shape[0] // 2
        if MIN_CLUSTER_VERTICES and MIN_CLUSTER_VERTICES > 0:
            adjL, adjR = build_surface_adjacency(n_vert)
        else:
            adjL, adjR = None, None

        for category, spec in CONJUNCTIONS.items():
            net_contour = None
            if PLOT_NETWORK_CONTOUR:
                key = CONJUNCTION_CONTOUR[category]
                ct = contour_terms[key]
                nb, net_thr = fdr_bool(net_z_surf[key], cortex, ct['side'],
                                       FDR_ALPHA, thr_override=ct['threshold'])
                net_contour = nb
                print(f"[SURFACE] {category}: contour {ct['name']} "
                      f"side={ct['side']} thr={net_thr:.3f} "
                      f"({int(nb.sum())} vertices)")
            compute_surface_category(category, spec, zcache, cortex,
                                     net_contour, adjL, adjR)


if __name__ == '__main__':
    main()