#!/usr/bin/env python3
"""
Compute the BH-FDR q-value corresponding to a certain Z* value, for a 
given Z-map, that are masked beforehand. It uses Nilearn's 
implementation of the Benjamini-Hochberg

Author: Ana Luisa Pinho
Email: agrilopi@uwo.ca

Creation: 20th of January 2026
Last Update: January 2026

Compatibility: Python 3.10.16, nilearn 0.11.1
"""

import os
import numpy as np
from nilearn.glm import fdr_threshold
from nilearn.maskers import NiftiMasker
from scipy.stats import norm


# =======================
# Functions
# =======================

def z_to_p(z, side="one"):
    z = np.asarray(z, dtype=float)
    if side == "one":
        return 1.0 - norm.cdf(z)
    if side == "two":
        return 2.0 * (1.0 - norm.cdf(np.abs(z)))
    raise ValueError("side must be 'one' or 'two'.")


def bh_qvalues(p):
    """Benjamini–Hochberg adjusted p-values (q-values)."""
    p = np.asarray(p, dtype=float)
    valid = np.isfinite(p)
    q = np.full_like(p, np.nan, dtype=float)

    pv = p[valid]
    m = pv.size
    if m == 0:
        return q

    order = np.argsort(pv)
    ps = pv[order]
    ranks = np.arange(1, m + 1, dtype=float)

    qs = ps * m / ranks
    qs = np.minimum.accumulate(qs[::-1])[::-1]
    qs = np.clip(qs, 0.0, 1.0)

    qv = np.empty_like(pv)
    qv[order] = qs
    q[valid] = qv
    return q


def q_for_zstar(z_values, z_star, side="one"):
    """
    Return q(Z*) under BH-FDR computed from the whole Z-map.
    """
    z = z_values[np.isfinite(z_values)]
    p = z_to_p(z, side=side)
    q = bh_qvalues(p)

    p_star = float(z_to_p(np.array([z_star]), side=side)[0])

    order = np.argsort(p)
    p_sorted = p[order]
    q_sorted = q[order]

    idx = int(np.searchsorted(p_sorted, p_star, side="left"))
    q_star = 1.0 if idx >= p_sorted.size else float(q_sorted[idx])

    return q_star, p_star


def compute_q_from_paths(zmap_path, mask_path, z_star,
                         side="one", alpha=0.05):
    masker = NiftiMasker(mask_img=mask_path)
    z_vals = masker.fit_transform(zmap_path).ravel()
    z_vals = z_vals[np.isfinite(z_vals)]

    z_thr = float(fdr_threshold(z_vals, alpha=alpha))
    q_star, p_star = q_for_zstar(z_vals, z_star, side=side)

    return z_thr, q_star, p_star, z_vals


# =======================
# PARAMETERS 
# =======================

# Parent directories
home_dir = os.path.expanduser("~")
inputs_dir = os.path.join(home_dir,
                          'mygit', 'music_sdtb', 'tdtb_analysis',
                          'imaging_analysis', 'results')

group_functional = os.path.join(inputs_dir, 'parametric_tests', 'volume',
                                'allmain_tasks')
group_anat = os.path.join(inputs_dir, 'spm_volume_files', 'group_anat')

# =======================
# Inputs
# =======================
zmap_path = os.path.join(group_functional, 
                         '1_encoding', '1_encoding_zmap.nii.gz')
wb_gmask = os.path.join(group_anat, 'group_mask_noskull.nii')

# Z value to convert
z_star = 6.8

# "one" = one-sided (greater-than), "two" = two-sided
side = "one"

# Only used to report the corresponding FDR Z-threshold
alpha = 0.05


# =======================
# Run
# =======================
if __name__ == "__main__":

    z_thr, q_star, puncorr_star, z_vals = compute_q_from_paths(
        zmap_path=zmap_path,
        mask_path=wb_gmask,
        z_star=z_star,
        side=side,
        alpha=alpha,
    )

    print("=== Nilearn voxel-wise BH-FDR ===")
    print(f"Z-map: {zmap_path}")
    print(f"Mask:  {wb_gmask}")
    print(f"Voxels in mask: {z_vals.size}")
    print(f"Z range: [{z_vals.min():.4f}, {z_vals.max():.4f}]")
    print(f"Side: {side}")
    print(f"FDR Z-threshold (alpha={alpha}): {z_thr:.6f}")
    print(f"Z*={z_star:.6f} -> p={puncorr_star:.6e} -> q={q_star:.6f}")