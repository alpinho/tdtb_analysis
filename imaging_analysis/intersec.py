"""
Intersect contrast with a binarized atlas

Author: Ana Luisa Pinho
Email: agrilopi@uwo.ca

Creation: 17th of November 2025
Last Update: November 2025

Compatibility: Python 3.10.16, nilearn 0.11.1
"""

import os
from nilearn import image

# ----------------------------
# Paths
# ----------------------------
work_dir = os.path.dirname(os.path.abspath(__file__))

contrast_dir = os.path.join(work_dir, 'results', 'parametric_tests', 'volume', 
                            'allmain_tasks', '1_encoding')
bg_dir = os.path.join(work_dir, 'atlases', 'fsl_atlases')

# ----------------------------
# Input files
# ----------------------------
contrast_file = os.path.join(contrast_dir, '1_encoding_zmap_gmmasked.nii.gz')
bg_mask_file = os.path.join(bg_dir, 'hos_dstr_bh_mask.nii.gz')

# ----------------------------
# Load images
# ----------------------------
contrast_img = image.load_img(contrast_file)
bg_mask_img = image.load_img(bg_mask_file)

# ------------------------------------------------------------------
# Make sure the mask lives in the contrast grid
#  - If shapes differ: resample the MASK to the CONTRAST (nearest)
#  - If shapes match: use the mask as is
# ------------------------------------------------------------------
if contrast_img.shape != bg_mask_img.shape:
    print("Resampling mask to contrast space (nearest)...")
    bg_mask_img = image.resample_to_img(
        bg_mask_img, contrast_img, interpolation="nearest"
    )

# At this point, bg_mask_img has the same shape as contrast_img
bg_mask_data = bg_mask_img.get_fdata() > 0

# ------------------------------------------------------------------
# Intersect: keep contrast values only where mask > 0
#  IMPORTANT: we NEVER resample the contrast here, we only
#  multiply its data array by the mask array.
# ------------------------------------------------------------------
contrast_data = contrast_img.get_fdata()
bg_contrast_data = contrast_data * bg_mask_data  # voxel-wise product

# Create new image in the SAME space as the original contrast
bg_contrast_img = image.new_img_like(contrast_img, bg_contrast_data)

# Save result
output_dir = os.path.join(work_dir, 'results', 'control_contrasts')
out_file = os.path.join(output_dir, 'contrast_basal_ganglia_only.nii.gz')
bg_contrast_img.to_filename(out_file)
print(f"Saved basal ganglia-only contrast to: {out_file}")
