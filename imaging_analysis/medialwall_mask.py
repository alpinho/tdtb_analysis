"""
Script to generate the medial wall mask

Author: Ana Luisa Pinho
Email: agrilopi@uwo.ca

Creation: 5th of March 2025
Last Update: March 2025

Compatibility: Python 3.10.14
"""

import os
import nibabel as nib
import numpy as np

from nibabel.gifti import GiftiDataArray, GiftiImage


# ========================== FUNCTIONS =================================

def generate_medialwall_mask(pial_path, white_path, thickness_thresh=1.0):
    """
    Generate a medial wall mask based on the distance between the pial
    and white surfaces.
    Vertices with a distance (cortical thickness) greater than
    thickness_thresh are considered cortical; those with a lower distance
    are marked as medial wall (non-cortical).

    Parameters:
      pial_path: str
          Path to the pial surface GIFTI file.
      white_path: str
          Path to the white surface GIFTI file.
      thickness_thresh: float
          Minimum distance required for a vertex to be considered part of
          the cortex.

    Returns:
      mask: np.array of booleans with shape (N,), where True indicates a
            cortical vertex.
    """

    # Load pial and white surfaces
    pial_img = nib.load(pial_path)
    white_img = nib.load(white_path)

    # Extract vertex coordinates
    pial_coords = pial_img.darrays[0].data  # shape: (N, 3)
    white_coords = white_img.darrays[0].data  # shape: (N, 3)

    # Compute Euclidean distances between corresponding vertices
    # (proxy for cortical thickness)
    distances = np.linalg.norm(pial_coords - white_coords, axis=1)

    # Create a binary mask:
    # True for vertices with thickness greater than the threshold
    mask = distances > thickness_thresh

    return mask


def save_mask(mask, output_path):
    """
    Save a boolean mask as a GIFTI file.

    Parameters:
      mask: np.array of booleans.
      output_path: str, file path to save the mask.
    """

    # Convert boolean mask to int32 (0 or 1),...
    # ... since GIFTI supports uint8, int32, and float32
    mask_da = GiftiDataArray(data=mask.astype(np.int32))
    mask_img = GiftiImage(darrays=[mask_da])
    nib.save(mask_img, output_path)

    print(f"Medial wall mask saved at: {output_path}")


# ============================ INPUTS ==================================
# Define your main folder
fslr32k_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'fslr32k_meshes')

# Define the templates folder and surface file names
templates_folder = os.path.join(fslr32k_folder, 'templates')
lh_pial_path = os.path.join(templates_folder,
                            'tpl-fs32k_hemi-L_pial.surf.gii')
lh_white_path = os.path.join(templates_folder,
                             'tpl-fs32k_hemi-L_white.surf.gii')
rh_pial_path = os.path.join(templates_folder,
                            'tpl-fs32k_hemi-R_pial.surf.gii')
rh_white_path = os.path.join(templates_folder,
                             'tpl-fs32k_hemi-R_white.surf.gii')

# Output folder
medialwall_folder = os.path.join(fslr32k_folder, 'medialwall_masks')

threshold = 1.
suffix = '1'


# ============================ RUN =====================================

if __name__ == '__main__':

    # Create output folder if does not exist
    os.makedirs(medialwall_folder, exist_ok=True)

    # --- Generate and save the left medial wall mask ---
    lh_mask = generate_medialwall_mask(
        lh_pial_path, lh_white_path, thickness_thresh=threshold)
    lh_mask_output = os.path.join(
        medialwall_folder, 'fs_LR.32k.L.medialwall.mask' + suffix + '.gii')
    save_mask(lh_mask, lh_mask_output)

    # --- Generate and save the right medial wall mask ---
    rh_mask = generate_medialwall_mask(
        rh_pial_path, rh_white_path, thickness_thresh=threshold)
    rh_mask_output = os.path.join(
        medialwall_folder, 'fs_LR.32k.R.medialwall.mask' + suffix + '.gii')
    save_mask(rh_mask, rh_mask_output)
