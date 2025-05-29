import os
import numpy as np
from nilearn import image

# List of subject IDs
subjects = [3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26, 28,
            29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
# subjects = [3, 26]

# Folder and naming config
mask_folder = 'roi_analyses_rwls_hrf128_wb/all/dorsal_striatum/hos/individual_roi_masks'
roi_name = 'dstr'
hemi = 'bh'

# Loop over each subject
for subject in subjects:
    mask_filename = f'i8a_sub-{subject:02d}_{roi_name}_{hemi}_mask.nii.gz'
    mask_path = os.path.join(mask_folder, mask_filename)

    try:
        # Load mask
        mask_img = image.load_img(mask_path)
        mask_data = mask_img.get_fdata()

        # Count voxels with value 1
        voxel_count = np.sum(mask_data == 1)
        print(f"Subject {subject:02d}: {int(voxel_count)} voxels")
    
    except FileNotFoundError:
        print(f"Subject {subject:02d}: Mask file not found at {mask_path}")
