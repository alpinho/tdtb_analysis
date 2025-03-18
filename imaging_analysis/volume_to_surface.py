"""
Script to do the volume to surface projection of data from the
 Music-SDTB project

Author: Ana Luisa Pinho
Email: agrilopi@uwo.ca

Creation: 24th of February 2025
Last Update: March 2025

Compatibility: Python 3.10.14, nilearn 0.11.1

Note: The all pipeline of this script only works for surf_files saved as
      cifti in fs_LR32k.
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import nitools as nt

import plotly.graph_objects as go
import plotly.io as pio

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

from scipy import stats
from nilearn.image import load_img
from nilearn.surface import vol_to_surf
from nilearn.maskers import NiftiMasker
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm.thresholding import fdr_threshold
from Functional_Fusion.util import smooth_fs32k_data
from SUITPy import flatmap


# ========================== FUNCTIONS =================================

def get_imeshes(derivatives_dir, subjects, surfspace='fslr32k'):

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
                    surf_dir, surfspace='fslr32k', save='gifti'):

    # Paths of the NON-NORMALIZED individual contrast map for all subjects
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
        contrast = all_contrasts[contrast_key].replace(' ', '-')
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
                    'sub-{sb:02d}_'.format(sb=sb)
                    + task_key.replace('_', '-')
                    + '_'
                    + contrast.lower()
                    + '_'
                    + surfspace
                    + '.hem-L.func.gii',
                ),
            )
            nib.save(
                GIFTIR,
                os.path.join(
                    surf_dir,
                    'sub-{sb:02d}_'.format(sb=sb)
                    + task_key.replace('_', '-')
                    + '_'
                    + contrast.lower()
                    + '_'
                    + surfspace
                    + '.hem-R.func.gii',
                ),
            )
        else:
            assert save == 'cifti'
            # Create CIFTI
            CIFTI = nt.cifti.join_giftis_to_cifti([GIFTIL, GIFTIR],
                                                  mask=[None, None])
            # Save CIFT file
            nib.save(
                CIFTI,
                os.path.join(
                    surf_dir,
                    f'sub-{sb:02d}_{task_key.replace("_", "-")}_'
                    f'{contrast.lower()}_{surfspace}.dscalar.nii'
                )
            )


def get_isurf_gifti(surf_dir, subjects, task_key, contrast,
                    surfspace='fslr32k'):

    # Paths of individual files per hemisphere
    gifti_left = [
        os.path.join(
            surf_dir,
            f"sub-{sub:02d}_"
            f"{task_key.replace('_', '-')}_"
            f"{contrast}_{surfspace}.hem-L.func.gii",
        )
        for sub in subjects
    ]

    gifti_right = [
        os.path.join(
            surf_dir,
            f"sub-{sub:02d}_"
            f"{task_key.replace('_', '-')}_"
            f"{contrast}_{surfspace}.hem-R.func.gii",
        )
        for sub in subjects
    ]

    return gifti_left, gifti_right


def get_isurf_cifti(surf_dir, subjects, task_key, contrast,
                    surfspace='fslr32k'):

    # Paths of individual files
    cifti_file = [
        os.path.join(
            surf_dir,
            f"sub-{sub:02d}_"
            f"{task_key.replace('_', '-')}_"
            f"{contrast}_{surfspace}.dscalar.nii",
        )
        for sub in subjects
    ]

    return cifti_file


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


def group_surf(surf_dir, subjects, task_key, contrast_tag, surfspace='fslr32k'):

    contrast = contrast_tag.lower().replace(' ', '-')

    # Get paths of files with individual functional data projected on...
    # ... the surface
    cifti_paths = get_isurf_cifti(surf_dir, subjects, task_key, contrast,
                                  surfspace=surfspace)

    # Substitute nan's by zeros and smooth the data
    smoothed_data = np.array([smooth_fs32k_data(cifti_path, smooth=8,
                                                kernel='fwhm',
                                                return_data_only=True)
                              for cifti_path in cifti_paths])

    # Remove the middle dimension
    smoothed_data = np.squeeze(smoothed_data, axis=1)

    # Substitute nan's by 0's
    smoothed_data[np.isnan(smoothed_data)] = 0

    # Calculate the one sample t-test
    tvals, _ = stats.ttest_1samp(smoothed_data, 0, axis=0,
                                 alternative='greater')

    # Compute z-values from t-values
    zvals = zval_conversion(tvals, len(subjects)-1)

    # Substitute nan's by 0's
    zvals[np.isnan(zvals)] = 0

    return zvals


def mask_cortical_activation(activation_data, medial_wall_mask_path):
    """
    Masks activation data using a medial wall mask so that only cortical
    vertices retain their activation. Vertices where the mask is 0
    (non-cortical) will be set to 0.
    
    Parameters:
      activation_data : np.array
          1D array of activation values for each vertex.
      medial_wall_mask_path : str
          Path to the medial wall mask GIFTI file
    
    Returns:
      masked_activation : np.array
          Activation values with non-cortical vertices set to 0.
    """
    
    # Load the medial wall mask
    mask_img = nib.load(medial_wall_mask_path)
    
    # Depending on the file, the data may be stored...
    # ... in the first data array.
    mask_data = mask_img.darrays[0].data
    
    # Ensure mask is boolean
    # (True = cortical, False = non-cortical)
    cortex_mask = mask_data.astype(bool)
    
    # Apply the mask: set activation to 0 for non-cortical vertices.
    masked_activation = activation_data.copy()
    masked_activation[~cortex_mask] = 0
    
    return masked_activation


def whole_brain_fdr(derivatives_dir, subjects, task_key, contrast_key, gmask):

    # Paths of the NORMALIZED individual contrast map for all subjects
    encoding_maps = [os.path.join(derivatives_dir, 'sub-%02d' % sub,
                                  'estimates', task_key, 'ffx_rwls_dbb_hrf128',
                                  'wcon_%04d' % contrast_key + '.nii')
                     for sub in subjects]

    # Create design matrix (one-sample t-test)
    design_matrix = pd.DataFrame([1] * len(encoding_maps), columns=['intercept'])

    # Initialize a NiftiMasker (it will create an implicit mask from the Z-map)
    masker = NiftiMasker(mask_img=gmask)

    # Initialize and fit the SecondLevelModel
    second_level_model = SecondLevelModel(mask_img=masker, smoothing_fwhm=8)
    second_level_model = second_level_model.fit(encoding_maps,
                                                design_matrix=design_matrix)

    # Compute the Z-Map
    z_map = second_level_model.compute_contrast(output_type='z_score')

    # Extract voxel values using fit_transform()
    z_values = masker.fit_transform(z_map)  # Output shape: (1, p)

    # Get FDR threshold at alpha = 0.05 (5% false discovery rate)
    # One side: greater than (so, no need to divide by 2)
    fdr_thresh = fdr_threshold(z_values.ravel(), alpha=0.05)

    # Print the estimated FDR threshold
    print(f'Estimated FDR threshold: {fdr_thresh}')

    return fdr_thresh


def plot_flatmap(stats, threshold, task_key, contrast_tag, output_dir,
                 hemi=['L', 'R'], colormap='viridis', vmax=10):

    contrast = contrast_tag.lower().replace(' ', '-')
    task_name = task_key.replace('_', '-')

    # Set paths
    _base_dir = os.path.dirname(os.path.abspath(__file__))
    _surf_dir = os.path.join(_base_dir, 'fslr32k_meshes')

    borders = {'L': os.path.join(_surf_dir, 'borders', 'fs_LR.32k.L.border'),
               'R': os.path.join(_surf_dir, 'borders', 'fs_LR.32k.R.border')
               }
    underlays = {'L': os.path.join(_surf_dir, 'flat',
                                   'fs_LR.32k.L.shape.gii'),
                 'R': os.path.join(_surf_dir, 'flat',
                                   'fs_LR.32k.R.shape.gii')
                 }
    surfaces = {'L': os.path.join(_surf_dir, 'flat',
                                  'fs_LR.32k.L.flat.surf.gii'),
                'R': os.path.join(_surf_dir, 'flat',
                                  'fs_LR.32k.R.flat.surf.gii')
                }

    # Define figure with two subplots
    fig, axs = plt.subplots(1, len(hemi), figsize=(8, 4),
                            gridspec_kw={'wspace': 0.05})
    for ax, stat, h in zip(axs, stats, hemi):
        plt.sca(ax)
        ax = flatmap.plot(stat,
                          surf=surfaces[h],
                          underlay=underlays[h],
                          undermap='gray',
                          underscale=[-1.5, 1],
                          threshold=threshold,
                          cmap=colormap,
                          borders=borders[h],
                          new_figure=False,
                          frame=None
                          )

    # Define lower bound of color limits
    vmin = threshold

    # Make colorbar
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=colormap)
    cbar = fig.colorbar(sm, ax=[axs[0], axs[1]], orientation='horizontal',
                        fraction=0.05, pad=0.02)

    # Add label below colorbar
    cbar.set_label('Z-values', fontsize=12, labelpad=8)

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
    output_name = \
        f'group_{task_name}_{contrast}_flat_fslr32k.png' if len(hemi) == 2 \
        else f'group_{task_name}_{contrast}_flat_fslr32k_{hemi[0]}.png'
    output_path = os.path.join(output_dir, output_name)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)


def split_and_save_sulc_cifti(cifti_path, output_dir):
    """
    Load a CIFTI-2 dscalar file containing sulcal depth, split the data
    into left and right hemispheres, and save them as GIFTI metric files
    with the correct metadata for Workbench.

    Parameters
    ----------
    cifti_path : str
        Path to the CIFTI file (.dscalar.nii).
    output_dir : str
        Directory where the GIFTI files will be saved.

    The files will be saved as:
      sulc.L.32k_fs_LR.gii   (for CortexLeft)
      sulc.R.32k_fs_LR.gii   (for CortexRight)
    """
    # Load the CIFTI file
    cifti = nib.load(cifti_path)
    data = cifti.get_fdata() # Expected shape: (num_maps, num_vertices)

    # Get the brain models (a list of brain model objects)
    brain_models = cifti.header.get_index_map(1).brain_models

    # Initialize variables for left and right hemisphere data
    lh_data = None
    rh_data = None

    # Loop through the brain models to extract data by hemisphere
    for bm in brain_models:
        if bm.brain_structure.upper() == 'CIFTI_STRUCTURE_CORTEX_LEFT':
            lh_data = data[0, bm.index_offset:bm.index_offset + bm.index_count]
        elif bm.brain_structure.upper() == 'CIFTI_STRUCTURE_CORTEX_RIGHT':
            rh_data = data[0, bm.index_offset:bm.index_offset + bm.index_count]

    if lh_data is None or rh_data is None:
        error_message = (
            "Could not find both left and right cortical data in "
            "the CIFTI file."
        )
        raise ValueError(error_message)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define output file paths
    gifti_l_path = os.path.join(output_dir, 'fs_LR.32k.L.sulc.dscalar.gii')
    gifti_r_path = os.path.join(output_dir, 'fs_LR.32k.R.sulc.dscalar.gii')

    # Define metadata for left and right hemispheres
    meta_l_dict = {
        "Name": "Sulcal Depth",
        "GeometricType": "Metric",
        "Caret-Version": "5.64",
        "encoding": "GZipBase64Binary",
        "structure": "CortexLeft",
    }
    meta_r_dict = {
        "Name": "Sulcal Depth",
        "GeometricType": "Metric",
        "Caret-Version": "5.64",
        "encoding": "GZipBase64Binary",
        "structure": "CortexRight",
    }

    # Convert dictionaries to GiftiMetaData objects
    meta_l = nib.gifti.GiftiMetaData(meta_l_dict)
    meta_r = nib.gifti.GiftiMetaData(meta_r_dict)

    # Create and save left hemisphere GIFTI file
    gifti_img_l = nib.gifti.GiftiImage()
    darray_l = nib.gifti.GiftiDataArray(lh_data.astype(np.float32))
    darray_l.intent = 2005  # NIFTI_INTENT_SHAPE for surface-based data
    darray_l.meta = meta_l
    gifti_img_l.add_gifti_data_array(darray_l)
    nib.save(gifti_img_l, gifti_l_path)

    # Create and save right hemisphere GIFTI file
    gifti_img_r = nib.gifti.GiftiImage()
    darray_r = nib.gifti.GiftiDataArray(rh_data.astype(np.float32))
    darray_r.intent = 2005
    darray_r.meta = meta_r
    gifti_img_r.add_gifti_data_array(darray_r)
    nib.save(gifti_img_r, gifti_r_path)

    print("Saved Left Hemisphere sulc:", gifti_l_path)
    print("Saved Right Hemisphere sulc:", gifti_r_path)


def grid_sample_border_vertices_snap(coords, cell_size=5.0):
    """
    Divide the space into a 3D grid with cells of size `cell_size`
    and for each cell that contains border vertices, compute the centroid
    and then snap it to the nearest original vertex in that cell.
    
    Parameters:
      coords : (N, 3) array of border vertex coordinates.
      cell_size : float, the size of each grid cell
                  (in the same units as your surface).
    
    Returns:
      representative : (M, 3) array of representative vertex coordinates
      (one per occupied cell).
    """
    coords = np.array(coords)
    # Determine grid indices for each vertex.
    min_coords = np.min(coords, axis=0)
    grid_indices = np.floor((coords - min_coords) / cell_size).astype(int)
 
    # Group indices by grid cell.
    cell_dict = {}
    for i, idx in enumerate(grid_indices):
        key = tuple(idx)
        if key not in cell_dict:
            cell_dict[key] = []
        cell_dict[key].append(i)
    
    # For each cell, compute the centroid and then pick the vertex...
    # ... closest to the centroid.
    representative = []
    for key, indices in cell_dict.items():
        cell_points = coords[indices]
        centroid = np.mean(cell_points, axis=0)
        distances = np.linalg.norm(cell_points - centroid, axis=1)
        min_index_in_cell = np.argmin(distances)
        # Snap: choose the actual vertex from the cell.
        representative.append(coords[indices[min_index_in_cell]])
    
    return np.array(representative)


def generate_sphere(res=3, radius=1.0):
    """
    Generate a UV sphere mesh with subdivisions defined by `res`
    and given radius.
    
    Returns:
      verts : numpy array of shape (N,3) for sphere vertices.
      faces : numpy array of shape (M,3) for triangular faces.
    """
    verts = []
    faces = []
    for i in range(res + 1):
        theta = np.pi * i / res  # 0 to pi
        for j in range(2 * res + 1):
            phi = 2 * np.pi * j / (2 * res)  # 0 to 2pi
            x = radius * np.sin(theta) * np.cos(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = radius * np.cos(theta)
            verts.append([x, y, z])
    verts = np.array(verts)
    n_verts_per_row = 2 * res + 1
    for i in range(res):
        for j in range(2 * res):
            idx = i * n_verts_per_row + j
            idx_next = idx + n_verts_per_row
            faces.append([idx, idx + 1, idx_next])
            faces.append([idx + 1, idx_next + 1, idx_next])
    return np.array(verts), np.array(faces, dtype=int)


def replicate_spheres_for_vertices(coords, sphere_verts, sphere_faces):
    """
    Replicate a sphere mesh (sphere_verts, sphere_faces)
    at each coordinate in `coords`.
    
    Returns:
      all_verts : combined vertices from all replicated spheres.
      all_faces : combined faces (with proper index offsets).
    """
    all_verts = []
    all_faces = []
    vert_count = 0
    for c in coords:
        shifted = sphere_verts + c
        all_verts.append(shifted)
        all_faces.append(sphere_faces + vert_count)
        vert_count += len(sphere_verts)
    return np.vstack(all_verts), np.vstack(all_faces)


def plotly_surfmap(
        sulc_path, borders_path, surf_path, data, threshold, outfname,
        gray_scale=[[0, 'rgb(105,105,105)'], [1, 'rgb(211,211,211)']],
        resolution=3, radius=0.5, plot_title=None, cmap='viridis',
        cbar_title='Z-values', cell_size=5.0, marker_size=5, borders=False):
    """
    Generates an interactive Plotly HTML render of an inflated brain
    surface with:
      - Sulcal depth background.
      - Thresholded activation overlay.
      - Border markers computed via grid-based sampling with snapping to
        the nearest border vertex.
    
    Parameters:
      sulc_path: str, path to the sulcal depth GIFTI file (.gii).
      borders_path: str, path to the borders GIFTI file (.label.gii).
      surf_path: str, path to the inflated surface mesh (.surf.gii).
      data: np.array, activation values per vertex.
      threshold: float, activation threshold.
      outfname: str, output filename (without extension) for the HTML
                file.
      gray_scale: list, colorscale for sulcal depth.
      resolution: int, resolution for sphere used as border markers.
      radius: float, radius of the sphere used for border markers.
      plot_title: str, title of the figure.
      cmap: str, colormap for activation overlay.
      cbar_title: str, label for the activation colorbar.
      cell_size: float, grid cell size for sampling border vertices.
      marker_size: int, size (scale factor) for the sphere markers.
    """
    # Load sulcal depth.
    sulc_img = nib.load(sulc_path)
    sulc_data = sulc_img.darrays[0].data

    # Load surface mesh.
    surf_img = nib.load(surf_path)
    surf_coords = surf_img.darrays[0].data  # shape: (N,3)
    faces = surf_img.darrays[1].data        # shape: (M,3)

    # Load border data.
    borders_img = nib.load(borders_path)
    borders_data = borders_img.darrays[0].data
    border_indices = np.where(borders_data > 0)[0]
    # Extract border vertex coordinates.
    border_coords = surf_coords[border_indices]

    # Determine active faces
    # (where all vertices have activation >= threshold).
    active_face_mask = np.all(data[faces] >= threshold, axis=1)
    active_faces = faces[active_face_mask]

    # Create sulcal background surface.
    background_surface = go.Mesh3d(
        x=surf_coords[:, 0],
        y=surf_coords[:, 1],
        z=surf_coords[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        intensity=sulc_data,
        colorscale=gray_scale,
        cmin=np.min(sulc_data),
        cmax=np.max(sulc_data),
        showscale=False,
        opacity=1,
        name='Sulc Background'
    )

    # Create activation overlay.
    activation_surface = go.Mesh3d(
        x=surf_coords[:, 0],
        y=surf_coords[:, 1],
        z=surf_coords[:, 2],
        i=active_faces[:, 0],
        j=active_faces[:, 1],
        k=active_faces[:, 2],
        intensity=data,
        colorscale=cmap,
        cmin=threshold,
        cmax=np.max(data),
        colorbar=dict(title=cbar_title, x=0.8, y=0.5, len=0.5),
        showscale=True,
        opacity=1,
        name='Activation Map'
    )

    # Grid-sample the border vertices:
    # compute one representative per cell,
    # then snap to the nearest vertex in that cell.
    sampled_border_coords = grid_sample_border_vertices_snap(
        border_coords, cell_size=cell_size)

    # Now, generate a sphere mesh (for a single dot).
    sphere_verts, sphere_faces = generate_sphere(res=resolution, radius=radius)
    # Replicate the sphere at each sampled border vertex.
    all_verts, all_faces = replicate_spheres_for_vertices(
        sampled_border_coords, sphere_verts, sphere_faces)
    border_spheres = go.Mesh3d(
        x=all_verts[:, 0],
        y=all_verts[:, 1],
        z=all_verts[:, 2],
        i=all_faces[:, 0],
        j=all_faces[:, 1],
        k=all_faces[:, 2],
        color='black',
        opacity=1.0,
        flatshading=True,
        name='Border Markers',
        showscale=False
    )

    # Combine all layers into a figure.
    if borders:
        fig = go.Figure(data=[background_surface, activation_surface,
                              border_spheres])
    else:
        fig = go.Figure(data=[background_surface, activation_surface])

    fig.update_layout(
        title=plot_title,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )

    # Save as interactive HTML.
    pio.write_html(fig, outfname, auto_open=True)
    print(f"Saved HTML figure: {outfname}.html")


# ============================ INPUTS ===================================

# Subjects without pilot
SUBJECTS = [3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26, 28,
            29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

# Relative path for output folders
surf_folder = 'results/surface_files'
contrasts_folder = 'results/control_contrasts'

task_tag = 'All Tasks'
contrast_name = 'Encoding'

# ========================= PARAMETERS =================================

# Parent directories
home = os.path.expanduser('~')
music = os.path.join(home, 'diedrichsen_data/data/Cerebellum/music-sdtb')
derivatives_folder = os.path.join(music, 'derivatives')
wb_gmask = os.path.join(derivatives_folder, 'group', 'anat',
                        'group_mask_noskull.nii')

# ###################### fs_LR32k Meshes ###############################
fslr32k_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'fslr32k_meshes')
lh_veryinflated = os.path.join(fslr32k_folder, 'templates',
                               'tpl-fs32k_hemi-L_veryinflated.surf.gii')
rh_veryinflated = os.path.join(fslr32k_folder, 'templates',
                               'tpl-fs32k_hemi-R_veryinflated.surf.gii')
sulc_folder = os.path.join(fslr32k_folder, 'sulc')
lr_sulc_path = os.path.join(sulc_folder,
                            'fs_LR.32k.LR.sulc.dscalar.nii')
lh_sulc_path = os.path.join(sulc_folder,
                            'fs_LR.32k.L.sulc.dscalar.gii')
rh_sulc_path = os.path.join(sulc_folder,
                            'fs_LR.32k.R.sulc.dscalar.gii')
lh_borders_path = os.path.join(fslr32k_folder, 'borders',
                               'fs_LR.32k.L.border.label.gii')
rh_borders_path = os.path.join(fslr32k_folder, 'borders',
                               'fs_LR.32k.R.border.label.gii')

mask_suffix = '1'
lh_medial_wall_mask_path = os.path.join(
    fslr32k_folder, 'medialwall_masks',
    'fs_LR.32k.L.medialwall.mask' + mask_suffix + '.gii')
rh_medial_wall_mask_path = os.path.join(
    fslr32k_folder, 'medialwall_masks',
    'fs_LR.32k.R.medialwall.mask' + mask_suffix + '.gii')

# ######################################################################

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

# ============================ RUN =====================================

if __name__ == '__main__':

    # Compute individual gifti/cifti files with the volume to surface...
    # ... projection of the contrast map
    individual_surf(derivatives_folder, SUBJECTS, task_id, contrast_id,
                    surf_folder, surfspace='fslr32k', save='gifti')
    individual_surf(derivatives_folder, SUBJECTS, task_id, contrast_id,
                    surf_folder, surfspace='fslr32k', save='cifti')

    # Compute group func cifti
    z_values = group_surf(surf_folder, SUBJECTS, task_id, contrast_name,
                          surfspace='fslr32k')

    # Split results into the two hemispheres
    zvals_lh = np.split(z_values, 2, axis=0)[0]
    zvals_rh = np.split(z_values, 2, axis=0)[1]

    # For each hemisphere, mask the activation values so that only...
    # ... cortical vertices retain their value.
    zvals_lh_masked = mask_cortical_activation(
        zvals_lh, lh_medial_wall_mask_path)
    zvals_rh_masked = mask_cortical_activation(
        zvals_rh, rh_medial_wall_mask_path)

    # Create and save z-maps gifti files
    cname = contrast_name.replace('_', '-')
    for zm, structure, hemi in zip([zvals_lh_masked, zvals_rh_masked],
                                   ['CortexLeft', 'CortexRight'],
                                   ['lh', 'rh']):
        gifti_img = nt.gifti.make_func_gifti(
            zm, anatomical_struct=structure, column_names=[cname])
        # Save the data
        nib.save(
            gifti_img,
            os.path.join(
                surf_folder,
                'group_'
                + task_id.replace('_', '-')
                + '_'
                + cname.lower().replace(' ', '-')
                + '_'
                + 'fslr32k.' + hemi[0].capitalize() + '.func.gii',
            ),
        )

    # # ################## Plot ##################
    # Note: This plotting only works for surfspace='fslr32k'

    # Create contrasts folder if it does not exist
    os.makedirs(contrasts_folder, exist_ok=True)

    # Open gifti
    zmap_lh = nib.load(
        os.path.join(
            surf_folder,
            'group_'
            + task_id.replace('_', '-')
            + '_'
            + contrast_name.lower().replace(' ', '-')
            + '_'
            + 'fslr32k.L.func.gii',
        )
    )
    zvals_lh_masked = zmap_lh.darrays[0].data
    zmap_rh = nib.load(
        os.path.join(
            surf_folder,
            'group_'
            + task_id.replace('_', '-')
            + '_'
            + contrast_name.lower().replace(' ', '-')
            + '_'
            + 'fslr32k.R.func.gii',
        )
    )
    zvals_rh_masked = zmap_rh.darrays[0].data

    # Compute whole-brain fdr threshold of volumetric data
    thresh = whole_brain_fdr(derivatives_folder, SUBJECTS, task_id,
                             contrast_id, wb_gmask)

    # # ################ Plot static flatmap #############################
 
    split_maps = [zvals_lh_masked, zvals_rh_masked]
    zvals_masked = np.concatenate((zvals_lh_masked, zvals_rh_masked))
    v_max = np.amax(zvals_masked[~np.isnan(zvals_masked)])

    print(f'Maximum Z value is: {v_max}')
    plot_flatmap(split_maps, thresh, task_id, contrast_name,
                 contrasts_folder, hemi=['L', 'R'], colormap='viridis',
                 vmax=v_max)

    # # ################## Plot dynamic map ##############################

    # Create Left and Right sulc gifti files
    # split_and_save_sulc_cifti(lr_sulc_path, sulc_folder)
    
    # Left Hemisphere
    lh_output_path = os.path.join(
        contrasts_folder,
        (
            'group_'
            + task_id.replace('_', '-')
            + '_'
            + contrast_name.lower().replace(' ', '-')
            + '_lh_veryinflated_fslr32k.html'
        ),
    )
    plotly_surfmap(
        sulc_path=lh_sulc_path,
        borders_path=lh_borders_path,
        surf_path=lh_veryinflated,
        data=zvals_lh_masked,
        threshold=thresh,
        outfname=lh_output_path,
        resolution=5,
        radius=.65,
        plot_title=contrast_name + '- Left Hemisphere',
        cmap='viridis',
        cbar_title='Z-values',
        cell_size=3.5,         # adjust this for the desired dot sparsity
        marker_size=3,         # adjust for the size of the dots
        borders=False
        )

    # Right Hemisphere
    rh_output_path = os.path.join(
        contrasts_folder,
        (
            'group_'
            + task_id.replace('_', '-')
            + '_'
            + contrast_name.lower().replace(' ', '-')
            + '_rh_veryinflated_fslr32k.html'
        ),
    )
    plotly_surfmap(
        sulc_path=rh_sulc_path,
        borders_path=rh_borders_path,
        surf_path=rh_veryinflated,
        data=zvals_rh_masked,
        threshold=thresh,
        outfname=rh_output_path,
        resolution=5,
        radius=.65,
        plot_title=contrast_name + '- Right Hemisphere',
        cmap='viridis',
        cbar_title='Z-values',
        cell_size=3.5,         # adjust this for the desired dot sparsity
        marker_size=3,         # adjust for the size of the dots
        borders=False
        )
