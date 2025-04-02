"""
Script to project cluster of activation in the basal ganglia with a
refined and smoothed surface mesh for smoother visualization
(no additional smoothing of the projected data is applied).

Author: Ana Luisa Pinho
Email: agrilopi@uwo.ca

Creation: 30th of March 2025
Last Update: April 2025

Compatibility: Python 3.10.16, nilearn 0.11.1
"""

import sys
import os

import nibabel as nib
import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.io as pio
import trimesh

from scipy.ndimage import map_coordinates
from nilearn.image import load_img
from nilearn.surface import load_surf_mesh
from nilearn.maskers import NiftiMasker
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm.thresholding import fdr_threshold

# setting path
sys.path.append('../')
# importing
from utils import zval_conversion


# ========================== FUNCTIONS =================================

def smooth_surf_data_custom(data, coords, faces, n_iter=5):
    """
    Smooth surface data using iterative neighbor averaging.

    Parameters
    ----------
    data : np.ndarray, shape (N,)
        Data values for each vertex.
    coords : np.ndarray, shape (N, 3)
        Coordinates for each vertex.
    faces : np.ndarray, shape (M, 3)
        Face connectivity.
    n_iter : int, optional
        Number of smoothing iterations. Default is 5.

    Returns
    -------
    smoothed_data : np.ndarray, shape (N,)
        Smoothed data.
    """
    n_vertices = coords.shape[0]
    # Build dictionary of neighbors.
    neighbors = {i: set() for i in range(n_vertices)}
    for face in faces:
        v0, v1, v2 = face
        neighbors[v0].update([v1, v2])
        neighbors[v1].update([v0, v2])
        neighbors[v2].update([v0, v1])
    smoothed_data = data.copy()
    for _ in range(n_iter):
        new_data = np.zeros_like(smoothed_data)
        for i in range(n_vertices):
            neigh = list(neighbors[i])
            if neigh:
                new_data[i] = (
                    smoothed_data[i] + np.mean(smoothed_data[neigh])
                ) / 2.0
            else:
                new_data[i] = smoothed_data[i]
        smoothed_data = new_data

    return smoothed_data


def max_preserving_smooth(data, size=3, sigma=1):
    """
    Smooth data with a maximum filter followed by a Gaussian filter,
    so that peak values are preserved while noise is reduced.
    
    Parameters
    ----------
    data : np.ndarray
        1D array of data values (per vertex).
    size : int, optional
        Size of the maximum filter window (default is 3).
    sigma : float, optional
        Standard deviation for the Gaussian filter (default is 1).
        
    Returns
    -------
    smoothed_data : np.ndarray
        The smoothed data array.
    """
    from scipy.ndimage import maximum_filter, gaussian_filter
    data_max = maximum_filter(data, size=size)
    data_smooth = gaussian_filter(data_max, sigma=sigma)
    return data_smooth


def refine_and_smooth_mesh(surf_gii_path, output_path, iterations=2,
                           lamb=0.5, smooth_iters=15):
    """
    Refines a surface mesh by subdividing triangles and applying
    Laplacian smoothing for a smoother visualization.

    Parameters
    ----------
    surf_gii_path : str
        Path to the input surface GIFTI (.surf.gii) file.
    output_path : str
        Path to save the refined & smoothed surface.
    iterations : int, optional
        Number of subdivision iterations (default 2).
    lamb : float, optional
        Smoothing strength for Laplacian smoothing (default 0.5).
    smooth_iters : int, optional
        Number of Laplacian smoothing iterations (default 15).

    Returns
    -------
    refined_coords : np.ndarray
        Smoothed vertex coordinates (float32).
    refined_faces : np.ndarray
        Smoothed faces (int32).
    """
    surf = nib.load(surf_gii_path)
    coords = surf.darrays[0].data
    faces = surf.darrays[1].data.astype(np.int32)
    mesh = trimesh.Trimesh(vertices=coords, faces=faces, process=False)
    for _ in range(iterations):
        mesh = mesh.subdivide()
    trimesh.smoothing.filter_laplacian(mesh, lamb=lamb,
                                       iterations=smooth_iters)
    refined_coords = mesh.vertices.astype(np.float32)
    refined_faces = mesh.faces.astype(np.int32)
    refined_surf = nib.gifti.GiftiImage()
    refined_surf.add_gifti_data_array(nib.gifti.GiftiDataArray(
        data=refined_coords, intent='NIFTI_INTENT_POINTSET'))
    refined_surf.add_gifti_data_array(nib.gifti.GiftiDataArray(
        data=refined_faces, intent='NIFTI_INTENT_TRIANGLE'))
    nib.save(refined_surf, output_path)
    return refined_coords, refined_faces


def build_surf_gii(coord_path, topo_path, output_path, spec_file_path=None,
                   spec_info=None):
    """
    Build a complete surf.gii file by combining coordinate and topology
    GIFTI files, optionally embedding spec information as metadata.

    Parameters
    ----------
    coord_path : str
        Path to the coordinate GIFTI file.
    topo_path : str
        Path to the topology GIFTI file.
    output_path : str
        Path to save the combined surf.gii file.
    spec_file_path : str, optional
        Path to the spec file; if provided, its metadata is merged.
    spec_info : dict or str, optional
        Additional spec information to embed as metadata.

    Returns
    -------
    None
    """
    coord_img = nib.load(coord_path)
    coords = coord_img.darrays[0].data
    topo_img = nib.load(topo_path)
    faces = topo_img.darrays[0].data.astype(np.int32)
    surf_img = nib.gifti.GiftiImage()
    da_coords = nib.gifti.GiftiDataArray(data=coords,
                                         intent='NIFTI_INTENT_POINTSET')
    surf_img.add_gifti_data_array(da_coords)
    da_faces = nib.gifti.GiftiDataArray(data=faces,
                                        intent='NIFTI_INTENT_TRIANGLE')
    surf_img.add_gifti_data_array(da_faces)
    if spec_file_path is not None and os.path.isfile(spec_file_path):
        if spec_file_path.lower().endswith('.gii'):
            try:
                spec_img = nib.load(spec_file_path)
                for key, value in spec_img.meta.items():
                    surf_img.meta[key] = value
            except Exception:
                with open(spec_file_path, 'r') as f:
                    surf_img.meta['SpecContent'] = f.read()
        else:
            with open(spec_file_path, 'r') as f:
                surf_img.meta['SpecContent'] = f.read()
    if spec_info is not None:
        if isinstance(spec_info, dict):
            for key, value in spec_info.items():
                surf_img.meta[key] = str(value)
        else:
            surf_img.meta['SpecInfo'] = str(spec_info)
    nib.save(surf_img, output_path)


def compute_sulc_gii(surf_gii_path, sulc_gii_path):
    """
    Computes sulcal depth (curvature) from a surface mesh and saves it
    as a sulc.gii file.

    Parameters
    ----------
    surf_gii_path : str
        Path to the input surface GIFTI (.surf.gii) file.
    sulc_gii_path : str
        Path to save the sulc GIFTI (.gii) file.
    """
    surf = nib.load(surf_gii_path)
    coords = surf.darrays[0].data
    faces = surf.darrays[1].data.astype(np.int32)
    n_vertices = coords.shape[0]
    # Compute simple curvature (proxy for sulcal depth)
    curvature = np.zeros(n_vertices, dtype=np.float32)
    for face in faces:
        for i in range(3):
            v1, v2 = face[i], face[(i + 1) % 3]
            curvature[v1] += np.linalg.norm(coords[v2] - coords[v1])
            curvature[v2] += np.linalg.norm(coords[v1] - coords[v2])
    curvature /= np.amax(curvature)  # Normalize to [0, 1]
    sulc_img = nib.gifti.GiftiImage()
    sulc_img.add_gifti_data_array(nib.gifti.GiftiDataArray(data=curvature))
    nib.save(sulc_img, sulc_gii_path)


def create_inner_mesh(surf_gii_path, scale=0.95, output_path=None):
    """
    Creates a pseudo inner mesh by shifting the mesh to its centroid,
    scaling inward, then shifting back.

    Parameters
    ----------
    surf_gii_path : str
        Path to the input surface GIFTI (.surf.gii) file.
    scale : float, optional
        Scaling factor (<1 scales inward). Default is 0.95.
    output_path : str, optional
        If provided, saves the inner mesh to this file.

    Returns
    -------
    inner_mesh : nibabel.gifti.GiftiImage
        The pseudo inner mesh.
    """
    surf = nib.load(surf_gii_path)
    coords = surf.darrays[0].data
    faces = surf.darrays[1].data.astype(np.int32)

    # 1) Compute the centroid of the vertices (mean of coords).
    centroid = coords.mean(axis=0)

    # 2) Shift the mesh so that centroid is at origin.
    shifted_coords = coords - centroid

    # 3) Scale the shifted mesh.
    scaled_coords = shifted_coords * scale

    # 4) Shift back by adding the centroid.
    inner_coords = scaled_coords + centroid

    # 5) Build the new GIFTI for the inner mesh.
    inner_mesh = nib.gifti.GiftiImage()
    da_coords = nib.gifti.GiftiDataArray(
        data=inner_coords, intent="NIFTI_INTENT_POINTSET"
    )
    da_faces = nib.gifti.GiftiDataArray(
        data=faces, intent="NIFTI_INTENT_TRIANGLE"
    )
    inner_mesh.add_gifti_data_array(da_coords)
    inner_mesh.add_gifti_data_array(da_faces)

    # 6) Optionally, save if an output path is specified.
    if output_path is not None:
        nib.save(inner_mesh, output_path)

    return inner_mesh


def vol_to_surf(nifti_img,
                outer_mesh,
                inner_mesh,
                method='single_nearest',
                n_samples=20,
                extrapolation_factor=1.0
                ):
    """
    A single function that supports three main volume-to-surface
    projection approaches:
      1) Single-sample nearest neighbor ('single_nearest')
      2) Single-sample trilinear ('single_linear')
      3) Multi-sample line maximum-intensity projection with nearest
         neighbor ('multisample_max_nearest')
      4) Multi-sample line maximum-intensity projection with trilinear
         ('multisample_max_linear')

    Parameters
    ----------
    nifti_img : nibabel.Nifti1Image
        A 3D or 4D volume from which to sample.
    outer_mesh : str or (coords, faces)
        The outer mesh (pial). If a filepath (e.g., a .gii file), loaded
        via nilearn.surface.load_surf_mesh.
    inner_mesh : str or (coords, faces)
        The inner mesh. If a filepath, loaded similarly.
        For single-sample approaches, inner_mesh is ignored.
    method : {"single_nearest", "single_linear",
              "multisample_max_nearest", "multisample_max_linear"}
        Which approach to use:
          - "single_nearest": single-sample from outer coords with
                              nearest neighbor
          - "single_linear": single-sample from outer coords with
                             trilinear
          - "multisample_max_nearest": multi-sample line from
                                       outer->inner with nearest neighbor
                                       at each sample, then take max
          - "multisample_max_linear": multi-sample line from outer->inner
                                      with trilinear at each sample, then
                                      take max
    n_samples : int, optional
        Number of sample points along the line from outer to inner
        (only used if method starts with "multisample_"). Default=20.
    extrapolation_factor : float, optional
        Extends the line beyond the inner mesh. e.g., 1.0 = exactly
        outer->inner, 1.5 = 50% beyond. Only used in multi-sample modes.
        Default=1.0.

    Returns
    -------
    proj_data : np.ndarray, shape (n_vertices,) or (n_vertices, n_volumes)
        Projected intensity for each vertex. If the NIfTI is 4D, shape is
        (n_vertices, n_volumes).
    """

    vol_data = nifti_img.get_fdata()  # 3D or 4D
    affine = nifti_img.affine
    inv_affine = np.linalg.inv(affine)

    # Decide interpolation order
    if method in ("single_nearest", "multisample_max_nearest"):
        order = 0  # nearest neighbor
    elif method in ("single_linear", "multisample_max_linear"):
        order = 1  # trilinear
    else:
        raise ValueError("Invalid method: " + method)

    # Load outer mesh
    outer = load_surf_mesh(outer_mesh)
    outer_coords = outer[0]  # shape=(n_vertices,3)
    n_vertices = outer_coords.shape[0]

    # Prepare output array
    if vol_data.ndim == 3:
        proj_data = np.zeros(n_vertices, dtype=np.float64)
    else:
        # 4D => e.g. shape=(x, y, z, n_volumes)
        n_vols = vol_data.shape[3]
        proj_data = np.zeros((n_vertices, n_vols), dtype=np.float64)

    # Case 1 & 2: Single-sample approach uses just the outer coords
    if method.startswith("single_"):
        # Convert outer_coords to voxel indices
        voxel_coords = nib.affines.apply_affine(inv_affine, outer_coords)
        # Sample the volume using the chosen interpolation order
        sampled_values = map_coordinates(
            vol_data,
            voxel_coords.T,
            order=order,
            mode="nearest"
        )
        # shape => (n_vertices,) or (n_vertices, n_vols) if 4D
        proj_data[:] = sampled_values

    # Case 3 & 4: Multi-sample line approach with maximum-intensity
    #             projection
    else:
        # We need to load the inner mesh
        inner = load_surf_mesh(inner_mesh)
        inner_coords = inner[0]

        # Fractions from 0..extrapolation_factor
        fractions = np.linspace(0, extrapolation_factor, n_samples)

        for v in range(n_vertices):
            pial_pt = outer_coords[v]
            inner_pt = inner_coords[v]

            # line_coords: shape (n_samples, 3)
            line_coords = (
                np.outer(1 - fractions, pial_pt) +
                np.outer(fractions, inner_pt)
            )

            # Convert to voxel indices
            voxel_coords = nib.affines.apply_affine(inv_affine, line_coords)

            # Sample
            sampled_values = map_coordinates(
                vol_data,
                voxel_coords.T,
                order=order,
                mode="nearest"
            )
            # shape => (n_samples,) or (n_samples, n_vols) if 4D

            # Maximum-intensity projection along this line
            if vol_data.ndim == 3:
                proj_data[v] = sampled_values.max()
            else:
                proj_data[v] = sampled_values.max(axis=0)

    return proj_data


def custom_view_surf_with_bg(surf_gii_path, surf_map, bg_map, threshold, vmin,
                             vmax, overlay_colorscale='viridis',
                             bg_colorscale=[[0, "rgb(200,200,200)"],
                                            [1, "rgb(240,240,240)"]],
                             bg_smoothing_iterations=0,
                             lighting_params=None, title="3D Surface"):
    """
    Render a surface with two layers:
      1. A background layer (e.g., sulcal depth) with a lighter gray
         texture.
      2. An activation overlay (only for faces with all
         vertices >= threshold).

    Parameters
    ----------
    surf_gii_path : str
        Path to the surface GIFTI file.
    surf_map : np.ndarray
        Activation data per vertex.
    bg_map : str or np.ndarray
        Background data (e.g., sulcal values); if a string, treated as file
        path.
    threshold : float
        Activation threshold.
    vmin : float
        Minimum intensity for the activation colormap.
    vmax : float
        Maximum intensity for the activation colormap.
    overlay_colorscale : str or list, optional
        Colormap for the activation overlay.
    bg_colorscale : list, optional
        Colormap for the background layer.
    bg_smoothing_iterations : int, optional
        Number of smoothing iterations for background data.
    lighting_params : dict, optional
        Lighting parameters (default provided if None).
    title : str, optional
        Title for the plot.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The 3D surface figure.
    """
    surf = nib.load(surf_gii_path)
    vertices = surf.darrays[0].data
    faces = surf.darrays[1].data.astype(np.int32)
    i = faces[:, 0]
    j = faces[:, 1]
    k = faces[:, 2]

    if isinstance(bg_map, str):
        bg_img = nib.load(bg_map)
        bg_data = bg_img.darrays[0].data
    else:
        bg_data = bg_map

    if bg_smoothing_iterations > 0:
        bg_data = smooth_surf_data_custom(bg_data, vertices, faces,
                                          n_iter=bg_smoothing_iterations)

    # Create background layer.
    background_mesh = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        intensity=bg_data,
        colorscale=bg_colorscale,
        cmin=np.amin(bg_data),
        cmax=np.amax(bg_data),
        showscale=False,
        opacity=1,
        flatshading=False,
        name='Sulc Background'
    )

    active_face_mask = np.all(surf_map[faces] >= threshold, axis=1)
    active_faces = faces[active_face_mask]
    overlay_mesh = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=active_faces[:, 0],
        j=active_faces[:, 1],
        k=active_faces[:, 2],
        intensity=surf_map,
        colorscale=overlay_colorscale,
        cmin=vmin,
        cmax=vmax,
        showscale=True,
        colorbar=dict(title='Activation'),
        opacity=1,
        name='Activation Overlay'
    )

    if lighting_params is None:
        lighting_params = dict(ambient=0.4, diffuse=0.8,
                               specular=0.3, roughness=0.2)
    background_mesh.lighting = lighting_params
    overlay_mesh.lighting = lighting_params

    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, t=50, b=0)
    )
    fig = go.Figure(data=[background_mesh, overlay_mesh], layout=layout)
    return fig


def compute_zmap(derivatives_dir, subjects, task_key, contrast_key, gmask,
                 out_path, threshold=.05):

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

    # Save the Z-map
    z_map.to_filename(out_path)

    # Extract voxel values using fit_transform()
    z_values = masker.fit_transform(z_map)  # Output shape: (1, p)

    # Get FDR threshold at alpha = 0.05 (5% false discovery rate)
    # One side: greater than (so, no need to divide by 2)
    fdr_thresh = fdr_threshold(z_values.ravel(), alpha=threshold)

    # Print the estimated FDR threshold
    print(f'Estimated FDR threshold: {fdr_thresh}')

    # Get maximum peak value
    z_max = np.amax(z_values[~np.isnan(z_values)])

    # Print z_max
    print(f'Maximum Z value is: {z_max}')

    return fdr_thresh, z_max


# ============================ INPUTS ===================================

dstr_meshes_folder = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'dstr_meshes'
)

# Original CARET files (for coordinates, topology, spec)
lh_dstr_coord_path = os.path.join(dstr_meshes_folder, 'lh.striatum.coord.gii')
lh_dstr_topo_path = os.path.join(dstr_meshes_folder, 'lh.striatum.topo.gii')
lh_dstr_spec_path = os.path.join(dstr_meshes_folder, 'lh.striatum.spec.gii')

rh_dstr_coord_path = os.path.join(dstr_meshes_folder, 'rh.striatum.coord.gii')
rh_dstr_topo_path = os.path.join(dstr_meshes_folder, 'rh.striatum.topo.gii')
rh_dstr_spec_path = os.path.join(dstr_meshes_folder, 'rh.striatum.spec.gii')

# Surface and related files
lh_dstr_surf_path = os.path.join(dstr_meshes_folder, 'lh.dstr.surf.gii')
lh_dstr_rs_surf_path = os.path.join(dstr_meshes_folder,
                                    'lh.dstr-refsmooth.surf.gii')
lh_dstr_inner_surf_path = os.path.join(dstr_meshes_folder,
                                       'lh.dstr-inner.surf.gii')
# Compute sulc from the refined surface for consistent vertex count
lh_dstr_sulc_path = os.path.join(dstr_meshes_folder, 'lh.dstr.sulc.gii')

rh_dstr_surf_path = os.path.join(dstr_meshes_folder, 'rh.dstr.surf.gii')
rh_dstr_rs_surf_path = os.path.join(dstr_meshes_folder,
                                    'rh.dstr-refsmooth.surf.gii')
rh_dstr_inner_surf_path = os.path.join(dstr_meshes_folder,
                                       'rh.dstr-inner.surf.gii')
rh_dstr_sulc_path = os.path.join(dstr_meshes_folder, 'rh.dstr.sulc.gii')

# Subjects without pilot
SUBJECTS = [3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26, 28,
            29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

# ###############################################

# Note: These inputs are specific to the projection of the individual
#       ROIs overlay

# # Overlay image (NIfTI)
# activation_map = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)),
#     'roi_analyses_rwls_hrf128_wb_puncorr',
#     'all',
#     'dorsal_striatum',
#     'hos',
#     'overlaid_masks',
#     'i8a_dstr_bh_mask.nii.gz'
# )

# # Output folder for HTML
# outputs_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),
#                               'results', 'irois')
# outfile_prefix = 'iroi_dstr_surf'

# # Define threshold and intensity range.
# THRESHOLD = 1 / len(SUBJECTS)  # vmin
# VMAX = 1

# OVERLAY_CMAP = 'cividis'

# ###############################################

# Note: These inputs are specific to the projection of the contrast
#       "Encoding vs. Rest"

home = os.path.expanduser('~')
derivatives_folder = os.path.join(
    home,
    'diedrichsen_data',
    'data',
    'Cerebellum',
    'music-sdtb',
    'derivatives'
)
group_folder = os.path.join(derivatives_folder, 'group')
wb_gmask_path = os.path.join(group_folder, 'anat', 'group_mask_noskull.nii')

volfile_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'results', 'volume_files')
activation_map = os.path.join(volfile_folder,
                              'group_allmain-tasks_encoding_wb_zmap.nii')

THRESHOLD, VMAX = compute_zmap(derivatives_folder, SUBJECTS, 'allmain_tasks',
                               1, wb_gmask_path, activation_map,
                               threshold=.05)

# Output folder for HTML
outputs_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'results', 'control_contrasts')
outfile_prefix = 'group_allmain-tasks_encoding_dstr'

OVERLAY_CMAP = 'viridis'


# ============================ RUN =====================================

if __name__ == "__main__":

    for coord, topo, spec, surf, rs_surf, inner_surf, sulc, hem in zip(
            [lh_dstr_coord_path, rh_dstr_coord_path],
            [lh_dstr_topo_path, rh_dstr_topo_path],
            [lh_dstr_spec_path, rh_dstr_spec_path],
            [lh_dstr_surf_path, rh_dstr_surf_path],
            [lh_dstr_rs_surf_path, rh_dstr_rs_surf_path],
            [lh_dstr_inner_surf_path, rh_dstr_inner_surf_path],
            [lh_dstr_sulc_path, rh_dstr_sulc_path],
            ['lh', 'rh']):

        # Build the original surface
        build_surf_gii(coord, topo, surf, spec_file_path=spec)
        # Refine and smooth the surface mesh
        refine_and_smooth_mesh(surf, rs_surf, iterations=2, lamb=0.5,
                               smooth_iters=15)
        # Compute sulcal depth from the refined and smoothed surface
        compute_sulc_gii(rs_surf, sulc)
        # Create an inner mesh from the refined and smoothed surface
        _ = create_inner_mesh(rs_surf, scale=.82, output_path=inner_surf)

        # Load the activation map
        dstr_activation_img = load_img(activation_map)
        # Project the overlay volume onto the refined surface.
        # surf_data = vol_to_surf_max(dstr_activation_img, rs_surf, inner_surf,
        #                             n_samples=1000, extrapolation_factor=0.)

        surf_data = vol_to_surf(
            dstr_activation_img,
            rs_surf,
            inner_surf, # ignored for single-sampled methods
            method='multisample_max_linear',
            n_samples=20,
            extrapolation_factor=1.
        )

        # Load sulcal data from the refined sulc file.
        sulc_img = nib.load(sulc)
        sulc_data = sulc_img.darrays[0].data
        if surf_data.shape != sulc_data.shape:
            print(
                f'Shape mismatch: '
                f'surf_data {surf_data.shape} vs. '
                f'sulc_data {sulc_data.shape}'
            )

        # Create the custom Plotly figure.
        fig = custom_view_surf_with_bg(
            surf_gii_path=rs_surf,
            surf_map=surf_data,
            bg_map=sulc_data,
            threshold=THRESHOLD,
            vmin=THRESHOLD,
            vmax=VMAX,
            overlay_colorscale=OVERLAY_CMAP,
            bg_colorscale=[[0, 'rgb(200,200,200)'], [1, 'rgb(240,240,240)']],
            bg_smoothing_iterations=0,
            lighting_params={'ambient': .4, 'diffuse': .8, 'specular': .3,
                             'roughness': .2},
            title='3D Dorsal Striatum'
        )

        # Define output HTML file path.
        output_html_path = os.path.join(
            outputs_folder, outfile_prefix + '_' + hem + '.html')

        pio.write_html(fig, output_html_path)
        print(f'Interactive HTML saved at: {output_html_path}')
        fig.show()
