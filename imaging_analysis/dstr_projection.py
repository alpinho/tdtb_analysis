"""
Script to project cluster of activation in the basal ganglia with a
 refined and smoothed surface mesh for smoother visualization
 (no additional smoothing of the projected data is applied).

Author: Ana Luisa Pinho
Email: agrilopi@uwo.ca

Creation: 30th of March 2025
Last Update: March 2025

Compatibility: Python 3.10.16, nilearn 0.11.1
"""

import os
import nibabel as nib
import numpy as np

import plotly.graph_objects as go
import plotly.io as pio
import trimesh

from scipy.ndimage import map_coordinates
from nilearn.image import load_img
from nilearn.surface import load_surf_mesh


# ========================== HELPER FUNCTIONS ==========================

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
    Smooth data with a maximum filter followed by a Gaussian filter so
    that peak values are preserved while noise is reduced.
    
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


def refine_and_smooth_mesh(surf_gii_path, output_path, iterations=1, lamb=0.5,
                           smooth_iters=10):
    """
    Refines a surface mesh by subdividing its triangles and then applies
    Laplacian smoothing to reduce high-frequency variations, yielding a
    smoother geometry.

    Parameters
    ----------
    surf_gii_path : str
        Path to the input surface GIFTI (.surf.gii) file.
    output_path : str
        Path where the refined & smoothed surface GIFTI file will be
        saved.
    iterations : int, optional
        Number of subdivision iterations to perform. Default is 1.
    lamb : float, optional
        Smoothing strength for Laplacian smoothing. Default is 0.5.
    smooth_iters : int, optional
        Number of Laplacian smoothing iterations. Default is 10.

    Returns
    -------
    refined_coords : np.ndarray
        Refined and smoothed vertex coordinates (float32).
    refined_faces : np.ndarray
        Refined faces (int32).
    """
    surf = nib.load(surf_gii_path)
    coords = surf.darrays[0].data
    faces = surf.darrays[1].data.astype(np.int32)
    mesh = trimesh.Trimesh(vertices=coords, faces=faces, process=False)
    for _ in range(iterations):
        mesh = mesh.subdivide()
    # Apply Laplacian smoothing.
    trimesh.smoothing.filter_laplacian(mesh, lamb=lamb,
                                       iterations=smooth_iters)
    refined_coords = mesh.vertices.astype(np.float32)
    refined_faces = mesh.faces.astype(np.int32)
    refined_surf = nib.gifti.GiftiImage()
    da_coords = nib.gifti.GiftiDataArray(
        data=refined_coords, intent='NIFTI_INTENT_POINTSET'
    )
    da_faces = nib.gifti.GiftiDataArray(
        data=refined_faces, intent='NIFTI_INTENT_TRIANGLE'
    )
    refined_surf.add_gifti_data_array(da_coords)
    refined_surf.add_gifti_data_array(da_faces)
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
    Computes a simple vertex-wise curvature measure
    (proxy for sulcal depth) from a surf.gii file and saves it as a
    sulc.gii file.

    Parameters
    ----------
    surf_gii_path : str
        Path to the input surface GIFTI (.surf.gii) file.
    sulc_gii_path : str
        Path to save the sulc GIFTI (.gii) file.
    """
    surf = nib.load(surf_gii_path)
    if len(surf.darrays) < 2:
        raise ValueError("Surface file must include both vertices and faces.")
    coords = surf.darrays[0].data
    faces = surf.darrays[1].data.astype(np.int32)
    n_vertices = coords.shape[0]
    neighbors = {i: [] for i in range(n_vertices)}
    for face in faces:
        for i in range(3):
            v1 = face[i]
            v2 = face[(i + 1) % 3]
            neighbors[v1].append(v2)
            neighbors[v2].append(v1)
    curvature = np.zeros(n_vertices, dtype=np.float32)
    for i in range(n_vertices):
        neigh_idx = np.array(neighbors[i])
        if neigh_idx.size > 0:
            diff = coords[neigh_idx] - coords[i]
            curvature[i] = np.linalg.norm(np.mean(diff, axis=0))
        else:
            curvature[i] = 0.0
    sulc_img = nib.gifti.GiftiImage()
    sulc_data = nib.gifti.GiftiDataArray(data=curvature)
    sulc_img.add_gifti_data_array(sulc_data)
    nib.save(sulc_img, sulc_gii_path)


def create_inner_mesh(surf_gii_path, scale=0.95, output_path=None):
    """
    Creates a pseudo inner mesh by scaling vertex coordinates inward.

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
    inner_coords = coords * scale
    inner_mesh = nib.gifti.GiftiImage()
    da_coords = nib.gifti.GiftiDataArray(data=inner_coords,
                                         intent='NIFTI_INTENT_POINTSET')
    da_faces = nib.gifti.GiftiDataArray(data=faces,
                                        intent='NIFTI_INTENT_TRIANGLE')
    inner_mesh.add_gifti_data_array(da_coords)
    inner_mesh.add_gifti_data_array(da_faces)
    if output_path is not None:
        nib.save(inner_mesh, output_path)

    return inner_mesh


def vol_to_surf_max(nifti_img, surf_mesh, inner_mesh, n_samples=20,
                    extrapolation_factor=1.0):
    """
    Custom volume-to-surface projection using maximum intensity sampling.
    Samples along the line from pial to inner
    (extended by extrapolation_factor) and returns the maximum intensity.

    Parameters
    ----------
    nifti_img : nibabel.Nifti1Image
        Volume image from which to sample.
    surf_mesh : str or tuple
        Pial surface mesh.
    inner_mesh : str or tuple
        Inner surface mesh (pseudo inner mesh).
    n_samples : int, optional
        Number of sample points along the line (default 20).
    extrapolation_factor : float, optional
        Factor by which to extend the line (default 1.0).

    Returns
    -------
    proj_data : np.ndarray
        Array of shape (n_vertices,) with the maximum intensity per
        vertex.
    """
    vol_data = nifti_img.get_fdata()
    affine = nifti_img.affine
    inv_affine = np.linalg.inv(affine)
    pial = load_surf_mesh(surf_mesh)
    inner = load_surf_mesh(inner_mesh)
    pial_coords = pial[0]
    inner_coords = inner[0]
    n_vertices = pial_coords.shape[0]
    proj_data = np.zeros(n_vertices, dtype=np.float64)
    fractions = np.linspace(0, extrapolation_factor, n_samples)
    for idx in range(n_vertices):
        line_coords = (np.outer(1 - fractions, pial_coords[idx]) +
                       np.outer(fractions, inner_coords[idx]))
        voxel_coords = nib.affines.apply_affine(inv_affine, line_coords)
        sampled_values = map_coordinates(vol_data, voxel_coords.T,
                                         order=1, mode='nearest')
        proj_data[idx] = np.max(sampled_values)

    return proj_data


def custom_view_surf_with_bg(surf_gii_path, surf_map, bg_map, threshold, vmin,
                             vmax, overlay_colorscale="viridis",
                             bg_colorscale=[[0, "rgb(200,200,200)"],
                                            [1, "rgb(240,240,240)"]],
                             bg_smoothing_iterations=0,
                             lighting_params=None, title="3D Surface"):
    """
    Render a surface with two layers:
      1. A background layer (e.g., sulcal depth) displayed in a lighter
         gray texture.
      2. An activation overlay (only for faces with all
         vertices >= threshold).

    Parameters
    ----------
    surf_gii_path : str
        Path to the surface GIFTI file.
    surf_map : np.ndarray
        Activation data per vertex.
    bg_map : str or np.ndarray
        Background data (e.g., sulcal values); if a string, treated as
        file path.
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
        cmin=np.min(bg_data),
        cmax=np.max(bg_data),
        showscale=False,
        opacity=1,
        flatshading=False,  # Allow texture to show.
        name="Sulc Background"
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
        colorbar=dict(title="Activation"),
        opacity=1,
        name="Activation Overlay"
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
            aspectmode="data"
        ),
        margin=dict(l=0, r=0, t=50, b=0)
    )
    fig = go.Figure(data=[background_mesh, overlay_mesh], layout=layout)
    
    return fig


# ============================ INPUTS ===================================

dstr_meshes_folder = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "dstr_meshes"
)

lh_dstr_coord_path = os.path.join(dstr_meshes_folder, "lh.striatum.coord.gii")
lh_dstr_topo_path = os.path.join(dstr_meshes_folder, "lh.striatum.topo.gii")
lh_dstr_spec_path = os.path.join(dstr_meshes_folder, "lh.striatum.spec.gii")
lh_dstr_surf_path = os.path.join(dstr_meshes_folder, "lh.dstr.surf.gii")
lh_dstr_inner_surf_path = os.path.join(dstr_meshes_folder,
                                       "lh.dstr-inner.surf.gii")
lh_dstr_sulc_path = os.path.join(dstr_meshes_folder, "lh.dstr.sulc.gii")

rh_dstr_coord_path = os.path.join(dstr_meshes_folder, "rh.striatum.coord.gii")
rh_dstr_topo_path = os.path.join(dstr_meshes_folder, "rh.striatum.topo.gii")
rh_dstr_spec_path = os.path.join(dstr_meshes_folder, "rh.striatum.spec.gii")
rh_dstr_surf_path = os.path.join(dstr_meshes_folder, "rh.dstr.surf.gii")
rh_dstr_inner_surf_path = os.path.join(dstr_meshes_folder,
                                       "rh.dstr-inner.surf.gii")
rh_dstr_sulc_path = os.path.join(dstr_meshes_folder, "rh.dstr.sulc.gii")

lh_dstr_overlay_masks_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "roi_analyses_rwls_hrf128_wb_puncorr",
    "all",
    "dorsal_striatum",
    "hos",
    "overlaid_masks",
    "i8a_dstr_bh_mask_gmmasked.nii.gz"
)

# Relative path for output folder
irois_folder = 'results/irois'


# ============================ RUN =====================================

if __name__ == "__main__":
    # Create surf file for left hemisphere.
    build_surf_gii(lh_dstr_coord_path, lh_dstr_topo_path, lh_dstr_surf_path,
                   spec_file_path=lh_dstr_spec_path)
    # Compute sulc GIFTI file for left hemisphere.
    compute_sulc_gii(lh_dstr_surf_path, lh_dstr_sulc_path)
    # Refine the surface mesh to increase vertex density and smooth geometry.
    refined_surf_path = os.path.join(dstr_meshes_folder,
                                     "lh.striatum.refined_smoothed.surf.gii")
    refine_and_smooth_mesh(lh_dstr_surf_path, refined_surf_path, iterations=2,
                           lamb=.5, smooth_iters=15)
    # Create an inner mesh from the refined (and smoothed) surface.
    _ = create_inner_mesh(refined_surf_path, scale=0.95,
                          output_path=lh_dstr_inner_surf_path)

    # Load the NIfTI overlay image.
    lh_dstr_overlay_img = load_img(lh_dstr_overlay_masks_path)

    # Project the overlay volume onto the refined left hemisphere surface.
    lh_surf_data = vol_to_surf_max(lh_dstr_overlay_img, refined_surf_path,
                                   lh_dstr_inner_surf_path,
                                   n_samples=10, extrapolation_factor=5.)

    # Define activation threshold and intensity range.
    threshold = 1 / 31
    vmin = threshold
    vmax = 1

    # Create the custom Plotly figure using the sulc file as background.
    fig = custom_view_surf_with_bg(
        surf_gii_path=refined_surf_path,
        surf_map=lh_surf_data,
        bg_map=lh_dstr_sulc_path,
        threshold=threshold,
        vmin=vmin,
        vmax=vmax,
        overlay_colorscale="cividis",
        bg_colorscale=[[0, "rgb(200,200,200)"], [1, "rgb(240,240,240)"]],
        bg_smoothing_iterations=0,
        lighting_params=dict(ambient=0.4, diffuse=0.8,
                             specular=0.3, roughness=0.2),
        title="3D Basal Ganglia: Fraction of Participants"
    )

    # Define the output HTML file path
    output_html_path = os.path.join( irois_folder, "iroi_dstr_surf_lh.html")

    # Save the figure as an interactive HTML file
    pio.write_html(fig, output_html_path)

    # Display the figure in the browser
    fig.show()
