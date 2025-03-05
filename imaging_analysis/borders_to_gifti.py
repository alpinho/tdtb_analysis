"""
Script to convert workbench fs_LR32k borders files in gifti files

Author: Ana Luisa Pinho
Email: agrilopi@uwo.ca

Creation: 4th of March 2025
Last Update: March 2025

Compatibility: Python 3.10.14
"""

import xml.etree.ElementTree as ET
import numpy as np
import nibabel as nib
from nibabel.gifti import GiftiDataArray, GiftiImage

# Define file paths
border_file = "fslr32k_meshes/flat/fs_LR.32k.L.border"
output_gii = "fslr32k_meshes/flat/fs_LR.32k.L.border.label.gii"

### Read and Parse the XML Border File ###
print(f"Reading XML border file: {border_file}")

# Parse XML
tree = ET.parse(border_file)
root = tree.getroot()

### Extract Border Vertex Indices ###
print("Extracting border vertex indices...")

border_vertices = []

# Find all <Vertices> elements in the XML
for vertices_elem in root.findall(".//Vertices"):
    if vertices_elem.text:  # Ensure it has text
        # Convert space-separated numbers into a list of integers
        vertices = [int(v) for v in vertices_elem.text.split()]
        border_vertices.extend(vertices)  # Add to the list

border_vertices = np.array(border_vertices, dtype=int)

# Display extracted border vertices
print("Extracted border vertices:", border_vertices[:10])  # Show first 10

# Handle empty extraction
if len(border_vertices) == 0:
    print("No border vertices extracted! Check XML structure.")
    exit()

### Convert Border Data to a GIFTI Label File ###
print(f"Converting border data to {output_gii}...")

# Get total number of vertices (adjust this based on your mesh)
num_vertices = 32492  # Ensure this matches fs_LR32k resolution

# Create a binary mask for border vertices
border_label = np.zeros(num_vertices)
border_label[border_vertices] = 1  # Mark border vertices

# Convert to GiftiDataArray
gifti_data = GiftiDataArray(border_label.astype(np.float32))

# Create a GiftiImage
gifti_image = GiftiImage(darrays=[gifti_data])

# Save as a GIFTI label file
nib.save(gifti_image, output_gii)

print(f"Successfully saved border file as {output_gii}")
