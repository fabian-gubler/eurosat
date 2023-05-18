import os
import numpy as np
import rasterio
from rasterio.transform import Affine

prefix = "/data/"
basedir = os.path.join(prefix, "eurosat/data")

parent_input_folder = os.path.join(basedir, "testset")
parent_output_folder = os.path.join(basedir, "testset_tiff")

# Create output folder if it does not exist
os.makedirs(parent_output_folder, exist_ok=True)

    
# Create output subdirectory if it does not exist
os.makedirs(parent_output_folder, exist_ok=True)

# Loop through all files in the current input subdirectory
for filename in os.listdir(parent_input_folder):
    # Check if the file is a NPY
    if filename.endswith('.npy'):
        # Set input and output file paths for the current image
        input_filepath = os.path.join(parent_input_folder, filename)
        output_filepath = os.path.join(parent_output_folder, filename.replace('.npy', '.tif'))

        # Load the NPY file
        image_data = np.load(input_filepath)

        # Create a dictionary to store metadata
        metadata = {
            'driver': 'GTiff',
            'height': image_data.shape[1],
            'width': image_data.shape[2],
            'count': image_data.shape[0],
            'dtype': image_data.dtype,
            'crs': '+proj=latlong',
            'transform': rasterio.Affine(1.0, 0, 0, 0, 1.0, 0),
        }

        # Write the .tif file from the numpy array
        with rasterio.open(output_filepath, 'w', **metadata) as dest:
            dest.write(image_data)
