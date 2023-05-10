# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.2-dev
#   kernelspec:
#     display_name: eurosat
#     language: python
#     name: python3
# ---

import os
import glob
import numpy as np
import rasterio as rio
from rasterio.plot import reshape_as_image
import matplotlib.pyplot as plt
# %matplotlib inline

# ## Convert all images from tif to npy

# +
import os
import numpy as np
import rasterio

parent_input_folder = '/Users/svenschnydrig/Documents/Coding Challenge/data/EuroSAT_MS'
parent_output_folder = '/Users/svenschnydrig/Documents/Coding Challenge/data/EuroSAT_MS_NPY'

os.makedirs(parent_output_folder, exist_ok=True)

for subdir in os.listdir(parent_input_folder):
    input_folder = os.path.join(parent_input_folder, subdir)
    output_folder = os.path.join(parent_output_folder, subdir)
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith('.tif'):
            input_filepath = os.path.join(input_folder, filename)
            output_filepath = os.path.join(output_folder, filename.replace('.tif', '.npy'))

            with rasterio.open(input_filepath) as src:
                image_data = src.read()
                np.save(output_filepath, image_data)
# -

# ## Remove band 10 from train data

# +
parent_input_folder = '/Users/svenschnydrig/Documents/Coding Challenge/data/EuroSAT_MS_NPY'
parent_output_folder = '/Users/svenschnydrig/Documents/Coding Challenge/data/EuroSAT_MS_NPY_wo_B10'

os.makedirs(parent_output_folder, exist_ok=True)

for subdir in os.listdir(parent_input_folder):
    input_folder = os.path.join(parent_input_folder, subdir)
    output_folder = os.path.join(parent_output_folder, subdir)
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith('.npy'):
            input_filepath = os.path.join(input_folder, filename)
            output_filepath = os.path.join(output_folder, filename)

            image_data = np.load(input_filepath)
            image_data_no_b10 = np.delete(image_data, 9, axis=0)  # Remove B10 band (assuming it's the 10th band)
            np.save(output_filepath, image_data_no_b10)
# -

# ## Sanity check

# +
# Load the .npy file
image_data = np.load("/Users/svenschnydrig/Documents/Coding Challenge/data/EuroSAT_MS_NPY_wo_B10/AnnualCrop/AnnualCrop_1.npy")

# Get the number of bands (assuming bands are in the first dimension)
print(image_data.shape[0])

# +
# Load the .npy file
image_data = np.load("/Users/svenschnydrig/Documents/Coding Challenge/data/EuroSAT_MS_NPY/AnnualCrop/AnnualCrop_1.npy")

# Get the number of bands (assuming bands are in the first dimension)
print(image_data.shape[0])
# -

# ## Fix differing band order between train and test data

# +
import numpy as np
import os
import shutil

def change_band_order(data):
    original_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    desired_order = [0, 1, 2, 3, 4, 5, 6, 7, 11, 8, 9, 10]
    
    ordered_data = np.empty_like(data)
    for i, band_idx in enumerate(desired_order):
        ordered_data[i] = data[band_idx]
        
    return ordered_data

def process_images(source_folder, dest_folder):
    for root, _, files in os.walk(source_folder):
        for file in files:
            if file.endswith(".npy"):
                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, source_folder)
                dest_subfolder = os.path.join(dest_folder, rel_path)
                dest_path = os.path.join(dest_subfolder, file)

                os.makedirs(dest_subfolder, exist_ok=True)
                image_data = np.load(src_path)
                ordered_data = change_band_order(image_data)
                np.save(dest_path, ordered_data)

source_folder = '/Users/svenschnydrig/Documents/Coding Challenge/data/EuroSAT_MS_NPY_wo_B10'
ordered_folder = '/Users/svenschnydrig/Documents/Coding Challenge/data/EuroSAT_MS_NPY_wo_B10_ordered'

if os.path.exists(ordered_folder):
    shutil.rmtree(ordered_folder)

process_images(source_folder, ordered_folder)

# -

# ## Sanity check

print(np.load('/Users/svenschnydrig/Documents/Coding Challenge/data/EuroSAT_MS_NPY_wo_B10/AnnualCrop/AnnualCrop_1.npy')[11])

print(np.load('/Users/svenschnydrig/Documents/Coding Challenge/data/EuroSAT_MS_NPY_wo_B10_ordered/AnnualCrop/AnnualCrop_1.npy')[8])
