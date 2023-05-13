# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.2-dev
#   kernelspec:
#     display_name: eurosat-10
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

# In this code snippet, we first import the necessary libraries and set the input and output folder paths. We create the output folder if it does not exist and then loop through all subdirectories in the input folder. For each subdirectory, we create an output subdirectory and loop through all files within it. We check if a file is a TIFF image and, if so, open it using rasterio, read the image data into a NumPy array, and save the array to a .npy file.

# +
# Import necessary libraries
import os
import numpy as np
import rasterio

prefix = "/home/paperspace"
basedir = os.path.join(prefix, "eurosat/data")

# Set input and output folder paths
parent_input_folder = os.path.join(basedir, "EuroSAT_MS")
parent_output_folder = os.path.join(basedir, "EuroSAT_MS_NPY")

# Create output folder if it does not exist
os.makedirs(parent_output_folder, exist_ok=True)

# Loop through all subdirectories in the input folder
for subdir in os.listdir(parent_input_folder):
    # Set input and output folder paths for the current subdirectory
    input_folder = os.path.join(parent_input_folder, subdir)
    output_folder = os.path.join(parent_output_folder, subdir)
    
    # Create output subdirectory if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all files in the current input subdirectory
    for filename in os.listdir(input_folder):
        # Check if the file is a TIFF image
        if filename.endswith('.tif'):
            # Set input and output file paths for the current image
            input_filepath = os.path.join(input_folder, filename)
            output_filepath = os.path.join(output_folder, filename.replace('.tif', '.npy'))

            # Open the input TIFF image using rasterio
            with rasterio.open(input_filepath) as src:
                # Read the image data into a NumPy array
                image_data = src.read()
                # Save the NumPy array to a .npy file
                np.save(output_filepath, image_data)
# -

# ## Load train data into x array and labels into y array

# +
import os
import numpy as np

data_dir = "/home/paperspace/eurosat/data/EuroSAT_MS_NPY"
# data_dir = "/Users/svenschnydrig/Documents/Coding Challenge/data/EuroSAT_MS_NPY"
class_names = sorted(os.listdir(data_dir))

x = []
y = []
for i, class_name in enumerate(class_names):
    print(class_name)
    print(i)
    class_dir = os.path.join(data_dir, class_name)
    print(class_dir)
    for filename in os.listdir(class_dir):
        #print(filename)
        filepath = os.path.join(class_dir, filename)
        #print(filename)
        data = np.load(filepath)
        x.append(data)
        y.append(i)
x = np.stack(x, axis=0)
y = np.array(y)

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
np.save('/home/paperspace/eurosat/data/preprocessed/x.npy', x)
# np.save('/Users/svenschnydrig/Documents/Coding Challenge/data/preprocessed/y.npy', y)
# -

# # Load testdata into array x_testset

# +
import numpy as np
import os
import re

# initialize an empty list
x_testset = []

# specify the directory you want to load from
directory = '/home/paperspace/eurosat/data/testset/test'
# directory = '/Users/svenschnydrig/Documents/Coding Challenge/data/testset/test'

# function to convert a filename to a number (ignoring the .npy extension)
def filename_to_number(filename):
    return int(re.sub(r'[^\d]', '', filename.split('.')[0]))

# get a list of all .npy files in the directory
files = [f for f in os.listdir(directory) if f.endswith('.npy')]

# sort the list of files by the numeric part of the filename
files.sort(key=filename_to_number)

# loop over the sorted list of files
for filename in files:
    # load the .npy file and append it to the list
    file_path = os.path.join(directory, filename)
    #print(file_path)
    x_testset.append(np.load(file_path))

# convert the list to a numpy array
x_testset = np.array(x_testset)
# -

# ## Train Data - change dimension, delete B10 and reorder bands

# +
x = np.transpose(x, (0, 2, 3, 1))
x.shape

#(num_images, height, width, bands)
x = np.delete(x, 9, axis=3)

# your current order: B1, B2, B3, B4, B5, B6, B7, B8, B9, B11, B12, B8A
current_order = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
# your desired order: B1, B2, B3, B4 ,B5, B6, B7, B8, B8A, B9, B11, B12
desired_order = np.array([0, 1, 2, 3, 4, 5, 6, 7, 11, 8, 9, 10])

# reorder the bands
x = x[:, :, :, desired_order]
# -

# ## Add band index features

# +
import numpy as np

# B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12
# 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11

def calculate_indices(x):
    # Define a very small number
    epsilon = np.float32(1e-6)

    # Assign bands to variables and convert them to float32
    blue = x[..., 1].astype(np.float32)
    green = x[..., 2].astype(np.float32)
    red = x[..., 3].astype(np.float32)
    nir = x[..., 7].astype(np.float32)
    swir1 = x[..., 10].astype(np.float32)
    swir2 = x[..., 11].astype(np.float32)

    # Calculate indices
    NDVI = (nir - red) / (nir + red + epsilon)
    # EVI = 2.5 * ((nir - red) / ((nir + 6*red - 7.5*blue) + 1 + epsilon)) something is wrong here
    NDWI = (green - nir) / (green + nir + epsilon)
    NDBI = (swir1 - nir) / (swir1 + nir + epsilon)
    NDSI = (green - swir1) / (green + swir1 + epsilon)
    L = np.float32(0.5)  # soil brightness correction factor
    SAVI = ((nir - red) / (nir + red + L)) * (1 + L)
    MNDWI = (green - swir1) / (green + swir1 + epsilon)

    # Reshape the indices to match the shape of the original data
    NDVI = NDVI[..., np.newaxis]
    #EVI = EVI[..., np.newaxis]
    NDWI = NDWI[..., np.newaxis]
    NDBI = NDBI[..., np.newaxis]
    NDSI = NDSI[..., np.newaxis]
    SAVI = SAVI[..., np.newaxis]
    MNDWI = MNDWI[..., np.newaxis]

    # Concatenate the original data with the new bands
    x = np.concatenate((x, NDVI, NDWI, NDBI, NDSI, SAVI, MNDWI), axis=-1)

    return x

# Now we can call this function with our ndarray
#x = calculate_indices(x)
x_testset = calculate_indices(x_testset)
# -

x_testset.dtype

# ## Get overview of data

# +
import numpy as np

# Set printing options.
np.set_printoptions(suppress=True, precision=2)

def compute_band_statistics(x):
    # Compute the band statistics: min, mean, median, and max.
    band_min = np.min(x, axis=(0, 1, 2))
    band_mean = np.mean(x, axis=(0, 1, 2))
    band_median = np.median(x, axis=(0, 1, 2))
    band_max = np.max(x, axis=(0, 1, 2))

    return band_min, band_mean, band_median, band_max

band_min, band_mean, band_median, band_max = compute_band_statistics(x)

# Print the statistics for each band.
for i in range(len(band_min)):
    print(f"Statistics for band {i+1}:")
    print(f"Min: {band_min[i]}")
    print(f"Mean: {band_mean[i]}")
    print(f"Median: {band_median[i]}")
    print(f"Max: {band_max[i]}")
    print()

# +
import numpy as np

# Set printing options.
np.set_printoptions(suppress=True, precision=2)

def compute_band_statistics(x):
    # Compute the band statistics: min, mean, median, and max.
    band_min = np.min(x, axis=(0, 1, 2))
    band_mean = np.mean(x, axis=(0, 1, 2))
    band_median = np.median(x, axis=(0, 1, 2))
    band_max = np.max(x, axis=(0, 1, 2))

    return band_min, band_mean, band_median, band_max

band_min, band_mean, band_median, band_max = compute_band_statistics(x_testset)

# Print the statistics for each band.
for i in range(len(band_min)):
    print(f"Statistics for band {i+1}:")
    print(f"Min: {band_min[i]}")
    print(f"Mean: {band_mean[i]}")
    print(f"Median: {band_median[i]}")
    print(f"Max: {band_max[i]}")
    print()
# -

# ## Normalise data

x_original = x.copy()
x_testset_original = x_testset.copy()

# # Normalise by constant division / 28000

# +
x[..., :12] = x[..., :12] / 28000.0
x_testset[..., :12] = x_testset[..., :12] / 28000.0

np.save('/home/paperspace/eursat/data/preprocessed/x_28k.npy', x)
np.save('/home/paperspace/eursat/data/preprocessed/x_testset_28k.npy', x_testset)

# np.save('/Users/svenschnydrig/Documents/Coding Challenge/data/preprocessed/x_28k.npy', x)
# np.save('/Users/svenschnydrig/Documents/Coding Challenge/data/preprocessed/x_testset_28k.npy', x_testset)
# -

# ## Normalise by dividing by max value of each band

# +
x = x_original.copy()
x_testset = x_testset_original.copy()

# Assuming x is your data array
for i in range(12):
    max_val = np.max(x[..., i])
    x[..., i] = x[..., i] / max_val
    x_testset[..., i] = x_testset[..., i] / max_val

np.save('/home/paperspace/eursat/data/preprocessed/x_max.npy', x)
np.save('/home/paperspace/eursat/data/preprocessed/x_testset_max.npy', x_testset)

# np.save('/Users/svenschnydrig/Documents/Coding Challenge/data/preprocessed/x_max.npy', x)
# np.save('/Users/svenschnydrig/Documents/Coding Challenge/data/preprocessed/x_testset_max.npy', x_testset)
# -

# ## Normalising by dividing by max value of test bands

# +
x_testset = x_testset_original.copy()

# Assuming x is your data array
for i in range(12):
    max_val = np.max(x_testset[..., i])
    x_testset[..., i] = x_testset[..., i] / max_val

np.save('/home/paperspace/eursat/data/preprocessed/x_testset_max_test.npy', x_testset)
# np.save('/Users/svenschnydrig/Documents/Coding Challenge/data/preprocessed/x_testset_max_test.npy', x_testset)
# -

# ## Min-Max 0-1

# +
x = x_original.copy()
x_testset = x_testset_original.copy()

# Get the minimum and maximum pixel values for the first twelve bands in each image.
min_vals = np.min(x[..., :12], axis=(0, 1, 2), keepdims=True)
max_vals = np.max(x[..., :12], axis=(0, 1, 2), keepdims=True)

# Subtract the minimum and divide by the range to normalize between 0 and 1.
x[..., :12] = (x[..., :12] - min_vals) / (max_vals - min_vals)
x_testset[..., :12] = (x_testset[..., :12] - min_vals) / (max_vals - min_vals)

np.save('/home/paperspace/eursat/data/preprocessed/x_minmax_0_1.npy', x)
np.save('/home/paperspace/eursat/data/preprocessed/x_testset_minmax_0_1.npy', x_testset)
# np.save('/Users/svenschnydrig/Documents/Coding Challenge/data/preprocessed/x_minmax_0_1.npy', x)
# np.save('/Users/svenschnydrig/Documents/Coding Challenge/data/preprocessed/x_testset_minmax_0_1.npy', x_testset)
# -

# ## Min-Max -1-1

# +
x = x_original.copy()
x_testset = x_testset_original.copy()

# Get the minimum and maximum pixel values for the first twelve bands in each image.
min_vals = np.min(x[..., :12], axis=(0, 1, 2), keepdims=True)
max_vals = np.max(x[..., :12], axis=(0, 1, 2), keepdims=True)

# Subtract the minimum, divide by the range, shift by -0.5 and multiply by 2.
# This will normalize the data between -1 and 1.
x[..., :12] = 2 * ((x[..., :12] - min_vals) / (max_vals - min_vals)) - 1
x_testset[..., :12] = 2 * ((x_testset[..., :12] - min_vals) / (max_vals - min_vals)) - 1

np.save('/home/paperspace/eursat/data/preprocessed/x_minmax_-1_1.npy', x)
np.save('/home/paperspace/eursat/data/preprocessed/x_testset_minmax_-1_1.npy', x_testset)
# np.save('/Users/svenschnydrig/Documents/Coding Challenge/data/preprocessed/x_minmax_-1_1.npy', x)
# np.save('/Users/svenschnydrig/Documents/Coding Challenge/data/preprocessed/x_testset_minmax_-1_1.npy', x_testset)
# -

# ## Standard scaler

# +
x = x_original.copy()
x_testset = x_testset_original.copy()

# Get the mean and standard deviation for the first twelve bands across all images.
mean_vals = np.mean(x[..., :12], axis=(0, 1, 2), keepdims=True)
std_vals = np.std(x[..., :12], axis=(0, 1, 2), keepdims=True)

# Subtract the mean and divide by the standard deviation for standard scaling.
x[..., :12] = (x[..., :12] - mean_vals) / std_vals
x_testset[..., :12] = (x_testset[..., :12] - mean_vals) / std_vals

np.save('/home/paperspace/eursat/data/preprocessed/x_std.npy', x)
np.save('/home/paperspace/eursat/data/preprocessed/x_testset_std.npy', x_testset)

# np.save('/Users/svenschnydrig/Documents/Coding Challenge/data/preprocessed/x_std.npy', x)
# np.save('/Users/svenschnydrig/Documents/Coding Challenge/data/preprocessed/x_testset_std.npy', x_testset)
# -

# ## Normalise all bands instead of just first 12 bands

# ### min max

# +
x = x_original.copy()
x_testset = x_testset_original.copy()

# Get the minimum and maximum pixel values for each band in each image.
min_vals = np.min(x, axis=(0, 1, 2), keepdims=True)
max_vals = np.max(x, axis=(0, 1, 2), keepdims=True)

# Subtract the minimum and divide by the range to normalize between 0 and 1.
x = (x - min_vals) / (max_vals - min_vals)
x_testset = (x_testset - min_vals) / (max_vals - min_vals)

np.save('/home/paperspace/eursat/data/preprocessed/x_minmax_0_1_all.npy', x)
np.save('/home/paperspace/eursat/data/preprocessed/x_testset_minmax_0_1_all.npy', x_testset)

# np.save('/Users/svenschnydrig/Documents/Coding Challenge/data/preprocessed/x_minmax_0_1_all.npy', x)
# np.save('/Users/svenschnydrig/Documents/Coding Challenge/data/preprocessed/x_testset_minmax_0_1_all.npy', x_testset)
# -

# ### standard

# +
x = x_original.copy()
x_testset = x_testset_original.copy()

# Get the mean and standard deviation for each band across all images.
mean_vals = np.mean(x, axis=(0, 1, 2), keepdims=True)
std_vals = np.std(x, axis=(0, 1, 2), keepdims=True)

# Subtract the mean and divide by the standard deviation for standard scaling.
x = (x - mean_vals) / std_vals
x_testset = (x_testset - mean_vals) / std_vals

np.save('/home/paperspace/eursat/data/preprocessed/x_std_all.npy', x)
np.save('/home/paperspace/eursat/data/preprocessed/x_testset_std_all.npy', x_testset)

# np.save('/Users/svenschnydrig/Documents/Coding Challenge/data/preprocessed/x_std_all.npy', x)
# np.save('/Users/svenschnydrig/Documents/Coding Challenge/data/preprocessed/x_testset_std_all.npy', x_testset)
