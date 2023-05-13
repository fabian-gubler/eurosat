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

# Given your use case of EuroSAT satellite image classification from Sentinel-2, I recommend the following indices:
#
# NDVI (Normalized Difference Vegetation Index): Since Sentinel-2 has high-resolution red and near-infrared bands, NDVI will be helpful in distinguishing between vegetated and non-vegetated areas.
#
# NDWI (Normalized Difference Water Index) or MNDWI (Modified Normalized Difference Water Index): Both indices are useful for detecting water bodies. You may need to experiment with both to see which performs better for your specific study area.
#
# NDBI (Normalized Difference Built-up Index): This index is useful for detecting built-up and urban areas in your classification.
#
# These three indices should provide a good starting point for classifying land cover types in EuroSAT images using Sentinel-2 data. However, it's essential to evaluate their performance and adjust the selection based on the results you obtain. If you encounter difficulty distinguishing between certain land cover classes, you may want to experiment with additional indices like EVI, SAVI, or the Tasseled Cap indices (TCB, TCG, and TCW) to improve your classification accuracy.

# Yes, including some of the other bands by themselves can be beneficial for your land cover classification model, as they can provide additional information that may not be captured by the RGB bands or the indices mentioned earlier.
#
# Sentinel-2 satellite has 13 spectral bands, each with different wavelength ranges and resolutions. These bands can help discriminate specific land cover types and surface properties. Some of the bands worth considering include:
#
# Near-Infrared (NIR) band (Band 8): This band is sensitive to vegetation and can provide complementary information to the NDVI or other vegetation indices.
#
# Shortwave Infrared (SWIR) bands (Bands 11 and 12): These bands are useful for detecting moisture content, differentiating between various soil and rock types, and identifying cloud and snow cover.
#
# Red-Edge bands (Bands 5, 6, and 7): These bands capture the reflectance in the red-edge region of the spectrum, which is sensitive to chlorophyll content in vegetation. Including red-edge bands can improve the discrimination of vegetation types and stress conditions.
#
# It's important to note that incorporating more bands can increase the complexity of the model and potentially the computation time. Therefore, it is essential to strike a balance between including relevant bands and maintaining computational efficiency. Perform feature selection techniques or dimensionality reduction methods, such as Principal Component Analysis (PCA), to identify the most informative bands and indices for your specific classification problem.
#
# In summary, including additional bands by themselves can potentially improve the accuracy of your model, but it is crucial to evaluate their contribution to the classification performance and manage the model's complexity.

# ## All indices

# +
import numpy as np
import os
import shutil
from tqdm import tqdm


prefix = "/home/paperspace"
basedir = os.path.join(prefix, "eurosat/data/")

# B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12.

def calculate_indices(file_path, output_path):
    data = np.load(file_path)

    # Assigning bands to variables
    B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12 = data

    # Calculate indices

    # Normalized Difference Vegetation Index (NDVI)
    NDVI = (B8 - B4) / (B8 + B4 + 1e-6)

    # Enhanced Vegetation Index (EVI)
    EVI = 2.5 * ((B8 - B4) / ((B8 + 6 * B4 - 7.5 * B2 + 1) + 1e-6))

    # Normalized Difference Water Index (NDWI)
    NDWI = (B3 - B8) / (B3 + B8 + 1e-6)

    # Normalized Difference Built-up Index (NDBI)
    NDBI = (B11 - B8) / (B11 + B8 + 1e-6)

    # Normalized Difference Snow Index (NDSI)
    NDSI = (B3 - B11) / (B3 + B11 + 1e-6)

    # Soil Adjusted Vegetation Index (SAVI)
    L = 0.5  # soil brightness correction factor; varies between 0 (very low vegetation) and 1 (dense vegetation)
    SAVI = ((B8 - B4) / (B8 + B4 + L + 1e-6)) * (1 + L)

    # Calculate MNDWI (Modified Normalized Difference Water Index)
    MNDWI = (B3 - B11) / (B3 + B11 + 1e-6)

    # Calculate Tasseled Cap Indices (Brightness, Greenness, Wetness)
    TCB = 0.2043 * B2 + 0.4158 * B3 + 0.5524 * B4 + 0.5741 * B8 + 0.3124 * B11 + 0.2303 * B12
    TCG = -0.1603 * B2 - 0.2819 * B3 - 0.4934 * B4 + 0.7940 * B8 - 0.0002 * B11 - 0.1446 * B12
    TCW = 0.0315 * B2 + 0.2021 * B3 + 0.3102 * B4 + 0.1594 * B8 - 0.6806 * B11 - 0.6109 * B12

    # These calculated indices can now be used for classification tasks.

    # Add a new singleton dimension to the calculated arrays along the first axis
    NDVI = np.expand_dims(NDVI, axis=0)
    EVI = np.expand_dims(EVI, axis=0)
    NDWI = np.expand_dims(NDWI, axis=0)
    NDBI = np.expand_dims(NDBI, axis=0)
    NDSI = np.expand_dims(NDSI, axis=0)
    SAVI = np.expand_dims(SAVI, axis=0)
    MNDWI = np.expand_dims(MNDWI, axis=0)
    TCB = np.expand_dims(TCB, axis=0)
    TCG = np.expand_dims(TCG, axis=0)
    TCW = np.expand_dims(TCW, axis=0)

    # Stack the arrays along the first dimension to create the enhanced data array
    enhanced_data = np.concatenate((data, NDVI, EVI, NDWI, NDBI, NDSI, SAVI, MNDWI, TCB, TCG, TCW), axis=0)
    # Change the data type to float32
    enhanced_data = enhanced_data.astype('float32')

    # The shape of enhanced_data should be (12 + 10, 64, 64) = (22, 64, 64)
    #print(enhanced_data.shape)

    # Save the enhanced data as a new .npy file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, enhanced_data)


def process_folder(input_path, output_path):
    for root, dirs, files in tqdm(list(os.walk(input_path)), desc='Processing directories'):
        for file in tqdm(files, desc='Processing files'):
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, input_path)
                output_file_path = os.path.join(output_path, relative_path)
                calculate_indices(file_path, output_file_path)
        for dir in tqdm(dirs, desc='Creating directories'):
            src_dir = os.path.join(root, dir)
            dst_dir = os.path.join(output_path, os.path.relpath(src_dir, input_path))
            os.makedirs(dst_dir, exist_ok=True)

input_folder_path = os.path.join(basedir, 'EuroSAT_MS_NPY_wo_B10_ordered')
output_folder_path = os.path.join(basedir, 'EuroSAT_MS_NPY_wo_B10_ordered_features_float32')

process_folder(input_folder_path, output_folder_path)
# -

# ## Most promising indices

# +
import numpy as np
import os
import shutil

# B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12.

def calculate_indices(file_path, output_path):
    data = np.load(file_path)

    # Assigning bands to variables
    B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12 = data

    # Calculate indices

    # Normalized Difference Vegetation Index (NDVI)
    NDVI = (B8 - B4) / (B8 + B4 + 1e-6)

    # Enhanced Vegetation Index (EVI)
    EVI = 2.5 * ((B8 - B4) / ((B8 + 6 * B4 - 7.5 * B2 + 1) + 1e-6))

    # Normalized Difference Water Index (NDWI)
    NDWI = (B3 - B8) / (B3 + B8 + 1e-6)

    # Normalized Difference Built-up Index (NDBI)
    NDBI = (B11 - B8) / (B11 + B8 + 1e-6)

    # Normalized Difference Snow Index (NDSI)
    NDSI = (B3 - B11) / (B3 + B11 + 1e-6)

    # Soil Adjusted Vegetation Index (SAVI)
    L = 0.5  # soil brightness correction factor; varies between 0 (very low vegetation) and 1 (dense vegetation)
    SAVI = ((B8 - B4) / (B8 + B4 + L + 1e-6)) * (1 + L)

    # Calculate MNDWI (Modified Normalized Difference Water Index)
    MNDWI = (B3 - B11) / (B3 + B11 + 1e-6)

    # Calculate Tasseled Cap Indices (Brightness, Greenness, Wetness)
    TCB = 0.2043 * B2 + 0.4158 * B3 + 0.5524 * B4 + 0.5741 * B8 + 0.3124 * B11 + 0.2303 * B12
    TCG = -0.1603 * B2 - 0.2819 * B3 - 0.4934 * B4 + 0.7940 * B8 - 0.0002 * B11 - 0.1446 * B12
    TCW = 0.0315 * B2 + 0.2021 * B3 + 0.3102 * B4 + 0.1594 * B8 - 0.6806 * B11 - 0.6109 * B12

    # These calculated indices can now be used for classification tasks.

    # Add a new singleton dimension to the calculated arrays along the first axis
    NDVI = np.expand_dims(NDVI, axis=0)
    EVI = np.expand_dims(EVI, axis=0)
    NDWI = np.expand_dims(NDWI, axis=0)
    NDBI = np.expand_dims(NDBI, axis=0)
    NDSI = np.expand_dims(NDSI, axis=0)
    SAVI = np.expand_dims(SAVI, axis=0)
    MNDWI = np.expand_dims(MNDWI, axis=0)
    TCB = np.expand_dims(TCB, axis=0)
    TCG = np.expand_dims(TCG, axis=0)
    TCW = np.expand_dims(TCW, axis=0)

    # Stack the arrays along the first dimension to create the enhanced data array
    enhanced_data = np.concatenate((data, NDVI, EVI, NDWI, NDBI, NDSI, SAVI, MNDWI, TCB), axis=0)
    # Change the data type to float32
    enhanced_data = enhanced_data.astype('float32')

    # The shape of enhanced_data should be (12 + 10, 64, 64) = (22, 64, 64)
    #print(enhanced_data.shape)

    # Save the enhanced data as a new .npy file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, enhanced_data)


def process_folder(input_path, output_path):
    for root, dirs, files in tqdm(list(os.walk(input_path)), desc='Processing directories'):
        for file in tqdm(files, desc='Processing files'):
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, input_path)
                output_file_path = os.path.join(output_path, relative_path)
                calculate_indices(file_path, output_file_path)
        for dir in tqdm(dirs, desc='Creating directories'):
            src_dir = os.path.join(root, dir)
            dst_dir = os.path.join(output_path, os.path.relpath(src_dir, input_path))
            os.makedirs(dst_dir, exist_ok=True)

input_folder_path = os.path.join(basedir, 'EuroSAT_MS_NPY_wo_B10_ordered')
output_folder_path = os.path.join(basedir, 'EuroSAT_MS_NPY_wo_B10_ordered_features_float32')

process_folder(input_folder_path, output_folder_path)
