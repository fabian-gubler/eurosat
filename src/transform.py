import os
import numpy as np
import tensorflow as tf

# Source folder
src_folder = "/data/eurosat/data/testset/"

def get_rgb_values(filename):
    # Load the image
    multispectral_image = np.load(os.path.join(src_folder, filename))

    # Reorder the channels to match the RGB order (assuming 0-based indexing)
    rgb_image = multispectral_image[:, :, [3, 2, 1]]

    # Normalize the pixel values based on ImageNet mean and standard deviation
    rgb_image = ((rgb_image - np.min(rgb_image, axis=(0, 1))) /
                        (np.max(rgb_image, axis=(0, 1)) - np.min(rgb_image, axis=(0, 1)))) * 255

    # Convert the image array to 8-bit unsigned integers
    rgb_image = rgb_image.astype(np.uint8)

    normalized_image = tf.keras.applications.resnet50.preprocess_input(rgb_image)

    return normalized_image

x_testset = []

# Loop over all files in the source folder
for filename in os.listdir(src_folder):
    # Check if the file is a .npy file
    if filename.endswith(".npy"):
        # Get the RGB values for the image
        normalized_image = get_rgb_values(filename)

        # Append the normalized image to x_testset list
        x_testset.append(normalized_image)


# Convert the list of images to a numpy array
x_testset = np.array(x_testset)

# Save the resulting array
np.save('/data/eurosat/data/preprocessed/x_rgb.npy', x_testset)

print(f'testset shape: {x_testset.shape}')

# ------------------------------------------------------------

# def equalize_band(band):
#     # Convert the band to 8-bit (if not already)
#     band = cv2.normalize(band, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#
#     # Perform histogram equalization
#     equalized_band = cv2.equalizeHist(band)
#
#     return equalized_band


# def get_rgb_values_equalized(filename):
#     # Load the image
#     multispectral_image = np.load(os.path.join(src_folder, filename))
#
#     # Reorder the channels to match the RGB order (assuming 0-based indexing)
#     rgb_image = multispectral_image[:, :, [2, 1, 0]]
#
#     # Apply histogram equalization to each band
#     equalized_image = np.stack([equalize_band(band) for band in cv2.split(rgb_image)], axis=-1)
#
#     # print(equalized_image)
#     return equalized_image


# def get_rgb_values(filename):
#     # Load the image
#     multispectral_image = np.load(os.path.join(src_folder, filename))
#
#     # Reorder the channels to match the RGB order (assuming 0-based indexing)
#     rgb_image = multispectral_image[:, :, [3, 2, 1]]
#
#     # Apply histogram equalization
#     img_yuv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2YUV)
#
#     # equalize the histogram of the Y channel
#     img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
#
#     # convert the YUV image back to RGB format
#     rgb_image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
#
#     normalized_image = tf.keras.applications.resnet50.preprocess_input(rgb_image)
#
#     # return normalized_image, image
#     return rgb_image, normalized_image


# def find_min_max_values(src_folder):
#     min_values = [np.inf, np.inf, np.inf]  # Initialize with "infinite" values
#     max_values = [
#         -np.inf,
#         -np.inf,
#         -np.inf,
#     ]  # Initialize with "negative infinite" values
#
#     for filename in os.listdir(src_folder):
#         if filename.endswith(".npy"):
#             # Load the image
#             multispectral_image = np.load(os.path.join(src_folder, filename))
#
#             # Reorder the channels to match the RGB order
#             rgb_image = multispectral_image[:, :, [3, 2, 1]]
#
#             # Update the min and max values for each band
#             for i in range(3):
#                 min_values[i] = min(np.min(rgb_image[:, :, i]), min_values[i])
#                 max_values[i] = max(np.max(rgb_image[:, :, i]), max_values[i])
#
#     return min_values, max_values
#
#
# min_values, max_values = find_min_max_values(src_folder)
# print("Min values for RGB bands: ", min_values)
# print("Max values for RGB bands: ", max_values)
#


# def get_rgb_values(filename):
#     # Load the image
#     multispectral_image = np.load(os.path.join(src_folder, filename))
#
#     # Reorder the channels to match the RGB order (assuming 0-based indexing)
#     rgb_image = multispectral_image[:, :, [3, 2, 1]]
#
#     # Define the min and max values for each band
#     band_min_values = [0, 0, 0]  # replace with the actual min values
#     band_max_values = [17826, 18894, 20506]  # replace with the actual max values
#
#     # Normalize the pixel values based on the predefined min and max values
#     for i in range(3):
#         rgb_image[:, :, i] = ((rgb_image[:, :, i] - band_min_values[i]) /
#                               (band_max_values[i] - band_min_values[i])) * 255
#
#     # Convert the image array to 8-bit unsigned integers
#     rgb_image = rgb_image.astype(np.uint8)
#
#     normalized_image = tf.keras.applications.resnet50.preprocess_input(rgb_image)
#
#     # return normalized_image, image
