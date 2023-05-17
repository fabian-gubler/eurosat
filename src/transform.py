import os

import numpy as np
from PIL import Image


# Source and destination folders
src_folder = "/data/eurosat/data/testset/"
dst_folder = "/data/eurosat/data/testset_rgb/"

# Create the destination folder if it doesn't exist
os.makedirs(dst_folder, exist_ok=True)


def get_rgb_values(filename):
    # Load the image
    multispectral_image = np.load(os.path.join(src_folder, filename))

    # Reorder the channels to match the RGB order (assuming 0-based indexing)
    rgb_image = multispectral_image[:, :, [3, 2, 1]]
    print(rgb_image)

    # Normalize the pixel values to the range [0, 255] for each channel separately
    normalized_image = (
        (rgb_image - np.min(rgb_image, axis=(0, 1)))
        / (np.max(rgb_image, axis=(0, 1)) - np.min(rgb_image, axis=(0, 1)))
    ) * 255

    # Convert the image array to 8-bit unsigned integers
    return normalized_image.astype(np.uint8)


# Loop over all files in the source folder
for filename in os.listdir(src_folder):

    # Check if the file is a .npy file
    if filename.endswith(".npy"):

        # Get the RGB values for the image
        rgb_image = get_rgb_values(filename)
        break

        # Save the image to the destination folder
        # Image.fromarray(rgb_image).save(os.path.join(dst_folder, filename[:-4] + ".png"))


