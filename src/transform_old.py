import os

import numpy as np

from PIL import Image




# Source and destination folders

src_folder = "/data/eurosat/data/testset/"

dst_folder = "/data/eurosat/data/testset_rgb/"




# Create the destination folder if it doesn't exist

os.makedirs(dst_folder, exist_ok=True)




# Loop over all files in the source folder

for filename in os.listdir(src_folder):

# Check if the file is a .npy file

    if filename.endswith(".npy"):

# Load the image

        multispectral_image = np.load(os.path.join(src_folder, filename))




# Reorder the channels to match the RGB order (assuming 0-based indexing)

        rgb_image = multispectral_image[:, :, [3, 2, 1]]




# Normalize the pixel values to the range [0, 255] for each channel separately

        normalized_image = ((rgb_image - np.min(rgb_image, axis=(0,1))) /

        (np.max(rgb_image, axis=(0,1)) - np.min(rgb_image, axis=(0,1)))) * 255




# Convert the image array to 8-bit unsigned integers

        normalized_image = normalized_image.astype(np.uint8)




# Create an RGB image from the normalized image array

        rgb_image = Image.fromarray(normalized_image)




# Save the RGB image as a .jpg file with high quality

        new_filename = os.path.splitext(filename)[0] + '.jpg'

        rgb_image.save(os.path.join(dst_folder, new_filename), quality=95)




        print("Conversion completed!")

