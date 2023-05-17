import os
import numpy as np
import matplotlib.pyplot as plt

# Source folder
src_folder = "/data/eurosat/data/testset/"

# ImageNet mean and standard deviation
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def get_rgb_values(filename):
    # Load the image
    multispectral_image = np.load(os.path.join(src_folder, filename))

    # Reorder the channels to match the RGB order (assuming 0-based indexing)
    rgb_image = multispectral_image[:, :, [3, 2, 1]]

    # Normalize the pixel values based on ImageNet mean and standard deviation
    normalized_image = ((rgb_image - np.min(rgb_image, axis=(0,1))) /
                        (np.max(rgb_image, axis=(0,1)) - np.min(rgb_image, axis=(0,1)))) * 255

    return normalized_image

# Loop over all files in the source folder
for filename in os.listdir(src_folder):
    # Check if the file is a .npy file
    if filename.endswith(".npy"):
        # Get the RGB values for the image
        rgb_image = get_rgb_values(filename)

        print(rgb_image)

        # Visualize the image
        # plt.imshow(rgb_image.astype(np.uint8))
        # plt.axis('off')
        # plt.show()
