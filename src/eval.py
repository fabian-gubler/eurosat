import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model('models/efficientnet_finetuned')

# Directory containing .npy test files
# test_dir = "/home/ubuntu/eurosat/data/testset"
test_dir = "/data/eurosat/data/testset"

# Get a list of all .npy files in the directory
test_files = [f for f in os.listdir(test_dir) if f.endswith(".npy")]

# Preallocate a numpy array for your predictions
predictions = np.zeros(len(test_files), dtype=int)

# Counter for files with values exceeding 4095
exceeding_files = 0

# Loop over the test files
for i, file in enumerate(test_files):
    # Load the multispectral image
    multispectral_image = np.load(os.path.join(test_dir, file))

    # Extract and normalize the RGB bands
    # rgb_image = multispectral_image[:, :, [3, 2, 1]]  # Extract RGB bands
    rgb_image = multispectral_image[:, :, [1, 2, 3]]  # Extract RGB bands
    rgb_image = rgb_image / 4095.0  # Normalize using the given maximum value
    rgb_image = np.clip(rgb_image, 0, 1)  # Clip the values to the range [0, 1]

    # Check if any value in the RGB bands exceeds 4095
    if np.max(rgb_image) > 4095:
        exceeding_files += 1

        # Normalize the pixel values based on ImageNet mean and standard deviation
    # rgb_image = ((rgb_image - np.min(rgb_image, axis=(0, 1))) /
    #                     (np.max(rgb_image, axis=(0, 1)) - np.min(rgb_image, axis=(0, 1)))) * 255

    # Convert the image array to 8-bit unsigned integers
    # rgb_image = rgb_image.astype(np.uint8)

    # Visualize the RGB image
    # plt.imshow(rgb_image)
    # plt.title(file)  # Show the file name as title
    # plt.show()

    # Ensure the image is of the correct size (adjust dimensions as necessary)
    rgb_image = tf.image.resize(rgb_image, [64, 64])

    # Add an extra dimension because the model expects a batch
    rgb_image = np.expand_dims(rgb_image, axis=0)

    # Make prediction
    prediction = model.predict(rgb_image)

    # Convert prediction probabilities to class label
    predicted_class = np.argmax(prediction, axis=1)

    # Store the prediction
    predictions[i] = predicted_class

# At this point, 'predictions' contains the predicted class for each .npy file
print(f"Number of files with RGB values exceeding 4095: {exceeding_files}")
