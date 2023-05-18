import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load your trained model
name = "/data/eurosat/models/efficientnet"
model = load_model(name)

# Directory containing .npy test files
# test_dir = "/home/ubuntu/eurosat/data/testset"
test_dir = "/data/eurosat/data/testset"

# Get a list of all .npy files in the directory
test_files = [f for f in os.listdir(test_dir) if f.endswith(".npy")]

# Preallocate a numpy array for your predictions
predictions = np.zeros(len(test_files), dtype=int)

# Define the classes
classes = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]


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

    # Visualize the RGB image
    plt.imshow(rgb_image)
    plt.title(f'Predicted class: {predicted_class}')
    plt.show()


# Map the class indices to actual class names
predicted_class_names = [classes[i] for i in predictions]

# Create a DataFrame for the test IDs and their predicted labels
df = pd.DataFrame(
    data={
        "test_id": np.arange(len(predicted_class_names)),
        "label": predicted_class_names,
    }
)

# Save the DataFrame to a CSV file
df.to_csv(name + ".csv", index=False)
