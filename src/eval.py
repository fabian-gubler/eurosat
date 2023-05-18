import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tqdm import tqdm

# Load your trained model
name = "efficientnet"
model = load_model(name)

# Directory containing .npy test files
# test_dir = "/home/ubuntu/eurosat/data/testset"
test_dir = "/home/ubuntu/eurosat/data/testset"

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
counter = 0

# Loop over the test files
for i, file in tqdm(enumerate(test_files), total=len(test_files), desc="Predicting"):

    # Look at first 10 files
    # if counter > 10:
    #     break

    # Load the multispectral image
    multispectral_image = np.load(os.path.join(test_dir, file))

    # Extract and normalize the RGB bands
    image = multispectral_image[:, :, [3, 2, 1]]  # Extract RGB bands
    image = image / 4095.0  # Normalize using the given maximum value

    image = np.clip(image, 0, 1)  # Clip the values to the range [0, 1]

    # Make it same format as training data
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.rot90(image)

    # Add an extra dimension because the model expects a batch
    image = np.expand_dims(image, axis=0)

    # Make prediction
    prediction = model.predict(image, verbose=0)

    # Convert prediction probabilities to class label
    predicted_class = np.argmax(prediction, axis=1)

    # Store the prediction
    predictions[i] = predicted_class

    # Visualize the RGB image
    predicted_class_name = [classes[i] for i in predicted_class]
    plt.imshow(image[0])
    plt.title(f'Predicted class: {predicted_class_name}')
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
df.to_csv("efficientnet.csv", index=False)
