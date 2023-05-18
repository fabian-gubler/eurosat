import glob
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from image_functions import preprocessing_image_ms as preprocessing_image
from skimage.io import imread
from tensorflow.keras.models import load_model
from tqdm import tqdm

# create directory for output
if not os.path.exists("../data/output"):
    os.makedirs("../data/output")

# output files
path_to_output_csv = "../data/output/predictions.csv"

# load model
path_to_model = "../data/models/vgg/vgg_ms_transfer_alternative_final.27-0.985.hdf5"
model = load_model(path_to_model)

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

print("preprocessing...")

# load test set
prefix = "/home/ubuntu/eurosat"
x_test = np.load(f"{prefix}/preprocessed/x_testset_std.npy")

# remove unwanted bands
bands_to_remove = [0, 9, 12, 13, 15, 16, 17]
x_test = np.delete(x_test, bands_to_remove, axis=3)

# preprocess the entire test set
x_test = preprocessing_image(x_test)

print("predicting...")

# Make predictions
predictions = model.predict(x_test)

# Get the class with highest probability for each test image
predicted_classes = np.argmax(predictions, axis=1)

# Map the class indices to actual class names
predicted_class_names = [classes[i] for i in predicted_classes]

# Create a DataFrame for the test IDs and their predicted labels
df = pd.DataFrame(
    data={
        "test_id": np.arange(len(predicted_class_names)),
        "label": predicted_class_names,
    }
)

# Save the DataFrame to a CSV file
df.to_csv(path_to_output_csv, index=False)
