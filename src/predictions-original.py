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

# +
import tensorflow as tf
import numpy as np
import pandas as pd

# Load the model
name = "EfficientNetB7_64x64x20_rot_20_shear_2.h5"
model = tf.keras.models.load_model(name)

# Assume that x_testset is your test dataset
# And we normalize it in the same way as you did for your training set
# x_testset = ...

x_testset = np.load('/Users/svenschnydrig/Documents/Coding Challenge/data/x_testset.npy')

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

# Make predictions
predictions = model.predict(x_testset)

# Get the class with highest probability for each test image
predicted_classes = np.argmax(predictions, axis=1)

# Map the class indices to actual class names
predicted_class_names = [classes[i] for i in predicted_classes]

# Create a DataFrame for the test IDs and their predicted labels
df = pd.DataFrame(data={
    'test_id': np.arange(len(predicted_class_names)),
    'label': predicted_class_names
})

# Save the DataFrame to a CSV file
df.to_csv(name+'.csv', index=False)
