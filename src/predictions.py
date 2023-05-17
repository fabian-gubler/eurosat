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
# name = "resnet50_std_wo_deeper_rgb.h5"
# model = tf.keras.models.load_model(name)

# Assume that x_testset is your test dataset
# And we normalize it in the same way as you did for your training set
# x_testset = ...

# eurosat_dir = "/home/ubuntu"
# x_testset = np.load(f"{eurosat_dir}/preprocessed/x_rgb.npy")

x_testset = np.load("/data/eurosat/data/preprocessed/x_rgb.npy")

print(x_testset.shape)
eurosat_dir = "/home/ubuntu/eurosat"
x_testset = np.load(f"{eurosat_dir}/preprocessed/x_rgb.npy")

# Delete B1 (at index 0) and three other bands (let's assume at indices 8, 9, and 10)

# x_testset = np.delete(x_testset, 0, axis=3)
# x_testset = np.delete(x_testset, 8, axis=3)

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
# predictions = model.predict(x_testset)
#
# # Get the class with highest probability for each test image
# predicted_classes = np.argmax(predictions, axis=1)
#
# # Map the class indices to actual class names
# predicted_class_names = [classes[i] for i in predicted_classes]
#
# # Create a DataFrame for the test IDs and their predicted labels
# df = pd.DataFrame(data={
#     'test_id': np.arange(len(predicted_class_names)),
#     'label': predicted_class_names
# })
#
# # Save the DataFrame to a CSV file
# df.to_csv(name+'.csv', index=False)
