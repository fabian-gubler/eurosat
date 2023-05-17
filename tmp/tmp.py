import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (Concatenate, Conv2D, Dense, Dropout,
                                     Flatten, GlobalAveragePooling2D, Input,
                                     MaxPooling2D)
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_datasets as tfds

# Load EuroSAT data
DATA_DIR = "../data"  # replace with your data directory
ds, ds_info = tfds.load("eurosat/rgb", with_info=True, split="train", data_dir=DATA_DIR)

# Preprocess the dataset
def preprocess(features):
    image = features["image"]
    # Convert the images to float type, also scale values from 0 to 1
    image = tf.image.convert_image_dtype(image, tf.float32)
    # Use the ResNet50 preprocess_input function
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image, features["label"]


ds = ds.map(preprocess)

# return the first 9 images from the dataset
for image, label in ds.take(9):
    print("Label: %d" % label.numpy())
    print("Image shape:", image.shape)
    print("Pixel values range from %.4f to %.4f" % (np.min(image), np.max(image)))
