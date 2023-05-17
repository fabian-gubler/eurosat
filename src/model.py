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
from tensorflow_datasets import tfds

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

# Split dataset into train and validation
train_size = int(0.7 * ds_info.splits["train"].num_examples)
val_size = int(0.3 * ds_info.splits["train"].num_examples)
train_dataset = ds.take(train_size).batch(32)
val_dataset = ds.skip(train_size).batch(32)

# Create a custom input layer for the 64x64x3 input
input_layer = tf.keras.layers.Input(shape=(64, 64, 3))

# Add a resizing layer to resize the input to the size ResNet expects
resizing_layer = tf.keras.layers.Resizing(224, 224, interpolation="bilinear")(
    input_layer
)

# Load the ResNet50 model without the top classification layer and with custom input
base_model = ResNet50(
    weights="imagenet", include_top=False, input_tensor=resizing_layer
)

# Add a custom classification layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)  # additional fully-connected layer
x = Dropout(0.2)(x)  # dropout for regularization
x = Dense(1024, activation="relu")(x)  # additional fully-connected layer
x = Dropout(0.2)(x)  # dropout for regularization

predictions = Dense(ds_info.features["label"].num_classes, activation="softmax")(x)

# Freeze all layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Unfreeze the last two residual blocks (9 layers)
for layer in base_model.layers[-9:]:
    layer.trainable = True

# Create the final model
model = Model(inputs=input_layer, outputs=predictions)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# Define the checkpoint callback
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="resnet50_std_wo_deeper_rgb.h5",
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    save_weights_only=False,
    verbose=1,
)

# Define the early stopping callback
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", mode="max", patience=5, verbose=1, restore_best_weights=True
)

# Fit the model with the augmented data
epochs = 20
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    callbacks=[checkpoint_callback, early_stopping_callback],
)
