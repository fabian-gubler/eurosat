# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.2-dev
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# from google.colab import drive
# drive.mount('/content/drive')

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (Concatenate, Conv2D, Dense, Dropout,
                                     Flatten, GlobalAveragePooling2D, Input,
                                     MaxPooling2D)
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm.keras import TqdmCallback

# x = np.load('/content/drive/My Drive/data/x_std.npy')
# y = np.load('/content/drive/My Drive/data/y.npy')

user = "paperspace"

print("loading data...")

# Assuming your data is stored in x and y
x = np.load(f"/home/{user}/eurosat/preprocessed/x_std.npy")
y = np.load(f"/home/{user}/eurosat/preprocessed/y.npy")

# Split the original dataset into RGB and additional bands
x_rgb = x[:,:,:, [3, 2, 1]].copy()  # RGB bands
x_additional = np.delete(x, [3, 2, 1], axis=3)  # All other bands

# Split the dataset into train and test sets
x_train_rgb, x_test_rgb, y_train, y_test = train_test_split(x_rgb, y, test_size=0.2, random_state=42)
x_train_additional, x_test_additional, _, _ = train_test_split(x_additional, y, test_size=0.2, random_state=42)  # We don't need to split y again

print("create custom input layer...")

# Create a custom input layer for the RGB and additional bands
input_layer_rgb = Input(shape=(64, 64, 3))
input_layer_additional = Input(shape=(64, 64, x_additional.shape[3]))

# Add a resizing layer to resize the RGB input to the size ResNet expects
resizing_layer = Resizing(224, 224, interpolation="Bilinear")(input_layer_rgb)

# Load the ResNet50 model without the top classification layer and with custom input
base_model_rgb = ResNet50(weights='imagenet', include_top=False, input_tensor=resizing_layer)

# For additional bands, create a separate ResNet50 model without pretrained weights
base_model_additional = ResNet50(weights=None, include_top=False, input_tensor=input_layer_additional)

# Ensure unique layer names
for i, layer in enumerate(base_model_additional.layers):
    layer._name = 'additional_' + str(i) + '_' + layer.name

for i, layer in enumerate(base_model_rgb.layers):
    layer._name = 'rgb_' + str(i) + '_' + layer.name


print("extract & combine features...")

# Extract features from the RGB and additional bands
features_rgb = GlobalAveragePooling2D()(base_model_rgb.output)
features_additional = GlobalAveragePooling2D()(base_model_additional.output)

# Combine the features and add a classification layer
combined_features = tf.keras.layers.concatenate([features_rgb, features_additional])
x = Dense(1024, activation="relu")(combined_features)
x = Dropout(0.2)(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.2)(x)
predictions = Dense(y.shape[1], activation="softmax")(x)

# Freeze all layers in the base models
for layer in base_model_rgb.layers:
    layer.trainable = False
for layer in base_model_additional.layers:
    layer.trainable = False

# Unfreeze the last two residual blocks (9 layers)
for layer in base_model_rgb.layers[-9:]:
    layer.trainable = True
for layer in base_model_additional.layers[-9:]:
    layer.trainable = True

print("create model...")

# Create the final model
model = Model(inputs=[input_layer_rgb, input_layer_additional], outputs=predictions)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.003),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

datagen = ImageDataGenerator(
    rotation_range=20,
    shear_range=0.2,  # added shear transformation
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # horizontal_flip=True,
    # vertical_flip=True,
    # zoom_range=0.2,  # added zoom
)


# + id="20Gp25hOJUmZ"
# Define the early stopping callback

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="resnet50_std_wo_deeper_batch_128.h5",
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    save_weights_only=False,
    verbose=1,
)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", mode="max", patience=5, verbose=1, restore_best_weights=True
)

print("train model...")

batch_size = 50
epochs = 20

class CombinedDataGenerator:
    def __init__(self, datagen, x_rgb, x_additional, y, batch_size):
        self.rgb_gen = datagen.flow(x_rgb, y, batch_size=batch_size)
        self.x_additional = x_additional
        self.batch_size = batch_size

    def __len__(self):
        return len(self.rgb_gen)

    def __getitem__(self, index):
        rgb_batch, y_batch = self.rgb_gen[index]
        additional_batch = self.x_additional[index*self.batch_size:(index+1)*self.batch_size]
        return [rgb_batch, additional_batch], y_batch

# Create an instance of the combined data generator
train_gen = CombinedDataGenerator(datagen, x_train_rgb, x_train_additional, y_train, batch_size)

# Now you can use the combined data generator with the fit function
model.fit(
    train_gen,
    steps_per_epoch=len(x_train_rgb) // batch_size,
    validation_data=([x_test_rgb, x_test_additional], y_test),
    epochs=epochs,
    callbacks=[TqdmCallback(verbose=1), checkpoint_callback, early_stopping_callback],
    verbose=0,
)
