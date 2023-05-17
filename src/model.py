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
from tensorflow.keras.applications.resnet50 import preprocess_input

import tensorflow_datasets as tfds


user = "paperspace"

print("loading data...")

# Assuming your data is stored in x and y
x = np.load(f"/home/{user}/eurosat/preprocessed/x_std.npy")
y = np.load(f"/home/{user}/eurosat/preprocessed/y.npy")

x_rgb = x[:,:,:, [3, 2, 1]].copy() # shape: (27000, 64, 64, 3)



# Split the dataset into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x_rgb, y, test_size=0.2, random_state=42)

# Create a custom input layer for the 64x64x20 input
input_layer = Input(shape=(64, 64, 3))

# Add a resizing layer to resize the input to the size ResNet expects
resizing_layer = Resizing(224, 224, interpolation="Bilinear")(input_layer)

# Load the ResNet101 model without the top classification layer and with custom input
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=resizing_layer)

# Add a custom classification layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)  # additional fully-connected layer
x = Dropout(0.2)(x)  # dropout for regularization, 0.1-0.5, start small
x = Dense(1024, activation='relu')(x)  # additional fully-connected layer
x = Dropout(0.2)(x)  # dropout for regularization, 0.1-0.5, start small
#x = Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)  # L2 regularization with a factor of 0.01, 0.0001 to 0.1, start small

predictions = Dense(y.shape[1], activation='softmax')(x)

# Freeze all layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Unfreeze the last two residual blocks (9 layers)
for layer in base_model.layers[-9:]:
    layer.trainable = True

# Create the final model
model = Model(inputs=input_layer, outputs=predictions)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.003), loss='categorical_crossentropy', metrics=['accuracy'])

# Create data augmentation generator
datagen = ImageDataGenerator(
    rotation_range=20,
    shear_range=0.2,  # added shear transformation
    #width_shift_range=0.2,
    #height_shift_range=0.2,
    #horizontal_flip=True,
    #vertical_flip=True,
    #zoom_range=0.2,  # added zoom
)

# Define the checkpoint callback
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath='resnet50_std_wo_deeper_rgb.h5', monitor='val_accuracy', mode='max', save_best_only=True, save_weights_only=False, verbose=1)
# Define the early stopping callback
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=5, verbose=1, restore_best_weights=True)

# Fit the model with the augmented data
batch_size = 50
epochs = 20
model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
          steps_per_epoch=len(x_train) // batch_size,
          validation_data=(x_test, y_test),
          epochs=epochs,
          callbacks=[checkpoint_callback, early_stopping_callback])  
