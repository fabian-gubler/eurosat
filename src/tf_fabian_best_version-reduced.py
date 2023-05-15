import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (Dense, Dropout, GlobalAveragePooling2D,
                                     Input, BatchNormalization)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm.keras import TqdmCallback

user = "paperspace"

# Assuming your data is stored in x and y
x = np.load(f"/home/{user}/eurosat/preprocessed/x_std.npy")
y = np.load(f"/home/{user}/eurosat/preprocessed/y.npy")

# Check the shape of the input data
print(f"Original shape of x: {x.shape}")

# Reduce the data size for faster experimentation, use only 10% of data
x, _, y, _ = train_test_split(x, y, test_size=0.9, random_state=42)

# Delete B1 (at index 0) and three other bands (let's assume at indices 8, 9, and 10)
x = np.delete(x, 0, axis=3)
x = np.delete(x, 8, axis=3)

# Check the shape of the input data after deleting the bands
print(f"Shape of x after deleting the bands: {x.shape}")

# Ensure that the depth of the input data is 16 after deleting the bands
assert x.shape[3] == 18, "The depth of the input data must be 18"

# Split the dataset into train and test sets

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)


# Create a custom input layer for the 64x64x20 input

input_layer = Input(shape=(64, 64, 18))


# Load the ResNet50 model without the top classification layer and with custom input
base_model = ResNet50(weights=None, include_top=False, input_tensor=input_layer)


# Add a custom classification layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)
x = BatchNormalization()(x)  # added batch normalization
x = Dropout(0.2)(x)
x = Dense(1024, activation="relu")(x)
x = BatchNormalization()(x)  # added batch normalization
x = Dropout(0.2)(x)

# x = Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)  # L2 regularization with a factor of 0.01, 0.0001 to 0.1, start small


predictions = Dense(y.shape[1], activation="softmax")(x)


# Create the final model

model = Model(inputs=input_layer, outputs=predictions)


# Create a learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.003,
    decay_steps=10000,
    decay_rate=0.9)

# Compile the model with SGD and learning rate schedule
model.compile(
    optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=lr_schedule),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)



# Create data augmentation generator

# Compile the model

model.compile(
    optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.003),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)


# Define the checkpoint callback

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="resnet50_std_wo_deeper_batch_128.h5",
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

batch_size = 128

epochs = 12 # <--- increase after successful generalization

model.fit(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch=len(x_train) // batch_size,
    validation_data=(x_test, y_test),
    epochs=epochs,
    callbacks=[TqdmCallback(verbose=1), checkpoint_callback, early_stopping_callback],
    verbose=0,
)  # set verbose=0 to prevent standard output


# Save the model

model.save("resnet50.h5")
