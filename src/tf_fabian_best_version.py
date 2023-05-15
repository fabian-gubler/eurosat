import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tqdm.keras import TqdmCallback
from tensorflow.keras import regularizers

# Assuming your data is stored in x and y
x = np.load("/home/paperspace/eurosat/preprocessed/x_std.npy")
y = np.load("/home/paperspace/eurosat/preprocessed/y.npy")


# Delete B1 (at index 0)
x = np.delete(x, 0, axis=3)
x = np.delete(x, 8, axis=3)


# Split the dataset into train and test sets

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)


# Create a custom input layer for the 64x64x20 input

input_layer = Input(shape=(64, 64, 16))


# Load the ResNet50 model without the top classification layer and with custom input

base_model = ResNet50(weights=None, include_top=False, input_tensor=input_layer)


# Add a custom classification layer

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(1024, activation="relu")(x)  # additional fully-connected layer

x = Dropout(0.2)(x)  # dropout for regularization, 0.1-0.5, start small

x = Dense(1024, activation="relu")(x)  # additional fully-connected layer

x = Dropout(0.2)(x)  # dropout for regularization, 0.1-0.5, start small

# x = Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)  # L2 regularization with a factor of 0.01, 0.0001 to 0.1, start small


predictions = Dense(y.shape[1], activation="softmax")(x)


# Create the final model

model = Model(inputs=input_layer, outputs=predictions)


# Compile the model

model.compile(
    optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.003),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)


# Create data augmentation generator

datagen = ImageDataGenerator(
    rotation_range=20,
    shear_range=0.2,  # added shear transformation
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # horizontal_flip=True,
    # vertical_flip=True,
    # zoom_range=0.2,  # added zoom
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

epochs = 20

model.fit(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch=len(x_train) // batch_size,
    validation_data=(x_test, y_test),
    epochs=epochs,
    callbacks=[TqdmCallback(verbose=1), checkpoint_callback, early_stopping_callback],
    verbose=0,
)  # set verbose=0 to prevent standard output


# Save the model

# model.save('resnet50.h5')
