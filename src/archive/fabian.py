# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.2-dev
# ---

# +
from tensorflow.keras.layers.experimental.preprocessing import Resizing

x = np.load('/Users/svenschnydrig/Documents/Coding Challenge/data/preprocessed/x_std.npy')
y = np.load('/Users/svenschnydrig/Documents/Coding Challenge/data/preprocessed/y.npy')

# Delete B1 (at index 0)
x = np.delete(x, 18, axis=3)
x = np.delete(x, 17, axis=3)
x = np.delete(x, 16, axis=3)
x = np.delete(x, 15, axis=3)
x = np.delete(x, 14, axis=3)
x = np.delete(x, 13, axis=3)
x = np.delete(x, 12, axis=3)
x = np.delete(x, 9, axis=3)
x = np.delete(x, 0, axis=3)

# Split the dataset into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=42)

# Create a custom input layer for the 64x64x20 input
input_layer = Input(shape=(64, 64, 11))

# Load the ResNet50 model without the top classification layer and with custom input
base_model = ResNet50(weights=None, include_top=False, input_tensor=input_layer)

# Add a custom classification layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(2048, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1))(x)  # additional fully-connected layer with L2 regularization with a factor of 0.01, 0.0001 to 0.1, start small
x = Dropout(0.5)(x)  # dropout for regularization, 0.1-0.5, start small
#x = Dropout(0.3)(x)  # dropout for regularization, 0.1-0.5, start small
#x = Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)  # L2 regularization with a factor of 0.01, 0.0001 to 0.1, start small
predictions = Dense(y.shape[1], activation='softmax')(x)

# Create the final model
model = Model(inputs=input_layer, outputs=predictions)
#model.compile(optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True), loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.003), loss='categorical_crossentropy', metrics=['accuracy'])

# Create data augmentation generator
datagen = ImageDataGenerator(rotation_range=20, shear_range=0.2,)

# Define the checkpoint callback
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath='resnet50_wo_indices_more_reg_bigger.h5', monitor='val_loss', mode='auto', save_best_only=True, save_weights_only=False, verbose=1)
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto', patience=5, verbose=1, restore_best_weights=True)

# Fit the model with the augmented data
batch_size = 128
epochs = 30
model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
          steps_per_epoch=len(x_train) // batch_size,
          validation_data=(x_test, y_test),
          epochs=epochs,
          callbacks=[checkpoint_callback, early_stopping_callback])

# +
from tensorflow.keras.layers.experimental.preprocessing import Resizing

# Assuming your data is stored in x and y
x = np.load('/Users/svenschnydrig/Documents/Coding Challenge/data/preprocessed/x_std.npy')
y = np.load('/Users/svenschnydrig/Documents/Coding Challenge/data/preprocessed/y.npy')

x_rgb = x[:,:,:, [3, 2, 1]].copy()

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

# Save the model
#model.save('resnet50.h5')
