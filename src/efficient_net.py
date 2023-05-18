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
#     language: python
#     name: python3
# ---

# https://github.com/derevirn/lulc-keras/blob/master/EfficientNet_Finetuned.ipynb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers, callbacks
from tensorflow.keras.applications import ResNet50, MobileNet
from efficientnet.tfkeras import EfficientNetB0, EfficientNetB4, EfficientNetB5
import tensorflow_datasets as tfds
from sklearn.metrics import classification_report, confusion_matrix

# +
BATCH_SIZE = 64

ds, ds_info  = tfds.load('eurosat/rgb', with_info=True, split='train')
figure = tfds.show_examples(ds, ds_info)
# -

(ds_train, ds_test) = tfds.load('eurosat/rgb',  with_info=False,
                                         split=['train[:80%]', 'train[80%:]'],
                                         as_supervised=True)


# +
def convert(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32) # Cast and normalize the image to [0,1]
    return image, label

def augment(image,label):
    image,label = convert(image, label)
    image = tf.image.convert_image_dtype(image, tf.float32) # Cast and normalize the image to [0,1]
    image = tf.image.rot90(image)
    image = tf.image.random_hue(image, max_delta=0.2)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.5) # Random brightness
    return image,label

ds_train = ds_train.cache()
ds_train = ds_train.shuffle(int(ds_info.splits['train'].num_examples * 0.8))
ds_train = ds_train.map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.repeat()
ds_train = ds_train.batch(BATCH_SIZE)
# -

ds_test = ds_test.map(convert, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.cache()
ds_test = ds_test.batch(BATCH_SIZE)

# +
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

#for layer in base_model.layers[:-100]:
#    layer.trainable = False

#for layer in base_model.layers:
#    print(layer.name,layer.trainable)

#base_model.trainable = False

# +
model = models.Sequential()
model.add(base_model)
model.add(layers.GlobalMaxPooling2D())

model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation = 'softmax'))

model.summary()


# +
steps = int(ds_info.splits['train'].num_examples * 0.8) // BATCH_SIZE

sgd = optimizers.SGD(lr = 1e-03, momentum = 0.9, nesterov = True)
adam = optimizers.Adam(lr = 1e-03)

early_stop = callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min',
            patience = 5, restore_best_weights = True, verbose = 1)

reduce_lr = callbacks.ReduceLROnPlateau(monitor = 'val_loss', mode = 'min',
            patience = 2, factor = 0.5, min_lr = 1e-06, verbose = 1)

model.compile(optimizer=adam, loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(ds_train, validation_data=ds_test, epochs=50,
                    steps_per_epoch = steps, callbacks=[early_stop, reduce_lr])

# -

model.save('models\efficientnet_finetuned')

test_loss, test_acc = model.evaluate(ds_test, verbose=1)

# +
fig = plt.figure(figsize=(12,8))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# +
fig = plt.figure(figsize=(12,8))

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# +
from utils import plot_confusion_matrix

predictions = []
labels = []
for test_images, test_labels in ds_test:
  probabilities = model(test_images.numpy())
  predictions.extend(tf.argmax(probabilities, axis=1))
  labels.extend(test_labels.numpy())

classes = ds_info.features['label'].names

# plot_confusion_matrix(labels, predictions, classes, normalize=True)
# -

