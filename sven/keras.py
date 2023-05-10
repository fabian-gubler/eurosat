# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.2-dev
#   kernelspec:
#     display_name: eurosat
#     language: python
#     name: python3
# ---

# +
import os
import numpy as np

data_dir = "/Users/svenschnydrig/Documents/Coding Challenge/data/EuroSAT_MS_NPY_wo_B10_ordered"
class_names = sorted(os.listdir(data_dir))

x = []
y = []
for i, class_name in enumerate(class_names):
    print(class_name)
    print(i)
    class_dir = os.path.join(data_dir, class_name)
    print(class_dir)
    for filename in os.listdir(class_dir):
        print(filename)
        filepath = os.path.join(class_dir, filename)
        print(filename)
        data = np.load(filepath)
        x.append(data)
        y.append(i)
x = np.stack(x, axis=0)
y = np.array(y)


# -

# # Important the order of bands is:
# ### B1 - 0, B2 - 1, B3 - 2, B4 - 3, B5 - 4, B6 - 5, B7 - 6, B8 - 7, B8A - 8, B9 - 9, B11 - 10, B12 - 11, NDVI - 12, EVI - 13, NDWI - 14, NDBI - 15, NDSI - 16, SAVI - 17, MNDWI - 18, TCB - 19
#

# +
import tensorflow as tf

def create_tf_dataset(x, y, batch_size=32, shuffle=False): # maybe it would be better to set shuffle to True
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        dataset = dataset.shuffle(len(x))
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

# Create the dataset
x = x.astype("float32")  # It's a good practice to convert the data to float32 for better compatibility with TensorFlow
batch_size = 32
tf_dataset = create_tf_dataset(x, y, batch_size, shuffle=False)
# -

tf.data.experimental.cardinality(tf_dataset)

# ## Sanity checks

# Get the first element and label from the TensorFlow dataset
first_element_tf, first_label_tf = next(iter(tf_dataset))
print("First element (TensorFlow):", first_element_tf.numpy()[0])  # Convert the tensor to a numpy array and print the first element
print("First label (TensorFlow):", first_label_tf.numpy()[0])  # Convert the tensor to a numpy array and print the first label

# +
# Get the 3001st element and label from the TensorFlow dataset
index_3001 = 2999  # Index of the 3001st element (0-based indexing)
batch_size = 32

# Calculate which batch the 3001st element belongs to and its index within that batch
batch_index = index_3001 // batch_size
element_index = index_3001 % batch_size

# Get the batch containing the 3001st element
for i, (elements, labels) in enumerate(tf_dataset):
    if i == batch_index:
        element_3001 = elements[element_index].numpy()
        label_3001 = labels[element_index].numpy()
        break

print("3001st element (TensorFlow):", element_3001)
print("3001st label (TensorFlow):", label_3001)

# +
import tensorflow as tf


dataset = tf.data.Dataset.from_tensor_slices((x, y))
# -

gpu = len(tf.config.list_physical_devices('GPU'))>0
print ("GPU is","available" if gpu else "NOT AVAILABLE")
