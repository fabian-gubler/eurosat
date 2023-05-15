import numpy as np
import tensorflow as tf

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_raw(value).numpy()]))

def serialize_example(feature0, feature1):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.
    feature = {
        'feature0': _bytes_feature(feature0),
        'feature1': _bytes_feature(feature1),
    }

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def npy_to_tfrecord(x_path, y_path, tfrecord_file):
    # Load the .npy files
    x = np.load(x_path)
    y = np.load(y_path)

    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        for i in range(len(x)):
            example = serialize_example(x[i], y[i])
            writer.write(example)

# Call the function to create the TFRecord file
npy_to_tfrecord('/home/paperspace/eurosat/preprocessed/x_std.npy', '/home/paperspace/eurosat/preprocessed/y.npy', 'data.tfrecord')
