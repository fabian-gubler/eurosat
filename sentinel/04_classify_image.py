import numpy as np
from skimage.io import imread
from tensorflow.keras.models import load_model
from tqdm import tqdm
import os
import glob
import pandas as pd


# input files
path_to_images = "../data/testset/"    # Folder with test images
path_to_model = "../data/models/vgg_rgb_transfer_final.56.hdf5"

# output files
path_to_output_csv = "../data/output/predictions.csv"

# Create DataFrame to store results
results_df = pd.DataFrame(columns=["Image", "Label", "Probability"])

# read model
model = load_model(path_to_model)

# Iterate over each test image
for path_to_image in glob.glob(os.path.join(path_to_images, '*.npy')):

    # read image
    if path_to_image.endswith('.npy'):
        image = np.load(path_to_image)
    else:
        image = np.array(imread(path_to_image), dtype=float)

    _, num_cols_unpadded, _ = image.shape

    # get input shape of model
    _, input_rows, input_cols, input_channels = model.layers[0].input_shape
    _, output_classes = model.layers[-1].output_shape
    in_rows_half = int(input_rows/2)
    in_cols_half = int(input_cols/2)

    # import correct preprocessing
    if input_channels is 3:
        from image_functions import preprocessing_image_rgb as preprocessing_image
    else:
        from image_functions import preprocessing_image_ms as preprocessing_image

    # pad image
    image = np.pad(image, ((input_rows, input_rows),
                        (input_cols, input_cols),
                        (0, 0)), 'symmetric')

    # don't forget to preprocess
    image = preprocessing_image(image)
    num_rows, num_cols, _ = image.shape

    # sliding window over image
    image_classified_prob = np.zeros((num_rows, num_cols, output_classes))
    row_images = np.zeros((num_cols_unpadded, input_rows,
                        input_cols, input_channels))
    for row in tqdm(range(input_rows, num_rows-input_rows), desc="Processing..."):
        # get all images along one row
        for idx, col in enumerate(range(input_cols, num_cols-input_cols)):
            # cut small image patch
            row_images[idx, ...] = image[row-in_rows_half:row+in_rows_half,
                                        col-in_cols_half:col+in_cols_half, :]
        # classify images
        row_classified = model.predict(row_images, batch_size=1024, verbose=0)
        # put them to final image
        image_classified_prob[row, input_cols:num_cols-input_cols, :] = row_classified

    # crop padded final image
    image_classified_prob = image_classified_prob[input_rows:num_rows-input_rows,
                                                input_cols:num_cols-input_cols, :]
    image_classified_label = np.argmax(image_classified_prob, axis=-1)
    image_classified_prob = np.sort(image_classified_prob, axis=-1)[..., -1]

    # Store image name, label and probability to the DataFrame
    image_name = os.path.basename(path_to_image)
    results_df = results_df.append({"Image": image_name, 
                                    "Label": image_classified_label, 
                                    "Probability": image_classified_prob}, 
                                    ignore_index=True)

# Save results to CSV
results_df.to_csv(path_to_output_csv, index=False)
