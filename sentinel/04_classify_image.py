import numpy as np
from skimage.io import imread
from tensorflow.keras.models import load_model
from tqdm import tqdm
import os
import glob
import pandas as pd


# input files
path_to_images = "../data/testset"
path_to_model = "../data/models/vgg/vgg_ms_transfer_alternative_final.27-0.985.hdf5"

# create directory for output
if not os.path.exists("../data/output"):
    os.makedirs("../data/output")

# output files
path_to_output_csv = "../data/output/predictions.csv"

# Create DataFrame to store results
results_df = pd.DataFrame(columns=["Image", "Label", "Probability"])

# read model
model = load_model(path_to_model)

# Define the classes
classes = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

# Iterate over each test image
for path_to_image in glob.glob(os.path.join(path_to_images, '*.npy')):

# for i, file in tqdm(enumerate(test_files), total=len(test_files), desc="Predicting"):


    # read image
    if path_to_image.endswith('.npy'):
        image = np.load(path_to_image)
    else:
        image = np.array(imread(path_to_image), dtype=float)

    missing_band = np.zeros((image.shape[0], image.shape[1], 1))
    image = np.concatenate((image[..., :10], missing_band, image[..., 10:]), axis=-1)

    print(image.shape) # Shape: (64, 64, 13)


    # get input shape of model
    _, input_rows, input_cols, input_channels = model.layers[0].input_shape[0]
    _, output_classes = model.layers[-1].output_shape

    # import correct preprocessing
    if input_channels is 3:
        from image_functions import preprocessing_image_rgb as preprocessing_image
    else:
        from image_functions import preprocessing_image_ms as preprocessing_image

    # don't forget to preprocess
    image = preprocessing_image(image)

    # Add an extra dimension because the model expects a batch
    image = np.expand_dims(image, axis=0)

    # Make prediction
    prediction = model.predict(image, verbose=0)
    print("predicted")

    # Convert prediction probabilities to class label
    predicted_class = np.argmax(prediction, axis=1)

    predicted_class_name = [classes[i] for i in predicted_class]

    print(f'Prediction: {predicted_class_name}')

    # image_name = os.path.basename(path_to_image)
    # results_df = results_df.append({"Image": image_name, 
    #                                 "Label": image_classified_label, 
    #                                 "Probability": image_classified_prob}, 
    #                                 ignore_index=True)

# Save results to CSV
results_df.to_csv(path_to_output_csv, index=False)
