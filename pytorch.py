# --- TODO ---
# Learn more about expected size of model (might improve performance and increase understanding)

# ResNet50, the model you are using, is designed to take in 3-channel (RGB) images of a certain size (usually 224x224). However, the EuroSAT dataset provides images with 13 channels.

# --- SOURCE START ---
# %%
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from tqdm.notebook import tqdm
import seaborn as sns
import random


from sklearn.metrics import confusion_matrix
from sklearn import model_selection
from sklearn.model_selection import train_test_split


import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split, SubsetRandomSampler
from torchvision.transforms import transforms
from torchvision.utils import make_grid
import torch.nn.functional as F

import matplotlib

matplotlib.rcParams["axes.grid"] = False
# --- SOURCE END ---

# %%
import glob

# %%
IDX_CLASS_LABELS = {
    0: 'AnnualCrop',
    1: 'Forest', 
    2: 'HerbaceousVegetation',
    3: 'Highway',
    4: 'Industrial',
    5: 'Pasture',
    6: 'PermanentCrop',
    7: 'Residential',
    8: 'River',
    9: 'SeaLake'
}

# %%
CLASSES = [
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

# %%
eurosat_dir = "/home/fabian/eurosat/data/"

# %%
train_dir = os.path.join(eurosat_dir, "EuroSAT_MS_NPY_wo_B10_ordered_features_float32")
samples = glob.glob(os.path.join(train_dir, "*", "*.npy"))
print(len(samples))

# %%
test_dir = os.path.join(eurosat_dir, "testset")
test_samples = glob.glob(os.path.join(test_dir, "*.npy"))
print(len(test_samples))

# %%
class EuroSATDataset(Dataset):
    def __init__(self, file_list, labels_dict, transform=None):
        self.file_list = file_list
        self.labels_dict = labels_dict
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = np.load(img_path)

        print("Shape of the numpy array: ", img.shape)
        print("Type of the numpy array: ", img.dtype)

        label = self.get_label_from_path(img_path)
        if self.transform:
            img = self.transform(img)
            return img, label

    def get_label_from_path(self, path):
        for i, class_label in self.labels_dict.items():
            if class_label in path:
                return i
        raise ValueError(f"No valid label found in path: {path}")


# %%
# Define a transform function to normalize the images

# However, the EuroSAT dataset images have 13 channels. The normalization values here are specific to ImageNet and RGB images, and you would need to compute your own mean and standard deviation for the EuroSAT dataset. For the sake of starting the training process, you can use a mean and standard deviation of 0.5 for each of the 13 channels, but it is recommended to calculate these values specifically for your dataset.

# Adjust the size of mean and std to match the number of channels in the EuroSAT dataset
mean = [0.5]*13
std = [0.5]*13

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# In the future, you can calculate the specific mean and standard deviation for the EuroSAT dataset, which may improve your model's performance.

# To calculate the mean and standard deviation, you would need to load all your training data, convert it to tensors, then use torch.mean() and torch.std() respectively. Keep in mind that this process might be time and memory consuming depending on the size of your dataset.

# Here's a rough example of how you could calculate the mean and standard deviation:

# Load all training data into a single tensor (this might require a lot of memory)

# all_data = torch.cat([data for data, _ in train_dataset])
#
# # Calculate the mean and std along the channel dimension
# mean = torch.mean(all_data, dim=[0, 2, 3])
# std = torch.std(all_data, dim=[0, 2, 3])
#
# print(mean)
# print(std)


# %%
# Create datasets
train_dataset = EuroSATDataset(samples, IDX_CLASS_LABELS, transform)
test_dataset = EuroSATDataset(test_samples, IDX_CLASS_LABELS, transform)

# %%
# Create data loaders
batch_size = 64  # You can adjust this according to your GPU memory

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %%
# Split the indices of the training dataset to create a validation dataset
val_split = 0.2  # Use 20% of the data for validation
train_len = len(train_dataset)
indices = list(range(train_len))
np.random.shuffle(indices)

# Determine the indices for the split
split = int(np.floor(val_split * train_len))
train_idx, val_idx = indices[split:], indices[:split]

# Create Samplers for the training and validation subsets
train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)

# Create the DataLoaders for training and validation subsets
train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
val_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler)

# %%
# Choose the device (GPU or CPU) for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the model
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(CLASSES))  # Change the last layer for your number of classes
model = model.to(device)  # Move model to device (GPU or CPU)

# Define the loss function (criterion)
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# %%
# Get a batch of training data
inputs, classes = next(iter(train_dl))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

print("Shape of the inputs: ", inputs.shape)
print("Type of the inputs: ", type(inputs))


# %%
# Now you can call the training function
# def train_model(model, criterion, optimizer, num_epochs=25):
#     for epoch in range(num_epochs):
#         print('Epoch {}/{}'.format(epoch, num_epochs - 1))
#         print('-' * 10)
#
#         # Each epoch has a training and validation phase
#         for phase in ['train', 'val']:
#             if phase == 'train':
#                 model.train()  # Set model to training mode
#                 dataloader = train_dl
#             else:
#                 model.eval()   # Set model to evaluate mode
#                 dataloader = val_dl
#
#             running_loss = 0.0
#             running_corrects = 0
#
#             # Iterate over data
#             for inputs, labels in dataloader:
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)
#
#                 # zero the parameter gradients
#                 optimizer.zero_grad()
#
#                 # forward
#                 # track history if only in train
#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(inputs)
#                     _, preds = torch.max(outputs, 1)
#                     loss = criterion(outputs, labels)
#
#                     # backward + optimize only if in training phase
#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()
#
#                 # statistics
#                 running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data)
#
#             epoch_loss = running_loss / len(dataloader.dataset)
#             epoch_acc = running_corrects.double() / len(dataloader.dataset)
#
#             print('{} Loss: {:.4f} Acc: {:.4f}'.format(
#                 phase, epoch_loss, epoch_acc))
#
#     return model
#
# # Train the model
# model_ft = train_model(model, criterion, optimizer, num_epochs=25)
#
