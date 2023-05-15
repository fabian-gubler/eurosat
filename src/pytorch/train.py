import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import EuroSatDataset
from model import get_model
from tqdm import tqdm

# Get the number of classes from the dataset
num_classes = len(EuroSatDataset.IDX_CLASS_LABELS)

# Get the model
model = get_model(num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Decide which device to use (use GPU if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# Move the model to the chosen device
model = model.to(device)

# Define base directory and batch size
prefix = "/data/eurosat/"
prefix = "/home/paperspace/eurosat/"
base_dir = os.path.join(prefix, "data/EuroSAT_MS_NPY_wo_B10_ordered_features_float32")


# Parameters
num_epochs = 20 # Adjust based on overfitting/underfitting observations
batch_size = 64 # You can adjust this according to your GPU memory
val_size = 0.2 # Proportion of training set to use as validation set

# Create dataset instances
train_dataset = EuroSatDataset(base_dir=base_dir, train=True, val_size=val_size)
val_dataset = EuroSatDataset(base_dir=base_dir, train=False, val_size=val_size)


# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
            dataloader = train_loader
        else:
            model.eval()   # Set model to evaluate mode
            dataloader = val_loader

        running_loss = 0.0
        running_corrects = 0

        # Create a progress bar
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        
        # Iterate over data
        for i, (inputs, labels) in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # Update progress bar
            progress_bar.set_description(f"Loss: {loss.item()} Acc: {torch.sum(preds == labels.data)}")
            
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    print()

print('Training complete.')

# After training
model_save_dir = "../models"
model_save_path = os.path.join(model_save_dir, "resnet18.pth")



# Ensure the directory exists
os.makedirs(model_save_dir, exist_ok=True)

torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
