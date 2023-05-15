import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet50_Weights

def get_model(num_classes):
    # Load the pretrained model
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # Adjust the first layer
    model.conv1 = nn.Conv2d(20, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    # Adjust the last layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model
