from torchvision.transforms import Compose, ToTensor
from dataset import EuroSatDataset
from model import get_model
from torch.autograd import Variable
import torch
import pandas as pd
import os

# Get the number of classes from the dataset
num_classes = len(EuroSatDataset.IDX_CLASS_LABELS)


def create_submission():
    model = get_model(num_classes)
    model.load_state_dict(torch.load("../models/resnet18.pth"))
    model.eval()

    # Prepare dataset
    test_dir = "../data/testset"
    transform_test = Compose([
        ToTensor(),
    ])
    
    test_dataset = EuroSatDataset(test_dir, is_test=True, transform=transform_test)

    # Prepare data loader
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    # Prepare submission dataframe
    submission_df = pd.DataFrame(columns=["test_id", "label"])

    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            inputs = Variable(inputs)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            test_id = test_dataset.filenames[i]
            predicted_label = EuroSatDataset.IDX_CLASS_LABELS[predicted.item()]
            submission_df = submission_df.append({"test_id": test_id, "label": predicted_label}, ignore_index=True)

    submission_df.to_csv("../data/submission.csv", index=False)

if __name__ == "__main__":
    create_submission()
