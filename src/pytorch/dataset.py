from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import os


class EuroSatDataset(Dataset):
    IDX_CLASS_LABELS = {
        0: "AnnualCrop",
        1: "Forest",
        2: "HerbaceousVegetation",
        3: "Highway",
        4: "Industrial",
        5: "Pasture",
        6: "PermanentCrop",
        7: "Residential",
        8: "River",
        9: "SeaLake",
    }

    def __init__(self, base_dir, train=True, transform=None, val_size=0.2):
        self.transform = transform
        self.train = train
        self.data = []
        self.labels = []

        for label_idx, label_name in self.IDX_CLASS_LABELS.items():
            class_dir = os.path.join(base_dir, label_name)
            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)
                self.data.append(file_path)
                self.labels.append(label_idx)

        (
            self.data_train,
            self.data_val,
            self.labels_train,
            self.labels_val,
        ) = train_test_split(
            self.data, self.labels, test_size=val_size, random_state=42
        )

    def __len__(self):
        if self.train:
            return len(self.data_train)
        else:
            return len(self.data_val)

    def __getitem__(self, idx):
        if self.train:
            file_path = self.data_train[idx]
            label = self.labels_train[idx]
        else:
            file_path = self.data_val[idx]
            label = self.labels_val[idx]

        sample = np.load(file_path)

        if self.transform:
            sample = self.transform(sample)

        return sample, label
