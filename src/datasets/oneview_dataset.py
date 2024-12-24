import cv2
import numpy as np
import pandas as pd
from pydicom.pixels import pixel_array
from torch.utils.data import Dataset


class OneViewDataset(Dataset):
    def __init__(
        self, csv: pd.DataFrame, transform=None, target_transform=None
    ) -> None:
        """
        Constructor of the class

        :param csv: The dataframe to store in the class
        :param transform: The image transformations before returning it
        :param target_transform: The target transformations before returning it
        """
        self.csv = csv
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        """Return size of dataset"""
        return len(self.csv)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        """
        Get item from index

        :param idx: Index of item to be returned
        :return values: The image and label of the index
        """
        row = self.csv.iloc[idx]

        # -- get image
        image_path = row["path"]
        if image_path.endswith(".dcm"):
            image = pixel_array(image_path)
        else:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # -- get label
        label = row["target"]

        # -- apply transformations
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def weights(self) -> list[float]:
        """
        Return the weights of each image in the dataset

        :return weights: The weights of each image
        """
        values = self.csv["target"].value_counts()
        weights = [1 / values[i] for i in self.csv["target"].values]
        return weights
