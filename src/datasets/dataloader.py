"""
This module constains the codes to load the datasets converted and return the
dataloader of the selected datasets to use
"""

import logging
import os
import sys

import cv2
import numpy as np
import pandas as pd
from pydicom.pixels import pixel_array
from sklearn.model_selection import train_test_split
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
    WeightedRandomSampler,
)

from src.datasets.inbreast import get_inbreast

logger = logging.getLogger()


class ImageDataset(Dataset):
    def __init__(
        self, csv: pd.DataFrame, transform=None, target_transform=None
    ) -> None:
        """
        Constructor of the class

        :param csv: The dataframe to store in the class
        :param transform: The image transformations before returning it
        :param target_transform: The taregt transformations before returning it
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
            image = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR)

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


def split_dataset(
    dataset: pd.DataFrame, split: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split datasets into training (train/validation) and testing sets

    :param dataset: DataFrame of the entire dataset
    :param split: Value to maintain in the training dataframe
    :param seed: Value to achieve reproducible results
    :return dataframes: The train and test dataframes
    """
    train_df, test_df = [], []
    for _, group in dataset.groupby("dataset"):
        train_aux, test_aux = train_test_split(
            group, train_size=split, random_state=seed
        )
        train_df.append(train_aux)
        test_df.append(test_aux)
    train_df = pd.concat(train_df, ignore_index=True)
    test_df = pd.concat(test_df, ignore_index=True)
    return train_df, test_df


def get_dataframe(
    datasets: list[str], datasets_path: str, seed: int, split: float = 0.9
) -> pd.DataFrame:
    """
    Return dataframe with items of each dataset selected.

    :param datasets: List with the names of each dataset to use in the training
    :param datasets_path: Path to the folder with datasets. The folder structure
    must contain the names available for 'datasets', and each dataset must have
    a csv file named 'metadata.csv' and a folder with the images named 'images'
    :param seed: Value for reproducible results
    :param split: Quantity to maintain in training dataframe
    :return dataframe: A DataFrame object containing the targets and paths.
    """
    func = {"inbreast": get_inbreast}
    total_df = pd.DataFrame(columns=["target", "path", "dataset"])

    # -- merge all datasets
    for dataset in datasets:
        logger.info(f"Loading dataset {dataset}")
        dataset_path = os.path.join(datasets_path, dataset)
        if not os.path.exists(dataset_path):
            logger.error(f"Path {dataset_path} does not exist")
            sys.exit()

        dataset_df = func[dataset](
            os.path.join(dataset_path, "metadata.csv"),
            os.path.join(dataset_path, "images"),
        )
        dataset_df["dataset"] = dataset
        total_df = pd.concat([total_df, dataset_df], ignore_index=True)

    total_df["target"] = total_df["target"].astype(int)
    total_df["path"] = total_df["path"].astype(str)
    return split_dataset(total_df, split, seed)


def get_dataloader(
    dataset: ImageDataset,
    batch_size: int,
    sampler_cfg: str = "",
    workers: int = 0,
    shuffle: bool = False,
) -> DataLoader:
    """
    Get instance of dataloader according to dataset

    :param dataset: Instance of the ImageDataset
    :param batch_size: The number of items inside a batch
    :param sampler_cfg: The name of the sampler to use. Options are 'weightened','random' and 'sequential'.
    If none, no sampler will be used
    :param workers: Number of workers to load the data
    :param shuffle: Whether to shuffle the data before creating the dataloader
    :
    """
    num_samples = len(dataset)
    samplers = {
        "weightened": WeightedRandomSampler(
            weights=dataset.weights(), num_samples=num_samples
        ),
        "random": RandomSampler(data_source=dataset, num_samples=num_samples),
        "sequential": SequentialSampler(data_source=dataset),
    }
    sampler = samplers.get(sampler_cfg, None)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=workers,
    )
    return dataloader
