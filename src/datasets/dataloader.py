
from src.datasets.inbreast import get_inbreast
import pandas as pd
import os
from pydicom.pixels import pixel_array
import numpy as np
from torch.utils.data import Dataset, Dataloader, WeightedRandomSampler, RandomSampler, SequentialSampler


class ImageDataset(Dataset):
    def __init__(self, csv: pd.DataFrame, transform: None, target_transform: None) -> None:
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
        image_path = row['path']
        image = pixel_array(image_path)

        # -- get label
        label = row['target']

        # -- apply transformations
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label



def get_dataframe(datasets: list[str], datasets_path: str) -> pd.DataFrame:
    """
    Return dataframe with items of each dataset selected.

    :param datasets: List with the names of each dataset to use in the training
    :param datasets_path: Path to the folder with datasets. The folder structure
    must contain the names available for 'datasets', and each dataset must have
    a csv file named 'metadata.csv' and a folder with the images named 'images'
    :return dataframe: A DataFrame object containing the targets and paths.
    """
    func = {
        "inbreast": get_inbreast
    }
    entire_df = pd.DataFrame(columns=["target", "path"])

    # -- merge all datasets
    for dataset in datasets:
        dataset_path = os.path.join(datasets_path, dataset)
        dataset_df = func[dataset](os.path.join(dataset_path, "metadata.csv"), os.path.join(dataset_path, "images"))
        entire_df = pd.concat(entire_df, dataset_df)

    return entire_df


def get_dataloader(dataset: ImageDataset, batch_size: int, sampler_cfg: str = '', workers: int = 0, shuffle: bool = False) -> Dataloader:
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
    samplers = {
        "weightened": WeightedRandomSampler(dataset, ),
        "random": RandomSampler,
        "sequential": SequentialSampler,
        "": None
    }
    sampler = samplers[sampler_cfg]
    dataloader = Dataloader(dataset, batch_size, shuffle, sampler, num_workers=workers)
    return dataloader