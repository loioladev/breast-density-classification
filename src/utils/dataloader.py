"""
This module constains the codes to load the datasets converted and return the
dataloader of the selected datasets to use
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    WeightedRandomSampler,
)

from src.datasets.inbreast_converter import InBreastConverter
from src.datasets.bmcd_converter import BMCDConverter
from src.datasets.rsna_converter import RSNAConverter
from src.datasets.miniddsm_converter import MiniDDSMConverter
from src.datasets.vindr_converter import VinDrConverter
from src.datasets.oneview_dataset import OneViewDataset

logger = logging.getLogger()


def distribution_split_dataset(dataset: pd.DataFrame, mode: str, seed: int) -> pd.DataFrame:
    """
    Split the dataset based on the distribution of the target.

    :param dataset: The dataset to split
    :param mode: The mode to split the dataset. Options are 'random', 'sequential' and 'balanced'
    :param seed: The seed for reproducibility
    :return dataset: The dataset split
    """
    if mode == 'sequential':
        return dataset.sort_values(by=["target", "dataset"])
    elif mode == 'random':
        return dataset.sample(frac=1, random_state=seed)
    
    # -- balanced mode
    df = dataset.copy()
    target_column = "target"
    dataset_column = "dataset"
    target_counts = df[target_column].value_counts()
    min_target_count = target_counts.min()
    
    balanced_samples = []

    # -- process each target value
    for target_value in target_counts.index:
        target_subset = df[df[target_column] == target_value]
        
        # -- calculate dataset proportions for current target
        dataset_proportions = (target_subset[dataset_column].value_counts() / 
                             len(target_subset))
        
        # -- calculate number of samples needed from each dataset
        samples_per_dataset = (dataset_proportions * min_target_count).round().astype(int)
        
        # -- adjust samples to exactly match min_target_count
        while samples_per_dataset.sum() != min_target_count:
            if samples_per_dataset.sum() < min_target_count:
                # -- add one to largest proportion
                idx = dataset_proportions.idxmax()
                samples_per_dataset[idx] += 1
            else:
                # -- subtract one from smallest non-zero value
                non_zero_idx = samples_per_dataset[samples_per_dataset > 0].idxmin()
                samples_per_dataset[non_zero_idx] -= 1
        
        # -- sample from each dataset according to calculated proportions
        target_balanced = pd.DataFrame()
        for dataset, n_samples in samples_per_dataset.items():
            if n_samples > 0:
                dataset_subset = target_subset[target_subset[dataset_column] == dataset]
                sampled = dataset_subset.sample(n=n_samples, random_state=42)
                target_balanced = pd.concat([target_balanced, sampled])
        
        balanced_samples.append(target_balanced)
    
    # -- combine all balanced samples
    final_df = pd.concat(balanced_samples, axis=0)
    
    return final_df




def split_dataset(
    dataset: pd.DataFrame, split: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split datasets into training (train/validation) and testing sets.
    The dataframe must have the column 'dataset' to split the data
    evenly between the datasets.

    :param dataset: DataFrame of the entire dataset
    :param split: Value to maintain in the training dataframe
    :param seed: Value to achieve reproducible results
    :return dataframes: The train and test dataframes
    """
    # -- for each dataset, split the data
    train_df, test_df = [], []
    for _, group in dataset.groupby("dataset"):
        train_aux, test_aux = train_test_split(
            group, test_size=1.0-split, random_state=seed
        )
        train_df.append(train_aux)
        test_df.append(test_aux)

    # -- merge lists in a single dataframe
    train_df = pd.concat(train_df, ignore_index=True)
    test_df = pd.concat(test_df, ignore_index=True)

    return train_df, test_df


def get_dataframe(
    datasets: list[str], datasets_path: str, seed: int, split_mode: str = 'random', split: float = 0.9
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return dataframe with items of each dataset selected.

    :param datasets: List with the names of each dataset to use in the training
    :param datasets_path: Path to the folder with datasets. The folder structure
    must contain the names available for 'datasets', and each dataset must have
    a csv file named 'metadata.csv' and a folder with the images named 'images'
    :param seed: Value for reproducible results
    :param split_mode: The mode to split the dataset. Options are 'random', 'sequential' and 'balanced'
    :param split: Quantity to maintain in training dataframe
    :return dataframe: A DataFrame object containing the targets and paths.
    """
    total_df = pd.DataFrame(columns=["target", "path", "dataset"])
    func = {
        "inbreast": InBreastConverter,
        "bmcd": BMCDConverter,
        "rsna": RSNAConverter,
        "miniddsm": MiniDDSMConverter,
        "vindr": VinDrConverter,
    }

    # -- merge all datasets
    not_found = []
    for dataset in datasets:
        logger.info(f"Loading dataset {dataset}")
        dataset_path = Path(datasets_path) / dataset
        if not dataset_path.exists():
            raise ValueError(f"Path {dataset_path} does not exist")
        
        if dataset not in func:
            raise ValueError(f"Dataset {dataset} not found in the available datasets")

        dataset_df = func[dataset].get_dataset(dataset_path / "metadata.csv", dataset_path / "images")   

        # -- check if path exists
        for image_path in dataset_df["path"]:
            image_path = Path(image_path)
            if not image_path.exists():
                not_found.append(image_path)
                print(str(image_path))

        dataset_df["dataset"] = dataset
        total_df = pd.concat([total_df, dataset_df], ignore_index=True)

    # -- remove items not found
    if not_found:
        logger.warning(f"Images not found: {len(not_found)}")
        logger.debug(f"Images not found: {not_found}")
        dataset_df = dataset_df[~dataset_df["path"].isin(not_found)]

    # -- assure that the target is an integer and the path is a string
    total_df["target"] = total_df["target"].astype(int)
    total_df["path"] = total_df["path"].astype(str)

    # -- split dataset based on the data distribution defined
    distribution_df = distribution_split_dataset(total_df, split_mode, seed)

    # -- split the dataset into training and testing
    train_df, test_df = split_dataset(distribution_df, split, seed)

    # -- add items of total_df that are not in distribution_df to the test_df
    test_df = pd.concat([test_df, total_df[~total_df.index.isin(distribution_df.index)]], ignore_index=True)

    return train_df, test_df


def get_dataloader(
    dataset: OneViewDataset,
    batch_size: int,
    sampler_cfg: str = "",
    workers: int = int(os.cpu_count() * 0.8),
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
        persistent_workers=True,
        pin_memory=True,
        # drop_last=True,  # -- drop last batch if it is not complete
    )
    return dataloader


def cross_validation(
    dataframe: pd.DataFrame, seed: int, folds: int = 5, id_to_label: dict = {}
) -> list:
    """
    Define the folds for cross-validation by creating a new column in the dataframe

    :param dataframe: The dataframe with the dataset
    :param seed: The seed for reproducibility
    :param folds: The number of folds to create
    :param id_to_label: The dictionary to convert the target to label
    :return: The dataset with the folds
    """
    # -- create new column "fold"
    dataframe = dataframe.copy()
    dataframe["fold"] = -1
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (_, val_) in enumerate(skf.split(X=dataframe, y=dataframe.target)):
        dataframe.loc[val_, "fold"] = fold

    # -- print distribution of classes in each fold
    logger.info("Printing distribution of classes in each fold")
    for fold in range(folds):
        values = dataframe[dataframe["fold"] == fold]["target"].values
        classes, counts = np.unique(values, return_counts=True)
        distribution = " ".join(
            [
                f"{id_to_label.get(_cls, _cls)} - {cnt} |"
                for _cls, cnt in zip(classes, counts)
            ]
        )
        logger.info(f"Fold {fold}: {distribution}")

    return dataframe
