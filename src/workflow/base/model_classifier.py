import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from src.datasets.oneview_dataset import OneViewDataset
from src.utils.config import set_device
from src.utils.dataloader import get_dataloader
from src.utils.logging import CSVLogger

logger = logging.getLogger()


class BaseModelClassifier(ABC):
    def __init__(
        self,
        folder: str,
        model,
        criterion,
        optimizer,
        scheduler,
        transforms,
        target_transforms,
        batch_size,
        workers,
        sampler,
        early_stopping,
    ) -> None:
        self.run_folder = Path(folder)
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.batch_size = batch_size
        self.workers = workers
        self.sampler = sampler
        self.early_stopping = early_stopping
        self.device = set_device()

    def load_data(self, dataframe: pd.DataFrame, fold: int) -> dict:
        """
        Load dataloaders for training and validation, according to folds for
        cross validation.

        :param dataframe: The dataframe to split
        :param fold: The fold to use for validation
        :return dataloaders: The dataloaders for training and validation
        """
        fold_train_df = dataframe[dataframe["fold"] != fold]
        fold_val_df = dataframe[dataframe["fold"] == fold]

        # TODO: log fold distribution

        train_class = OneViewDataset(
            fold_train_df, self.transforms["train"], self.target_transforms["train"]
        )
        val_class = OneViewDataset(
            fold_val_df, self.transforms["val"], self.target_transforms["val"]
        )

        train_loader = get_dataloader(
            train_class, self.batch_size, self.sampler, workers=self.workers
        )
        val_loader = get_dataloader(val_class, self.batch_size, workers=self.workers)

        dataloaders = {"train": train_loader, "val": val_loader}
        return dataloaders

    def create_logger(self, folder_path: str, metrics: dict) -> CSVLogger:  # TODO: abc
        """
        Create an instance of the CSV logger to store the metrics

        :param folder_path: The path to the folder to store the metrics
        :param metrics: The metrics to store
        :return: The instance of the CSV logger
        """
        logger_metrics = []
        for metric_type in metrics:
            if metric_type in ["confusion"]:
                logger_metrics.append(("%s", metric_type))
                continue
            logger_metrics.append(("%.5f", metric_type))

        csv_logger = CSVLogger(
            folder_path,
            ("%d", "epoch"),
            ("%.5f", "loss"),
            ("%s", "time"),
            *logger_metrics,
        )
        return csv_logger

    @abstractmethod
    def train_models(
        self, dataframe: pd.DataFrame, kfolds: int, epochs: int, metrics: list
    ) -> None:
        """
        Initialize training of the binary classification models

        :param dataframe: The dataframe to use for training
        :param kfolds: The number of folds to use for cross validation
        :param epochs: The number of epochs to train the models
        :param metrics: The MetricCollection to use for evaluation
        """
        pass

    @abstractmethod
    def test_models(self, dataframe: pd.DataFrame) -> None:
        """Test models metrics in a independent dataset"""
        pass
