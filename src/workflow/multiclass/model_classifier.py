import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.datasets.oneview_dataset import OneViewDataset
from src.utils.dataloader import get_dataloader
from src.utils.logging import convert_time
from src.utils.plotting import plot_confusion_matrix, plot_metrics
from src.workflow.base.model_classifier import BaseModelClassifier
from src.workflow.multiclass.model_trainer import MulticlassModelTrainer
from src.workflow.multiclass.model_tester import MultiClassModelTester

logger = logging.getLogger()


class MultiClassClassifier(BaseModelClassifier):
    def train_models(
        self, dataframe: pd.DataFrame, kfolds: int, epochs: int, metrics: list
    ) -> None:
        """
        Initialize training of the Multiclass classification models

        **Note**: The dataframe must have a column named "target" with the target
        classes. The target classes must be integers and must be sorted in ascending
        order to ensure consistency (and the code was made this way).

        :param dataframe: The dataframe to use for training
        :param kfolds: The number of folds to use for cross validation
        :param epochs: The number of epochs to train the models
        :param metrics: The MetricCollection to use for evaluation
        """
        since_start = time.time()
        logger.info(f"Starting training for multiclass model")

        # -- store pretrained states in order to reset models after each fold
        model_start = self.model.state_dict()
        optimizer_start = self.optimizer.state_dict()
        scheduler_start = self.scheduler.state_dict() if self.scheduler else None

        for fold in range(kfolds):
            since_fold = time.time()
            logger.info(f"Starting training on fold {fold+1}/{kfolds}")

            # -- create storage of fold
            folder_path = self.run_folder / f"fold_{fold}"
            folder_path.mkdir(parents=True, exist_ok=True)

            # -- create dataloaders and logger
            dataloaders = self.load_data(dataframe, fold)
            csv_logger = self.create_logger(folder_path, metrics)

            # -- start fold training
            trainer = MulticlassModelTrainer(
                self.model,
                self.criterion,
                self.optimizer,
                self.scheduler,
                csv_logger,
                folder_path,
            )
            trainer.train(epochs, metrics, dataloaders)

            # -- reset models to pretrained version
            self.model.load_state_dict(model_start)
            self.optimizer.load_state_dict(optimizer_start)
            if self.scheduler:
                self.scheduler.load_state_dict(scheduler_start)

            train_metrics = pd.read_csv(folder_path / "train_metrics.csv", sep=';')
            plot_metrics(train_metrics, folder_path / "train_metrics.png")
            val_metrics = pd.read_csv(folder_path / "val_metrics.csv", sep=';')
            plot_metrics(val_metrics, folder_path / "val_metrics.png")

            logger.info(
                f"Folder {fold} trained in {convert_time(time.time() - since_fold)}"
            )
        logger.info(
            f"Multiclass models classification training completed in {convert_time(time.time() - since_start)}"
        )

    def test_models(self, dataframe: pd.DataFrame) -> None:
        """
        Models testing step for multiclass classification

        :param dataframe: The dataframe to use for testing
        """
        logger.info("Start testing multiclass models")

        # -- load dataloader
        dataset = OneViewDataset(
            dataframe, self.transforms["test"], self.target_transforms["test"]
        )
        dataloader = get_dataloader(
            dataset, self.batch_size, workers=self.workers
        )

        # -- initialize tester instance
        tester = MultiClassModelTester(self.model, self.run_folder, dataloader, self.device)

        # -- obtain predictions for the target
        logits = tester.test()

        predictions = np.argmax(logits, axis=1)

        logger.info("Multiclass models testing completed")

        # -- calculate the metrics
  
        ground_truth = dataframe["target"].tolist()
        report = classification_report(
            ground_truth,
            predictions,
            target_names=["A", "B", "C", "D"],
            digits=3,
            zero_division=0,
        )
        logger.info(report)
        with open(self.run_folder / "test_metrics.txt", "w") as f:
            f.write(report)

        # -- calculate confusion matrix
        cm = confusion_matrix(ground_truth, predictions)
        plot_confusion_matrix(cm, self.run_folder)
