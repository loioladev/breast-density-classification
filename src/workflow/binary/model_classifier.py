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
from src.utils.plotting import plot_confusion_matrix
from src.workflow.base.model_classifier import BaseModelClassifier
from src.workflow.binary.model_tester import BinaryModelTester
from src.workflow.binary.model_trainer import BinaryModelTrainer

logger = logging.getLogger()


class BinaryModelClassifier(BaseModelClassifier):
    def train_models(
        self, dataframe: pd.DataFrame, kfolds: int, epochs: int, metrics: list
    ) -> None:
        """
        Initialize training of the binary classification models

        **Note**: The dataframe must have a column named "target" with the target
        classes. The target classes must be integers and must be sorted in ascending
        order to ensure consistency (and the code was made this way).

        :param dataframe: The dataframe to use for training
        :param kfolds: The number of folds to use for cross validation
        :param epochs: The number of epochs to train the models
        :param metrics: The MetricCollection to use for evaluation
        """
        since_binary = time.time()

        # -- obtain targets in **ascending order**
        targets = dataframe["target"].unique().tolist()
        targets = sorted(targets)

        for target in targets:
            since_target = time.time()
            logger.info(f"Starting training for target {target}")

            # -- initialize target folder
            binary_folder = Path(self.run_folder) / str(target)
            binary_folder.mkdir(parents=True, exist_ok=True)

            # -- create a binary dataframe for the target
            binary_df = dataframe.copy()
            binary_df["target"] = binary_df["target"].apply(
                lambda x: 1 if x == target else 0
            )

            # -- TODO: print distribution of target

            # -- store pretrained states in order to reset models after each fold
            model_start = self.model.state_dict()
            optimizer_start = self.optimizer.state_dict()
            scheduler_start = self.scheduler.state_dict() if self.scheduler else None

            for fold in range(kfolds):
                logger.info(f"Starting training on fold {fold+1}/{kfolds}")

                # -- create storage of fold
                folder_path = binary_folder / f"fold_{fold}"
                folder_path.mkdir(parents=True, exist_ok=True)

                # -- create dataloaders and logger
                dataloaders = self.load_data(binary_df, fold)
                csv_logger = self.create_logger(folder_path, metrics)

                # -- start fold training
                trainer = BinaryModelTrainer(
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

            logger.info(
                f"Models of target {target} trained in {convert_time(time.time() - since_target)}"
            )
        logger.info(
            f"Binary models classification training completed in {convert_time(time.time() - since_binary)}"
        )

    def test_models(self, dataframe: pd.DataFrame) -> None:
        """
        Models testing step for binary classification

        :param dataframe: The dataframe to use for testing
        """
        logger.info("Start testing binary models")

        results = []
        targets = sorted(dataframe["target"].unique().tolist())
        for target in targets:
            logger.info(f"Starting testing for target {target}")

            # -- create a binary dataframe for the target
            binary_df = dataframe.copy()
            binary_df["target"] = binary_df["target"].apply(
                lambda x: 1 if x == target else 0
            )

            # -- load dataloader
            binary_class = OneViewDataset(
                binary_df, self.transforms["val"], self.target_transforms["val"]
            )
            dataloader = get_dataloader(
                binary_class, self.batch_size, workers=self.workers
            )

            # -- initialize tester instance
            target_folder = self.run_folder / str(target)
            binary_tester = BinaryModelTester(
                self.model, target_folder, dataloader, self.device
            )

            # -- obtain predictions for the target
            target_predictions = binary_tester.test()
            threshold = binary_tester.find_optimal_threshold(target_predictions)
            results.append((target_predictions, threshold))
            logger.info(
                f"Test in target {target} done. Threshold {threshold:.2f} obtained"
            )
        logger.info("Binary models testing completed")

        # -- calculate the metrics
        best_predictions = self.select_best_predictions(dataframe, results)
        ground_truth = dataframe["target"].tolist()
        report = classification_report(
            ground_truth,
            best_predictions,
            target_names=["A", "B", "C", "D"],
            digits=3,
            zero_division=0,
        )
        logger.info(report)

        # -- calculate confusion matrix
        cm = confusion_matrix(ground_truth, best_predictions)
        plot_confusion_matrix(cm, self.run_folder)

    def select_best_predictions(
        self, dataframe: pd.DataFrame, values: list[float]
    ) -> list[int]:
        """
        Select best predictions between the models based on the threshold values

        :param values: The list of predictions and thresholds
        """
        # -- initialize variable
        n_samples = len(dataframe)
        n_classes = len(values)
        best_predictions = np.full(n_samples, 0, dtype=int)


        for i in range(n_samples):
            # -- extract predictions and thresholds for this sample
            sample_predictions = [values[j][0][i] for j in range(n_classes)]
            sample_thresholds = [values[j][1] for j in range(n_classes)]

            # -- find indices where prediction meets or exceeds threshold
            thresholds_found = [sample_predictions[i] - sample_thresholds[i] for i in range(n_classes)]

            best_predictions[i] = np.argmax(thresholds_found)
         

        return best_predictions
