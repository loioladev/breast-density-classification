import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from tqdm import tqdm

from src.datasets.dataloader import (
    ImageDataset,
    get_dataloader,
)
from src.utils.config import set_device
from src.utils.logging import CSVLogger, convert_time
from src.utils.plotting import plot_confusion_matrix

logger = logging.getLogger()


class BaseModelTrainer(ABC):
    """Base class for model training implementations"""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        csv_logger: CSVLogger,
        save_path: str | Path,
    ) -> None:
        """
        Constructor for the BaseModelTrainer

        :param model: The model instance
        :param csv_logger: The CSVLogger instance
        :param criterion: The loss function
        :param optimizer: The optimizer instance
        :param scheduler: The scheduler instance
        :param save_path: Folder path to storage training results
        """
        self.device = set_device()
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.csv_logger = csv_logger
        self.log_path = Path(save_path)

    def save_epoch(self, path: Path, is_best: bool, loss: float, epoch: int) -> None:
        """
        Save model, optimizer, scheduler and training states to the path

        :param path: The path to save the model
        :param is_best: If the model is the best found
        :param loss: Loss obtained in the epoch
        :param epoch: Epoch number
        """
        states = {
            "loss": loss,
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        save_type = "best" if is_best else "last"
        save_path = path / f"{save_type}.pt"
        torch.save(states, save_path)
        logger.debug(f"Model '{save_type}' saved to {save_path}")

    def process_metrics(self, metrics: dict) -> list[str | float]:
        """
        Convert metrics from a MetricCollection to a list of values

        :param metrics: The metrics dictionary
        :return values: The list of values obtained for each metric
        """
        values = []
        for key, value in metrics.items():
            if key.endswith("confusion"):
                values.append((key, str(value.cpu().tolist()).replace("\n", " ")))
                continue
            values.append((key, value.item()))

        # -- log metrics for the user and return
        for key, value in values:
            logger.debug(f"{key}: {value}")
        return values

    def train(
        self, epochs: int, metrics: MetricCollection, dataloaders: dict[DataLoader]
    ) -> None:
        """
        Start the training of the model

        :param epochs: The number of epochs to train
        :param metrics: The metrics to use for the training
        :param dataloaders: The dataloaders for the training, inside a dictionary
        with keys 'train' and 'val'
        """
        # -- initialize variables
        since = time.time()
        best_epoch = 0
        best_loss = float("inf")

        # -- initialize metrics
        train_metrics = metrics.clone(prefix="train_").to(self.device)
        val_metrics = metrics.clone(prefix="val_").to(self.device)

        # -- iterate over the epochs
        for epoch in range(epochs):
            since_epoch_train = time.time()
            logger.info(f"Epoch {epoch + 1}/{epochs}")

            # -- run the training epoch
            train_loss_res, train_metrics_res = self.epoch(
                "train", dataloaders["train"], train_metrics
            )
            logger.info(
                f"Training completed in {convert_time(time.time() - since_epoch_train)}"
            )
            train_metrics_val = self.process_metrics(train_metrics_res)

            # -- run the validation epoch
            since_epoch_val = time.time()
            with torch.no_grad():
                val_loss_res, val_metrics_res = self.epoch(
                    "val", dataloaders["val"], val_metrics
                )
                val_metrics_val = self.process_metrics(val_metrics_res)
            logger.info(
                f"Validation completed in {convert_time(time.time() - since_epoch_val)}"
            )

            if self.scheduler:
                self.scheduler.step(val_loss_res)

            # -- save the epoch state
            if val_loss_res < best_loss:
                best_loss = val_loss_res
                best_epoch = epoch
                self.save_epoch(self.log_path, True, val_loss_res, epoch)
                logger.info(f"Model improved with loss {val_loss_res:.4f}")
            self.save_epoch(self.log_path, False, val_loss_res, epoch)

            # -- log the results
            logger.info(
                f"Epoch {epoch + 1} completed in {convert_time(time.time() - since_epoch_train)}"
            )
            logger.info(f"Training loss: {train_loss_res:.4f}")
            logger.info(f"Validation loss: {val_loss_res:.4f}")
            for phase, elapsed_time, metrics in [
                ("train", since_epoch_train, train_metrics_val),
                ("val", since_epoch_val, val_metrics_val),
            ]:
                self.csv_logger.log(
                    phase,
                    *[
                        epoch,
                        val_loss_res,
                        elapsed_time,
                        *[value for _, value in metrics],
                    ],
                )
            train_metrics.reset()
            val_metrics.reset()

        logger.info(f"Training completed in {convert_time(time.time() - since)}")
        logger.info(f"Best loss {best_loss:.4f} at epoch {best_epoch}")

    def epoch(
        self, phase: str, dataloader: dict[DataLoader], metrics: MetricCollection
    ) -> tuple[float, MetricCollection]:
        """
        Run a single epoch for the model, with possibility of training or validation

        :param phase: The phase of the epoch, 'train' or 'val'
        :param dataloader: The dataloader for the phase
        :param metrics: The metrics to use for the phase
        :return tuple: The loss and metrics for the phase
        """
        running_loss = 0.0
        self.model.train() if phase == "train" else self.model.eval()
        metrics.train() if phase == "train" else metrics.eval()

        # -- iterate over the dataloader
        for inputs, labels in tqdm(dataloader, desc=f"{phase}"):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # -- forward operation
            with torch.set_grad_enabled(phase == "train"):
                outputs, loss = self.epoch_iteration(inputs, labels)
                if phase == "train":
                    loss.backward()
                    self.optimizer.step()

            # -- update the metrics
            self.optimizer.zero_grad()
            running_loss += loss.item()
            metrics.update(outputs, labels)

        # -- calculate the loss and metrics
        epoch_loss = running_loss / len(dataloader.dataset)
        return epoch_loss, metrics.compute()

    @abstractmethod
    def epoch_iteration(
        self, inputs: list[torch.Tensor], labels: list[torch.Tensor]
    ) -> tuple[list, torch.Tensor]:
        """
        Obtain the loss and update the model parameters

        :param inputs: The input data
        :param labels: The target labels
        :return results: The outputs and loss obtained in the iteration
        """
        pass


class BinaryModelTrainer(BaseModelTrainer):
    """Model Trainer for binary classification problems"""

    def epoch_iteration(
        self, inputs: list[torch.Tensor], labels: list[torch.Tensor]
    ) -> tuple[list, torch.Tensor]:
        """
        Obtain the loss and update the model parameters

        :param inputs: The input data
        :param labels: The target labels
        :return results: The outputs and loss obtained in the iteration
        """
        outputs = self.model(inputs).squeeze()
        loss = self.criterion(outputs, labels)
        return outputs, loss


class BaseModelTester(ABC):
    """Base class for model testing implementations"""

    def __init__(
        self, model: nn.Module, folder: str, dataloader: DataLoader, device: str
    ) -> None:
        """
        Constructor for the BaseModelTester

        :param model: Model instance
        :param folder: The folder to save the results
        :param dataloader: The dataloader object used for testing
        :param device: The device to use for the testing
        """
        self.model = model
        self.folder = Path(folder)
        self.dataloader = dataloader
        self.device = device

    def evaluate(self, model: nn.Module) -> np.ndarray:
        """
        Evaluate a single model and return the probabilities

        :param model: The model to evaluate
        :param dataloader: The dataloader to evaluate
        :return: The probabilities of the model for the test set
        """
        model.eval()
        model.to(self.device)
        all_probs = []
        with torch.no_grad():
            for inputs, _ in tqdm(self.dataloader, desc="Testing"):
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                probabilities = torch.sigmoid(outputs)
                all_probs.extend(probabilities.cpu().numpy())
        all_probs = np.array(all_probs)
        return all_probs

    @abstractmethod
    def test(self) -> list[float]:
        """Test models and save the results"""
        pass


class BinaryModelTester(BaseModelTester):
    """Binary class to test the model in a binary classification problem"""

    def find_optimal_threshold(self, preds, metric: str = "f1") -> float:
        """
        Find optimal classification threshold for the weights

        :param preds: The sigmoid value of the prediction
        :param labels: The ground-truth value of the target
        :param metric: The metric used as reference
        :return threshold: The optimal threshold for the predictions
        """
        # -- obtain labels of dataloader
        all_labels = []
        [all_labels.extend(labels) for _, labels in self.dataloader]

        # -- obtain threshold values to iterate
        thresholds = np.arange(0.1, 1.0, 0.01)

        metric_functions = {
            "acc": accuracy_score,
            "f1": f1_score,
            "pr": precision_score,
            "re": recall_score,
        }
        if metric not in metric_functions:
            raise ValueError("Value must be in 'acc', 'f1', 'pr' or 're'")

        # -- calculate the best score accodingly to the metric
        scores = []
        score_function = metric_functions[metric]
        for threshold in thresholds:
            y_pred = (preds >= threshold).astype(int)
            score = score_function(all_labels, y_pred)
            scores.append(score)
        return thresholds[np.argmax(scores)]

    def test(self) -> list:
        """
        Obtain model predictions for testing

        :return: The probabilities for each label
        """
        # -- obtain logits from the models
        probabilities = []
        for fold in os.listdir(self.folder):
            model_path = self.folder / Path(fold) / Path("best.pt")
            model_info = torch.load(model_path, weights_only=True)
            self.model.load_state_dict(model_info["model"])
            probabilities.append(self.evaluate(self.model))

        # -- get mean of probabilities
        probabilities = np.array(probabilities)
        probabilities = np.mean(probabilities, axis=0)
        return probabilities


class BaseClassification(ABC):
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

        train_class = ImageDataset(
            fold_train_df, self.transforms["train"], self.target_transforms["train"]
        )
        val_class = ImageDataset(
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
            ("%d", "time"),
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


class BinaryClassification(BaseClassification):
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
        since_binary = time.time()

        # -- obtain targets in **ascending order**
        targets = dataframe["target"].unique().tolist()
        targets = sorted(targets)
        for target in targets:
            since_target = time.time()
            logger.info(f"Starting training for target {target}")

            # -- initialize target folder
            binary_folder = Path(self.run_folder) / str(target)
            os.makedirs(binary_folder, exist_ok=True)

            # -- create a binary dataframe for the target
            binary_df = dataframe.copy()
            binary_df["target"] = binary_df["target"].apply(
                lambda x: 1 if x == target else 0
            )
            # TODO: log distribution of target

            # -- store pretrained states
            model_start = self.model.state_dict()
            optimizer_start = self.optimizer.state_dict()
            scheduler_start = self.scheduler.state_dict() if self.scheduler else None

            for fold in range(kfolds):
                logger.info(f"Starting training on fold {fold+1}/{kfolds}")

                # -- create storage of fold
                folder_path = os.path.join(binary_folder, f"fold_{fold}")
                os.makedirs(folder_path, exist_ok=True)

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
        """Test models metrics in a independent dataset"""
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
            binary_class = ImageDataset(
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
        Select best predictions based on the threshold values

        :param values: The list of predictions and thresholds
        """
        # -- initialize variable
        n_samples = len(dataframe)
        n_classes = len(values)
        best_predictions = np.full(n_samples, -1, dtype=int)

        for i in range(n_samples):
            # -- extract predictions and thresholds for this sample
            sample_predictions = np.array([values[j][0][i] for j in range(n_classes)])
            sample_thresholds = np.array([values[j][1] for j in range(n_classes)])

            # -- find indices where prediction meets or exceeds threshold
            valid_indices = np.where(sample_predictions >= sample_thresholds)[0]

            if len(valid_indices) > 0:
                # -- if any predictions meet threshold, take the last (highest index)
                best_predictions[i] = valid_indices[-1]
            else:
                # -- if no predictions meet threshold, find the closest to threshold
                threshold_distances = sample_thresholds - sample_predictions
                best_predictions[i] = np.argmin(threshold_distances)

        return best_predictions
