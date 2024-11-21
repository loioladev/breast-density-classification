import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from tqdm import tqdm

from src.utils.config import set_device
from src.utils.logging import CSVLogger, convert_time

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
