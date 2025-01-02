import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path

import torch
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
        early_stopping: int,
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
        self.early_stopping = early_stopping
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
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
        }
        save_type = "best" if is_best else "last"
        save_path = path / f"{save_type}.pt"
        torch.save(states, save_path)
        logger.debug(f"Model '{save_type}' saved to {save_path}")

    def process_metrics(self, metrics: dict) -> list[str | float]:
        """
        Convert metrics from a MetricCollection to a list of tuples,
        where each tuple contains the metric name and its value.

        :param metrics: The metrics dictionary
        :return values: The list of values obtained for each metric
        """
        values = []
        for key, value in metrics.items():
            if key.endswith("confusion"):
                values.append((key, str(value.cpu().tolist()).replace("\n", " ")))
                continue
            values.append((key, value.item()))

        # -- log metrics for the user
        for key, value in values:
            logger.debug(f"{key}: {value}")

        return values

    def train(
        self,
        epochs: int,
        metric_collection: MetricCollection,
        dataloaders: dict[DataLoader],
    ) -> None:
        """
        Start the training of the model

        :param epochs: The number of epochs to train
        :param metrics: The metrics to use for the training
        :param dataloaders: The dataloaders for the training, inside a dictionary
        with keys 'train' and 'val'
        """
        # -- initialize variables
        best_epoch = 0
        since = time.time()
        best_loss = float("inf")

        # -- initialize metrics
        train_metrics = metric_collection.clone(prefix="train_").to(self.device)
        val_metrics = metric_collection.clone(prefix="val_").to(self.device)
        early_stop = 0

        # -- iterate over the epochs
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")

            # -- run the training step
            since_epoch_train = time.time()
            train_loss, metric_results = self.epoch(
                "train", dataloaders["train"], train_metrics
            )
            train_metric_results = self.process_metrics(metric_results)

            # -- calculate the time taken for the training step
            train_time = convert_time(time.time() - since_epoch_train)
            logger.info(f"Training completed in {train_time}")

            # -- log training step to the CSV file
            train_info = [epoch, train_loss, train_time, *[value for _, value in train_metric_results]]
            self.csv_logger.log("train", *train_info)

            # -- run the validation step
            since_epoch_val = time.time()
            with torch.no_grad():
                val_loss, metric_results = self.epoch(
                    "val", dataloaders["val"], val_metrics
                )
                val_metric_results = self.process_metrics(metric_results)

            # -- calculate the time taken for the validation step
            val_time = convert_time(time.time() - since_epoch_val)
            logger.info(f"Validation completed in {val_time}")

            # -- log validation step to the CSV file
            val_info = [epoch, val_loss, val_time, *[value for _, value in val_metric_results]]
            self.csv_logger.log("val", *val_info)

            # -- update the learning rate
            if self.scheduler:
                self.scheduler.step(val_loss)

            # -- save the best epoch state
            if val_loss < best_loss:
                early_stop = 0
                best_loss = val_loss
                best_epoch = epoch
                self.save_epoch(self.log_path, True, val_loss, epoch)
                logger.info(f"Model improved with loss {val_loss:.4f}")
            else:
                early_stop += 1

            # -- save the last epoch state
            self.save_epoch(self.log_path, False, val_loss, epoch)


            # -- log the results to the console
            logger.info(
                f"Epoch {epoch + 1} completed in {convert_time(time.time() - since_epoch_train)}"
            )
            logger.info(f"Training loss: {train_loss:.4f}")
            logger.info(f"Validation loss: {val_loss:.4f}")
            train_metrics.reset()
            val_metrics.reset()

            # -- check for early stopping
            if early_stop >= self.early_stopping:
                logger.info("Early stopping activated")
                break

        logger.info(f"Training completed in {convert_time(time.time() - since)}")
        logger.info(f"Best loss {best_loss:.4f} at epoch {best_epoch}")

    def epoch(
        self, phase: str, dataloader: dict[DataLoader], metrics: MetricCollection
    ) -> tuple[float, MetricCollection]:
        """
        Run a single epoch for the model, with possibility of training or validation steps

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
            inputs = inputs.to(self.device).float()
            labels = labels.to(self.device).long()

            # -- forward operation
            with torch.set_grad_enabled(phase == "train"):
                outputs, loss = self.epoch_iteration(inputs, labels)
                if phase == "train":
                    loss.backward()
                    self.optimizer.step()

            # -- update the metrics
            running_loss += loss.item()
            metrics.update(outputs, labels)
            self.optimizer.zero_grad()

        # -- calculate the loss and metrics
        epoch_loss = running_loss / len(dataloader.dataset)
        return epoch_loss, metrics.compute()

    @abstractmethod
    def epoch_iteration(
        self, inputs: list[torch.Tensor], labels: list[torch.Tensor]
    ) -> tuple[list, torch.Tensor]:
        """
        Obtain the loss value of the batch

        :param inputs: The input data
        :param labels: The target labels
        :return results: The outputs and loss obtained in the iteration
        """
        pass
