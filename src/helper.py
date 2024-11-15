import logging
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchvision.transforms import v2
from tqdm import tqdm

from src.utils.config import set_device
from src.utils.logging import CSVLogger

logger = logging.getLogger()


def get_transformations(res: tuple) -> dict[v2.Compose]:
    """
    Get transformations for the dataloader

    :param res: The resolution of the images
    :return: The transformations for the dataloader
    """
    # TODO: add new transformations
    transforms = {
        "train": v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(res),
                v2.ToDtype(torch.float32, scale=True),
            ],
        ),
        "val": v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(res),
                v2.ToDtype(torch.float32, scale=True),
            ],
        ),
    }
    target_transforms = {
        "train": lambda x: torch.tensor(x, dtype=torch.float32),
        "val": lambda x: torch.tensor(x, dtype=torch.float32)
    }
    return transforms, target_transforms


class Training:
    """
    Cl
    """

    def __init__(
        self,
        model: nn.Module,
        csv_logger: CSVLogger,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        log_path: Path,
    ) -> None:
        device = set_device()
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.csv_logger = csv_logger
        self.log_path = Path(log_path)

    def train(
        self, epochs: int, metrics: MetricCollection, dataloaders: dict[DataLoader]
    ) -> None:
        """
        Start the training of the model

        :param epochs: The number of epochs to train
        :param metrics: The metrics to use for the training
        """
        # -- initialize variables
        best_epoch = 0
        best_loss = float("inf")
        device = set_device()
        since = time.time()

        # -- initialize metrics
        train_metrics = metrics.clone(prefix="train_").to(device)
        val_metrics = metrics.clone(prefix="val_").to(device)

        # -- iterate over the epochs
        for epoch in range(epochs):
            since_epoch = time.time()
            logger.info(f"Epoch {epoch}/{epochs}")

            train_loss_res, train_metrics_res = self.run_epoch(
                "train", dataloaders["train"], train_metrics
            )
            logger.info(f"Training completed in {time.time() - since_epoch:.2f}s")

            with torch.no_grad():
                val_loss, val_metrics_res = self.run_epoch(
                    "val", dataloaders["val"], val_metrics
                )
            logger.info(f"Validation completed in {time.time() - since_epoch:.2f}s")

            if self.scheduler:
                self.scheduler.step(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                self.save_epoch(self.log_path, True, val_loss, epoch)
                logger.info(f"Model improved with loss {val_loss:.4f}")
            self.save_epoch(self.log_path, False, val_loss, epoch)

            # -- log the results
            time_elapsed = time.time() - since_epoch
            logger.info(f"Epoch complete in {time_elapsed // 60}m {time_elapsed % 60}s")

            logger.info(f"Train loss: {train_loss_res:.4f} | Val loss: {val_loss:.4f}")
            train_metrics_val = self.process_metrics(train_metrics_res)
            val_metrics_val = self.process_metrics(val_metrics_res)

            self.csv_logger.log(
                "train", *[epoch, val_loss, time_elapsed, *train_metrics_val]
            )
            self.csv_logger.log("val", *[epoch, val_loss, time_elapsed, *val_metrics_val])

            train_metrics.reset()
            val_metrics.reset()

        time_elapsed = time.time() - since
        logger.info(f"Training complete in {time_elapsed // 60}m {time_elapsed % 60}s")
        logger.info(f"Best loss: {best_loss:.4f} at epoch {best_epoch}")
        return self.model

    def run_epoch(
        self, phase: str, dataloader: dict[DataLoader], metrics: MetricCollection
    ) -> None:
        """
        Run the epoch for the model

        :param phase: The phase of the epoch, train or val
        :param dataloader: The dataloader for the phase
        :param metrics: The metrics for the phase
        :return: The loss and metrics for the phase
        """
        # -- initialize variables
        running_loss = 0.0
        device = set_device()
        self.model.train() if phase == "train" else self.model.eval()
        metrics.train() if phase == "train" else metrics.eval()

        # -- iterate over the dataloader
        for inputs, labels in tqdm(dataloader, desc=f"{phase}"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # -- forward operation
            with torch.set_grad_enabled(phase == "train"):
                # TODO: this is only for binary classification, squeeze is not general
                outputs = self.model(inputs).squeeze()
                loss = self.criterion(outputs, labels) 
                if phase == "train":
                    loss.backward()
                    self.optimizer.step()

            self.optimizer.zero_grad()
            running_loss += loss.item()
            metrics.update(outputs, labels)

        epoch_loss = running_loss / len(dataloader.dataset)
        return epoch_loss, metrics.compute()

    def save_epoch(self, path: Path, is_best: bool, loss: float, epoch: int) -> None:
        """
        Save the model to the path
        """
        training_results = {
            "loss": loss,
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        type_save = "best" if is_best else "last"
        path_save = path / f"{type_save}.pt"
        torch.save(training_results, path_save)
        logger.debug(f"Model {type_save} saved to {path_save}")

    def process_metrics(self, metrics: dict) -> list:
        """
        Convert metrics to a list of values

        :param metrics: The metrics dictionary
        :return: The list of values
        """
        values = []
        for key in metrics.keys():
            if not key.endswith("confusion"):
                values.append((key, metrics[key].item()))
                continue
            confusion_text = str(metrics[key].item())
            values.append((key, confusion_text.replace(r"\n", " ")))
        logger.info()
        return values
    