import logging
import time
from pathlib import Path
from abc import ABC, abstractmethod
import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchvision.transforms import v2
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

            logger.info(f"Training loss: {train_loss_res:.5f}")
            logger.info(f"Validation loss: {val_loss:.5f}")
            train_metrics_val = self.process_metrics(train_metrics_res)
            val_metrics_val = self.process_metrics(val_metrics_res)

            self.csv_logger.log(
                "train", *[epoch, val_loss, time_elapsed, *[value for _, value in train_metrics_val]]
            )
            self.csv_logger.log("val", *[epoch, val_loss, time_elapsed, *[value for _, value in val_metrics_val]])

            train_metrics.reset()
            val_metrics.reset()

        time_elapsed = time.time() - since
        logger.info(f"Training complete in {time_elapsed // 60}m {time_elapsed % 60}s")
        logger.info(f"Best loss: {best_loss:.4f} at epoch {best_epoch}")
        return self.model

    def run_epoch(
        self, phase: str, dataloader: dict[DataLoader], metrics: MetricCollection
    ) -> tuple[float, MetricCollection]:
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
        for key, value in metrics.items():
            if not key.endswith("confusion"):
                values.append((key, value.item()))
            else:
                values.append((key, str(value.cpu().tolist()).replace('\n', ' ')))

        # -- log the metrics
        for key, value in values:
            logger.debug(f"{key}: {value}")

        return values
    # TODO: print metrics over time


class BaseModelTester(ABC):
    """Base class for model testing implementations"""
    def __init__(self, model: nn.Module, folder: str, dataloader: DataLoader, device: str) -> None:
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
    def find_optimal_threshold(self, preds, metric: str = 'f1') -> float:
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
            'acc': accuracy_score,
            'f1': f1_score,
            'pr': precision_score,
            're': recall_score
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
            self.model.load_state_dict(model_info['model'])
            probabilities.append(self.evaluate(self.model))
        
        # -- get mean of probabilities
        probabilities = np.array(probabilities)
        probabilities = np.mean(probabilities, axis=0)
        return probabilities
