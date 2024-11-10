"""
This module contains the functions to configure some of the global varibales to
be used in the training process.
"""

import logging
import random
import sys

import numpy
import torch
import torch.nn as nn
from focal_loss.focal_loss import (
    FocalLoss,  # https://github.com/mathiaszinnen/focal_loss_torch
)
from torch.optim import Optimizer, lr_scheduler

logger = logging.getLogger()


class ConfigManager:
    @staticmethod
    def get_optimizer(model: nn.Module, optimizer: str, args: dict) -> Optimizer:
        """
        Get optimizer instance

        :param model: The model instance
        :param optimizer: The optimizer name
        :param args: The arguments for the optimizer
        :return optimizer: The optimizer instance
        """
        optimizers = {
            "adamw": torch.optim.AdamW,
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
            "rsmprop": torch.optim.RMSprop,
        }
        if optimizer not in optimizers:
            logger.error(f"Optimizer {optimizer} not implemented")
            sys.exit(1)
        optimizer = optimizers[optimizer](model.parameters(), **args)
        return optimizer

    @staticmethod
    def get_scheduler(
        optimizer: Optimizer, scheduler: str, args: dict
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Get scheduler instance

        :param optimizer: The optimizer instance
        :param scheduler: The scheduler name
        :param args: The arguments for the scheduler
        :return scheduler: The scheduler instance
        """
        schedulers = {
            "step": lr_scheduler.StepLR,
            "multistep": lr_scheduler.MultiStepLR,
            "plateau": lr_scheduler.ReduceLROnPlateau,
            "cosine": lr_scheduler.CosineAnnealingLR,
            "wamup_cosine": lr_scheduler.CosineAnnealingWarmRestarts,
        }
        if scheduler not in schedulers:
            logger.error(f"Scheduler {scheduler} not implemented")
            sys.exit(1)

        scheduler = schedulers[scheduler](optimizer, **args)
        return scheduler

    @staticmethod
    def get_loss(loss: str, weights: list, args: dict) -> nn.Module:
        """
        Get loss function instance

        :param loss: The loss function name
        :param weights: The weights for the loss function
        :param args: The arguments for the loss function
        :return loss: The loss function instance
        """
        losses = {
            "cross_entropy": nn.CrossEntropyLoss,
            "mse": nn.MSELoss,
            "focal": FocalLoss,
        }
        if loss not in losses:
            logger.error(f"Loss {loss} not implemented")
            sys.exit(1)

        if loss == "cross_entropy" and not args["weight"]:
            del args["weight"]
        # TODO: review this line, improving it
        if loss == "focal":
            args["weights"] = torch.tensor(weights).to(set_device())

        loss = losses[loss](**args)
        return loss


def set_seed(seed: int) -> None:
    """
    Set seed for reproducibility

    :param seed: seed value
    """
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_device() -> torch.device:
    """
    Set device to GPU if available else CPU

    :return device: device to be used
    """
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    return device
