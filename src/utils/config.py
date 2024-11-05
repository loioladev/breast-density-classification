"""
This module contains the functions to configure some of the global varibales to
be used in the training process.
"""

import random

import numpy
import torch


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
