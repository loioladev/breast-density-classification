from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class BaseModelTester(ABC):
    """Base class for model test step implementation"""

    def __init__(
        self, model: nn.Module, folder: str, dataloader: DataLoader, device: str
    ) -> None:
        """
        Constructor for the BaseModelTester

        :param model: The object of the model to test
        :param folder: The folder to save the results
        :param dataloader: The dataloader object used for testing
        :param device: The device to use for the testing
        """
        self.model = model
        self.folder = Path(folder)
        self.dataloader = dataloader
        self.device = device

    def evaluate(self) -> np.ndarray:
        """
        Evaluate a single model and return the probabilities

        :return: The probabilities of the model for the test set
        """
        model = self.model
        model.eval()
        model.to(self.device)

        all_probs = []
        with torch.no_grad():
            for inputs, _ in tqdm(self.dataloader, desc="Testing"):
                inputs = inputs.to(self.device)
                probabilities = self.obtain_probabilities(model, inputs)
                all_probs.extend(probabilities.cpu().numpy())
        all_probs = np.array(all_probs)
        return all_probs

    @abstractmethod
    def obtain_probabilities(
        self, model: nn.Module, inputs: list[torch.Tensor]
    ) -> list:
        """
        Obtain model predictions for testing

        :param model: The model to evaluate
        :param inputs: The inputs to evaluate in the device
        :return: The probabilities of the model for the test set
        """
        pass

    @abstractmethod
    def test(self) -> list[float]:
        """
        Test models and save the results

        :return: The probabilities for each label
        """
        pass
