import os
from pathlib import Path

import numpy as np
import torch

from src.workflow.base.model_tester import BaseModelTester


class MultiClassModelTester(BaseModelTester):
    """MultiClass class to test the model in a multiclass classification problem"""

    def obtain_probabilities(
        self, model: torch.nn.Module, inputs: list[torch.Tensor]
    ) -> list:
        """
        Obtain model predictions for testing

        :param model: The model to evaluate
        :param inputs: The inputs to evaluate in the device
        :return: The probabilities of the model for the test set
        """
        outputs = model(inputs)
        probabilities = torch.softmax(outputs, dim=1)
        return probabilities
    

    def test(self) -> list[float]:
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
            probabilities.append(self.evaluate())

        # -- get mean of probabilities
        probabilities = np.array(probabilities)
        probabilities = np.mean(probabilities, axis=0)
        return probabilities
