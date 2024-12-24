import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.workflow.base.model_tester import BaseModelTester


class BinaryModelTester(BaseModelTester):
    """Binary class to test the model in a binary classification problem"""

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
        probabilities = torch.sigmoid(outputs)
        return probabilities

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
