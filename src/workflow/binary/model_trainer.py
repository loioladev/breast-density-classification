import torch

from src.workflow.base.model_trainer import BaseModelTrainer


class BinaryModelTrainer(BaseModelTrainer):
    """Model Trainer for binary classification problems"""

    def epoch_iteration(
        self, inputs: list[torch.Tensor], labels: list[torch.Tensor]
    ) -> tuple[list, torch.Tensor]:
        """
        Obtain the loss value for the batch

        :param inputs: The input data
        :param labels: The target labels
        :return results: The outputs and loss obtained in the iteration
        """
        outputs = self.model(inputs).squeeze()
        loss = self.criterion(outputs, labels)
        return outputs, loss
