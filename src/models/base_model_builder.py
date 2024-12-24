import logging
from abc import ABC, abstractmethod

import torch.nn as nn

logger = logging.getLogger()


class BaseModelBuilder(ABC):
    """Abstract base class for model builders"""

    @abstractmethod
    def build(self, num_classes: int, pretrained: bool = True) -> nn.Module:
        """
        Build and return a model

        :param num_classes: The number of classes for the model
        :param pretrained: Whether to load the pretrained weights
        :return model: The model instance
        """
        pass

    def modify_last_layer(self, model: nn.Module, num_classes: int) -> nn.Module:
        """
        Modify the last layer of the model to match the number of classes

        :param model: The model instance
        :param num_classes: The number of classes for the model
        :return model: The model instance with the last layer modified
        """
        layer = None
        if hasattr(model, "fc"):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
            layer = "fc"
        elif hasattr(model, "classifier"):
            if isinstance(model.classifier, nn.Linear):
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, num_classes)
            elif isinstance(model.classifier, nn.Sequential):
                in_features = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(in_features, num_classes)
            layer = "classifier"
        logger.debug(
            f"Last layer of attribute {layer} modified to {num_classes} classes"
        )
        return model
