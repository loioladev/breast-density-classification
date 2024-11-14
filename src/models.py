import logging
import sys
from abc import ABC, abstractmethod
from typing import Any

import torch.nn as nn
import torchvision.models as models

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
        logger.debug(f"Last layer of attribute {layer} modified to {num_classes} classes")
        return model


class ResNetBuilder(BaseModelBuilder):
    """Builder class for ResNet models"""

    def build(
        self, num_classes: int, model_size: int = 50, pretrained: bool = True
    ) -> nn.Module:
        """
        Build and return a ResNet model

        :param num_classes: The number of classes for the model
        :param model_size: The size of the ResNet model
        :param pretrained: Whether to load the pretrained weights
        :return model: The ResNet model instance
        """
        valid_sizes = [18, 34, 50, 101, 152]
        if model_size not in valid_sizes:
            logger.error(f"Invalid model size. Choose from {valid_sizes}")
            sys.exit(1)

        # -- get the model from torchvision and modify last layer
        model_fn = getattr(models, f"resnet{model_size}")
        model = model_fn(weights="DEFAULT" if pretrained else None)
        model = self.modify_last_layer(model, num_classes)
        return model


class ModelFactory:
    """Factory class for model builders"""

    def __init__(self) -> None:
        self.builders = {"resnet": ResNetBuilder()}

    def get_model(
        self, model_name: str, task_type: str, pretrained: bool = True, **kwargs: Any
    ) -> nn.Module:
        """
        Get a model based on name and task type

        :param model_name: Name of the model architecture
        :param task_type: 'binary' or 'multiclass'
        :param pretrained: Whether to load the pretrained weights
        :param kwargs: Additional arguments for the model builder
        :return model: The model instance
        """
        builder = self.builders.get(model_name, None)
        if builder is None:
            logger.error(f"Model {model_name} not found")
            sys.exit(1)

        if task_type not in ["binary", "multiclass"]:
            logger.error(f"Task type {task_type} not supported")
            sys.exit(1)

        num_classes = 1 if task_type == "binary" else kwargs.get("num_classes", 4)
        model = builder.build(
            num_classes,
            model_size=kwargs.get("model_size", None),
            pretrained=pretrained,
        )

        # -- add sigmoid for binary classification
        if task_type == "binary":
            model = nn.Sequential(model, nn.Sigmoid())

        return model
