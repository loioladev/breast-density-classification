from typing import Any

import torch.nn as nn
from src.models.resnet_builder import ResNetBuilder


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
            raise ValueError(f"Model {model_name} not found")

        if task_type not in ["binary", "multiclass"]:
            raise ValueError(f"Task type {task_type} not supported")

        num_classes = 1 if task_type == "binary" else kwargs.get("num_classes", 4)
        model = builder.build(
            num_classes,
            model_size=kwargs.get("model_size", None),
            pretrained=pretrained,
        )

        return model
