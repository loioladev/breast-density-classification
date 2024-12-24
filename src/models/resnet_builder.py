import torch.nn as nn
import torchvision.models as models
from src.models.base_model_builder import BaseModelBuilder


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
            raise ValueError(f"Invalid model size. Choose from {valid_sizes}")

        # -- get the model from torchvision and modify last layer
        model_fn = getattr(models, f"resnet{model_size}")
        model = model_fn(weights="DEFAULT" if pretrained else None)
        model = self.modify_last_layer(model, num_classes)
        return model
