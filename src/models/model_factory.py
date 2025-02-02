import torch.nn as nn
import torchvision.models as tmodels


class ModelFactory:
    """Factory class for model builders"""
    @staticmethod
    def _modify_last_layer(model: nn.Module, num_classes: int) -> nn.Module:
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
                layer = "classifier"
            elif isinstance(model.classifier, nn.Sequential):
                in_features = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(in_features, num_classes)
                layer = "classifier seq"
        return model

    @classmethod
    def get_model(cls, model_name: str, task_type: str, pretrained: bool = True) -> nn.Module:
        """
        Get a model based on name and task type

        :param model_name: Name of the model architecture
        :param task_type: 'binary' or 'multiclass'
        :param pretrained: Whether to load the pretrained weights
        :param kwargs: Additional arguments for the model builder
        :return model: The model instance
        """
        if task_type not in ["binary", "multiclass"]:
            raise ValueError(f"Task type {task_type} not supported")
        
        if model_name not in tmodels.__dict__:
            raise ValueError(f"Model {model_name} not found in torchvision models")
        
        weights = {
            "convnext_tiny": tmodels.ConvNeXt_Tiny_Weights.DEFAULT,
            "convnext_small": tmodels.ConvNeXt_Small_Weights.DEFAULT,
            "convnext_base": tmodels.ConvNeXt_Base_Weights.DEFAULT,
            "convnext_large": tmodels.ConvNeXt_Large_Weights.DEFAULT,
            "resnet18": tmodels.ResNet18_Weights.DEFAULT,
            "resnet34": tmodels.ResNet34_Weights.DEFAULT,
            "resnet50": tmodels.ResNet50_Weights.DEFAULT,
            "resnet101": tmodels.ResNet101_Weights.DEFAULT,
            "densenet121": tmodels.DenseNet121_Weights.DEFAULT,
            "densenet169": tmodels.DenseNet169_Weights.DEFAULT,
            "efficientnet_b0": tmodels.EfficientNet_B0_Weights.DEFAULT,
            "efficientnet_b1": tmodels.EfficientNet_B1_Weights.DEFAULT,
            "efficientnet_b2": tmodels.EfficientNet_B2_Weights.DEFAULT,

        }
        # -- get model function from torchvision
        model_fn = getattr(tmodels, model_name)

        # -- load model from torchvision
        pretrained = weights[model_name] if pretrained else None
        model = model_fn(weights=pretrained)

        # -- modify last layer to match number of classes
        num_classes = 1 if task_type == "binary" else 4
        model = cls._modify_last_layer(model, num_classes)
        
        # -- allow model to be trained on multiple GPUs
        # model = nn.DataParallel(model)

        return model
