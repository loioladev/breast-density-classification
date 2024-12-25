import torch
import torchvision.transforms.v2 as v2


def get_transformations(res: tuple) -> dict[v2.Compose]:
    """
    Get transformations for the dataloader

    :param res: The resolution of the images
    :return: The transformations for the dataloader
    """
    # TODO: add new transformations
    transforms = {
        "train": v2.Compose(
            [
                v2.ToImage(),
                v2.RandomRotation(20, fill=0, expand=True),
                v2.Resize(res),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ],
        ),
        "val": v2.Compose(
            [
                v2.ToImage(),
                v2.RandomRotation(20, fill=0, expand=True),
                v2.Resize(res),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ],
        ),
        "test": v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(res),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ],
        )
    }
    target_transforms = {
        "train": lambda x: torch.tensor(x, dtype=torch.float32),
        "val": lambda x: torch.tensor(x, dtype=torch.float32),
        "test": lambda x: x
    }
    return transforms, target_transforms