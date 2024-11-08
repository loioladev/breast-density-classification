import logging
import os

import torch
from torchvision.transforms import v2
import yaml

from src.datasets.dataloader import ImageDataset, get_dataframe, get_dataloader
from src.utils.config import set_device, set_seed
from src.utils.logging import CSVLogger
from src.utils.plotting import visualize_dataloader

logger = logging.getLogger()


def main(args: dict) -> None:
    """
    Training function for the script

    :param args: Dictionary containing the parameters for the training
    """
    # -- set device
    device = set_device()
    logger.info(f"Device set to {device}")

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- SEED
    seed = args["meta"]["seed"]
    set_seed(seed)
    logger.info(f"Seed set to {seed} for reproducible results")

    # -- FOLDERS
    training_folder = args["meta"]["training_folder"]
    os.makedirs(training_folder, exist_ok=True)
    folders = os.listdir(training_folder)
    log_folder = os.path.join(training_folder, "training_" + str(len(folders) + 1))
    os.makedirs(log_folder, exist_ok=True)

    # -- AUGMENTATION
    height = args["augmentation"]["height"]
    width = args["augmentation"]["width"]

    transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((height, width)),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    # -- DATA
    datasets_path = args["data"]["datasets_path"]
    datasets = args["data"]["datasets"]
    sampler = args["data"]["sampler"]
    batch_size = args["data"]["batch_size"]
    workers = args["data"]["workers"]

    train_df, test_df = get_dataframe(datasets, datasets_path, seed)
    logger.info("Datasets loaded successfully")

    train_class = ImageDataset(train_df, transform=transforms)
    dataloader = get_dataloader(train_class, batch_size, sampler, workers=workers)
    logger.info("Dataloader loaded successfully")

    dataloader_visual = os.path.join(log_folder, "dataloader.png")
    visualize_dataloader(
        dataloader, {0: "A", 1: "B", 2: "C", 3: "D"}, dataloader_visual
    )
    logger.info(f"Dataloader visualization saved in {dataloader_visual}")
    # ----------------------------------------------------------------------- #

    # -- save parameters
    dump = os.path.join(log_folder, "params.yaml")
    with open(dump, "w") as f:
        yaml.dump(args, f)
        logger.info(f"Training parameters stored in {dump}")

    # -- make csv logger
    csv_logger = CSVLogger(
        os.path.join(log_folder, "metrics.csv"),
        ("%d", "epoch"),
        ("%.5f", "loss"),
        ("%d", "time (min)"),
    )

    # -- load model

    # -- load optimizer and scheduler

    # -- load training checkpoint

    # -- training loop
