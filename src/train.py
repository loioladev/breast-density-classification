import logging
import os

import torch
import yaml

from src.utils.config import set_device, set_seed
from src.datasets.dataloader import get_dataloader, get_dataframe, ImageDataset

logger = logging.getLogger()


def main(args: dict) -> None:
    """
    Training function for the script

    :param args: Dictionary containing the parameters for the training
    """
    # -- set device
    device = set_device()
    logger.info(f"Set device to {device}")

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- SEED
    seed = args["meta"]["seed"]
    set_seed(seed)
    
    # -- LOGGING
    log_folder = args["logging"]["folder"]
    os.makedirs(log_folder, exist_ok=True)

    # -- DATA
    datasets_path = args["data"]["datasets_path"]
    datasets = args["data"]["datasets"]
    sampler = args["data"]["sampler"]
    batch_size = args["data"]["batch_size"]
    workers = args["data"]["workers"]
    train_df, test_df = get_dataframe(datasets, datasets_path)
    train_class = ImageDataset(train_df)
    dataloader = get_dataloader(train_class, batch_size, sampler, workers=workers)

    # -- save parameters
    dump = os.path.join(log_folder, "params.yaml")
    with open(dump, "w") as f:
        yaml.dump(args, f)
        logger.info(f"Training parameters stored in {dump}")
    # ----------------------------------------------------------------------- #
