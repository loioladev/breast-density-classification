import logging
import os

import torch
import yaml

from src.datasets.dataloader import ImageDataset, get_dataframe, get_dataloader
from src.utils.config import set_device, set_seed

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

    # -- LOGGING
    log_folder = args["logging"]["folder"]
    os.makedirs(log_folder, exist_ok=True)

    # -- DATA
    datasets_path = args["data"]["datasets_path"]
    datasets = args["data"]["datasets"]
    sampler = args["data"]["sampler"]
    batch_size = args["data"]["batch_size"]
    workers = args["data"]["workers"]

    logger.info(f"Loading datasets {', '.join(datasets)}")
    train_df, test_df = get_dataframe(datasets, datasets_path)
    train_class = ImageDataset(train_df)
    dataloader = get_dataloader(train_class, batch_size, sampler, workers=workers)
    logger.info("Datasets loaded successfully")

    # -- save parameters
    dump = os.path.join(log_folder, "params.yaml")
    with open(dump, "w") as f:
        yaml.dump(args, f)
        logger.info(f"Training parameters stored in {dump}")
    # ----------------------------------------------------------------------- #

    # TODO: add losses, schedulers, training
