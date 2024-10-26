import logging
import os

import torch
import yaml

from src.utils.config import set_device, set_seed

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

    dump = os.path.join(log_folder, "params.yaml")
    with open(dump, "w") as f:
        yaml.dump(args, f)
        logger.info(f"Training parameters stored in {dump}")
    # ----------------------------------------------------------------------- #


