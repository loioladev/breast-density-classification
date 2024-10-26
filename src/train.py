import logging
import os
import sys

import torch
import yaml


def main(args: dict) -> None:
    """
    """
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Set device
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        torch.cuda.set_device(device)
    logger.info(f"Device set to: {device}")


    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #


    
    log_folder = args["logging"]["folder"]
    dump = os.path.join(log_folder, "params.yaml")
    with open(dump, "w") as f:
        yaml.dump(args, f)
    logger.info(f"Training parameters stored in: {dump}")
    # ----------------------------------------------------------------------- #