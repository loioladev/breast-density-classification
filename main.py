import argparse
import logging

import yaml

from src.train import main as app_main


def create_parser() -> argparse.ArgumentParser:
    """
    Create parser for command line arguments

    :return parser: The parser object
    """
    parser = argparse.ArgumentParser(description="Breast density training")
    parser.add_argument(
        "-f",
        "--fname",
        type=str,
        help="name of config file to load",
        default="configs.yaml",
    )
    return parser


def main(fname: str) -> None:
    """
    Main function

    :param fname: name of config file to load
    """
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Load script params
    logger.info(f"Loading config file: {fname}")
    params = None
    with open(fname) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    logger.info("Params loaded successfully")

    # Start training
    app_main(args=params)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args.fname)
