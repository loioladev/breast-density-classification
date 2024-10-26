import argparse
import logging

import yaml

from src.datasets.inbreast import convert_inbreast
from src.train import main as app_main


def create_parser() -> argparse.ArgumentParser:
    """
    Create parser for command line arguments

    :return parser: The parser object
    """
    parser = argparse.ArgumentParser(description="Breast density training")
    subparser = parser.add_subparsers(
        title="Commands", help="Available commands", dest="command"
    )

    # -- convert
    parser_convert = subparser.add_parser("convert", help="Convert the dataset")
    parser_convert.add_argument(
        "dataset",
        type=str,
        choices=["inbreast"],
        help="Dataset to convert",
    )
    parser_convert.add_argument(
        "path",
        type=str,
        help="Path to the dataset",
    )
    parser_convert.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to the output directory",
        default="./data",
    )
    parser_convert.set_defaults(
        func=lambda args: convert(args.dataset, args.path, args.output)
    )

    # -- train
    parser_training = subparser.add_parser("train", help="Train the model")
    parser_training.add_argument(
        "-f",
        "--fname",
        type=str,
        help="name of config file to load",
        default="./configs/configs.yaml",
    )
    parser_training.set_defaults(func=lambda args: train(args.fname))
    return parser


def convert(dataset: str, path: str, output: str) -> None:
    """
    Convert the dataset to the training model format

    :param dataset: Name of dataset to convert
    :param path: Path to the dataset
    :param output: Path to the output directory
    """
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Start conversion
    if dataset == "inbreast":
        convert_inbreast(path, output)


def train(fname: str) -> None:
    """
    Training function for the model

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


def main() -> None:
    """Main function"""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
