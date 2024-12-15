import argparse
import logging

import yaml

from src.datasets.bmcd_converter import BMCDConverter
from src.datasets.inbreast_converter import InBreastConverter
from src.datasets.rsna_converter import RSNAConverter
from src.datasets.miniddsm_converter import MiniDDSMConverter
from src.train import main as app_main

LOGGER_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


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
        choices=["inbreast", "bmcd", "rsna", "vindr", "miniddsm"],
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
    parser_convert.add_argument(
        "-p",
        "--processes",
        type=int,
        help="Number of processes to use. Default to -1 (.8 of the CPU cores)",
        default=-1,
    )
    parser_convert.set_defaults(
        func=lambda args: convert(args.dataset, args.path, args.output, args.processes)
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


def convert(dataset: str, path: str, output: str, processes: int) -> None:
    """
    Convert the dataset to the training model format

    :param dataset: Name of dataset to convert
    :param path: Path to the dataset
    :param output: Path to the output directory
    :param processes: Number of processes to use
    """
    logging.basicConfig(
        format=LOGGER_FORMAT, datefmt="%Y-%m-%d %H:%M:%S", level=logging.DEBUG
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # -- start conversion
    datasets = {
        "inbreast": InBreastConverter,
        "bmcd": BMCDConverter,
        "rsna": RSNAConverter,
        "miniddsm": MiniDDSMConverter,
        "vindr": None,
    }

    converter = datasets.get(dataset, None)
    if not converter:
        logger.error(f"Dataset {dataset} not supported")
        return

    converter = converter(path, output)
    converter.convert_dataset(processes)


def train(fname: str) -> None:
    """
    Training function for the model

    :param fname: name of config file to load
    """
    logging.basicConfig(
        format=LOGGER_FORMAT, datefmt="%Y-%m-%d %H:%M:%S", level=logging.DEBUG
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # -- load script params
    logger.info(f"Loading config file: {fname}")
    params = None
    with open(fname) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    logger.info("Params loaded successfully")

    # -- start training
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
