"""
Convert INbreast dataset to the training model format. The dataset was obtained from
Kaggle and is available in the following link: https://www.kaggle.com/datasets/martholi/inbreast

The INbreast dataset contains DICOM files and a CSV file. The DICOM files are already
processed in DICOM format, so the conversion to PNG will be straightforward. The CSV
file contains the metadata of the DICOM files, which will be used to create the training
model format.

The way to use this module is to call the inbreast function, passing the path to the
extracted INbreast dataset folder.
"""

import logging
import multiprocessing
import os

import cv2
import numpy as np
import pandas as pd
import pydicom

from src.utils.processing import recort_breast_morp

logger = logging.getLogger()


def process_inbreast_image(dicom_path: str, output: str, dicom_ids: set[str]) -> None:
    """
    Convert the image from DICOM to PNG format

    :param dicom_path: Path to the DICOM directory
    :param output: Path to the output directory
    :param dicom_ids: Set of DICOM IDs to be processed
    """
    split_name = os.path.basename(dicom_path).split("_")
    if split_name[0] not in dicom_ids:
        return

    image = pydicom.pixel_array(dicom_path)

    # -- convert to 8-bit image
    ratio = np.amax(image) / 256
    image = (image / ratio).astype("uint8")

    # -- recort black space
    image, _ = recort_breast_morp(image)

    # -- save image in format filename_laterality_view.png
    filename = f"{split_name[0]}_{split_name[3]}_{split_name[4]}.png"
    cv2.imwrite(os.path.join(output, filename), image)


def process_dicom(
    dicom_path: str, output: str, dicom_ids: set[str], processes: int = 1
) -> None:
    """
    Process the DICOM files in the INbreast dataset

    :param dicom_path: Path to the DICOM directory
    :param output: Path to the output directory
    :param dicom_ids: Set of DICOM IDs to be processed
    :param processes: Number of processes to use
    """
    logger.info("Processing DICOM files")
    output = os.path.join(output, "images")
    os.makedirs(output, exist_ok=True)

    # -- read dicom files and convert to 8-bit images
    files = [os.path.join(dicom_path, file) for file in os.listdir(dicom_path)]
    with multiprocessing.Pool(processes) as pool:
        pool.starmap(
            process_inbreast_image,
            [(file, output, dicom_ids) for file in files],
        )
    logger.info(f"DICOM files done with {processes} processes")


def process_csv(csv_path: str, output: str) -> pd.DataFrame:
    """
    Process the CSV file in the INbreast dataset

    :param csv_path: Path to the CSV file
    :param output: Path to the output directory
    """
    logger.info("Processing CSV file")

    # -- read xls file
    df = pd.read_excel(csv_path)

    # -- filter columns
    df = df.rename(
        columns={
            "Laterality": "laterality",
            "View": "view",
            "ACR": "density",
            "File Name": "filename",
        }
    )
    df = df[["filename", "laterality", "view", "density"]]
    df.dropna(subset=["filename", "density"], inplace=True)

    # -- convert columns to correct format
    df["filename"] = df["filename"].astype(int)
    df["density"] = df["density"].apply(lambda x: str(x).strip())
    df = df[df["density"].isin(["1", "2", "3", "4"])]

    # -- log density distribution
    for density in df["density"].unique():
        logger.info(f"Density {density}: {len(df[df['density'] == density])} samples")

    # -- save csv file
    os.makedirs(output, exist_ok=True)
    df.to_csv(os.path.join(output, "metadata.csv"), index=False)

    logger.info("CSV file processed")
    return df


def convert_inbreast(path: str, output: str, processes: int = 1) -> None:
    """
    Convert the INbreast dataset to the training model format

    :param path: Path to the INbreast dataset
    :param output: Path to the output directory
    :param processes: Number of processes to use
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path {path} does not exist")

    dicom_path = os.path.join(path, "ALL-IMGS")
    csv_path = os.path.join(path, "INbreast.xls")

    df = process_csv(csv_path, output)
    dicom_ids = set(df["filename"].astype(str))
    process_dicom(dicom_path, output, dicom_ids, processes)
    logger.info("INbreast dataset processed")


def get_inbreast(csv_path: str, image_path: str) -> pd.DataFrame:
    """
    Get the INbreast dataset prepared for the training

    :param csv_path: Path to the CSV file
    :param image_path: Path to the images directory
    :return: DataFrame with the path and target columns
    """
    df = pd.read_csv(csv_path)
    df["target"] = df["density"].apply(lambda x: int(x) - 1)
    df["path"] = df.apply(
        lambda row: os.path.join(
            image_path, f"{row['filename']}_{row['laterality']}_{row['view']}.dcm"
        ),
        axis=1,
    )
    df = df[["path", "target"]]
    return df
