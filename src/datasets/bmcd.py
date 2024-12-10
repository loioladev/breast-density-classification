"""
Convert BDSM dataset to the training model format. The dataset was obtained from
the following link: https://zenodo.org/records/5036062

The way to use this module is to call the `bmcd` function, passing the path to the
extracted bcmd dataset folder (usually 'Dataset').
"""

import logging
import os

import cv2
import numpy as np
import pandas as pd
import pydicom
from pathlib import Path
import multiprocessing

from src.utils.processing import recort_breast_morp

logger = logging.getLogger()


def process_bmcd_images(dicom_path: Path, output: Path) -> None:
    """
    Process the DICOM file to the output directory

    :param dicom_path: Path to the DICOM file
    :param output: Path to the output directory
    """
    # -- read dicom file
    ds = pydicom.dcmread(dicom_path)

    # -- read image and convert to uint8
    image = pydicom.pixel_array(ds)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # -- crop breast region
    image, _ = recort_breast_morp(image)

    # -- flip image if it is right breast
    if ds.ImageLaterality == "R":
        image = cv2.flip(image, 1)

    # -- create new name for image and save it
    file_id = dicom_path.parent.name
    view = dicom_path.stem.split("_")[0]
    file_info = dicom_path.stem.split("_")[1]
    file_class = dicom_path.parent.parent.name[0]
    file_name = f"{file_class}_{file_id}_{view}_{file_info}.png".lower()
    cv2.imwrite(str(Path(output) / file_name), image)


def process_dicom(
    dicom_dir: str, output: str, processes: int = 1
) -> None:
    """
    Process the DICOM files in the INbreast dataset

    :param dicom_dir: Path to the DICOM directory
    :param output: Path to the output directory
    :param dicom_ids: Set of DICOM IDs to be processed
    :param processes: Number of processes to use
    """
    logger.info("Processing DICOM files")
    output = os.path.join(output, "images")
    os.makedirs(output, exist_ok=True)

    # -- read dicom files and convert to 8-bit images
    files = list(dicom_dir.rglob("*.dcm"))
    with multiprocessing.Pool(processes) as pool:
        pool.starmap(
            process_bmcd_images,
            [(file, output) for file in files],
        )
    logger.info(f"DICOM files done with {processes} processes")


def process_csv(xlsx_path: Path, dicom_dir: Path, output: Path) -> None:
    """
    Create a CSV file with the information of the DICOM files

    :param xlsx_path: Path to the XLSX file
    :param dicom_dir: Path to the DICOM directory
    :param output: Path to the output directory
    """

    normal_cases = pd.read_excel(xlsx_path, sheet_name="Normal_cases", skiprows=2)
    suspicious_cases = pd.read_excel(xlsx_path, sheet_name="Suspicious_cases", skiprows=2)

    files = list(dicom_dir.rglob("*.dcm"))
    data = []
    for file in files:
        ds = pydicom.dcmread(file, specific_tags=["ImageLaterality"])
        file_idx = file.parent.name
        view = file.stem.split("_")[0]
        file_info = file.stem.split("_")[1]
        file_class = file.parent.parent.name[0]

        if file_class == 's':
            xls = suspicious_cases
        else:
            xls = normal_cases
        density = xls.loc[xls["Folder #"] == int(file_idx), "BI-RADS categories for breast density"].values[0]
        
        data.append(
            {
                "case": file_class.lower(),
                "number": int(file_idx),
                "status": file_info.lower(),
                "laterality": ds.ImageLaterality.lower(),
                "view": view.lower(),
                "density": density.lower()
            }
        )
    dataframe = pd.DataFrame(data)
    dataframe.to_csv(output / "metadata.csv", index=False)
    logger.info("CSV file created")


def convert_bmcd(path: str, output: str, processes: int = 1) -> None:
    """
    Convert the BMCD dataset to PNG images and save them to the output directory
    with a CSV file containing the information of the images.

    :param path: Path to the INbreast dataset
    :param output: Path to the output directory
    :param processes: Number of processes to use
    """
    logger.info("Processing BMCD dataset")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Path {path} does not exist")
    
    path = Path(path)
    dicomdir = path / "Dataset"
    xlsx_path = path / "bmcd.xlsx"
    output_path = Path(output)

    if not os.path.exists(dicomdir):
        raise FileNotFoundError(f"Path {dicomdir} does not exist")
    
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"Path {xlsx_path} does not exist")
    
    process_dicom(dicomdir, output_path, processes)
    process_csv(xlsx_path, dicomdir, output_path)

    logger.info("BMCD dataset processed")

    
def get_bmcd(csv_path: str, image_path: str) -> pd.DataFrame:
    """
    Modify the BMCD dataset to the training model format, adding the columns
    'path' and 'target' to the DataFrame.

    :param csv_path: Path to the CSV file
    :param image_path: Path to the images directory
    :return: DataFrame with the path and target columns
    """
    df = pd.read_csv(csv_path)
    df["target"] = df["density"].apply(lambda x: {"a": 0, "b": 1, "c": 2, "d": 3}[x])
    df["path"] = df.apply(
        lambda row: os.path.join(
            image_path, f"{row['case']}_{row['number']}_{row['view']}_{row['status']}.png"
        ),
        axis=1,
    )
    df = df[["path", "target"]]
    return df
