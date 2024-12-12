"""
Convert InBreast dataset to the training model format. The dataset was obtained from
Kaggle and is available in the following link: https://www.kaggle.com/datasets/martholi/inbreast

The INbreast dataset contains DICOM files and a CSV file. The DICOM files are already
processed in DICOM format, so the conversion to PNG will be straightforward. The CSV
file contains the metadata of the DICOM files, which will be used to create the training
model format.

The dataset is prepocessed with CLAHE, which is a contrast enhancement algorithm.

The way to use this module is to call the `get_inbreast` function, passing the path to the
extracted INbreast dataset folder.
"""

import logging
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom

from src.datasets.base import BaseConverter
from src.utils.processing import left_side_breast, recort_breast_morp

logger = logging.getLogger()


class InBreastConverter(BaseConverter):
    """
    InBreast dataset converter
    """

    def process_dicom(self, dicom_path: Path, output: Path) -> None:
        """
        Convert the image from DICOM to PNG format

        :param dicom_path: Path to the DICOM directory
        :param output: Path to the output directory
        """
        image = pydicom.pixel_array(dicom_path)

        # -- convert to 8-bit image
        ratio = np.amax(image) / 256
        image = (image / ratio).astype("uint8")

        # -- flip image if it is right side breast
        if not left_side_breast(image):
            image = cv2.flip(image, 1)

        # -- recort black space
        image, _ = recort_breast_morp(image)

        # -- apply CLAHE contrast enhancement (https://www.sciencedirect.com/science/article/pii/S2352340920308222)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)

        # -- save image in format filename_laterality_view.png
        split_name = dicom_path.stem.split("_")
        if split_name[4] == "ML":
            split_name[4] = "MLO"
        filename = f"{split_name[0]}_{split_name[3]}_{split_name[4]}.png"
        cv2.imwrite(os.path.join(output, filename), image)

    def process_csv(self, csv_path: Path) -> None:
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

        # -- remove views not in mlo or cc
        df = df[df["view"].isin(["MLO", "CC"])]

        # -- convert columns to correct format
        df["filename"] = df["filename"].astype(int)
        df["density"] = df["density"].apply(lambda x: str(x).strip())
        df = df[df["density"].isin(["1", "2", "3", "4"])]

        # -- log density distribution
        for density in df["density"].unique():
            logger.info(
                f"Density {density}: {len(df[df['density'] == density])} samples"
            )

        # -- save csv file
        df.to_csv(self.dataset_output / "metadata.csv", index=False)
        logger.info("CSV file processed")

    def convert_dataset(self, workers: int = 1) -> None:
        """
        Convert the INbreast dataset to the training model format

        :param workers: Number of workers to use
        """
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Path {self.dataset_dir} does not exist")

        dicom_path = self.dataset_dir / "ALL-IMGS"
        csv_path = self.dataset_dir / "INbreast.xls"

        if not dicom_path.exists():
            raise FileNotFoundError(f"Path {dicom_path} does not exist")

        if not csv_path.exists():
            raise FileNotFoundError(f"Path {csv_path} does not exist")

        dicom_files = list(dicom_path.rglob("*.dcm"))
        self.start_dicom_conversion(dicom_files, workers)
        self.process_csv(csv_path)

        logger.info("InBreast dataset processed")

    @classmethod
    def get_inbreast(csv_path: str | Path, image_path: str | Path) -> pd.DataFrame:
        """
        Get the InBreast dataset for training, adding the columns 'path' and 'target'
        to the DataFrame.

        :param csv_path: Path to the CSV file
        :param image_path: Path to the images directory
        :return: DataFrame with the path and target columns
        """
        if isinstance(csv_path, str):
            csv_path = Path(csv_path)
        if isinstance(image_path, str):
            image_path = Path(image_path)

        df = pd.read_csv(csv_path)
        df["target"] = df["density"].apply(lambda x: int(x) - 1)
        df["path"] = df.apply(
            lambda row: image_path
            / f"{row['filename']}_{row['laterality']}_{row['view']}.png",
            axis=1,
        )
        df = df[["path", "target"]]
        return df
