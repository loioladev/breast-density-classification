"""
TODO: Add a description
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
import pydicom.pixels

from src.datasets.base_converter import BaseConverter
from src.utils.processing import apply_windowing, left_side_breast, recort_breast_morp

logger = logging.getLogger()


class RSNAConverter(BaseConverter):
    """
    RSNA dataset converter
    """

    def process_dicom(self, dicom_path: Path, output: Path) -> None:
        """
        Process the DICOM file to the output directory

        :param dicom_path: Path to the DICOM file
        :param output: Path to the output directory
        """
        # -- read dicom file
        image = pydicom.pixel_array(str(dicom_path))

        # -- apply windowing
        ds = pydicom.dcmread(str(dicom_path), stop_before_pixels=True)
        image = apply_windowing(image, ds)

        # -- normalize pixel values to 0-255
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # -- invert image
        if ds.PhotometricInterpretation == "MONOCHROME1":
            image = 255 - image

        # -- recort breast region
        image, _ = recort_breast_morp(image)

        # -- flip image if it is right side breast
        if not left_side_breast(image):
            image = cv2.flip(image, 1)

        # -- save image
        output_image = output / f"{dicom_path.parent.name}@{dicom_path.stem}.png"
        cv2.imwrite(str(output_image), image)

    def process_csv(self, csv_path: Path) -> None:
        """
        Create a CSV file with the information of the DICOM files

        :param csv_path: Path to the CSV file
        """
        df = pd.read_csv(csv_path)
        df = df[df["implant"] == 0]
        df = df[["patient_id", "image_id", "density", "laterality", "view", "age"]]
        df = df.dropna(subset=["density"])
        output = self.dataset_output / "metadata.csv"
        df.to_csv(output, index=False)

    def convert_dataset(self, processes: int = 1):
        """
        Convert the RSNA dataset to PNG images and save them to the output directory
        with a CSV file containing the information of the images.

        :param processes: Number of processes to use
        """
        logger.info("Processing RSNA dataset")

        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Path {self.dataset_dir} does not exist")

        dicom_dir = self.dataset_dir / "train_images"
        csv_path = self.dataset_dir / "train.csv"

        if not dicom_dir.exists():
            raise FileNotFoundError(f"Path {dicom_dir} does not exist")

        if not csv_path.exists():
            raise FileNotFoundError(f"Path {csv_path} does not exist")

        files = list(dicom_dir.rglob("*.dcm"))
        self.start_dicom_conversion(files, processes)
        self.process_csv(csv_path)

        logger.info("RSNA dataset processed")

    @classmethod
    def get_dataset(cls, csv_path: str | Path, image_path: str | Path) -> pd.DataFrame:
        """
        Modify the RSNA dataset to the training model format, adding the columns
        'path' and 'target' to the DataFrame.

        :param csv_path: Path to the CSV file
        :param image_path: Path to the images directory
        :return: DataFrame with the path and target columns
        """
        if isinstance(csv_path, str):
            csv_path = Path(csv_path)
        if isinstance(image_path, str):
            image_path = Path(image_path)

        df = pd.read_csv(csv_path)
        df["target"] = df["density"].apply(
            lambda x: {"A": 0, "B": 1, "C": 2, "D": 3}[x]
        )
        df["path"] = df.apply(lambda row: image_path / f"{row['patient_id']}@{row['image_id']}.png", axis=1)
        df = df[["path", "target"]]
        return df
