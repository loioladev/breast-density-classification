"""
TODO: add module description
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.datasets.base_converter import BaseConverter
from src.utils.processing import left_side_breast, recort_breast_morp

logger = logging.getLogger()


class MiniDDSMConverter(BaseConverter):
    """
    MiniDDSM dataset converter
    """
    def reduce_image_size(self, image: np.ndarray) -> np.ndarray:
        """
        Reduces the size of the image by removing the white borders from the image.
        The MiniDDSM has many artifacts and borders that are not part of the breast image,
        so this function removes the white borders from it.

        :param image: Image to be processed
        :return: Processed image
        """
        # -- remove 50 pixels from the left and 100 from the top and bottom
        height, width = image.shape[:2]
        top_crop = 120
        bottom_crop = 120
        left_crop = 60
        right_crop = 60

        # -- crop image
        image = image[
            top_crop:height-bottom_crop,  
            left_crop:width-right_crop            
        ]

        return image

    def process_dicom(self, dicom_path: Path, output: Path) -> None:
        """
        In MiniDDSM dataset, the DICOM files are already in PNG format, so the files
        are processed to be in the correct format for the training model.

        :param dicom_path: Path to the DICOM directory
        :param output: Path to the output directory
        """
        image = cv2.imread(str(dicom_path), cv2.IMREAD_GRAYSCALE)

        # -- convert to 8-bit image
        ratio = np.amax(image) / 256
        image = (image / ratio).astype("uint8")

        # -- flip image if it is right side breast
        if not left_side_breast(image):
            image = cv2.flip(image, 1)

        # -- remove white borders from the image by reducing the size
        image = self.reduce_image_size(image) 

        # -- recort black space
        image, _ = recort_breast_morp(image)

        # -- flip image if it is right side breast check again
        if not left_side_breast(image):
            image = cv2.flip(image, 1)

        # -- save image in format PNG as case_id_number.laterality_view.png
        output_path = output / dicom_path.name
        cv2.imwrite(str(output_path), image)

    def process_csv(self, csv_path: Path) -> None:
        """
        Process the CSV file in the INbreast dataset

        :param csv_path: Path to the CSV file
        :param output: Path to the output directory
        """
        logger.info("Processing CSV file")

        # -- read xls file
        df = pd.read_excel(csv_path)

        # -- create new dataframe 
        rows = []
        for _, row in df.iterrows():
            filename = Path(row["fileName"]).stem
            density = row["Density"]
            age = row["Age"]
            items = filename.split(".")
            case, id_, number = items[0].split("_")
            laterality, view = items[1].split("_")
            rows.append({
                    "case": case,
                    "id": id_,
                    "number": number,
                    "laterality": laterality,
                    "view": view,
                    "density": density,
                    "age": age,
            })

        # -- save csv file
        new_df = pd.DataFrame(rows)
        new_df.to_csv(self.dataset_output / "metadata.csv", index=False)

        logger.info("CSV file processed")

    def convert_dataset(self, workers: int = 1) -> None:
        """
        Convert the MiniDDSM dataset to the training model format

        :param workers: Number of workers to use
        """
        logger.info("Processing MiniDDSM dataset")
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Path {self.dataset_dir} does not exist")

        image_path = self.dataset_dir / "MINI-DDSM-Complete-PNG-16"
        csv_path = self.dataset_dir / "Data-MoreThanTwoMasks" / "Data-MoreThanTwoMasks.xlsx"

        if not image_path.exists():
            raise FileNotFoundError(f"Path {image_path} does not exist")

        if not csv_path.exists():
            raise FileNotFoundError(f"Path {csv_path} does not exist")

        logger.info("Processing MiniDDSM images")
        image_files = list(image_path.rglob(r"*[CC|MLO].png"))
        self.start_dicom_conversion(image_files, workers)
        self.process_csv(csv_path)

        logger.info("MiniDDSM dataset processed")

    @classmethod
    def get_dataset(cls, csv_path: str | Path, image_path: str | Path) -> pd.DataFrame:
        """
        Get the MiniDDSM dataset for training, adding the columns 'path' and 'target'
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
            / f"{row['case']}_{row['id']}_{row['number']}.{row['laterality']}_{row['view']}.png",
            axis=1,
        )
        df = df[["path", "target"]]
        return df
