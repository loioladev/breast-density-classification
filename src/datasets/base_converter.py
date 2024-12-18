import concurrent
from concurrent.futures import ProcessPoolExecutor
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from tqdm import tqdm


class BaseConverter(ABC):
    """
    Base class for dataset converters
    """

    def __init__(self, dataset_dir: str, dataset_output: str) -> None:
        """
        Constructor for the BaseConverter class

        :param dataset_dir: Path to the dataset directory
        :param dataset_output: Path to the output directory
        """
        self.dataset_dir = Path(dataset_dir)
        self.dataset_output = Path(dataset_output)

    def start_dicom_conversion(self, files: list[Path], workers: int = 1) -> None:
        """
        Start the DICOM conversion process with the given files. It creates a new directory
        called 'images' in the output directory and saves the converted images there

        :param files: List of DICOM files
        :param output: Path to the output directory
        :param workers: Number of workers to use with multiprocessing
        """
        output = self.dataset_output / "images"
        output.mkdir(parents=True, exist_ok=True)

        if workers == 1:
            for dicom_path in tqdm(files, desc="Processing DICOM Files"):
                self.process_dicom(dicom_path, output)
            return

        file_args = [(file, output) for file in files]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            # -- submit all tasks
            futures = [executor.submit(self.process_dicom, *args) for args in file_args]
            
            # -- wait for all tasks to finish and show a progress bar
            [_ for _ in tqdm(concurrent.futures.as_completed(futures), total=len(files), desc="Processing DICOMs")]


    @abstractmethod
    def process_dicom(self, dicom_path: Path, output: Path) -> None:
        """
        Convert the DICOM file to PNG with the respect transformations of the dataset
        and save it in the output directory. The images should be normalized and flipped
        to the right orientation.

        :param dicom_path: Path to the DICOM file
        :param output: Path to the output directory of images
        """
        pass

    @abstractmethod
    def process_csv(self) -> None:
        """Convert the original CSV file to the desired format and save it to the output directory."""
        pass

    @abstractmethod
    def convert_dataset(self, workers: int = 1) -> None:
        """
        Convert the original dataset to the desired format and save it to the output directory.
        The images should be saved in a folder called 'images' and the CSV file with the name
        'metadata.csv'. This function should call the 'start_dicom_conversion' and 'process_csv'

        :param workers: Number of workers to use
        """
        pass

    @classmethod
    @abstractmethod
    def get_dataset(cls, csv_path: str | Path, image_path: str | Path) -> pd.DataFrame:
        """
        From the converted CSV file, create a DataFrame with the columns 'path' and 'target'.

        :param csv_path: Path to the CSV file
        :param image_path: Path to the images directory
        :return: DataFrame with the path and target columns
        """
        pass
