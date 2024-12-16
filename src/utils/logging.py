import os
from pathlib import Path
import pandas as pd

class CSVLogger(object):
    """
    Class to handle the logging of the metrics
    """

    def __init__(self, log_path: str | Path, *argv: list) -> None:
        """
        Constructor of the class

        :param log_path: The path to the directory to save the metrics
        :param *argv: List of tuples, where each tuple contains the format
        and the label of the column
        """
        if isinstance(log_path, Path):
            log_path = str(log_path)
        
        self.train_path = log_path + "/train_metrics.csv"
        self.val_path = log_path + "/val_metrics.csv"
        self.types = [v[0] for v in argv]
        self.sep = ";"

        # -- print headers
        self.header(self.train_path, argv)
        self.header(self.val_path, argv)

    def header(self, path: str, argv: list) -> None:
        """
        Create the header of the metrics file

        :param path: The path to the file
        """
        with open(path, "+a") as f:
            for i, v in enumerate(argv, 1):
                print(v[1], end=self.sep if i < len(argv) else "\n", file=f)

    def log(self, phase: str, *argv: list) -> None:
        """
        Log metrics in to the file

        :param phase: The phase of the epoch, train or val
        :param *argv: The values of each column to print
        """
        fname = self.train_path if phase == "train" else self.val_path
        with open(fname, "+a") as f:
            for i, tv in enumerate(zip(self.types, argv), 1):
                end = self.sep if i < len(argv) else "\n"
                print(tv[0] % tv[1], end=end, file=f)


def create_folder(folder: str, experiment_name: str = "training") -> str:
    """
    Create a folder with the experiment name

    :param folder: The path to the folder to create the new folder
    :param experiment_name: The name of the experiment
    :return: The path to the new folder
    """
    os.makedirs(folder, exist_ok=True)
    folders = os.listdir(folder)
    experiment_len = len([f for f in folders if f.startswith(experiment_name)])
    log_folder = os.path.join(folder, f"{experiment_name}_{str(experiment_len + 1)}")
    os.makedirs(log_folder, exist_ok=True)
    return log_folder


def convert_time(seconds: int) -> str:
    """
    Convert seconds to a string with the time

    :param seconds: The seconds to convert
    :return: The time in the format HH:MM:SS
    """
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"


def log_csv_information(df: pd.DataFrame, output_path: Path, is_test: bool = False) -> None:
    """
    Save the information of the dataframe to files in the same directory.

    :param df: The dataframe to save
    :param output_path: The path to the output directory
    :param is_test: If the dataframe is to test
    """
    stats_text = []

    # -- overall statistics
    overall_counts = df['target'].value_counts().sort_index()
    stats_text.append("Overall Target Distribution:\n")
    stats_text.extend([f"Target {k}: {v}\n" for k, v in overall_counts.items()])

    # -- dataset-wise statistics
    stats_text.append("\nTarget Distribution by Dataset:\n")
    dataset_group = df.groupby(['dataset', 'target']).size().unstack(fill_value=0)
    for dataset in dataset_group.index:
        stats_text.append(f"Dataset: {dataset}\n")
        stats_text.extend([f"  Target {k}: {v}\n" for k, v in dataset_group.loc[dataset].items()])

    # -- fold-wise statistics
    if not is_test:
        stats_text.append("\nTarget Distribution by Fold:\n")
        fold_group = df.groupby(['fold', 'target']).size().unstack(fill_value=0)
        for fold in fold_group.index:
            stats_text.append(f"Fold: {fold}\n")
            stats_text.extend([f"  Target {k}: {v}\n" for k, v in fold_group.loc[fold].items()])

    # -- save statistics to a text file
    with open(output_path, "w") as f:
        f.writelines(stats_text)
