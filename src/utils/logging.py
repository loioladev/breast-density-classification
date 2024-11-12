import os


class CSVLogger(object):
    """
    Class to handle the logging of the metrics
    """

    def __init__(self, log_path: str, *argv: list) -> None:
        """
        Constructor of the class

        :param log_path: The path to the directory to save the metrics
        :param *argv: List of tuples, where each tuple contains the format
        and the label of the column
        """
        self.train_path = log_path + "/train_metrics.csv"
        self.val_path = log_path + "/val_metrics.csv"
        self.types = [v[0] for v in argv]

        # -- print headers
        self.header(self.train_path, argv)
        self.header(self.val_path, argv)

    def header(self, path: str, *argv: list) -> None:
        """
        Create the header of the metrics file

        :param path: The path to the file
        """
        with open(path, "+a") as f:
            for i, v in enumerate(argv, 1):
                print(v[1], end="," if i < len(argv) else "\n", file=f)
        return

    def log(self, phase: str, *argv: list) -> None:
        """
        Log metrics in to the file

        :param phase: The phase of the epoch, train or val
        :param *argv: The values of each column to print
        """
        fname = self.train_path if phase == "train" else self.val_path
        with open(fname, "+a") as f:
            for i, tv in enumerate(zip(self.types, argv), 1):
                end = "," if i < len(argv) else "\n"
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
