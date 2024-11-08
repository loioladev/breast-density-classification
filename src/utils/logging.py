class CSVLogger(object):
    """
    Class to handle the logging of the metrics
    """

    def __init__(self, fname: str, *argv: list) -> None:
        """
        Constructor of the class

        :param fname: The path to the file to save the metrics
        :param *argv: List of tuples, where each tuple contains the format
        and the label of the column
        """
        self.fname = fname
        self.types = []

        # -- print headers
        with open(self.fname, "+a") as f:
            for i, v in enumerate(argv, 1):
                self.types.append(v[0])
                print(v[1], end="," if i < len(argv) else "\n", file=f)

    def log(self, *argv: list) -> None:
        """
        Log metrics in to the file

        :param *argv: The values of each column to print
        """
        with open(self.fname, "+a") as f:
            for i, tv in enumerate(zip(self.types, argv), 1):
                end = "," if i < len(argv) else "\n"
                print(tv[0] % tv[1], end=end, file=f)
