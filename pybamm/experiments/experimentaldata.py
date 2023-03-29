#
# Experimental Data class
#

import pandas as pd

# import pybamm


class ExperimentalData():
    """
    This is a base class for experimental data.

    Parameters
    ----------
    filename: string, path object
        Location of data container
    format: string
        currently "csv"
    """

    allowed = ["csv"]

    def __init__(self, filename, format="csv"):
        if format.lower() not in self.allowed:
            raise ValueError(
                f"format '{format}' not allowed, only"
                "supported formats are {self.allowed}"
            )
        self.format = format.lower()
        self.filename = filename
        self.data = None
        self._load()


    def _load(self):

        if self.format == "csv":
            self._load_csv()

    def _load_csv(self):
        self.data = pd.read_csv(self.filename)


if __name__ == "__main__":
    data = ExperimentalData("test_data.csv")
    print(data.data.head())
