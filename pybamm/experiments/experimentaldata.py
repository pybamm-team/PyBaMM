#
# Experimental Data class
#

from collections import UserDict
import pandas as pd


class ExperimentalData(UserDict):
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
    mandatory_columns = ["Voltage [V]", "Time [s]", "Current [A]"]
    cycle_index_column = "Cycle"
    step_index_column = "Step"

    def __init__(self, filename, format="csv"):
        if format.lower() not in self.allowed:
            raise ValueError(
                f"format '{format}' not allowed, only"
                "supported formats are {self.allowed}"
            )
        self.format = format.lower()
        self.filename = filename
        self._data = None
        self._load()

    def _load(self):
        if self.format == "csv":
            self._load_csv()

        self._validate()

    def _load_csv(self):
        self._data = pd.read_csv(self.filename)

    def _validate(self):
        cols = self._data.columns
        for col in self.mandatory_columns:
            if col not in cols:
                raise ValueError(f"Mandatory column '{col}' not found")

        if self.cycle_index_column not in cols:
            self._data[self.cycle_index_column] = 0

        if self.step_index_column not in cols:
            self._data[self.step_index_column] = 0

    def __setitem__(self, key, value):
        raise ValueError("Not allowed to set values in ExperimentalData")

    def __getitem__(self, key):
        return self._data[key].values


if __name__ == "__main__":
    d = ExperimentalData("test_data.csv")
    print(d._data.head())
