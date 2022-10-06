#
# Functions to process parameter data (for Interpolants)
#
import os
import pandas as pd
import json
import numpy as np


def _process_name(name, path, ext):
    if not name.endswith(ext):
        name = name + ext

    # Set the path
    if path is not None:
        # first look in the specified path
        filename = os.path.join(path, name)
        if not os.path.exists(filename):
            # then look in the "data" subfolder
            filename = os.path.join(path, "data", name)
            if not os.path.exists(filename):
                raise FileNotFoundError(
                    "Could not find file '{}' in '{}' or '{}'".format(
                        name, path, os.path.join(path, "data")
                    )
                )
    else:
        filename = name
        # Split the name (in case original name was a path)
        _, name = os.path.split(filename)

    # Remove the extension from the name
    return (filename, name.split(".")[0])


def process_1D_data(name, path=None):
    """
    Process 1D data from a csv file
    """
    filename, name = _process_name(name, path, ".csv")

    data = pd.read_csv(
        filename, comment="#", skip_blank_lines=True, header=None
    ).to_numpy()
    # Save name and data
    return (name, ([data[:, 0]], data[:, 1]))


def process_2D_data(name, path=None):
    """
    Process 2D data from a JSON file
    """
    filename, name = _process_name(name, path, ".json")

    with open(filename, "r") as jsonfile:
        json_data = json.load(jsonfile)
    data = json_data["data"]
    data[0] = [np.array(el) for el in data[0]]
    data[1] = np.array(data[1])
    return (name, tuple(data))
