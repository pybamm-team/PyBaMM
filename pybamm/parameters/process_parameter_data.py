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


def process_2D_data_csv(name, path=None):
    """
    Process 2D data from a csv file. Assumes
    data is in the form of a three columns
    and that all data points lie on a regular
    grid. The first column is assumed to
    be the 'slowest' changing variable and
    the second column the 'fastest' changing
    variable, which is the C convention for
    indexing multidimensional arrays (as opposed
    to the Fortran convention where the 'fastest'
    changing variable comes first).

    Parameters
    ----------
    name : str
        The name to be given to the function
    path : str
        The path to the file where the three
        dimensional data is stored.

    Returns
    -------
    formatted_data: tuple
        A tuple containing the name of the function
        and the data formatted correctly for use
        within three-dimensional interpolants.
    """

    filename, name = _process_name(name, path, ".csv")

    df = pd.read_csv(filename)

    x1 = np.array(list(set(df.iloc[:, 0])))
    x2 = np.array(list(set(df.iloc[:, 1])))

    value = df.iloc[:, 2].to_numpy()

    x1.sort()
    x2.sort()

    x = (x1, x2)

    value_data = np.reshape(
        value,
        (len(x1), len(x2)),
        order="C",  # use the C convention
    )

    formatted_data = (name, (x, value_data))

    return formatted_data


def process_3D_data_csv(name, path=None):
    """
    Process 3D data from a csv file. Assumes
    data is in the form of four columns and
    that all data points lie on a
    regular grid. The first column is assumed to
    be the 'slowest' changing variable and
    the third column the 'fastest' changing
    variable, which is the C convention for
    indexing multidimensional arrays (as opposed
    to the Fortran convention where the 'fastest'
    changing variable comes first).

    Parameters
    ----------
    name : str
        The name to be given to the function
    path : str
        The path to the file where the three
        dimensional data is stored.

    Returns
    -------
    formatted_data: tuple
        A tuple containing the name of the function
        and the data formatted correctly for use
        within three-dimensional interpolants.
    """

    filename, name = _process_name(name, path, ".csv")

    df = pd.read_csv(filename)

    x1 = np.array(list(set(df.iloc[:, 0])))
    x2 = np.array(list(set(df.iloc[:, 1])))
    x3 = np.array(list(set(df.iloc[:, 2])))

    value = df.iloc[:, 3].to_numpy()

    x1.sort()
    x2.sort()
    x3.sort()

    x = (x1, x2, x3)

    value_data = np.reshape(
        value,
        (len(x1), len(x2), len(x3)),
        order="C",
    )

    formatted_data = (name, (x, value_data))

    return formatted_data
