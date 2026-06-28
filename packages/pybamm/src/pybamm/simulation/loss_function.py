import numpy as np
import pandas as pd

import pybamm

bdp_to_pybamm_mapping = {
    "Voltage / V": pybamm.Variable("Voltage [V]"),
    "Current / A": pybamm.Variable("Current [A]"),
}


def _data_comparison(data: pd.DataFame):
    data_times = data["Test Time / s"]
    data_values_list = []
    variables_list = []
    for column in data.columns:
        variable = bdp_to_pybamm_mapping.get(column, None)
        if variable is not None:
            variables_list.append(variable)
            data_values_list.append(data[column])
    if not variables_list:
        raise ValueError(
            "No variables found in the data. Please ensure that the data "
            "contains columns with any of the following names: "
            f"{list(bdp_to_pybamm_mapping.keys())}"
        )

    data_values = np.hstack(data_values_list)
    variables = pybamm.NumpyConcatenation(*variables_list)

    data_values = data["value"]
    data = pybamm.DiscreteTimeData(data_times, data_values, "sum of squares data")
    return data, variables


def sum_of_squares(data: pd.DataFrame):
    """
    A method to create a loss function with a sum-of-squared-error loss function, given some data in BDF format (https://battery-data-alliance.github.io/battery-data-format/) to fit against.
    """
    data, variables = _data_comparison(data)
    return pybamm.DiscreteTimeSum((data - variables) ** 2)
