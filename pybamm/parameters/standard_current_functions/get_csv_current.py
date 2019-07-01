#
# Load current profile from a csv file
#
import pybamm
import os
import pandas as pd
import numpy as np
import warnings
import scipy.interpolate as interp


# Load data from csv
pybamm_path = pybamm.root_dir()
data = pd.read_csv(
    os.path.join(pybamm_path, "input", "drive_cycles", "US06.csv"), comment="#"
).to_dict('list')

# Interpolate using Piecewise Cubic Hermite Interpolating Polynomial
# (does not overshoot non-smooth data)
current = interp.PchipInterpolator(data["time"], data["amplitude"])


def get_csv_current(t):
    """
    Calls the interpolating function created using the data from a user-supplied
    csv file at time t (seconds).
    """

    if np.min(t) < data["time"][0] or np.max(t) > data["time"][-1]:
        warnings.warn(
            "Requested time ({}) is outside of the data range [{}, {}]".format(
                t, data["time"][0], data["time"][-1]
            ),
            pybamm.ModelWarning,
        )

    return current(t)
