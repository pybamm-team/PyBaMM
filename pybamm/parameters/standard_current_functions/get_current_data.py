#
# Load current profile from a csv file
#
import pybamm
import os
import pandas as pd
import numpy as np
import warnings
import scipy.interpolate as interp


class GetCurrentData:
    """
    A class which loads a current profile from a csv file and creates an
    interpolating function which can be called during solve.

    For broadcast and mass_matrix, we follow the default behaviour from SpatialMethod.

    Parameters
    ----------
    filename : str
        The name of the file to load

    """

    def __init__(self, filename):
        if filename:
            # Load data from csv
            pybamm_path = pybamm.root_dir()
            data = pd.read_csv(
                os.path.join(pybamm_path, "input", "drive_cycles", filename),
                comment="#",
            ).to_dict("list")

            # Interpolate using Piecewise Cubic Hermite Interpolating Polynomial
            # (does not overshoot non-smooth data)
            self.current = interp.PchipInterpolator(data["time"], data["amplitude"])
            self.t_start = data["time"][0]
            self.t_end = data["time"][-1]
        else:
            raise pybamm.ModelError("No input file provided for current")

    def __call__(self, t):
        """
        Calls the interpolating function created using the data from user-supplied
        data file at time t (seconds).
        """

        if np.min(t) < self.t_start or np.max(t) > self.t_end:
            warnings.warn(
                "Requested time ({}) is outside of the data range [{}, {}]".format(
                    t, self.t_start, self.t_end
                ),
                pybamm.ModelWarning,
            )

        return self.current(t)
