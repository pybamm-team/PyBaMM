#
# Load current profile from a csv file
#
import pybamm
import os
import pandas as pd
import numpy as np
import warnings
import scipy.interpolate as interp


class GetCurrentData(pybamm.GetCurrent):
    """
    A class which loads a current profile from a csv file and creates an
    interpolating function which can be called during solve.

    Parameters
    ----------
    filename : str
        The name of the file to load.
    units : str, optional
        The units of the current data which is to be loaded. Can be "[]" for
        dimenionless data (default), or "[A]" for current in Amperes.
    current_scale : :class:`pybamm.Symbol` or float, optional
        The scale the current in Amperes if loading non-dimensional data. Default
        is to use the typical current I_typ

    **Extends:"": :class:`pybamm.GetCurrent`
    """

    def __init__(
        self, filename, units="[]", current_scale=pybamm.electrical_parameters.I_typ
    ):
        self.parameters = {"Current [A]": current_scale}
        self.parameters_eval = {"Current [A]": current_scale}

        # Load data from csv
        if filename:
            pybamm_path = pybamm.root_dir()
            data = pd.read_csv(
                os.path.join(pybamm_path, "input", "drive_cycles", filename),
                comment="#",
                skip_blank_lines=True,
            ).to_dict("list")

            self.time = np.array(data["time [s]"])
            self.units = units
            self.current = np.array(data["current " + units])
            # If voltage data is present, load it into the class
            try:
                self.voltage = np.array(data["voltage [V]"])
            except KeyError:
                self.voltage = None
        else:
            raise pybamm.ModelError("No input file provided for current")

    def __str__(self):
        return "Current from data"

    def interpolate(self):
        " Creates the interpolant from the loaded data "
        # If data is dimenionless, multiply by a typical current (e.g. data
        # could be C-rate and current_scale the 1C discharge current). Otherwise,
        # just import the current data.
        if self.units == "[]":
            current = self.parameters_eval["Current [A]"] * self.current
        elif self.units == "[A]":
            current = self.current
        else:
            raise pybamm.ModelError(
                "Current data must have units [A] or be dimensionless"
            )
        # Interpolate using Piecewise Cubic Hermite Interpolating Polynomial
        # (does not overshoot non-smooth data)
        self.current_interp = interp.PchipInterpolator(self.time, current)

    def __call__(self, t):
        """
        Calls the interpolating function created using the data from user-supplied
        data file at time t (seconds).
        """

        if np.min(t) < self.time[0] or np.max(t) > self.time[-1]:
            warnings.warn(
                "Requested time ({}) is outside of the data range [{}, {}]".format(
                    t, self.time[0], self.time[-1]
                ),
                pybamm.ModelWarning,
            )

        return self.current_interp(t)
