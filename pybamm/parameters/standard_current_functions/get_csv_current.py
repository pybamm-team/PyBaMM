#
# Load current profile from a csv file
#
import pybamm
import os
import pandas as pd
import numpy as np
import warnings


def get_csv_current(t):
    """
    Loads a current profile from a csv file and interpolate on the fly

    """
    pybamm_path = pybamm.root_dir()
    data = pd.read_csv(
        os.path.join(pybamm_path, "input", "drive_cycles", "US06.csv"), comment="#"
    )

    amplitude = data.values[:, 0]
    time = data.values[:, 1]
    if np.min(t) < time[0] or np.max(t) > time[-1]:
        warnings.warn(
            "Requested time ({}) is outside of the data range [{}, {}]".format(
                t, time[0], time[-1]
            ),
            pybamm.ModelWarning,
        )
    current = np.interp(t, time, amplitude)
    return current

    # def current(t):
    #    if np.min(t) < time[0] or np.max(t) > time[-1]:
    #        warnings.warn(
    #            "Requested time ({}) is outside of the data range [{}, {}]".format(
    #                t, time[0], time[-1]
    #            ),
    #            pybamm.ModelWarning,
    #        )
    #    return np.interp(t, time, amplitude)

    # return current
