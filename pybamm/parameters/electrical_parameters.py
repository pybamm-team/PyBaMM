#
# Standard electrical parameters
#
import pybamm
import numpy as np
from .base_parameters import BaseParameters


class ElectricalParameters(BaseParameters):
    """
    Standard electrical parameters

    Layout:
        1. Dimensional Parameters
        2. Dimensionless Parameters
    """

    def __init__(self):

        # Get geometric parameters
        self.geo = pybamm.geometric_parameters

        # Set parameters
        self._set_dimensional_parameters()
        # self._set_dimensionless_parameters()

    def _set_dimensional_parameters(self):
        """Defines the dimensional parameters."""

        self.Q = pybamm.Parameter("Nominal cell capacity [A.h]")
        self.n_electrodes_parallel = pybamm.Parameter(
            "Number of electrodes connected in parallel to make a cell"
        )
        self.n_cells = pybamm.Parameter(
            "Number of cells connected in series to make a battery"
        )
        self.voltage_low_cut = pybamm.Parameter("Lower voltage cut-off [V]")
        self.voltage_high_cut = pybamm.Parameter("Upper voltage cut-off [V]")

        # Current as a function of time
        self.current_with_time = pybamm.FunctionParameter(
            "Current function [A]", {"Time[s]": pybamm.t}
        )
        self.current_density_with_time = self.current_with_time / (
            self.n_electrodes_parallel * self.geo.A_cc
        )


electrical_parameters = ElectricalParameters()
