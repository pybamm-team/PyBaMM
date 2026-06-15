#
# Standard electrical parameters
#
import pybamm

from .base_parameters import BaseParameters


class ElectricalParameters(BaseParameters):
    """
    Standard electrical parameters
    """

    def __init__(self):
        # Get geometric parameters
        self.geo = pybamm.geometric_parameters

        # Set parameters
        self._set_parameters()

    def _set_parameters(self):
        """Defines the dimensional parameters."""

        self.Q = pybamm.Parameter("Nominal cell capacity [A.h]")
        self.n_cells = pybamm.Parameter(
            "Number of cells connected in series to make a battery"
        )
        self.voltage_low_cut = pybamm.Parameter("Lower voltage cut-off [V]")
        self.voltage_high_cut = pybamm.Parameter("Upper voltage cut-off [V]")
        self.ocp_soc_0 = pybamm.Parameter("Open-circuit voltage at 0% SOC [V]")
        self.ocp_soc_100 = pybamm.Parameter("Open-circuit voltage at 100% SOC [V]")
        # Current as a function of time
        self.current_with_time = pybamm.FunctionParameter(
            "Current function [A]", {"Time [s]": pybamm.t}
        )
        self.current_density_with_time = self.current_with_time / (self.geo.A_cc)

    def R_contact(self, T):
        """
        Series resistance [Ohm] as a function of cell temperature ``T`` [K].

        Despite the legacy name "Contact resistance [Ohm]", this is a lumped
        series resistance added to the terminal voltage
        (https://github.com/FaradayInstitution/BPX/issues/130); the name is kept
        to avoid a breaking rename.
        """
        return pybamm.FunctionParameter(
            "Contact resistance [Ohm]", {"Temperature [K]": T}
        )


electrical_parameters = ElectricalParameters()
