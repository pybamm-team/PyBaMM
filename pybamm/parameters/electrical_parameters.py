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
        self._set_dimensionless_parameters()

    def _set_dimensional_parameters(self):
        """Defines the dimensional parameters."""

        self.I_typ = pybamm.Parameter("Typical current [A]")
        self.Q = pybamm.Parameter("Nominal cell capacity [A.h]")
        self.Ratio_Li_SolConsumed = pybamm.Parameter("Ratio of Li-ion concentration change in electrolyte consider solvent consumption") # Mark Ruihe modify
        self.Int_ElelyExces_Ratio = pybamm.Parameter("Initial electrolyte excessive amount ratio")          # Mark Ruihe modify
        self.Ratio_Elely_Dryout   = pybamm.Parameter("Ratio of electrolyte dry out in jelly roll")          # Mark Ruihe modify
        self.Vol_Elely_JR         = pybamm.Parameter("Current total electrolyte volume in jelly roll [m3]") # Mark Ruihe modify
        self.Vol_Elely_Tot        = pybamm.Parameter("Current total electrolyte volume in whole cell [m3]") # Mark Ruihe modify
        self.C_rate = pybamm.AbsoluteValue(self.I_typ / self.Q)
        self.n_electrodes_parallel = pybamm.Parameter(
            "Number of electrodes connected in parallel to make a cell"
        )
        self.n_cells = pybamm.Parameter(
            "Number of cells connected in series to make a battery"
        )
        self.i_typ = pybamm.Function(
            np.abs, self.I_typ / (self.n_electrodes_parallel * self.geo.A_cc)
        )
        self.voltage_low_cut_dimensional = pybamm.Parameter("Lower voltage cut-off [V]")
        self.voltage_high_cut_dimensional = pybamm.Parameter(
            "Upper voltage cut-off [V]"
        )

        # Current as a function of *dimensional* time. The below is overwritten in
        # lithium_ion_parameters.py and lead_acid_parameters.py to use the correct
        # timescale used for non-dimensionalisation. For a base model, the user may
        # provide the typical timescale as a parameter.
        self.timescale = pybamm.Parameter("Typical timescale [s]")
        self.dimensional_current_with_time = pybamm.FunctionParameter(
            "Current function [A]", {"Time[s]": pybamm.t * self.timescale}
        )
        self.dimensional_current_density_with_time = (
            self.dimensional_current_with_time
            / (self.n_electrodes_parallel * self.geo.A_cc)
        )

    def _set_dimensionless_parameters(self):
        """Defines the dimensionless parameters."""

        self.current_with_time = (
            self.dimensional_current_with_time / self.I_typ * pybamm.sign(self.I_typ)
        )


electrical_parameters = ElectricalParameters()
