import pybamm


class EcmParameters:
    def __init__(self):
        self.timescale = pybamm.Scalar(1)

        self.cell_capacity = pybamm.Parameter("Cell capacity [A.h]")

        self._set_current_parameters()
        self._set_voltage_parameters()
        self._set_thermal_parameters()
        self._set_initial_condition_parameters()
        self._set_compatibility_parameters()

    def _set_current_parameters(self):
        self.dimensional_current_with_time = pybamm.FunctionParameter(
            "Current function [A]", {"Time [s]": pybamm.t * self.timescale}
        )

    def _set_voltage_parameters(self):
        self.voltage_high_cut = pybamm.Parameter("Upper voltage cut-off [V]")
        self.voltage_low_cut = pybamm.Parameter("Lower voltage cut-off [V]")

    def _set_thermal_parameters(self):
        self.cth_cell = pybamm.Parameter("Cell thermal mass [J/K]")
        self.k_cell_jig = pybamm.Parameter("Cell-jig heat transfer coefficient [W/K]")

        self.cth_jig = pybamm.Parameter("Jig thermal mass [J/K]")
        self.k_jig_air = pybamm.Parameter("Jig-air heat transfer coefficient [W/K]")

    def _set_compatibility_parameters(self):
        # These are parameters that for compatibility with
        # external circuits submodels
        self.Q = self.cell_capacity
        self.current_with_time = self.dimensional_current_with_time
        self.dimensional_current_density_with_time = self.dimensional_current_with_time
        self.I_typ = pybamm.Scalar(1)
        self.n_electrodes_parallel = pybamm.Scalar(1)
        self.A_cc = pybamm.Scalar(1)
        self.n_cells = pybamm.Scalar(1)

    def _set_initial_condition_parameters(self):
        self.initial_soc = pybamm.Parameter("Initial SoC")
        self.initial_T_cell = pybamm.Parameter("Initial cell temperature [degC]")
        self.initial_T_jig = pybamm.Parameter("Initial jig temperature [degC]")

    def T_amb(self, t):
        return pybamm.FunctionParameter("Ambient temperature [degC]", {"Time [s]": t})

    def ocv(self, soc):
        return pybamm.FunctionParameter("Open circuit voltage [V]", {"SoC": soc})

    def rcr_element(self, name, T_cell, current, soc):
        inputs = {"Cell temperature [degC]": T_cell, "Current [A]": current, "SoC": soc}

        return pybamm.FunctionParameter(name, inputs)

    def initial_rc_overpotential(self, element_number):
        return pybamm.Parameter(f"Element-{element_number} initial overpotential [V]")

    def dUdT(self, ocv, T_cell):
        inputs = {"Open circuit voltage [V]": ocv, "Cell temperature [degC]": T_cell}
        return pybamm.FunctionParameter("Entropic change [V/K]", inputs)
