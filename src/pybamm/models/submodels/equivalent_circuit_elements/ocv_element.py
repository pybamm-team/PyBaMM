import pybamm


class OCVElement(pybamm.BaseSubModel):
    """
    Open-circuit Voltage (OCV) element for
    equivalent circuits.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    options : dict, optional
        A dictionary of options to be passed to the model.
    """

    def __init__(self, param, options=None):
        super().__init__(param)
        self.model_options = options

    def build(self, submodels):
        soc = pybamm.Variable("SoC")
        ocv = self.param.ocv(soc)
        variables = {"SoC": soc, "Open-circuit voltage [V]": ocv}

        current = pybamm.CoupledVariable("Current [A]")
        self.coupled_variables.update({"Current [A]": current})

        T_cell = pybamm.CoupledVariable("Cell temperature [degC]")
        self.coupled_variables.update({"Cell temperature [degC]": T_cell})

        dUdT = self.param.dUdT(ocv, T_cell)

        T_cell_kelvin = pybamm.CoupledVariable("Cell temperature [K]")
        self.coupled_variables.update({"Cell temperature [K]": T_cell_kelvin})
        Q_rev = -current * T_cell_kelvin * dUdT

        variables.update(
            {
                "Entropic change [V/K]": dUdT,
                "Reversible heat generation [W]": Q_rev,
            }
        )

        cell_capacity = self.param.cell_capacity
        self.rhs = {soc: -current / cell_capacity / 3600}

        self.initial_conditions = {soc: self.param.initial_soc}
        self.variables.update(variables)

    def add_events_from(self, variables):
        soc = variables["SoC"]
        self.events += [
            pybamm.Event("Minimum SoC", soc),
            pybamm.Event("Maximum SoC", 1 - soc),
        ]
