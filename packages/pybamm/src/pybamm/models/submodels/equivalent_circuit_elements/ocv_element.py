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

    def build(self):
        current = self.coupled_variable("Current [A]")
        T_cell = self.coupled_variable("Cell temperature [degC]")
        T_cell_kelvin = self.coupled_variable("Cell temperature [K]")

        soc = pybamm.Variable("SoC")
        ocv = self.param.ocv(soc)
        dUdT = self.param.dUdT(ocv, T_cell)
        Q_rev = -current * T_cell_kelvin * dUdT
        self.variables.update(
            {
                "SoC": soc,
                "Open-circuit voltage [V]": ocv,
                "Entropic change [V/K]": dUdT,
                "Reversible heat generation [W]": Q_rev,
            }
        )

        self.rhs[soc] = -current / self.param.cell_capacity / 3600
        self.initial_conditions[soc] = self.param.initial_soc
        self.events += [
            pybamm.Event("Minimum SoC", soc),
            pybamm.Event("Maximum SoC", 1 - soc),
        ]
