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

    def get_fundamental_variables(self):
        T_cell = pybamm.Variable("Cell temperature [degC]")
        soc = pybamm.Variable("SoC")
        ocv = self.param.ocv(T_cell, soc)

        variables = {
            "Cell temperature [degC]": T_cell,
            "SoC": soc,
            "Open-circuit voltage [V]": ocv,
        }

        return variables

    def get_coupled_variables(self, variables):
        current = variables["Current [A]"]
        soc = variables["SoC"]
        T_cell = variables["Cell temperature [degC]"]
        T_cell_kelvin = variables["Cell temperature [K]"]

        dUdT = self.param.dUdT(T_cell, soc)

        Q_rev = -current * T_cell_kelvin * dUdT
        Q_hys = pybamm.Scalar(0)

        variables.update(
            {
                "Entropic change [V/K]": dUdT,
                "Reversible heat generation [W]": Q_rev,
                "Hysteresis heat generation [W]": Q_hys,
            }
        )

        return variables

    def set_rhs(self, variables):
        soc = variables["SoC"]
        current = variables["Current [A]"]
        cell_capacity = self.param.cell_capacity
        self.rhs = {soc: -current / cell_capacity / 3600}

    def set_initial_conditions(self, variables):
        soc = variables["SoC"]
        self.initial_conditions = {soc: self.param.initial_soc}

    def add_events_from(self, variables):
        soc = variables["SoC"]
        self.events += [
            pybamm.Event("Minimum SoC", soc),
            pybamm.Event("Maximum SoC", 1 - soc),
        ]
