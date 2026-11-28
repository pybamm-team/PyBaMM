import pybamm


class OCVHysteresisElement(pybamm.BaseSubModel):
    """
    Open-circuit voltage (OCV) element for equivalent circuits, extended with a
    one-state hysteresis model based on the approach of :footcite:t:`Plett2004`.
    The hysteresis state `h` (bounded in [-1, 1]) is governed by an ODE that
    depends on the terminal current.

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
        soc = pybamm.Variable("SoC")
        h = pybamm.Variable("Hysteresis state")

        variables = {
            "SoC": soc,
            "Hysteresis state": h,
        }
        return variables

    def get_coupled_variables(self, variables):
        current = variables["Current [A]"]
        soc = variables["SoC"]
        h = variables["Hysteresis state"]
        T_cell = variables["Cell temperature [degC]"]
        T_cell_kelvin = variables["Cell temperature [K]"]

        dUdT = self.param.dUdT(T_cell, soc)
        ocv_charge = self.param.ocv(T_cell, soc, "charge")
        ocv_discharge = self.param.ocv(T_cell, soc, "discharge")

        ocv_ave = 0.5 * (ocv_charge + ocv_discharge)
        ocv = 0.5 * (1.0 + h) * ocv_charge + 0.5 * (1.0 - h) * ocv_discharge

        Q_rev = -current * T_cell_kelvin * dUdT
        Q_hys = -current * (ocv - ocv_ave)

        variables.update(
            {
                "Entropic change [V/K]": dUdT,
                "Open-circuit voltage charge [V]": ocv_charge,
                "Open-circuit voltage discharge [V]": ocv_discharge,
                "Average open-circuit voltage [V]": ocv_ave,
                "Open-circuit voltage [V]": ocv,
                "Reversible heat generation [W]": Q_rev,
                "Hysteresis heat generation [W]": Q_hys,
            }
        )

        return variables

    def set_rhs(self, variables):
        soc = variables["SoC"]
        h = variables["Hysteresis state"]
        current = variables["Current [A]"]
        T_cell = variables["Cell temperature [degC]"]
        cell_capacity = self.param.cell_capacity

        gamma_charge = self.param.hysteresis_decay(T_cell, current, soc, "charge")
        gamma_discharge = self.param.hysteresis_decay(T_cell, current, soc, "discharge")

        i_sign = pybamm.sign(-current)
        signed_h = 1 - i_sign * h
        gamma = 0.5 * (1 - i_sign) * gamma_charge + 0.5 * (1 + i_sign) * gamma_discharge

        dhdt = gamma * (-current) / cell_capacity / 3600 * signed_h
        self.rhs[soc] = -current / cell_capacity / 3600
        self.rhs[h] = dhdt

    def set_initial_conditions(self, variables):
        soc = variables["SoC"]
        h = variables["Hysteresis state"]
        self.initial_conditions = {
            soc: self.param.initial_soc,
            h: self.param.initial_h,
        }

    def add_events_from(self, variables):
        soc = variables["SoC"]
        self.events += [
            pybamm.Event("Minimum SoC", soc),
            pybamm.Event("Maximum SoC", 1 - soc),
        ]
