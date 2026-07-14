import pybamm


class RCElement(pybamm.BaseSubModel):
    """
    Parallel Resistor-Capacitor (RC) element for
    equivalent circuits.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    element_number: int
        The number of the element (i.e. whether it
        is the first, second, third, etc. element)
    options : dict, optional
        A dictionary of options to be passed to the model.
    """

    def __init__(self, param, element_number, options=None):
        super().__init__(param)
        self.element_number = element_number
        self.model_options = options

    def get_fundamental_variables(self):
        vrc = pybamm.Variable(f"Element-{self.element_number} overpotential [V]")
        variables = {f"Element-{self.element_number} overpotential [V]": vrc}
        return variables

    def get_coupled_variables(self, variables):
        T_cell = variables["Cell temperature [degC]"]
        current = variables["Current [A]"]
        soc = variables["SoC"]

        r = self.param.rcr_element(
            f"R{self.element_number} [Ohm]", T_cell, current, soc
        )
        c = self.param.rcr_element(f"C{self.element_number} [F]", T_cell, current, soc)
        tau = r * c

        vrc = variables[f"Element-{self.element_number} overpotential [V]"]

        Q_irr = -current * vrc

        variables.update(
            {
                f"R{self.element_number} [Ohm]": r,
                f"C{self.element_number} [F]": c,
                f"tau{self.element_number} [s]": tau,
                f"Element-{self.element_number} "
                + "irreversible heat generation [W]": Q_irr,
            }
        )

        return variables

    def set_rhs(self, variables):
        vrc = variables[f"Element-{self.element_number} overpotential [V]"]
        current = variables["Current [A]"]

        r = variables[f"R{self.element_number} [Ohm]"]
        tau = variables[f"tau{self.element_number} [s]"]

        self.rhs = {
            vrc: -vrc / (tau) - current * r / tau,
        }

    def set_initial_conditions(self, variables):
        vrc = variables[f"Element-{self.element_number} overpotential [V]"]

        self.initial_conditions = {
            vrc: self.param.initial_rc_overpotential(self.element_number)
        }
