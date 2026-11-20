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

    def build(self):
        number = self.element_number
        T_cell = self.coupled_variable("Cell temperature [degC]")
        current = self.coupled_variable("Current [A]")
        soc = self.coupled_variable("SoC")

        vrc = pybamm.Variable(f"Element-{number} overpotential [V]")
        r = self.param.rcr_element(f"R{number} [Ohm]", T_cell, current, soc)
        c = self.param.rcr_element(f"C{number} [F]", T_cell, current, soc)
        tau = r * c
        Q_irr = -current * vrc

        self.variables.update(
            {
                f"Element-{number} overpotential [V]": vrc,
                f"R{number} [Ohm]": r,
                f"C{number} [F]": c,
                f"tau{number} [s]": tau,
                f"Element-{number} irreversible heat generation [W]": Q_irr,
            }
        )

        self.rhs[vrc] = -vrc / tau - current * r / tau
        self.initial_conditions[vrc] = self.param.initial_rc_overpotential(number)
