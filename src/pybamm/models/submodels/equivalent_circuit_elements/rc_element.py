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

    def build(self, submodels):
        vrc = pybamm.Variable(f"Element-{self.element_number} overpotential [V]")
        variables = {f"Element-{self.element_number} overpotential [V]": vrc}

        T_cell = pybamm.CoupledVariable("Cell temperature [degC]")
        self.coupled_variables.update({"Cell temperature [degC]": T_cell})
        current = pybamm.CoupledVariable("Current [A]")
        self.coupled_variables.update({"Current [A]": current})
        soc = pybamm.CoupledVariable("SoC")
        self.coupled_variables.update({"SoC": soc})

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

        self.rhs = {
            vrc: -vrc / (tau) - current * r / tau,
        }
        self.initial_conditions = {
            vrc: self.param.initial_rc_overpotential(self.element_number)
        }
        self.variables.update(variables)
