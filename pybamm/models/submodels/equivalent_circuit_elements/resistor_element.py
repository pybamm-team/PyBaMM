import pybamm


class ResistorElement(pybamm.BaseSubModel):
    def __init__(self, param, element_number, options=None):
        super().__init__(param)
        self.element_number = element_number
        self.model_options = options

    def get_coupled_variables(self, variables):

        T_cell = variables["Cell temperature [degC]"]
        current = variables["Current [A]"]
        soc = variables["SoC"]

        r = self.param.rcr_element(
            f"R{self.element_number} [Ohm]", T_cell, current, soc
        )

        overpotential = -current * r
        Q_irr = current**2 * r

        variables.update(
            {
                f"R{self.element_number} [Ohm]": r,
                f"Element-{self.element_number} overpotential [V]": overpotential,
                f"Element-{self.element_number} "
                + "irreversible heat generation [W]": Q_irr,
            }
        )

        return variables
