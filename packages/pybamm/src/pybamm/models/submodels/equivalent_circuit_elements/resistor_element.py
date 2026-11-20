import pybamm


class ResistorElement(pybamm.BaseSubModel):
    """
    Resistor element for equivalent circuits.

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
        T_cell = self.coupled_variable("Cell temperature [degC]")
        current = self.coupled_variable("Current [A]")
        soc = self.coupled_variable("SoC")

        r = self.param.rcr_element("R0 [Ohm]", T_cell, current, soc)
        overpotential = -current * r
        Q_irr = current**2 * r

        self.variables.update(
            {
                "R0 [Ohm]": r,
                "Element-0 overpotential [V]": overpotential,
                "Element-0 irreversible heat generation [W]": Q_irr,
            }
        )
