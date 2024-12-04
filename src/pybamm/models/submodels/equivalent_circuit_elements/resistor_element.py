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
        T_cell = pybamm.CoupledVariable("Cell temperature [degC]")
        self.coupled_variables.update({"Cell temperature [degC]": T_cell})
        current = pybamm.CoupledVariable("Current [A]")
        self.coupled_variables.update({"Current [A]": current})
        soc = pybamm.CoupledVariable("SoC")
        self.coupled_variables.update({"SoC": soc})

        r = self.param.rcr_element("R0 [Ohm]", T_cell, current, soc)

        overpotential = -current * r
        Q_irr = current**2 * r

        self.variables.update(
            {
                "R0 [Ohm]": r,
                "Element-0 overpotential [V]": overpotential,
                "Element-0 " + "irreversible heat generation [W]": Q_irr,
            }
        )
