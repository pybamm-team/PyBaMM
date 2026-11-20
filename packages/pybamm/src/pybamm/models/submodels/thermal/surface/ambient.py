#
# Class for ambient surface temperature submodel
import pybamm


class Ambient(pybamm.BaseSubModel):
    """
    Class for setting surface temperature equal to ambient temperature.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    options : dict, optional
        A dictionary of options to be passed to the model.

    """

    def __init__(self, param, options=None):
        super().__init__(param, options=options)

    def build(self):
        T_amb = self.coupled_variable(
            "Ambient temperature [K]", domain="current collector"
        )
        T_amb_av = self.coupled_variable("Volume-averaged ambient temperature [K]")
        self.variables.update(
            {
                "Surface temperature [K]": T_amb,
                "Volume-averaged surface temperature [K]": T_amb_av,
                "Environment total cooling [W]": pybamm.Scalar(0),
            }
        )
