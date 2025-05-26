import pybamm
from pybamm.models.submodels.thermal.base_thermal import BaseThermal


class FullThreeDimensional(BaseThermal):
    """
    Class for three-dimensional thermal submodel with constant heat source.

    This model solves the heat equation in 3D:
        rho_c_p * dT/dt = div(lambda * grad(T)) + Q

    where:
        - T is the temperature field in the 3D cell domain
        - rho_c_p is the volumetric heat capacity
        - lambda is the thermal conductivity
        - Q is the volumetric heat source

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    options : dict, optional
        A dictionary of options to be passed to the model.
    x_average : bool, optional
        Whether to include x-averaged variables. Default is False.
    """

    def __init__(self, param, options=None, x_average=False):
        options = options or {}
        options["dimensionality"] = 3
        super().__init__(param, options=options, x_average=x_average)
        self._geometry = (
            self.options.get("geometry options", {})
            .get("domains", {})
            .get("cell", {})
            .get("type", "rectangular")
        )
        pybamm.citations.register("Timms2021")
