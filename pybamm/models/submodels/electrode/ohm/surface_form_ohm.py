#
# Class for ohmic electrodes in the surface potential formulation
#
import pybamm


class SurfaceFormOhm(pybamm.BaseOhm):
    """Ohm's law + conservation of current for the current in the electrodes.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`pybamm.SubModel`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def set_coupled_variables(self, variables):

        param = self.param
        x_n = pybamm.standard_spatial_vars.x_n
        x_p = pybamm.standard_spatial_vars.x_p
        i_boundary_cc = variables["Current collector current density"]
        i_e = variables[self._domain + " electrolyte current density"]
        eps = variables[self._domain + " porosity"]

        i_s = i_boundary_cc - i_e

        if self._domain == "Negative":
            conductivity = param.sigma_n * (1 - eps) ** param.b
            phi_s = -pybamm.IndefiniteIntegral(i_s / conductivity, x_n)

        elif self._domain == "Positive":

            phi_e_s = variables["Separator electrolyte potential"]
            delta_phi_p = variables["Positive electrode surface potential difference"]

            conductivity = param.sigma_n * (1 - eps) ** param.b
            phi_s = (
                -pybamm.IndefiniteIntegral(i_s / conductivity, x_p)
                + pybamm.boundary_value(phi_e_s, "right")
                + pybamm.boundary_value(delta_phi_p, "left")
            )

        self._get_standard_potential_variables(phi_s)
        self._get_standard_current_variables(i_s)

        return variables

    @property
    def default_solver(self):
        """
        Create and return the default solver for this model
        """
        return pybamm.ScikitsDaeSolver()
