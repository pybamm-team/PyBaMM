#
# Composite model for Ohm's law in the electrode
#
import pybamm

from .base_ohm import BaseModel


class Composite(BaseModel):
    """An explicit composite leading and first order solution to solid phase
    current conservation with ohm's law. Note that the returned current density is
    only the leading order approximation.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        Either 'Negative electrode' or 'Positive electrode'


    **Extends:** :class:`pybamm.BaseOhm`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def get_coupled_variables(self, variables):

        i_boundary_cc = variables["Current collector current density"]

        # import parameters and spatial variables
        l_n = self.param.l_n
        l_p = self.param.l_p
        x_n = pybamm.standard_spatial_vars.x_n
        x_p = pybamm.standard_spatial_vars.x_p

        eps = variables[self.domain + " electrode porosity"]

        if self.domain == "Negative":
            sigma_eff = self.param.sigma_n * (1 - eps)
            phi_s = i_boundary_cc * x_n * (x_n - 2 * l_n) / (2 * sigma_eff * l_n)
            i_s = i_boundary_cc - i_boundary_cc * x_n / l_n

        elif self.domain == "Positive":
            ocp_p_av = variables["Average positive electrode open circuit potential"]
            eta_r_p_av = variables["Average positive electrode reaction overpotential"]
            phi_e_p_av = variables["Average positive electrolyte potential"]

            sigma_eff = self.param.sigma_p * (1 - eps)

            const = (
                ocp_p_av
                + eta_r_p_av
                + phi_e_p_av
                - (i_boundary_cc / 6 / l_p / sigma_eff) * (2 * l_p ** 2 - 6 * l_p + 3)
            )
            phi_s = const - i_boundary_cc * x_p / (2 * l_p * sigma_eff) * (
                x_p + 2 * (l_p - 1)
            )
            i_s = i_boundary_cc - i_boundary_cc * (1 - x_p) / l_p

        variables.update(self._get_standard_potential_variables(phi_s))
        variables.update(self._get_standard_current_variables(i_s))

        if self.domain == "Positive":
            variables.update(self._get_standard_whole_cell_current_variables(variables))

        return variables

    @property
    def default_solver(self):
        """
        Create and return the default solver for this model
        """
        return pybamm.ScikitsOdeSolver()
