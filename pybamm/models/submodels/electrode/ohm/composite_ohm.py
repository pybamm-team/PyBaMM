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

        eps = variables[self.domain + " electrode porosity"].orphans[0]

        if self._domain == "Negative":
            sigma_eff = self.param.sigma_n * (1 - eps) ** self.param.b
            phi_s = (
                pybamm.outer(i_boundary_cc, x_n * (x_n - 2 * l_n) / (2 * l_n))
                / sigma_eff
            )
            i_s = pybamm.outer(i_boundary_cc, 1 - x_n / l_n)

        elif self.domain == "Positive":
            delta_phi_p_av = variables[
                "Average positive electrode surface potential difference"
            ]
            phi_e_p_av = variables["Average positive electrolyte potential"]

            sigma_eff = self.param.sigma_p * (1 - eps) ** self.param.b

            const = (
                delta_phi_p_av
                + phi_e_p_av
                - (i_boundary_cc / sigma_eff) * (l_p - l_p ** 3 / 3)
            )

            phi_s = (
                pybamm.Broadcast(
                    const, ["positive electrode"], broadcast_type="primary"
                )
                - pybamm.outer(i_boundary_cc, x_p + (x_p - 1) ** 2 / (2 * l_p))
                / sigma_eff
            )
            i_s = pybamm.outer(i_boundary_cc, 1 - (1 - x_p) / l_p)

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
