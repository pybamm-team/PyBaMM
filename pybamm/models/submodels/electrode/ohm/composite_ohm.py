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

        eps = variables[
            "Leading-order x-averaged " + self.domain.lower() + " electrode porosity"
        ]
        phi_s_cn = variables["Negative current collector potential"]

        if self._domain == "Negative":
            sigma_eff = self.param.sigma_n * (1 - eps) ** self.param.b
            phi_s = pybamm.PrimaryBroadcast(
                phi_s_cn, "negative electrode"
            ) + pybamm.outer(
                i_boundary_cc / sigma_eff, x_n * (x_n - 2 * l_n) / (2 * l_n)
            )
            i_s = pybamm.outer(i_boundary_cc, 1 - x_n / l_n)

        elif self.domain == "Positive":
            delta_phi_p_av = variables[
                "X-averaged positive electrode surface potential difference"
            ]
            phi_e_p_av = variables["X-averaged positive electrolyte potential"]

            sigma_eff = self.param.sigma_p * (1 - eps) ** self.param.b

            const = (
                delta_phi_p_av
                + phi_e_p_av
                + (i_boundary_cc / sigma_eff) * (1 - l_p / 3)
            )

            phi_s = pybamm.PrimaryBroadcast(
                const, ["positive electrode"]
            ) - pybamm.outer(
                i_boundary_cc / sigma_eff, x_p + (x_p - 1) ** 2 / (2 * l_p)
            )
            i_s = pybamm.outer(i_boundary_cc, 1 - (1 - x_p) / l_p)

        variables.update(self._get_standard_potential_variables(phi_s))
        variables.update(self._get_standard_current_variables(i_s))

        if self.domain == "Positive":
            variables.update(self._get_standard_whole_cell_current_variables(variables))

        return variables

    def set_boundary_conditions(self, variables):

        phi_s = variables[self.domain + " electrode potential"]
        eps = variables["Leading-order " + self.domain.lower() + " electrode porosity"]
        i_boundary_cc = variables["Current collector current density"]

        if self.domain == "Negative":
            lbc = (pybamm.Scalar(0), "Dirichlet")
            rbc = (pybamm.Scalar(0), "Neumann")

        elif self.domain == "Positive":
            lbc = (pybamm.Scalar(0), "Neumann")
            sigma_eff = self.param.sigma_p * (1 - eps) ** self.param.b
            rbc = (
                i_boundary_cc / pybamm.boundary_value(-sigma_eff, "right"),
                "Neumann",
            )

        self.boundary_conditions[phi_s] = {"left": lbc, "right": rbc}

    @property
    def default_solver(self):
        """
        Create and return the default solver for this model
        """
        return pybamm.ScikitsOdeSolver()
