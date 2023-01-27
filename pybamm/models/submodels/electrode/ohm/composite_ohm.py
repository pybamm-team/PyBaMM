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
    options : dict, optional
        A dictionary of options to be passed to the model.

    **Extends:** :class:`pybamm.BaseOhm`
    """

    def __init__(self, param, domain, options=None):
        super().__init__(param, domain, options=options)

    def get_coupled_variables(self, variables):
        domain = self.domain
        param = self.param

        i_boundary_cc = variables["Current collector current density"]

        # import parameters and spatial variables
        l_n = param.n.l
        l_p = param.p.l
        x_n = pybamm.standard_spatial_vars.x_n
        x_p = pybamm.standard_spatial_vars.x_p

        tor = variables[f"X-averaged {domain} electrode transport efficiency"]
        phi_s_cn = variables["Negative current collector potential"]
        T = variables[f"X-averaged {domain} electrode temperature"]

        sigma_eff = self.domain_param.sigma(T) * tor
        if self._domain == "negative":
            phi_s = phi_s_cn + (i_boundary_cc / sigma_eff) * (
                x_n * (x_n - 2 * l_n) / (2 * l_n)
            )
            i_s = i_boundary_cc * (1 - x_n / l_n)

        elif self.domain == "positive":
            delta_phi_p_av = variables[
                "X-averaged positive electrode surface potential difference"
            ]
            phi_e_p_av = variables["X-averaged positive electrolyte potential"]

            const = (
                delta_phi_p_av
                + phi_e_p_av
                + (i_boundary_cc / sigma_eff) * (1 - l_p / 3)
            )

            phi_s = const - (i_boundary_cc / sigma_eff) * (
                x_p + (x_p - 1) ** 2 / (2 * l_p)
            )
            i_s = i_boundary_cc * (1 - (1 - x_p) / l_p)

        variables.update(self._get_standard_potential_variables(phi_s))
        variables.update(self._get_standard_current_variables(i_s))

        if self.domain == "positive":
            variables.update(self._get_standard_whole_cell_variables(variables))

        return variables

    def set_boundary_conditions(self, variables):
        domain, Domain = self.domain_Domain

        phi_s = variables[f"{Domain} electrode potential"]
        tor = variables[f"X-averaged {domain} electrode transport efficiency"]
        i_boundary_cc = variables["Current collector current density"]
        T = variables[f"X-averaged {domain} electrode temperature"]

        if self.domain == "negative":
            lbc = (pybamm.Scalar(0), "Dirichlet")
            rbc = (pybamm.Scalar(0), "Neumann")

        elif self.domain == "positive":
            lbc = (pybamm.Scalar(0), "Neumann")
            sigma_eff = self.param.p.sigma(T) * tor
            rbc = (-i_boundary_cc / sigma_eff, "Neumann")

        self.boundary_conditions[phi_s] = {"left": lbc, "right": rbc}
