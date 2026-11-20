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
    """

    def __init__(self, param, domain, options=None):
        super().__init__(param, domain, options=options)

    def build(self):
        coupled = self._coupled_current_collector_variable
        i_boundary_cc = coupled("Current collector current density [A.m-2]")
        sigma_eff = self._get_effective_conductivity(x_averaged=True)

        L_n = self.param.n.L
        L_p = self.param.p.L
        L_x = self.param.L_x
        x_n = pybamm.standard_spatial_vars.x_n
        x_p = pybamm.standard_spatial_vars.x_p

        if self.domain == "negative":
            phi_s_cn = coupled("Negative current collector potential [V]")
            phi_s = phi_s_cn + (i_boundary_cc / sigma_eff) * (
                x_n * (x_n - 2 * L_n) / (2 * L_n)
            )
            i_s = i_boundary_cc * (1 - x_n / L_n)
            lbc = (pybamm.Scalar(0), "Dirichlet")
            rbc = (pybamm.Scalar(0), "Neumann")
        else:
            delta_phi_p_av = self._get_x_averaged_surface_potential_difference()
            phi_e_p_av = coupled("X-averaged positive electrolyte potential [V]")
            const = (
                delta_phi_p_av
                + phi_e_p_av
                + (i_boundary_cc / sigma_eff) * (L_x - L_p / 3)
            )
            phi_s = const - (i_boundary_cc / sigma_eff) * (
                x_p + (x_p - L_x) ** 2 / (2 * L_p)
            )
            i_s = i_boundary_cc * (1 - (L_x - x_p) / L_p)
            lbc = (pybamm.Scalar(0), "Neumann")
            rbc = (-i_boundary_cc / sigma_eff, "Neumann")

        self.variables.update(self._get_standard_potential_variables(phi_s))
        self.variables.update(self._get_standard_current_variables(i_s))
        if self.domain == "positive":
            self.variables.update(self._get_standard_whole_cell_variables(i_s, phi_s))

        self.boundary_conditions[phi_s] = {"left": lbc, "right": rbc}
