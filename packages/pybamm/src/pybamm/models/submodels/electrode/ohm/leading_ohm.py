#
# Full model for Ohm's law in the electrode
#
import pybamm

from .base_ohm import BaseModel


class LeadingOrder(BaseModel):
    """An electrode submodel that employs Ohm's law the leading-order approximation to
    governing equations.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        Either 'negative' or 'positive'
    options : dict, optional
        A dictionary of options to be passed to the model.
    set_positive_potential :  bool, optional
        If True the battery model sets the positve potential based on the current.
        If False, the potential is specified by the user. Default is True.
    """

    def __init__(self, param, domain, options=None, set_positive_potential=True):
        super().__init__(
            param,
            domain,
            options=options,
            set_positive_potential=set_positive_potential,
        )

    def build(self):
        coupled = self._coupled_current_collector_variable
        i_boundary_cc = coupled("Current collector current density [A.m-2]")

        L_n = self.param.n.L
        L_p = self.param.p.L
        L_x = self.param.L_x
        x_n = pybamm.standard_spatial_vars.x_n
        x_p = pybamm.standard_spatial_vars.x_p

        if self.domain == "negative":
            phi_s_cn = coupled("Negative current collector potential [V]")
            phi_s = pybamm.PrimaryBroadcast(phi_s_cn, "negative electrode")
            i_s = i_boundary_cc * (1 - x_n / L_n)
        else:
            # recall delta_phi = phi_s - phi_e
            delta_phi_p_av = self._get_x_averaged_surface_potential_difference()
            phi_e_p_av = coupled("X-averaged positive electrolyte potential [V]")
            v = delta_phi_p_av + phi_e_p_av
            phi_s = pybamm.PrimaryBroadcast(v, ["positive electrode"])
            i_s = i_boundary_cc * (1 - (L_x - x_p) / L_p)

        self.variables.update(self._get_standard_potential_variables(phi_s))
        self.variables.update(self._get_standard_current_variables(i_s))
        if self.domain == "positive":
            self.variables.update(self._get_standard_whole_cell_variables(i_s, phi_s))

        self.boundary_conditions[phi_s] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (pybamm.Scalar(0), "Neumann"),
        }
