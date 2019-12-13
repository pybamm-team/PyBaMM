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
        Either 'Negative' or 'Positive'
    set_positive_potential :  bool, optional
        If True the battery model sets the positve potential based on the current.
        If False, the potential is specified by the user. Default is True.

    **Extends:** :class:`pybamm.electrode.ohm.BaseModel`
    """

    def __init__(self, param, domain, set_positive_potential=True):
        super().__init__(param, domain, set_positive_potential=set_positive_potential)

    def get_coupled_variables(self, variables):
        """
        Returns variables which are derived from the fundamental variables in the model.
        """
        i_boundary_cc = variables["Current collector current density"]
        phi_s_cn = variables["Negative current collector potential"]

        # import parameters and spatial variables
        l_n = self.param.l_n
        l_p = self.param.l_p
        x_n = pybamm.standard_spatial_vars.x_n
        x_p = pybamm.standard_spatial_vars.x_p

        if self.domain == "Negative":
            phi_s = pybamm.PrimaryBroadcast(phi_s_cn, "negative electrode")
            i_s = i_boundary_cc * (1 - x_n / l_n)

        elif self.domain == "Positive":
            # recall delta_phi = phi_s - phi_e
            delta_phi_p_av = variables[
                "X-averaged positive electrode surface potential difference"
            ]
            phi_e_p_av = variables["X-averaged positive electrolyte potential"]

            v = delta_phi_p_av + phi_e_p_av

            phi_s = pybamm.PrimaryBroadcast(v, ["positive electrode"])
            i_s = i_boundary_cc * (1 - (1 - x_p) / l_p)

        variables.update(self._get_standard_potential_variables(phi_s))
        variables.update(self._get_standard_current_variables(i_s))

        if self.domain == "Positive":
            variables.update(self._get_standard_whole_cell_variables(variables))

        return variables

    def set_boundary_conditions(self, variables):

        phi_s = variables[self.domain + " electrode potential"]

        lbc = (pybamm.Scalar(0), "Neumann")
        rbc = (pybamm.Scalar(0), "Neumann")

        self.boundary_conditions[phi_s] = {"left": lbc, "right": rbc}
