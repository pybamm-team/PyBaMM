#
# Full model for Ohm's law in the electrode
#
import pybamm

from .base_ohm import BaseOhm


class LeadingOhm(BaseOhm):
    """Leading-order model for ohm's law with conservation of current for the current
    in the electrodes.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        Either 'Negative electrode' or 'Positive electrode'

    *Extends:* :class:`pybamm.BaseOhm`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def get_derived_variables(self, variables):
        """
        Returns variables which are derived from the fundamental variables in the model.
        """
        i_boundary_cc = variables["Current collector current density"]

        # import parameters and spatial variables
        l_n = self.param.l_n
        l_p = self.param.l_p
        x_n = pybamm.standard_spatial_vars.x_n
        x_p = pybamm.standard_spatial_vars.x_p

        if self._domain == "Negative":
            phi_s = pybamm.Broadcast(0, ["negative electrode"])
            i_s = pybamm.outer(i_boundary_cc, 1 - x_n / l_n)

        elif self._domain == "Positive":
            ocp_p_av = variables["Average positive open circuit potential"]
            eta_r_p_av = variables["Average positive overpotential"]
            phi_e_p_av = variables["Average positive electrolyte potential"]

            v = ocp_p_av + eta_r_p_av + phi_e_p_av

            phi_s = pybamm.Broadcast(v, ["positive electrode"])
            i_s = pybamm.outer(i_boundary_cc, 1 - (1 - x_p) / l_p)

        else:
            pybamm.DomainError("Domain must be either: 'Negative' or 'Positive'")

        variables.update(self._get_standard_potential_variables(phi_s))
        variables.update(self._get_standard_current_variables(i_s))

        return variables

    @property
    def default_solver(self):
        """
        Create and return the default solver for this model
        """
        return pybamm.ScikitsOdeSolver()
