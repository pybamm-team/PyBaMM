#
# Bulter volmer class
#

import pybamm
import autograd.numpy as np
from .base_interface import BaseInterface


class BaseModel(BaseInterface):
    """
       Base submodel which implements the forward Butler-Volmer equation:

    .. math::
        j = j_0(c) * \\sinh(\\eta_r(c))

    Parameters
    ----------
    param :
        model parameters
    domain : str
        The domain to implement the model, either: 'Negative' or 'Positive'.


    **Extends:** :class:`pybamm.interface.BaseInterface`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def _get_delta_phi_s(self, variables):
        phi_s = variables[self.domain + " electrode potential"]
        phi_e = variables[self.domain + " electrolyte potential"]
        delta_phi_s = phi_s - phi_e
        delta_phi_s_av = pybamm.average(delta_phi_s)
        variables.update(
            self._get_standard_surface_potential_difference_variables(
                delta_phi_s, delta_phi_s_av
            )
        )
        return variables

    def get_coupled_variables(self, variables):
        i_boundary_cc = variables["Current collector current density"]
        # Calculate delta_phi_s from phi_s and phi_e if it isn't already known
        if self.domain + " electrode surface potential difference" not in variables:
            variables = self._get_delta_phi_s(variables)
        delta_phi_s = variables[self.domain + " electrode surface potential difference"]
        ocp = variables[self.domain + " electrode open circuit potential"]

        eta_r = delta_phi_s - ocp
        j0 = self._get_exchange_current_density(variables)

        if self.domain == "Negative":
            ne = self.param.ne_n
            j_av = i_boundary_cc / pybamm.geometric_parameters.l_n

        elif self.domain == "Positive":
            ne = self.param.ne_p
            j_av = -i_boundary_cc / pybamm.geometric_parameters.l_p

        j = 2 * j0 * pybamm.Function(np.sinh, (ne / 2) * eta_r)

        j0_av = pybamm.average(j0)

        # j = j_av + (j - pybamm.average(j))  # enforce true average

        eta_r_av = pybamm.average(eta_r)

        variables.update(self._get_standard_interfacial_current_variables(j, j_av))
        variables.update(self._get_standard_exchange_current_variables(j0, j0_av))
        variables.update(self._get_standard_overpotential_variables(eta_r, eta_r_av))

        if self.domain == "Positive":
            variables.update(
                self._get_standard_whole_cell_interfacial_current_variables(variables)
            )
            variables.update(
                self._get_standard_whole_cell_exchange_current_variables(variables)
            )

        return variables

    def _get_exchange_current_density(self, variables):
        raise NotImplementedError
